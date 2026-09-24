#!/usr/bin/env python3
"""Rewrite a measured sensor_noise.yaml to carry the VSLAM configs' IMU noise.

    python3 script/make_vslam_noise_yaml.py --preset inflated \
        --noise output/compare/simulation/exp-estimator-baseline/noise

    # then, per bag, with the EKF reading the rewritten model:
    python3 script/proprio_estimator.py run --estimator ekf \
        --bag <bag.db3> --noise <.../sensor_noise_inflated.yaml> \
        --out <run-dir> --allow-foreign-noise

WHY THIS EXISTS
    Table 4.4 compares an EKF whose noise was MEASURED from each bag against
    VSLAM whose noise was hand-set in a config file. Tuning is therefore an
    uncontrolled variable between the two estimator families: any difference in
    the table could be the estimator or could be the tuning. Re-running VSLAM at
    the EKF's settings needs Gazebo and a fresh log. Re-running the EKF at the
    VSLAM settings needs neither -- it reads the existing bags offline -- so this
    is the cheap half of the control, and it is the half that can be done before
    Wednesday.

    Only the four IMU noise fields move. Biases, wheel scales and wheel sigmas
    stay as measured, so the IMU noise model is the single treatment.

UNITS -- THE PART THAT IS EASY TO GET WRONG
    VINS and ORB state noise as a DENSITY per sqrt(Hz). proprio_estimator.py
    stores a PER-SAMPLE sigma, because its process-noise block multiplies by dt
    rather than sqrt(dt):

        G[2,1] = dt,  var contribution = sigma^2 dt^2 = (density^2/dt) dt^2

    which is the correct density^2*dt only when sigma = density / sqrt(dt).
    So sigma = density * sqrt(f), f = 200 Hz here, taken from the yaml's own
    imu_dt_s rather than assumed.

    gyro_bias_rw is the exception: it enters through G[4,2] = sqrt(dt), which is
    already the density convention, so gyr_w transfers with no conversion.

THE PRESETS ARE NOT WHAT THEIR NAMES SUGGEST
    'inflated' is the setting every reported VSLAM run used. Against the measured
    noise it is 6.5x LOOSE on the gyro but about 0.8x on the accelerometer --
    i.e. slightly TIGHT. It is not a uniform loosening, and a result from it
    should not be read as one.

    'truenoise' is the SDF's declared densities. The accelerometer figure there
    (0.004) matches the measured y and z axes but is 5.4x below the measured x
    axis, which is the driving axis and the one the filter integrates. See the
    note printed by --show.
"""

import argparse
import glob
import math
import os
import sys

import yaml

# Both VINS and ORB carry the SAME four values in each condition -- the ORB keys
# are NoiseGyro/NoiseAcc/GyroWalk/AccWalk but the numbers are identical -- so
# there is one VSLAM noise setting per condition, not two.
PRESETS = {
    # vins_fusion_ros2/config/wil_sim/stereo_imu.yaml
    # orbslam3_ros2/config/wil_sim/stereo_imu.yaml
    'inflated':  dict(gyr_n=0.002,      acc_n=0.02,      gyr_w=0.0001),
    # *_truenoise.yaml, taken from the SDF's declared densities
    'truenoise': dict(gyr_n=0.000339411, acc_n=0.00400222, gyr_w=0.00000173452),
}


def convert(noise, preset):
    """Return the yaml with only the IMU noise fields replaced."""
    p = PRESETS[preset]
    dt = float(noise['window']['imu_dt_s'])
    f = 1.0 / dt
    sg = p['gyr_n'] * math.sqrt(f)
    sa = p['acc_n'] * math.sqrt(f)

    imu = noise['imu']
    before = dict(gyro_sigma_xyz=list(imu['gyro_sigma_xyz']),
                  accel_sigma_xyz=list(imu['accel_sigma_xyz']),
                  gyro_bias_rw=imu['gyro_bias_rw'])

    # The EKF reads index 2 of the gyro and index 0 of the accelerometer, but all
    # three axes are set so the file cannot be misread as partly measured.
    imu['gyro_sigma_xyz'] = [sg, sg, sg]
    imu['accel_sigma_xyz'] = [sa, sa, sa]
    imu['gyro_noise_density'] = [p['gyr_n']] * 3
    imu['accel_noise_density'] = [p['acc_n']] * 3
    imu['gyro_bias_rw'] = p['gyr_w']
    imu['gyro_bias_rw_source'] = 'VSLAM config (%s preset), not measured' % preset

    noise['provenance']['note'] = (
        'IMU NOISE OVERRIDDEN from the %s VSLAM config by '
        'script/make_vslam_noise_yaml.py. Biases and the whole wheel block are '
        'still the measured values from this bag. Run with '
        '--allow-foreign-noise; the transfer IS the experiment.' % preset)
    noise['provenance']['override_preset'] = preset
    noise['provenance']['override_source'] = (
        'vins_fusion_ros2/config/wil_sim/stereo_imu%s.yaml'
        % ('' if preset == 'inflated' else '_truenoise'))
    return before, dict(sg=sg, sa=sa, gyr_w=p['gyr_w'])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--preset', choices=sorted(PRESETS), required=True)
    ap.add_argument('--noise', required=True,
                    help='a sensor_noise.yaml, or a directory searched recursively')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    if os.path.isdir(args.noise):
        files = sorted(glob.glob(os.path.join(args.noise, '**', 'sensor_noise.yaml'),
                                 recursive=True))
    else:
        files = [args.noise]
    if not files:
        raise SystemExit('no sensor_noise.yaml under %s' % args.noise)

    for src in files:
        noise = yaml.safe_load(open(src))
        before, after = convert(noise, args.preset)
        dst = src.replace('sensor_noise.yaml', 'sensor_noise_%s.yaml' % args.preset)
        tag = os.path.basename(os.path.dirname(src))
        print('%s' % tag)
        print('  gyro  sigma_z  %.6f -> %.6f rad/s      (x%.2f)'
              % (before['gyro_sigma_xyz'][2], after['sg'],
                 after['sg'] / before['gyro_sigma_xyz'][2]))
        print('  accel sigma_x  %.6f -> %.6f m/s^2      (x%.2f)'
              % (before['accel_sigma_xyz'][0], after['sa'],
                 after['sa'] / before['accel_sigma_xyz'][0]))
        print('  gyro bias rw   %.3g -> %.3g rad/s^2/sqrt(Hz)  (x%.1f)'
              % (before['gyro_bias_rw'], after['gyr_w'],
                 after['gyr_w'] / before['gyro_bias_rw']))
        if not args.dry_run:
            with open(dst, 'w') as f:
                yaml.safe_dump(noise, f, sort_keys=False, default_flow_style=False)
            print('  wrote %s' % os.path.relpath(dst))

    if args.dry_run:
        print('\n--dry-run: nothing written')
    else:
        print('\nRun the EKF against these with --allow-foreign-noise, into a NEW '
              'output directory. Do not overwrite exp-estimator-baseline.')


if __name__ == '__main__':
    sys.exit(main())
