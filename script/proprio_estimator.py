#!/usr/bin/env python3
"""Non-visual baseline estimators: wheel odometry, IMU dead reckoning, and an EKF.

    # 1. derive the noise model from the data (writes sensor_noise.yaml)
    python3 script/proprio_estimator.py calibrate --bag <bag.db3> --out <dir>

    # 2. run one estimator against that noise model
    python3 script/proprio_estimator.py run --estimator ekf \\
        --bag <bag.db3> --noise <sensor_noise.yaml> --out <run-dir>

WHY THIS EXISTS
    Chapter 4 currently reports VIO error against ground truth and nothing else,
    which tells a reader that ORB-SLAM3 was off by X metres but not whether X is
    good. A dead-reckoning floor supplies the missing scale: proprioception alone,
    no cameras, same bag, same ground truth. VIO that cannot beat it has not
    earned its compute.

WHY OFFLINE AND NOT A LIVE ROS NODE
    These three estimators are deterministic functions of the recorded data. Run
    live they would instead depend on replay rate, QoS drops and whatever else is
    contending for the CPU -- three sources of run-to-run variation that have
    nothing to do with the estimator being compared. Offline, the same bag gives
    the same numbers on every machine, which is what a baseline has to do.

    The cost is a threat to validity worth stating in the report: the VIO figures
    these are compared against WERE produced live, so any timing-induced error in
    those is present on one side of the comparison only.

THE ORIENTATION TRAP, WHICH IS THE ONE BUG THAT WOULD INVALIDATE EVERYTHING
    sensor_msgs/Imu.orientation in these bags is not an estimate. Measured over
    dataset_allsensor_000, it equals ground-truth yaw minus exactly 90 degrees,
    residual std 0.00000 deg, drift 0.00000 deg over 87 s. Gazebo's IMU plugin
    reads the true orientation out of the ECM and publishes it.

    Consuming that field would hand the "pure IMU" estimator a perfect, noise-free
    attitude reference that no real IMU has, and the resulting trajectory would be
    a measurement of ground truth, not of dead reckoning. So this file NEVER reads
    msg.orientation. Yaw comes from integrating angular_velocity.z, which does
    drift honestly -- 2.52 deg over 87 s on bag 000, 1.70 deg over 81.8 s on 001.

    If you ever see a pure-IMU heading that does not drift, this rule has been
    broken somewhere.

WHY THE COVARIANCES ARE DERIVED AND NOT READ
    Every covariance field in these bags is identically zero: 0 of 17416 IMU
    messages, 0 of 4355 wheel-odometry messages, 0 of 4354 ground-truth messages
    on dataset_allsensor_000. Gazebo Fortress does not populate them, and
    AckermannSteering has no noise parameter to populate them from.

    So `calibrate` measures the noise instead -- against ground truth, from the
    same bag the estimator will run on -- and writes it to a YAML the estimators
    read. That keeps the filter's R matrix an observed quantity with a stated
    provenance rather than a number chosen to make the output look good.

WHICH WHEEL ODOMETRY
    Three things in these bags could be called wheel odometry, and they are not
    equivalent:

      /model/../odometry        the AckermannSteering plugin integrating its own
                                exact joint states. Reproduces ground-truth path
                                length to 0.06%. This is the simulator marking its
                                own homework, not an encoder. Available here as
                                --wheel-source bagged, for reference only.

      joint_state + differential  rear-wheel speed difference over the track. Sounds
                                standard, measures badly: yaw-rate scale 1.22/1.14 and
                                residual 0.24/0.31 rad/s against ground truth, because
                                a 0.42 m track with a solid rear axle slips through
                                every turn.

      joint_state + bicycle     rear wheels for speed, steering joints for yaw rate.
                                Scale 1.011/0.953, residual 0.066/0.080 rad/s -- three
                                to four times better, and the default here. calibrate
                                picked it on both allsensor bags.

    `calibrate` reports all three so the choice stays evidence-based rather than
    inherited from this docstring.
"""

import argparse
import datetime
import json
import math
import os
import sqlite3
import subprocess
import sys

import numpy as np

try:
    import yaml
except ImportError:
    sys.exit("PyYAML missing. source /opt/ros/humble/setup.bash first.")

from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Geometry, from aws-robomaker-small-warehouse-world/models/ackermann_robot/model.sdf.
# Read off the <ackermann_steering> block, not guessed: kingpin_width 0.42,
# wheel_base 0.42, wheel_separation 0.42, wheel_radius 0.0585, steering_limit 0.6109.
WHEEL_RADIUS = 0.0585           # m
WHEEL_BASE = 0.42               # m, front axle to rear axle
WHEEL_SEPARATION = 0.42         # m, track width

T_IMU = '/imu'
T_GT = '/ground_truth/odometry'
T_JOINT = '/model/ackermann_robot_001/joint_state'
T_WHEEL_ODOM = '/model/ackermann_robot_001/odometry'

# Below this ground-truth speed the robot is treated as stationary, which is how
# the static window for bias estimation is found. 0.02 m/s is well under the
# 1.34 m/s mean driving speed and above the numerical noise in differenced poses.
STATIC_SPEED = 0.02             # m/s

# Gravity is removed from the body-x accelerometer channel only to the extent that
# the platform is level. Measured roll and pitch stay inside +/-0.002 deg for both
# bags, so the leak into body x is under 3.5e-4 m/s^2 and is ignored. If a future
# dataset has a ramp in it, this assumption has to be revisited, so it is checked
# at run time rather than assumed.
MAX_TILT_DEG = 1.0


# ---------------------------------------------------------------------------
# bag reading
# ---------------------------------------------------------------------------

def read_topic(bag, topic):
    """All messages on `topic`, in bag order, as (header_time_s, msg).

    Header stamps, not bag timestamps. The bag timestamp is wall time at the
    moment of recording; the header stamp is sim time, which is the only clock
    the ground truth and the estimate share. These bags were recorded at about
    0.67x real time, so the two differ by tens of seconds and mixing them would
    silently destroy every association.
    """
    con = sqlite3.connect(bag)
    try:
        row = con.execute("SELECT id,type FROM topics WHERE name=?", (topic,)).fetchone()
        if row is None:
            raise SystemExit("topic %s not in %s" % (topic, bag))
        tid, typename = row
        mt = get_message(typename)
        out = []
        for (blob,) in con.execute(
                "SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp", (tid,)):
            m = deserialize_message(bytes(blob), mt)
            h = m.header.stamp
            out.append((h.sec + h.nanosec * 1e-9, m))
        return out
    finally:
        con.close()


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def tilt_deg(q):
    roll = math.atan2(2.0 * (q.w * q.x + q.y * q.z), 1.0 - 2.0 * (q.x * q.x + q.y * q.y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (q.w * q.y - q.z * q.x))))
    return math.degrees(abs(roll)), math.degrees(abs(pitch))


def load_gt(bag):
    """Ground truth as a dict of arrays: t, x, y, yaw (unwrapped), v, w."""
    msgs = read_topic(bag, T_GT)
    t = np.array([a for a, _ in msgs])
    x = np.array([m.pose.pose.position.x for _, m in msgs])
    y = np.array([m.pose.pose.position.y for _, m in msgs])
    z = np.array([m.pose.pose.position.z for _, m in msgs])
    yaw = np.unwrap(np.array([yaw_of(m.pose.pose.orientation) for _, m in msgs]))
    quat = np.array([[m.pose.pose.orientation.w, m.pose.pose.orientation.x,
                      m.pose.pose.orientation.y, m.pose.pose.orientation.z]
                     for _, m in msgs])
    # The twist fields ARE populated here (4321 of 4354 rows nonzero on bag 000;
    # the zeros are the stationary samples), so they are used directly rather than
    # differentiated. Differentiated position is kept as a cross-check in calibrate.
    v = np.array([m.twist.twist.linear.x for _, m in msgs])
    w = np.array([m.twist.twist.angular.z for _, m in msgs])
    return dict(t=t, x=x, y=y, z=z, yaw=yaw, quat=quat, v=v, w=w)


def load_imu(bag):
    msgs = read_topic(bag, T_IMU)
    t = np.array([a for a, _ in msgs])
    acc = np.array([[m.linear_acceleration.x, m.linear_acceleration.y,
                     m.linear_acceleration.z] for _, m in msgs])
    gyr = np.array([[m.angular_velocity.x, m.angular_velocity.y,
                     m.angular_velocity.z] for _, m in msgs])
    # Orientation is read ONLY to assert the tilt assumption and to prove in the
    # log that it was not used for estimation. See the module docstring.
    tilt = np.array([tilt_deg(m.orientation) for _, m in msgs])
    return dict(t=t, acc=acc, gyr=gyr, tilt=tilt)


def load_joints(bag):
    """Wheel speed and steering angle derived from the 1 kHz joint feed."""
    msgs = read_topic(bag, T_JOINT)
    names = list(msgs[0][1].name)
    iL, iR = names.index('rear_left_wheel_joint'), names.index('rear_right_wheel_joint')
    iSL = names.index('front_left_steering_joint')
    iSR = names.index('front_right_steering_joint')
    t = np.array([a for a, _ in msgs])
    pos = np.array([[m.position[iL], m.position[iR]] for _, m in msgs])
    vel = np.array([[m.velocity[iL], m.velocity[iR]] for _, m in msgs])
    steer = np.array([0.5 * (m.position[iSL] + m.position[iSR]) for _, m in msgs])

    # Speed from the reported velocity field. Differencing positions gives the same
    # answer to 7e-5 m/s here (scale 0.978866 vs 0.978795 against ground truth), so
    # the simpler path is taken; encoder quantisation is a separate factor and
    # belongs in encoder_sim.py, not silently here.
    v = WHEEL_RADIUS * vel.mean(axis=1)

    # Yaw rate, both ways, so the caller can pick on evidence.
    dt = np.diff(t, prepend=t[0] - 1e-3)
    dt[dt <= 0] = 1e-3
    dpos = np.diff(pos, axis=0, prepend=pos[:1])
    w_diff = WHEEL_RADIUS * (dpos[:, 1] - dpos[:, 0]) / dt / WHEEL_SEPARATION
    w_bicycle = v * np.tan(steer) / WHEEL_BASE
    return dict(t=t, v=v, steer=steer, w_diff=w_diff, w_bicycle=w_bicycle,
                names=names)


def load_bagged_odom(bag):
    msgs = read_topic(bag, T_WHEEL_ODOM)
    t = np.array([a for a, _ in msgs])
    x = np.array([m.pose.pose.position.x for _, m in msgs])
    y = np.array([m.pose.pose.position.y for _, m in msgs])
    yaw = np.unwrap(np.array([yaw_of(m.pose.pose.orientation) for _, m in msgs]))
    v = np.array([m.twist.twist.linear.x for _, m in msgs])
    w = np.array([m.twist.twist.angular.z for _, m in msgs])
    return dict(t=t, x=x, y=y, yaw=yaw, v=v, w=w)


# ---------------------------------------------------------------------------
# calibrate
# ---------------------------------------------------------------------------

def static_window(gt):
    """(t_start, t_end) of the stationary period at the head of the run."""
    moving = np.where(np.abs(gt['v']) > STATIC_SPEED)[0]
    if len(moving) == 0:
        raise SystemExit("ground truth never exceeds %.3f m/s -- is this the right bag?"
                         % STATIC_SPEED)
    return gt['t'][0], gt['t'][moving[0]]


def fit_scale(ref, meas):
    """Least-squares gain of meas on ref through the origin, and residual sigma.

    Through the origin deliberately: an offset term would absorb a real sensor bias
    into the fit and report a scale error that is partly bias. For a rate sensor
    the physical model is meas = k*ref + noise with k=1 when the geometry is right.
    """
    k = float(np.dot(ref, meas) / np.dot(ref, ref))
    resid = meas - k * ref
    return k, float(np.std(resid))


def calibrate(args):
    gt, imu, js = load_gt(args.bag), load_imu(args.bag), load_joints(args.bag)
    t0s, t1s = static_window(gt)
    n_static = int(((imu['t'] >= t0s) & (imu['t'] < t1s)).sum())
    sm = (imu['t'] >= t0s) & (imu['t'] < t1s)

    tilt_max = float(imu['tilt'].max())
    dt_imu = float(np.median(np.diff(imu['t'])))

    # --- IMU, from the static window -------------------------------------------
    # Short: 0.66 s / 134 samples on bag 000. Enough for a noise sigma, NOT enough
    # for Allan-variance bias stability, so no bias-instability figure is claimed.
    acc_s, gyr_s = imu['acc'][sm], imu['gyr'][sm]
    gyro_bias = gyr_s.mean(axis=0)
    gyro_sigma = gyr_s.std(axis=0)
    acc_bias = acc_s.mean(axis=0).copy()
    gravity = float(acc_bias[2])
    acc_bias[2] -= gravity          # z carries gravity, not bias
    acc_sigma = acc_s.std(axis=0)

    # --- IMU, referenced to ground truth over the whole run ---------------------
    # The static window cannot see scale error or motion-dependent noise. This can.
    gw = np.interp(imu['t'], gt['t'], gt['w'])
    k_gyro, s_gyro_run = fit_scale(gw, imu['gyr'][:, 2])
    # Ground-truth forward acceleration by differentiating the twist.
    a_gt = np.gradient(gt['v'], gt['t'])
    a_gt_i = np.interp(imu['t'], gt['t'], a_gt)
    k_acc, s_acc_run = fit_scale(a_gt_i, imu['acc'][:, 0] - acc_bias[0])

    # --- wheel, referenced to ground truth --------------------------------------
    moving = np.abs(gt['v']) > 0.05
    v_w = np.interp(gt['t'], js['t'], js['v'])
    w_bi = np.interp(gt['t'], js['t'], js['w_bicycle'])
    w_di = np.interp(gt['t'], js['t'], js['w_diff'])
    k_v, s_v = fit_scale(gt['v'][moving], v_w[moving])
    k_wb, s_wb = fit_scale(gt['w'][moving], w_bi[moving])
    k_wd, s_wd = fit_scale(gt['w'][moving], w_di[moving])

    chosen = 'bicycle' if s_wb <= s_wd else 'differential'

    noise = {
        'provenance': {
            'bag': os.path.abspath(args.bag),
            'generated': datetime.datetime.now().isoformat(timespec='seconds'),
            'script': os.path.abspath(__file__),
            'git_commit': git_commit(ROOT),
            'note': ('Derived from data. Every covariance field in the source bag '
                     'is identically zero; Gazebo Fortress does not populate them.'),
        },
        'window': {
            'static_start_s': float(t0s), 'static_end_s': float(t1s),
            'static_duration_s': float(t1s - t0s), 'static_imu_samples': n_static,
            'run_start_s': float(gt['t'][0]), 'run_end_s': float(gt['t'][-1]),
            'imu_dt_s': dt_imu, 'max_tilt_deg': tilt_max,
        },
        'imu': {
            'gyro_bias_xyz': [float(v) for v in gyro_bias],
            'gyro_sigma_xyz': [float(v) for v in gyro_sigma],
            'gyro_noise_density': [float(v * math.sqrt(dt_imu)) for v in gyro_sigma],
            'accel_bias_xyz': [float(v) for v in acc_bias],
            'accel_sigma_xyz': [float(v) for v in acc_sigma],
            'accel_noise_density': [float(v * math.sqrt(dt_imu)) for v in acc_sigma],
            'gravity_z': gravity,
            'gyro_z_scale_vs_gt': k_gyro, 'gyro_z_sigma_vs_gt': s_gyro_run,
            'accel_x_scale_vs_gt': k_acc, 'accel_x_sigma_vs_gt': s_acc_run,
            # Random walk is not observable in a 0.66 s window. This is a stated
            # assumption, not a measurement, and is flagged as such so it cannot be
            # quoted in the report as an Allan-variance result.
            'gyro_bias_rw': args.gyro_bias_rw,
            'gyro_bias_rw_source': 'assumed (static window too short to measure)',
        },
        'wheel': {
            'speed_scale': k_v, 'speed_sigma': s_v,
            'yaw_rate_model': chosen,
            'yaw_rate_scale_bicycle': k_wb, 'yaw_rate_sigma_bicycle': s_wb,
            'yaw_rate_scale_differential': k_wd, 'yaw_rate_sigma_differential': s_wd,
            'wheel_radius': WHEEL_RADIUS, 'wheel_base': WHEEL_BASE,
            'wheel_separation': WHEEL_SEPARATION,
        },
    }

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, 'sensor_noise.yaml')
    with open(path, 'w') as f:
        yaml.safe_dump(noise, f, sort_keys=False, default_flow_style=False)

    print("static window      %.3f -> %.3f s  (%.2f s, %d IMU samples)"
          % (t0s, t1s, t1s - t0s, n_static))
    print("max tilt           %.4f deg  (planar assumption %s)"
          % (tilt_max, "holds" if tilt_max < MAX_TILT_DEG else "VIOLATED"))
    print()
    print("IMU (static window)")
    print("  gyro  bias  %s rad/s" % np.array2string(gyro_bias, precision=6))
    print("  gyro  sigma %s rad/s" % np.array2string(gyro_sigma, precision=6))
    print("  accel bias  %s m/s^2" % np.array2string(acc_bias, precision=6))
    print("  accel sigma %s m/s^2" % np.array2string(acc_sigma, precision=6))
    print("  gravity_z   %.5f m/s^2" % gravity)
    print("IMU (vs ground truth, whole run)")
    print("  gyro_z  scale %.6f  sigma %.6f rad/s" % (k_gyro, s_gyro_run))
    print("  accel_x scale %.6f  sigma %.6f m/s^2" % (k_acc, s_acc_run))
    print()
    print("WHEEL (vs ground truth, moving samples only)")
    print("  speed          scale %.6f  sigma %.6f m/s" % (k_v, s_v))
    print("  yaw bicycle    scale %.6f  sigma %.6f rad/s" % (k_wb, s_wb))
    print("  yaw differential scale %.6f  sigma %.6f rad/s" % (k_wd, s_wd))
    print("  -> yaw_rate_model: %s" % chosen)
    print()
    print("wrote %s" % path)


# ---------------------------------------------------------------------------
# estimators
# ---------------------------------------------------------------------------

def anchor(gt, t_start):
    """Initial (x, y, yaw) taken from ground truth at the estimator's first sample.

    Dead reckoning has no global reference: it can only say how the pose CHANGED.
    Anchoring the first pose to ground truth is the standard convention and is what
    makes the resulting error a measurement of drift rather than of an arbitrary
    frame choice. It is stated in run_metadata.csv because it is an alignment
    decision, not a neutral detail -- it gives all three estimators a free, exact
    starting pose that a real deployment would have to obtain some other way.
    """
    i = int(np.searchsorted(gt['t'], t_start))
    i = min(i, len(gt['t']) - 1)
    return float(gt['x'][i]), float(gt['y'][i]), float(gt['yaw'][i]), float(gt['z'][i])


def run_wheel(bag, noise, gt, args):
    """Integrate wheel speed and wheel yaw rate. No IMU, no correction.

    Covariance is propagated with the textbook odometry model: the pose Jacobian
    times the previous covariance, plus the control noise mapped through the motion
    model. Because nothing ever corrects it, P grows without bound -- which is the
    honest statement about dead reckoning and is exactly what the consistency test
    later checks against the real error.

    THE SCALE ERROR IS DELIBERATELY NOT CORRECTED
        calibrate measures the wheel speed and yaw-rate scales against ground truth.
        Measured: speed scale 1.0002 (bag 000) and 0.9985 (001) -- the wheel speed is
        essentially unbiased -- while the yaw-rate scale is 1.011 and 0.953, i.e.
        wrong by about 1% one way on one bag and 5% the other way on the other. The
        drift here is therefore dominated by HEADING, not by distance travelled, and
        the two bags should bend in opposite directions.

        Those numbers are RECORDED in sensor_noise.yaml and used to size the noise,
        but they are never divided out here. Correcting them would mean calibrating
        the estimator with the ground truth it is about to be scored against, and the
        resulting trajectory would flatter the baseline by exactly the amount of
        information smuggled in. The scale error is a real property of encoder
        odometry and it stays in.
    """
    if args.wheel_source == 'bagged':
        # The plugin's own integration, passed through unchanged. Reported for
        # reference only: it reproduces ground-truth path length to 0.06% because it
        # integrates the simulator's exact joint states with the exact geometry, so
        # it measures the plugin, not an encoder. Never quote it as "wheel odometry"
        # in a results table without that caveat attached.
        bo = load_bagged_odom(bag)
        rows = [(bo['t'][k], bo['x'][k], bo['y'][k], 0.0, bo['yaw'][k],
                 bo['v'][k], bo['w'][k]) for k in range(len(bo['t']))]
        nan = float('nan')
        cov = [(bo['t'][k],) + (nan,) * 8 for k in range(len(bo['t']))]
        return rows, cov, dict(wheel_source='bagged', updates=0,
                               covariance='none (plugin publishes zeros)')

    js = load_joints(bag)
    model = args.yaw_model or noise['wheel']['yaw_rate_model']
    w_src = js['w_bicycle'] if model == 'bicycle' else js['w_diff']

    sv = noise['wheel']['speed_sigma']
    sw = (noise['wheel']['yaw_rate_sigma_bicycle'] if model == 'bicycle'
          else noise['wheel']['yaw_rate_sigma_differential'])

    x, y, th, z0 = anchor(gt, js['t'][0])
    P = np.zeros((3, 3))
    rows, cov = [], []
    t_prev = js['t'][0]
    for k in range(len(js['t'])):
        t = js['t'][k]
        dt = t - t_prev
        t_prev = t
        if dt <= 0 or dt > 0.5:
            continue
        v, w = float(js['v'][k]), float(w_src[k])

        # Midpoint heading: integrating with the pre-update heading biases every
        # turn outward by w*dt/2, which over 117 m of driving is not a rounding
        # error. Costs nothing to do properly.
        th_mid = th + 0.5 * w * dt
        F = np.array([[1.0, 0.0, -v * math.sin(th_mid) * dt],
                      [0.0, 1.0, v * math.cos(th_mid) * dt],
                      [0.0, 0.0, 1.0]])
        B = np.array([[math.cos(th_mid) * dt, 0.0],
                      [math.sin(th_mid) * dt, 0.0],
                      [0.0, dt]])
        Q = B @ np.diag([sv ** 2, sw ** 2]) @ B.T
        P = F @ P @ F.T + Q

        x += v * math.cos(th_mid) * dt
        y += v * math.sin(th_mid) * dt
        th += w * dt
        rows.append((t, x, y, z0, th, v, w))
        cov.append((t, P[0, 0], P[1, 1], P[2, 2], float('nan'), float('nan'),
                    P[0, 1], P[0, 2], P[1, 2]))
    return rows, cov, dict(yaw_rate_model=model, speed_sigma=sv, yaw_sigma=sw,
                           updates=0)


def run_imu(bag, noise, gt, args):
    """Strapdown dead reckoning from the gyro and accelerometer alone.

    Reads angular_velocity and linear_acceleration. Does NOT read orientation --
    see the module docstring; that field is ground truth in disguise.

    Expect this to diverge badly. Double-integrated accelerometer error grows as
    t^2 even before the heading drifts, and heading drift turns speed error into
    position error in a direction that keeps changing. A large number here is the
    correct result, not a bug, and is the reason visual-inertial fusion exists.
    """
    imu = load_imu(bag)
    gb = np.array(noise['imu']['gyro_bias_xyz'])
    ab = np.array(noise['imu']['accel_bias_xyz'])
    sg = float(noise['imu']['gyro_sigma_xyz'][2])
    sa = float(noise['imu']['accel_sigma_xyz'][0])

    tilt_max = float(imu['tilt'].max())
    if tilt_max > MAX_TILT_DEG:
        raise SystemExit(
            "max tilt %.3f deg exceeds %.1f deg: the planar assumption that lets "
            "body-x be treated as gravity-free no longer holds. Gravity projection "
            "must be added before this bag can be used." % (tilt_max, MAX_TILT_DEG))

    x, y, th, z0 = anchor(gt, imu['t'][0])
    v = 0.0                       # the robot is stationary at the head of both bags
    P = np.zeros((4, 4))          # [x, y, theta, v]
    rows, cov = [], []
    t_prev = imu['t'][0]
    for k in range(len(imu['t'])):
        t = imu['t'][k]
        dt = t - t_prev
        t_prev = t
        if dt <= 0 or dt > 0.5:
            continue
        w = float(imu['gyr'][k, 2] - gb[2])
        a = float(imu['acc'][k, 0] - ab[0])

        th_mid = th + 0.5 * w * dt
        F = np.eye(4)
        F[0, 2] = -v * math.sin(th_mid) * dt
        F[0, 3] = math.cos(th_mid) * dt
        F[1, 2] = v * math.cos(th_mid) * dt
        F[1, 3] = math.sin(th_mid) * dt
        G = np.zeros((4, 2))
        G[0, 0] = 0.5 * math.cos(th_mid) * dt * dt
        G[1, 0] = 0.5 * math.sin(th_mid) * dt * dt
        G[3, 0] = dt
        G[2, 1] = dt
        P = F @ P @ F.T + G @ np.diag([sa ** 2, sg ** 2]) @ G.T

        x += v * math.cos(th_mid) * dt + 0.5 * a * math.cos(th_mid) * dt * dt
        y += v * math.sin(th_mid) * dt + 0.5 * a * math.sin(th_mid) * dt * dt
        v += a * dt
        th += w * dt
        rows.append((t, x, y, z0, th, v, w))
        cov.append((t, P[0, 0], P[1, 1], P[2, 2], P[3, 3], float('nan'),
                    P[0, 1], P[0, 2], P[1, 2]))
    return rows, cov, dict(gyro_bias_z=float(gb[2]), accel_bias_x=float(ab[0]),
                           max_tilt_deg=tilt_max, updates=0)


def run_ekf(bag, noise, gt, args):
    """EKF over [x, y, theta, v, b_gyro], IMU predicting and wheel correcting.

    STATE CHOICE
        Five states, planar. Roll and pitch stay inside 0.002 deg across both bags,
        so a full 15-state INS would spend nine states estimating quantities the
        platform does not have, and their covariances would be unobservable noise
        in the consistency test that is the point of the experiment.

    WHO DOES WHAT
        predict, 200 Hz IMU:  heading from the gyro minus its estimated bias, speed
                              from the accelerometer.
        update, 50 Hz wheel:  speed from the wheel encoders corrects v directly.
                              The wheel yaw rate does not correct theta -- it
                              corrects the GYRO BIAS, via the residual
                              (gyro_z - w_wheel), which is what b_gyro means. That
                              is the only reason the bias is observable at all.

    WHY THE BIAS UPDATE IS WRITTEN THIS WAY
        Correcting theta directly from a wheel-derived heading would require
        integrating the wheel yaw rate into a heading first, which is just the wheel
        estimator again, and the filter would inherit its unbounded drift. Treating
        the instantaneous rate difference as an observation of the bias keeps each
        sensor doing what it is actually good at: the wheels know rate, the gyro
        knows change, neither knows absolute heading, and the filter does not
        pretend otherwise. Heading is therefore still unobservable and its variance
        still grows -- correctly.
    """
    imu, js = load_imu(bag), load_joints(bag)
    model = args.yaw_model or noise['wheel']['yaw_rate_model']
    w_wheel_src = js['w_bicycle'] if model == 'bicycle' else js['w_diff']

    gb0 = float(noise['imu']['gyro_bias_xyz'][2])
    ab0 = float(noise['imu']['accel_bias_xyz'][0])
    sg = float(noise['imu']['gyro_sigma_xyz'][2])
    sa = float(noise['imu']['accel_sigma_xyz'][0])
    s_brw = float(noise['imu']['gyro_bias_rw'])
    sv_w = float(noise['wheel']['speed_sigma'])
    sw_w = (float(noise['wheel']['yaw_rate_sigma_bicycle']) if model == 'bicycle'
            else float(noise['wheel']['yaw_rate_sigma_differential']))

    x, y, th, z0 = anchor(gt, imu['t'][0])
    st = np.array([x, y, th, 0.0, gb0])
    # Initial covariance. Pose is anchored to ground truth so its variance starts at
    # zero; speed starts at zero because the robot is measurably stationary; the
    # bias prior is the spread of the static-window estimate, which is the only
    # honest thing to claim about it.
    P = np.diag([0.0, 0.0, 0.0, 1e-6, (sg / math.sqrt(max(1, noise['window']
                                                          ['static_imu_samples']))) ** 2])
    P[4, 4] = max(P[4, 4], 1e-8)

    # Merge the two streams into one time-ordered pass. 1 kHz joints against 200 Hz
    # IMU means several wheel samples per predict step; each is applied as its own
    # update at its own stamp rather than being averaged, so no measurement is
    # invented and none is dropped.
    events = [(t, 0, k) for k, t in enumerate(imu['t'])]
    if not args.no_wheel_update:
        stride = max(1, int(round(len(js['t']) / max(1, args.wheel_update_hz)
                                  / max(1e-9, js['t'][-1] - js['t'][0]))))
        events += [(js['t'][k], 1, k) for k in range(0, len(js['t']), stride)]
    events.sort(key=lambda e: (e[0], e[1]))

    rows, cov = [], []
    t_prev = events[0][0]
    n_upd = 0
    gyro_latest = float(imu['gyr'][0, 2])
    for t, kind, k in events:
        dt = t - t_prev
        t_prev = t
        if dt < 0 or dt > 0.5:
            continue
        if dt > 0:
            w = float(gyro_latest - st[4])
            a = float(imu['acc'][min(k, len(imu['acc']) - 1), 0] - ab0) if kind == 0 \
                else float(imu['acc'][max(0, np.searchsorted(imu['t'], t) - 1), 0] - ab0)
            th_mid = st[2] + 0.5 * w * dt
            F = np.eye(5)
            F[0, 2] = -st[3] * math.sin(th_mid) * dt
            F[0, 3] = math.cos(th_mid) * dt
            F[1, 2] = st[3] * math.cos(th_mid) * dt
            F[1, 3] = math.sin(th_mid) * dt
            F[2, 4] = -dt
            G = np.zeros((5, 3))
            G[0, 0] = 0.5 * math.cos(th_mid) * dt * dt
            G[1, 0] = 0.5 * math.sin(th_mid) * dt * dt
            G[3, 0] = dt
            G[2, 1] = dt
            G[4, 2] = math.sqrt(dt)      # random walk grows as sqrt(t), not t
            P = F @ P @ F.T + G @ np.diag([sa ** 2, sg ** 2, s_brw ** 2]) @ G.T

            st[0] += st[3] * math.cos(th_mid) * dt + 0.5 * a * math.cos(th_mid) * dt * dt
            st[1] += st[3] * math.sin(th_mid) * dt + 0.5 * a * math.sin(th_mid) * dt * dt
            st[3] += a * dt
            st[2] += w * dt

        if kind == 0:
            gyro_latest = float(imu['gyr'][k, 2])
            rows.append((t, st[0], st[1], z0, st[2], st[3], gyro_latest - st[4]))
            cov.append((t, P[0, 0], P[1, 1], P[2, 2], P[3, 3], P[4, 4],
                        P[0, 1], P[0, 2], P[1, 2]))
        else:
            z = np.array([float(js['v'][k]), float(gyro_latest - w_wheel_src[k])])
            H = np.zeros((2, 5))
            H[0, 3] = 1.0
            H[1, 4] = 1.0
            # The bias observation inherits BOTH sensors' noise: it is a difference
            # of a gyro sample and a wheel-derived rate, so its variance is the sum.
            R = np.diag([sv_w ** 2, sw_w ** 2 + sg ** 2])
            yres = z - H @ st
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            st = st + K @ yres
            # Joseph form: it stays symmetric and positive-definite over the ~16000
            # updates in a run, where (I-KH)P does not reliably. The covariance IS
            # the measurement here, so it cannot be allowed to rot.
            IKH = np.eye(5) - K @ H
            P = IKH @ P @ IKH.T + K @ R @ K.T
            P = 0.5 * (P + P.T)
            n_upd += 1

    return rows, cov, dict(yaw_rate_model=model, updates=n_upd,
                           speed_sigma=sv_w, yaw_sigma=sw_w,
                           gyro_bias_final=float(st[4]), gyro_bias_initial=gb0)


ESTIMATORS = {'wheel': run_wheel, 'imu': run_imu, 'ekf': run_ekf}
TREE = {'wheel': 'output_wheel', 'imu': 'output_imu', 'ekf': 'output_ekf'}


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------

def git_commit(path):
    try:
        return subprocess.check_output(
            ['git', '-C', path, 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return 'unknown'


def write_vio(path, rows):
    """vio.csv, the 11-column format the existing analysis chain already reads.

    t_ns, x, y, z, qw, qx, qy, qz, vx, vy, vz -- see docs/report/slam_logging_schema.md.
    Matching it exactly is what lets vio_metrics.py, plot_compare.py and
    plot_summary.py treat these baselines as just three more systems.
    """
    with open(path, 'w') as f:
        for (t, x, y, z, th, v, w) in rows:
            qw, qz = math.cos(th / 2.0), math.sin(th / 2.0)
            f.write("%d,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f\n"
                    % (int(round(t * 1e9)), x, y, z, qw, 0.0, 0.0, qz,
                       v * math.cos(th), v * math.sin(th), 0.0))


def write_cov(path, cov):
    with open(path, 'w') as f:
        f.write("t_ns,var_x,var_y,var_yaw,var_v,var_bgyro,cov_xy,cov_xyaw,cov_yyaw\n")
        for r in cov:
            f.write("%d," % int(round(r[0] * 1e9)))
            f.write(",".join("%.9g" % v for v in r[1:]) + "\n")


def write_gt(path, gt):
    with open(path, 'w') as f:
        for i in range(len(gt['t'])):
            q = gt['quat'][i]
            f.write("%d,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f\n"
                    % (int(round(gt['t'][i] * 1e9)), gt['x'][i], gt['y'][i], gt['z'][i],
                       q[0], q[1], q[2], q[3],
                       gt['v'][i] * math.cos(gt['yaw'][i]),
                       gt['v'][i] * math.sin(gt['yaw'][i]), 0.0))


def write_metadata(path, meta):
    with open(path, 'w') as f:
        f.write("key,value\n")
        for k, v in meta.items():
            f.write('%s,"%s"\n' % (k, v))


def do_run(args):
    if args.estimator not in ESTIMATORS:
        raise SystemExit("unknown estimator %s" % args.estimator)
    with open(args.noise) as f:
        noise = yaml.safe_load(f)

    nb = noise['provenance']['bag']
    if os.path.abspath(args.bag) != nb and not args.allow_foreign_noise:
        raise SystemExit(
            "sensor_noise.yaml was derived from\n  %s\nbut this run is on\n  %s\n"
            "The noise model is per-dataset: the wheel speed scale differs by 0.8%% "
            "between the two allsensor bags. Re-run `calibrate` on this bag, or pass "
            "--allow-foreign-noise if transferring the model IS the experiment." % (nb, args.bag))

    gt = load_gt(args.bag)
    rows, cov, extra = ESTIMATORS[args.estimator](args.bag, noise, gt, args)
    if not rows:
        raise SystemExit("estimator produced no poses -- check the bag's topics")

    os.makedirs(args.out, exist_ok=True)
    write_vio(os.path.join(args.out, 'vio.csv'), rows)
    write_cov(os.path.join(args.out, 'covariance.csv'), cov)
    write_gt(os.path.join(args.out, 'ground_truth.csv'), gt)

    t = np.array([r[0] for r in rows])
    meta = {
        'experiment_id': args.experiment_id,
        'estimator': args.estimator,
        'status': 'completed',
        'bag': os.path.abspath(args.bag),
        'noise_model': os.path.abspath(args.noise),
        'output_dir': os.path.abspath(args.out),
        'command': ' '.join(sys.argv),
        'generated': datetime.datetime.now().isoformat(timespec='seconds'),
        'git_commit_superproject': git_commit(ROOT),
        'wheel_radius_m': WHEEL_RADIUS,
        'wheel_base_m': WHEEL_BASE,
        'wheel_separation_m': WHEEL_SEPARATION,
        'initial_pose_source': 'ground truth at first estimator sample (dead '
                               'reckoning has no global reference)',
        'imu_orientation_used': 'NO - field equals ground-truth yaw and is excluded '
                                'by design; yaw is integrated from angular_velocity.z',
        'pose_count': len(rows),
        'first_stamp_s': float(t[0]),
        'last_stamp_s': float(t[-1]),
        'duration_s': float(t[-1] - t[0]),
        'mean_rate_hz': float(len(t) / max(1e-9, t[-1] - t[0])),
        'gt_count': len(gt['t']),
        'gt_first_stamp_s': float(gt['t'][0]),
        'gt_last_stamp_s': float(gt['t'][-1]),
        'source_covariances_populated': 'none (all zero in bag)',
    }
    meta.update({('param_' + k): v for k, v in extra.items()})
    write_metadata(os.path.join(args.out, 'run_metadata.csv'), meta)

    print("%s: %d poses, %.2f s, %.1f Hz -> %s"
          % (args.estimator, len(rows), t[-1] - t[0],
             len(t) / max(1e-9, t[-1] - t[0]), args.out))
    for k, v in extra.items():
        print("   %-20s %s" % (k, v))
    print("   files: vio.csv  covariance.csv  ground_truth.csv  run_metadata.csv")


# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    c = sub.add_parser('calibrate', help='derive the noise model from a bag')
    c.add_argument('--bag', required=True, help='absolute path to the .db3')
    c.add_argument('--out', required=True, help='directory to write sensor_noise.yaml')
    c.add_argument('--gyro-bias-rw', type=float, default=1e-5,
                   help='gyro bias random-walk sigma (rad/s/sqrt(s)). ASSUMED, not '
                        'measured: the static window is far too short for an Allan '
                        'variance. Default 1e-5.')
    c.set_defaults(func=calibrate)

    r = sub.add_parser('run', help='run one estimator over a bag')
    r.add_argument('--estimator', required=True, choices=sorted(ESTIMATORS))
    r.add_argument('--bag', required=True)
    r.add_argument('--noise', required=True, help='sensor_noise.yaml from calibrate')
    r.add_argument('--out', required=True, help='output directory for this run')
    r.add_argument('--experiment-id', default='exp-estimator-baseline')
    r.add_argument('--yaw-model', choices=['bicycle', 'differential'], default=None,
                   help='override the model chosen by calibrate')
    r.add_argument('--wheel-source', choices=['joints', 'bagged'], default='joints',
                   help="wheel estimator input. 'joints' derives odometry from the "
                        "encoder feed (the honest baseline). 'bagged' passes through "
                        "the AckermannSteering plugin's own integration, which is "
                        "the simulator marking its own homework -- reference only.")
    r.add_argument('--wheel-update-hz', type=float, default=50.0,
                   help='EKF wheel correction rate, subsampled from the 1 kHz joint '
                        'feed. Default 50.')
    r.add_argument('--no-wheel-update', action='store_true',
                   help='EKF with prediction only -- collapses it to the IMU '
                        'estimator plus a bias prior. Diagnostic.')
    r.add_argument('--allow-foreign-noise', action='store_true',
                   help='permit a noise model derived from a different bag')
    r.set_defaults(func=do_run)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
