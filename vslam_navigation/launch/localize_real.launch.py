"""Relocalisation on a prior real-rig map in ONE command: ORB-SLAM3
stereo-inertial front end plus RTAB-Map in localize mode against an existing
database, RViz as the GUI.

    ros2 launch vslam_navigation localize_real.launch.py map_name:=realv7_001_prior
    ros2 bag play "<another bag of the same site>" --rate 0.5     # no --clock

What happens: ORB-SLAM3 starts a fresh map of its own (its world frame is
wherever it initialises) and publishes orbslam3_world -> imu_link; RTAB-Map
opens map/<map_name>/<map_name>.db read-only, adds NO nodes, and once a frame
is recognised against the prior map it publishes map -> orbslam3_world (the
correction) and /localization_pose. Until that first recognition the robot is
only in the odometry frame, which is the expected state, not a failure.

    map_name     the prior database map/<map_name>/<map_name>.db (must exist).
    calibration  config directory for ORB and the shim, as in mapping_real.
                 MUST be the one the prior map was built with: the stored
                 visual words come from the rectified images.
    read_only    true (default) keeps the prior database byte-identical.
                 RTAB-Map refuses read-only on a database whose sessions never
                 merged; a failed extend needs read_only:=false.

Evidence is written to map/<map_name>/localize_<run_name>.csv by
localization_recorder.py (every /localization_pose with an inside-extent flag,
the map->odom correction at 2 Hz, and a summary line at shutdown).

Judging the result (design record, Section 6d): a localisation covariance
below 9999 is NOT a correctness signal -- one sim run reported 98 % localised
with poses 161 m outside a 14 x 21 m building. Check that /localization_pose
lands inside the map's extent and that map -> orbslam3_world stays stable
between recognitions. loop_closure is forced on: relocalisation is the
loop-closure detector.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from ament_index_python.packages import get_package_prefix
from launch.actions import (DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription,
                            OpaqueFunction, TimerAction)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

WORKSPACE = os.path.expanduser('~/wil_project')
ORB_LAUNCH = os.path.join(get_package_share_directory('orbslam3_ros2'),
                          'launch', 'wil_stereo_imu.launch.py')
RTAB_LAUNCH = os.path.join(get_package_share_directory('vslam_navigation'),
                           'launch', 'rtabmap.launch.py')
RECORDER = os.path.join(get_package_prefix('vslam_navigation'), 'lib', 'vslam_navigation',
                        'localization_recorder.py')


def _setup(context, *args, **kwargs):
    cfg = lambda k: LaunchConfiguration(k).perform(context)  # noqa: E731
    map_name = cfg('map_name')
    calibration = cfg('calibration')
    orb_cfg = os.path.join(WORKSPACE, 'orbslam3_ros2', 'config', calibration, 'stereo_imu.yaml')
    shim_cfg = os.path.join(WORKSPACE, 'vins_fusion_ros2', 'config', calibration)
    for p in (orb_cfg, os.path.join(shim_cfg, 'stereo_imu.yaml'),
              os.path.join(shim_cfg, 'cam0_fisheye.yaml')):
        if not os.path.isfile(p):
            raise RuntimeError(f'calibration {calibration!r}: missing {p}')
    database = os.path.join(WORKSPACE, 'map', map_name, f'{map_name}.db')
    if not os.path.isfile(database):
        raise RuntimeError(f'prior map {database} does not exist; build it with '
                           f'mapping_real.launch.py map_name:={map_name} first')
    run_name = cfg('run_name') or f'{map_name}_localize'
    orb_out = os.path.join(WORKSPACE, 'output', 'output_orb', 'real', run_name)

    print(f'[localize_real] prior map  = {database}  (read_only {cfg("read_only")})')
    print(f'[localize_real] calibration= {calibration}: {orb_cfg}')
    print(f'[localize_real] orb output = {orb_out}')

    orb_args = {
        'config_file': orb_cfg,
        'output_path': orb_out,
        'use_viewer': cfg('use_viewer'),
        'imu_best_effort': cfg('imu_best_effort'),
        'experiment_id': run_name,
    }
    if cfg('orb_imu_accel_scale'):
        orb_args['imu_accel_scale'] = cfg('orb_imu_accel_scale')
    orb = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(ORB_LAUNCH),
        launch_arguments=orb_args.items())

    rtab = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(RTAB_LAUNCH),
        launch_arguments={
            'mode': 'localize',
            'read_only': cfg('read_only'),
            'frontend': 'orb',
            'camera_model': 'fisheye',
            'left_cam': cfg('left_cam'),
            'rect_scale': cfg('rect_scale'),
            'hfov_deg': cfg('hfov_deg'),
            'config_dir': shim_cfg,
            'imu_height': cfg('imu_height'),
            'database_path': database,
            # Pinned on purpose. Launch arguments of an earlier include stay in
            # the shared context, so without this the ORB include's output_path
            # leaked in here and RTAB-Map's trajectory writer overwrote ORB's
            # vio.csv with the prior map's graph (4 Nov, 14:37 run).
            'output_path': '',
            'detection_rate': cfg('detection_rate'),
            'shim_rate': cfg('shim_rate'),
            'optimize_max_error': cfg('optimize_max_error'),
            'loop_closure': 'true',
            'rviz': cfg('rviz'),
        }.items())

    # Evidence of what localisation did: /localization_pose and the map->odom
    # correction, each pose checked against the prior map's extent.
    record_csv = os.path.join(WORKSPACE, 'map', map_name, f'localize_{run_name}.csv')
    recorder = ExecuteProcess(
        cmd=['python3', RECORDER, '--ros-args',
             '-p', 'use_sim_time:=true',
             '-p', f'csv:={record_csv}',
             '-p', f'database:={database}',
             '-p', 'odom_frame:=orbslam3_world',
             '-p', 'base_frame:=base_link',
             '-p', f'margin:={cfg("extent_margin")}'],
        output='screen')
    print(f'[localize_real] record     = {record_csv}')

    return [orb, TimerAction(period=float(cfg('rtabmap_delay')), actions=[rtab, recorder])]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('map_name', description='prior map: map/<map_name>/<map_name>.db'),
        DeclareLaunchArgument('run_name', default_value='',
                              description='ORB run directory name; empty = <map_name>_localize'),
        DeclareLaunchArgument('calibration', default_value='wil',
                              description='must match the calibration the prior map was built with'),
        DeclareLaunchArgument('read_only', default_value='true',
                              description='keep the prior database unchanged (Mem/LocalizationReadOnly)'),
        DeclareLaunchArgument('orb_imu_accel_scale', default_value=''),
        DeclareLaunchArgument('imu_best_effort', default_value='true'),
        DeclareLaunchArgument('use_viewer', default_value='false'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('left_cam', default_value='1'),
        DeclareLaunchArgument('rect_scale', default_value='0.5'),
        DeclareLaunchArgument('hfov_deg', default_value='110'),
        DeclareLaunchArgument('imu_height', default_value='0.11'),
        DeclareLaunchArgument('detection_rate', default_value='1.0'),
        DeclareLaunchArgument('shim_rate', default_value='2.0'),
        DeclareLaunchArgument('optimize_max_error', default_value='3.0',
                              description='RGBD/OptimizeMaxError: relocalisation hypotheses whose '
                                          'optimisation error ratio exceeds this are rejected.'),
        DeclareLaunchArgument('rtabmap_delay', default_value='5.0'),
        DeclareLaunchArgument('extent_margin', default_value='2.0',
                              description='metres beyond the prior map\'s node extent a pose may lie and still count as inside'),
        OpaqueFunction(function=_setup),
    ])
