"""Real-rig map collection in ONE command: ORB-SLAM3 stereo-inertial front end
plus RTAB-Map behind it (construct or extend), RViz as the GUI.

    ros2 launch vslam_navigation mapping_real.launch.py map_name:=realv7_002
    ros2 bag play "<bag>" --rate 0.5          # no --clock: the bags record one

Replaces the two-terminal procedure (orbslam3_ros2 wil_stereo_imu.launch.py in
one, rtabmap.launch.py in the other). Frame names, the imu_link -> base_link
transform and the fisheye rectification are all implied by the real rig, so
only the map name and the calibration set need naming:

    map_name     output/database land in map/<map_name>/<map_name>.db and
                 map/<map_name>/vio.csv (RTAB-Map's optimised graph); the ORB
                 trajectory in output/output_orb/real/<map_name>_mapping/.
    mode         construct (new map, database wiped) | extend (append a session
                 to map/<map_name>/<map_name>.db, which must exist).
    calibration  name of the config directory used by BOTH front end and shim:
                 orbslam3_ros2/config/<calibration>/stereo_imu.yaml and
                 vins_fusion_ros2/config/<calibration>/ (fisheye intrinsics +
                 stereo_imu.yaml). 'wil' is the live calibration;
                 'wil_311026_selfcal' the self-calibrated 31 Oct mount. Both are
                 read from the SOURCE tree so an uninstalled variant works.

RTAB-Map is started `rtabmap_delay` seconds after ORB-SLAM3. Until ORB has
initialised its IMU (2-3 s on this rig) there is no odometry transform, and
every stereo pair that arrives before it blocks RTAB-Map for the full
wait_for_transform (5 s); a handful of such frames put the input queue behind
its 10 s TF buffer and the run yields zero nodes (rtabmap_ros design record,
2026-10-29). Starting the back end late sidesteps that without touching the
timeout.

Live co-running needs the laptop cool: at 92 degC the CPU throttled to
0.8-1.3 GHz and the same pipeline ran fifteen times slower (4 Nov: the cooling
fan was the fix, not any parameter).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            OpaqueFunction, TimerAction)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

WORKSPACE = os.path.expanduser('~/wil_project')
ORB_LAUNCH = os.path.join(get_package_share_directory('orbslam3_ros2'),
                          'launch', 'wil_stereo_imu.launch.py')
RTAB_LAUNCH = os.path.join(get_package_share_directory('vslam_navigation'),
                           'launch', 'rtabmap.launch.py')


def _setup(context, *args, **kwargs):
    cfg = lambda k: LaunchConfiguration(k).perform(context)  # noqa: E731
    map_name = cfg('map_name')
    mode = cfg('mode').lower()
    if mode not in ('construct', 'extend'):
        raise RuntimeError(f"mode must be construct|extend for mapping, got {mode!r}")
    calibration = cfg('calibration')
    orb_cfg = os.path.join(WORKSPACE, 'orbslam3_ros2', 'config', calibration, 'stereo_imu.yaml')
    shim_cfg = os.path.join(WORKSPACE, 'vins_fusion_ros2', 'config', calibration)
    for p in (orb_cfg, os.path.join(shim_cfg, 'stereo_imu.yaml'),
              os.path.join(shim_cfg, 'cam0_fisheye.yaml')):
        if not os.path.isfile(p):
            raise RuntimeError(f'calibration {calibration!r}: missing {p}')
    map_dir = os.path.join(WORKSPACE, 'map', map_name)
    database = os.path.join(map_dir, f'{map_name}.db')
    orb_out = cfg('orb_output_path') or os.path.join(
        WORKSPACE, 'output', 'output_orb', 'real', f'{map_name}_mapping')

    print(f'[mapping_real] map        = {database}  (mode {mode})')
    print(f'[mapping_real] calibration= {calibration}: {orb_cfg}')
    print(f'[mapping_real] orb output = {orb_out}')

    orb_args = {
        'config_file': orb_cfg,
        'output_path': orb_out,
        'use_viewer': cfg('use_viewer'),
        'imu_best_effort': cfg('imu_best_effort'),
        'experiment_id': f'{map_name}_mapping',
    }
    # Only override the front end's own IMU scale when one is given, so the
    # calibration constant keeps living in one place (the ORB launch). The
    # argument is NOT named imu_accel_scale on purpose: launch configurations
    # are inherited by included launches, and a parent argument with the same
    # name as the child's would hand the child our empty default -- which is
    # exactly what killed orbslam3_node with "parameter 'imu_accel_scale' has
    # invalid type ... string" on every run of 4 Nov until this was found.
    if cfg('orb_imu_accel_scale'):
        orb_args['imu_accel_scale'] = cfg('orb_imu_accel_scale')
    orb = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(ORB_LAUNCH),
        launch_arguments=orb_args.items())

    rtab = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(RTAB_LAUNCH),
        launch_arguments={
            'mode': mode,
            'frontend': 'orb',
            'camera_model': 'fisheye',
            'left_cam': cfg('left_cam'),
            'rect_scale': cfg('rect_scale'),
            'hfov_deg': cfg('hfov_deg'),
            'config_dir': shim_cfg,
            'imu_height': cfg('imu_height'),
            'output_path': map_dir + os.sep,   # explicit: an earlier include's value would leak in
            'database_path': database,
            'grid_range_max': cfg('grid_range_max'),
            'detection_rate': cfg('detection_rate'),
            'shim_rate': cfg('shim_rate'),
            'loop_closure': cfg('loop_closure'),
            'rviz': cfg('rviz'),
        }.items())

    delay = float(cfg('rtabmap_delay'))
    return [orb, TimerAction(period=delay, actions=[rtab])]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('map_name', default_value='real_map',
                              description='map/<map_name>/<map_name>.db; also names the ORB run.'),
        DeclareLaunchArgument('mode', default_value='construct',
                              description='construct (new map) | extend (append to the existing db).'),
        DeclareLaunchArgument('calibration', default_value='wil',
                              description="config directory name shared by ORB and the shim: 'wil' "
                                          "(live) or 'wil_311026_selfcal' (31 Oct mount)."),
        DeclareLaunchArgument('orb_output_path', default_value='',
                              description='ORB trajectory directory; empty = output/output_orb/real/<map_name>_mapping'),
        DeclareLaunchArgument('orb_imu_accel_scale', default_value='',
                              description="empty = the ORB launch's own default (the measured rig value)."),
        DeclareLaunchArgument('imu_best_effort', default_value='true',
                              description='Best-effort QoS on the IMU topic, as the real recordings need.'),
        DeclareLaunchArgument('use_viewer', default_value='false',
                              description='ORB-SLAM3 Pangolin viewer. Off: RViz is the GUI and the viewer costs CPU.'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('left_cam', default_value='1',
                              description='Physically left camera index (cam1 since the 19 Oct remount).'),
        DeclareLaunchArgument('rect_scale', default_value='0.5'),
        DeclareLaunchArgument('hfov_deg', default_value='110'),
        DeclareLaunchArgument('imu_height', default_value='0.11',
                              description='IMU height above the floor [m] for the base_link transform.'),
        DeclareLaunchArgument('grid_range_max', default_value='6.0',
                              description='Grid/RangeMax [m]. Measured depth noise suggests 3.0 at rect_scale 0.5.'),
        DeclareLaunchArgument('detection_rate', default_value='1.0'),
        DeclareLaunchArgument('shim_rate', default_value='2.0'),
        DeclareLaunchArgument('loop_closure', default_value='true',
                              description='RTAB-Map closures. Required true for extend (sessions stitch by closure).'),
        DeclareLaunchArgument('rtabmap_delay', default_value='5.0',
                              description='Seconds after ORB-SLAM3 before RTAB-Map starts (IMU init first).'),
        OpaqueFunction(function=_setup),
    ])
