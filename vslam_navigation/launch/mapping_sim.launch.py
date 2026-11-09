"""Simulation map collection in ONE command: ORB-SLAM3 stereo-inertial front end
plus RTAB-Map behind it (construct or extend), RViz as the GUI. The simulation
counterpart of mapping_real.launch.py, so both domains are two terminals:

    ros2 launch vslam_navigation mapping_sim.launch.py \
        map_name:=dataset_allsensor_realistic_01_map loop_closure:=false
    ros2 bag play "/media/ambushee/New Volume/wil_rescue/dataset_allsensor_realistic_01_map" --rate 0.5

Play the bag WITHOUT --clock: the simulation bags record /clock as a topic, so
replay republishes it and rosbag2's own clock would fight it.

WHAT DIFFERS FROM mapping_real.launch.py, and why this file exists at all
  * camera_model is pinhole. A gz camera is an ideal pinhole: cam0_pinhole.yaml
    carries zero distortion and both cameras share intrinsics, so the raw frames
    already satisfy what rectification would produce and the shim publishes them
    as "rect" untouched. The real rig's Kannala-Brandt remap (left_cam,
    hfov_deg, rect_scale) has no meaning here and is not exposed.
  * the front end is orbslam3_ros2 wil_sim_stereo_imu.launch.py, which defaults
    to config/wil_sim/stereo_imu.yaml and sets use_sim_time true to match the
    bag's /clock. The real-rig launch sets it false.
  * ORB broadcasts orbslam3_world -> base_footprint in simulation, not
    -> imu_link, so no imu_link -> base_link transform is published and there is
    no imu_height to measure. publish_base_tf stays 'auto', which acts only on
    the fisheye path.
  * image_transport is pinned to compressed: the simulation bags carry only
    /camN/image_raw/compressed, and both nodes decode in process rather than
    through a republisher.

LOOP CLOSURE ON MAPPING BAGS. loop_closure defaults true because extend stitches
sessions by closure, but the realistic mapping bags are known to accept FALSE
closures: realistic_map and realistic_mapv2 held 0.15 m of error until closures
of 3.4, 28.9 and 27.2 m appeared, and on these bags any correction over 1 m is
false (rtabmap_ros design record). Pass loop_closure:=false when building a
prior from a realistic mapping bag, and check the link types afterwards.

RTAB-Map starts rtabmap_delay seconds after ORB-SLAM3, for the same reason as on
the rig: until the IMU has initialised there is no odometry transform, and each
stereo pair arriving before it blocks the back end for the full
wait_for_transform, which can put the input queue behind its TF buffer and yield
a zero-node database.
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
                          'launch', 'wil_sim_stereo_imu.launch.py')
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
    # cam0_pinhole.yaml is the discriminator: a fisheye calibration directory has
    # cam0_fisheye.yaml instead, and passing one here would silently apply rig
    # intrinsics to simulation frames.
    for p in (orb_cfg, os.path.join(shim_cfg, 'stereo_imu.yaml'),
              os.path.join(shim_cfg, 'cam0_pinhole.yaml'),
              os.path.join(shim_cfg, 'cam1_pinhole.yaml')):
        if not os.path.isfile(p):
            raise RuntimeError(
                f'calibration {calibration!r}: missing {p}\n'
                f'  mapping_sim needs a PINHOLE calibration directory '
                f'(cam0_pinhole.yaml / cam1_pinhole.yaml); use mapping_real.launch.py '
                f'for the fisheye rig.')
    map_dir = os.path.join(WORKSPACE, 'map', map_name)
    database = os.path.join(map_dir, f'{map_name}.db')
    if mode == 'extend' and not os.path.isfile(database):
        raise RuntimeError(f'mode extend: {database} does not exist yet; use mode:=construct')
    orb_out = cfg('orb_output_path') or os.path.join(
        WORKSPACE, 'output', 'output_orb', 'simulation', f'{map_name}_mapping')

    loop = cfg('loop_closure').lower() in ('true', '1', 'yes')
    print(f'[mapping_sim] map        = {database}  (mode {mode})')
    print(f'[mapping_sim] calibration= {calibration}: {orb_cfg}')
    print(f'[mapping_sim] orb output = {orb_out}')
    print(f'[mapping_sim] closures   = {"ON" if loop else "OFF (closure-free prior)"}')
    if loop and mode == 'construct':
        print('[mapping_sim] NOTE: realistic mapping bags accept false closures '
              '(3.4 / 28.9 / 27.2 m seen); pass loop_closure:=false for a prior.')

    # config_file is deliberately NOT a launch argument of this file: launch
    # configurations are inherited by included launches, so an argument sharing
    # the child's name hands the child our default. That is what killed
    # orbslam3_node on every run of 4 Nov via imu_accel_scale. Same reason
    # orb_output_path is renamed -- rtabmap.launch.py also has output_path.
    orb_args = {
        'config_file': orb_cfg,
        'output_path': orb_out,
        'use_viewer': cfg('use_viewer'),
        'experiment_id': f'{map_name}_mapping',
        'image_transport': 'compressed',   # sim bags are compressed-only
    }
    orb = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(ORB_LAUNCH),
        launch_arguments=orb_args.items())

    rtab = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(RTAB_LAUNCH),
        launch_arguments={
            'mode': mode,
            'frontend': 'orb',             # -> orbslam3_world / base_footprint
            'camera_model': 'pinhole',
            'config_dir': shim_cfg,
            'output_path': map_dir + os.sep,  # explicit: an earlier include's value would leak in
            'database_path': database,
            'grid_range_max': cfg('grid_range_max'),
            'detection_rate': cfg('detection_rate'),
            'shim_rate': cfg('shim_rate'),
            'loop_closure': cfg('loop_closure'),
            'rviz': cfg('rviz'),
        }.items())

    return [orb, TimerAction(period=float(cfg('rtabmap_delay')), actions=[rtab])]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('map_name', default_value='sim_map',
                              description='map/<map_name>/<map_name>.db; also names the ORB run.'),
        DeclareLaunchArgument('mode', default_value='construct',
                              description='construct (new map) | extend (append to the existing db).'),
        DeclareLaunchArgument('calibration', default_value='wil_sim',
                              description="pinhole config directory shared by ORB and the shim: "
                                          "'wil_sim' (default) or another wil_sim* variant."),
        DeclareLaunchArgument('orb_output_path', default_value='',
                              description='ORB trajectory directory; empty = '
                                          'output/output_orb/simulation/<map_name>_mapping'),
        DeclareLaunchArgument('use_viewer', default_value='false',
                              description='ORB-SLAM3 Pangolin viewer. Off: RViz is the GUI and the viewer costs CPU.'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('grid_range_max', default_value='10.0',
                              description='Grid/RangeMax [m]. The sim stereo pair is noiseless, so unlike '
                                          'the rig the range is not depth-noise limited.'),
        DeclareLaunchArgument('detection_rate', default_value='1.0'),
        DeclareLaunchArgument('shim_rate', default_value='4.0',
                              description='Stereo pairs per second into RTAB-Map.'),
        DeclareLaunchArgument('loop_closure', default_value='true',
                              description='RTAB-Map closures. Required true for extend. Pass false when '
                                          'building a prior from a realistic mapping bag (false closures).'),
        DeclareLaunchArgument('rtabmap_delay', default_value='5.0',
                              description='Seconds after ORB-SLAM3 before RTAB-Map starts (IMU init first).'),
        OpaqueFunction(function=_setup),
    ])
