"""The full SIMULATION navigation stack in ONE command: Gazebo, ORB-SLAM3 or VINS
with RTAB-Map relocalising on a prior map, Nav2 with the pose gate, and RViz.
The simulation counterpart of navigate_real.launch.py.

    ros2 launch vslam_navigation navigate_sim.launch.py map_name:=realistic_01_prior
    # then set a goal in RViz with the 2D Goal Pose tool (/goal_pose)

Order of events: Gazebo starts and spawns the robot; the front end starts and
initialises its IMU; RTAB-Map loads the prior and relocalises; the pose gate
waits out the init transient and a stability window, then STARTS Nav2; a goal set
in RViz is planned on the prior map and tracked; /cmd_vel reaches the robot
through the Gazebo bridge, and Nav2 is paused again if the localisation goes bad.

WHY THIS IS NOT aws_robomaker_small_warehouse_world nav2.launch.py
    That file navigates on GROUND TRUTH: it includes localization_tf.launch.py,
    whose map -> odom is an exact static identity published by
    ground_truth_localization. It is the right tool for testing Nav2 itself,
    because localisation error is then exactly zero. It cannot answer the
    question this file exists for -- whether the robot can navigate on a map it
    built and relocalised into by itself, which is what the real rig has to do.

    The two are mutually exclusive, not composable: both publish map -> odom.

DIFFERENCES FROM THE REAL RIG, all of them structural rather than tuning
    * no ackermann_bridge: gz AckermannSteering takes a Twist on
      /model/<robot>/cmd_vel and small_warehouse's cmd_vel_bridge already
      forwards /cmd_vel to it. The rig needs /cmd_velocity and /cmd_steering in
      its own units instead.
    * use_sim_time is true throughout, and Gazebo is the clock.
    * ground truth is still bridged to /ground_truth/odometry, for SCORING only.
      Nothing in the control path consumes it, and no node publishes a transform
      from it. That is what makes the localisation error measurable here and not
      on the rig.
    * the chassis differs: wheel_base 0.42 m and a 35 deg steering limit, so the
      car turns in a 1.2 m circle rather than the rig's 2.7 m. See
      params/nav2_sim.yaml.

Every argument of every include is passed explicitly, because launch
configurations leak between includes otherwise.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

WORKSPACE = os.path.expanduser('~/wil_project')
SHARE = get_package_share_directory('vslam_navigation')
WAREHOUSE = get_package_share_directory('aws_robomaker_small_warehouse_world')


def _setup(context, *args, **kwargs):
    cfg = lambda k: LaunchConfiguration(k).perform(context)  # noqa: E731
    truthy = lambda k: cfg(k).lower() in ('true', '1', 'yes')  # noqa: E731
    map_name = cfg('map_name')
    map_dir = os.path.join(WORKSPACE, 'map', map_name)
    database = os.path.join(map_dir, f'{map_name}.db')
    if not os.path.isfile(database):
        raise RuntimeError(
            f'prior map {database} does not exist.\n'
            f'  Build one first:  ros2 launch vslam_navigation mapping_sim.launch.py '
            f'map_name:={map_name} loop_closure:=false')
    run_name = cfg('run_name') or f'{map_name}_navigate'

    print(f'[navigate_sim] prior    = {database}')
    print(f'[navigate_sim] frontend = {cfg("frontend")}')
    print(f'[navigate_sim] gazebo   = {"starting" if truthy("start_sim") else "assumed already running"}'
          f' (headless {cfg("headless")})')

    acts = []
    if truthy('start_sim'):
        # First, and as an include rather than a copy: its _override_speed_limits
        # OpaqueFunction mutates IGN_GAZEBO_RESOURCE_PATH and MUST run before gz
        # starts. bridge_cmd_vel is how Nav2 reaches the robot; bridge_ground_truth
        # stays on for scoring but nothing publishes TF from it.
        acts.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(WAREHOUSE, 'launch', 'small_warehouse.launch.py')),
            launch_arguments={
                'use_sim_time': 'true',
                'headless': cfg('headless'),
                'bridge_sensors': 'true',
                'compressed_images': 'true',
                'bridge_cmd_vel': 'true',
                'cmd_vel_bridge_topic': '/cmd_vel',
                'bridge_ground_truth': 'true',   # scoring only; publishes no transform
                'bridge_wheel_odom': 'false',
                'bridge_joint_states': 'false',  # localize_sim runs joint_state_publisher
                'bridge_model_poses': 'false',
                'robot_name': cfg('robot_name'),
                'verbosity': cfg('verbosity'),
                # Only override the world when one is named, so the default stays
                # small_warehouse.launch.py's own and cannot drift from it.
                **({'world': cfg('world')} if cfg('world') else {}),
            }.items()))

    localize = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(SHARE, 'launch', 'localize_sim.launch.py')),
        launch_arguments={
            'map_name': map_name,
            'run_name': run_name,
            'frontend': cfg('frontend'),
            'read_only': 'true',
            'start_frontend': 'true',
            'image_transport': 'compressed',
            'ground_height': cfg('ground_height'),
            'optimize_max_error': cfg('optimize_max_error'),
            'detection_rate': cfg('detection_rate'),
            'shim_rate': cfg('shim_rate'),
            'publish_joint_states': 'true',
            'extent_margin': cfg('extent_margin'),
            'rviz': 'false',                      # one RViz, with the navigation config
        }.items())

    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(SHARE, 'launch', 'nav2_sim.launch.py')),
        launch_arguments={
            'use_sim_time': 'true',
            'map_yaml': '',                       # /map comes from RTAB-Map
            'database': database,
            'odom_frame': 'odom',
            'base_frame': 'base_link',
            'autostart': 'false',
            'pose_gate': 'true',
            'init_ignore_s': cfg('init_ignore_s'),
            'max_step': cfg('max_step'),
            'extent_margin': cfg('extent_margin'),
            'gate_log': os.path.join(map_dir, f'gate_{run_name}.csv'),
        }.items())

    # Gazebo needs to be up and the front end initialised before the back end and
    # Nav2 are useful; starting them late costs nothing and avoids a queue of
    # stereo pairs each blocking on a transform that does not exist yet.
    delay = float(cfg('stack_delay'))
    acts.append(TimerAction(period=delay, actions=[localize]))
    acts.append(TimerAction(period=delay + float(cfg('nav2_delay')), actions=[nav2]))

    if truthy('rviz'):
        acts.append(TimerAction(period=delay, actions=[Node(
            package='rviz2', executable='rviz2', name='nav_rviz', output='screen',
            arguments=['-d', os.path.join(SHARE, 'rviz', 'navigation.rviz')],
            parameters=[{'use_sim_time': True}])]))
    return acts


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('map_name', description='prior map: map/<map_name>/<map_name>.db'),
        DeclareLaunchArgument('run_name', default_value='',
                              description='names the recorder CSV and the gate log; empty = <map_name>_navigate'),
        DeclareLaunchArgument('frontend', default_value='orb', description='orb | vins'),
        DeclareLaunchArgument('start_sim', default_value='true',
                              description='false reuses a Gazebo already running.'),
        DeclareLaunchArgument('headless', default_value='true',
                              description='true is the cheap path (README: RTF 0.90 vs 0.40). '
                                          'RViz is the GUI anyway.'),
        DeclareLaunchArgument(
            'world', default_value='',
            description='MUST be the world the prior map was built in, or the prior '
                        'does not describe the place the robot is driving. Empty keeps '
                        'small_warehouse.launch.py own default (small_warehouse.world). '
                        'The realistic worlds live in '
                        'worlds/small_warehouse/ as small_warehouse_static_realistic_'
                        'nofloortexture.world (layout U) and ..._nofloortexture_01.world '
                        '(layout I); the realistic_01 bags are layout I.'),
        DeclareLaunchArgument('robot_name', default_value='ackermann_robot_001'),
        DeclareLaunchArgument('verbosity', default_value='3'),
        DeclareLaunchArgument('ground_height', default_value='0.29'),
        DeclareLaunchArgument('optimize_max_error', default_value='10.0'),
        DeclareLaunchArgument('detection_rate', default_value='1.0'),
        DeclareLaunchArgument('shim_rate', default_value='4.0'),
        DeclareLaunchArgument('init_ignore_s', default_value='8.0',
                              description='seconds of localisation output the gate ignores while the '
                                          'front end initialises.'),
        DeclareLaunchArgument('max_step', default_value='1.0',
                              description='map->odom correction step above which the gate drops [m].'),
        DeclareLaunchArgument('extent_margin', default_value='2.0'),
        DeclareLaunchArgument('stack_delay', default_value='12.0',
                              description='seconds after Gazebo before the localisation stack starts.'),
        DeclareLaunchArgument('nav2_delay', default_value='8.0',
                              description='further seconds before Nav2 starts.'),
        DeclareLaunchArgument('rviz', default_value='true'),
        OpaqueFunction(function=_setup),
    ])
