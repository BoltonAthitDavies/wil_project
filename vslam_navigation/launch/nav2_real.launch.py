"""Nav2 for the real rig on a prior RTAB-Map map, plus the two WiL nodes that
sit between Nav2 and the world: the pose gate (starts/pauses Nav2 on the
soundness of the localisation) and the Ackermann bridge (/cmd_vel -> the
robot's /cmd_velocity and /cmd_steering).

Normally included by navigate_real.launch.py. On its own it is the OFFLINE
test harness: with map_yaml:= a map_server serves the grid and static
transforms can stand in for the SLAM chain, so planner, controller, behaviour
tree and bridge are exercised with no robot and no estimator:

    ros2 launch vslam_navigation nav2_real.launch.py use_sim_time:=false \
        map_yaml:=/path/prior.yaml database:=map/<m>/<m>.db bridge_dry_run:=true

Frame contract (what the SLAM side must provide):
    map -> orbslam3_world     RTAB-Map localize mode (the correction)
    orbslam3_world -> imu_link  ORB-SLAM3
    imu_link -> base_link     rtabmap.launch.py (publish_base_tf)
    /map                      RTAB-Map use_saved_map (latched), or map_server here
    /orbslam3/odometry        ORB-SLAM3 (bt_navigator / velocity_smoother odom)

cmd_vel chain, same as the simulation: controller -> cmd_vel_nav ->
velocity_smoother -> /cmd_vel -> ackermann_bridge -> robot.
"""
import os

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch_ros.actions import Node
from nav2_common.launch import RewrittenYaml

PKG = 'vslam_navigation'
SHARE = get_package_share_directory(PKG)
LIB = os.path.join(get_package_prefix(PKG), 'lib', PKG)


def _setup(context, *args, **kwargs):
    cfg = lambda k: LaunchConfiguration_perform(k, context)  # noqa: E731
    params_file = cfg('params_file')
    use_sim_time = cfg('use_sim_time').lower() in ('true', '1', 'yes')
    map_yaml = cfg('map_yaml')
    database = cfg('database')
    autostart = cfg('autostart').lower() in ('true', '1', 'yes')
    gate = cfg('pose_gate').lower() in ('true', '1', 'yes')
    bridge = cfg('bridge').lower() in ('true', '1', 'yes')
    bridge_dry = cfg('bridge_dry_run').lower() in ('true', '1', 'yes')

    lifecycle_nodes = ['controller_server', 'smoother_server', 'planner_server',
                       'behavior_server', 'bt_navigator', 'velocity_smoother']
    if map_yaml:
        lifecycle_nodes.insert(0, 'map_server')

    configured = RewrittenYaml(
        source_file=params_file, root_key='',
        param_rewrites={'use_sim_time': 'true' if use_sim_time else 'false',
                        'yaml_filename': map_yaml,
                        'default_nav_through_poses_bt_xml': os.path.join(
                            SHARE, 'behavior_trees', 'nav_through_poses_ackermann.xml')},
        convert_types=True)

    print(f'[nav2_real] params     = {params_file}')
    print(f'[nav2_real] map        = {"map_server " + map_yaml if map_yaml else "/map from RTAB-Map"}')
    print(f'[nav2_real] lifecycle  = {"autostart" if autostart else "started by pose_gate" if gate else "MANUAL (ros2 service call)"}')
    print(f'[nav2_real] bridge     = {"dry run" if bridge_dry else "LIVE: publishes /cmd_velocity /cmd_steering" if bridge else "off"}')

    nodes = []
    if map_yaml:
        nodes.append(Node(package='nav2_map_server', executable='map_server', name='map_server',
                          output='screen', parameters=[configured]))
    nodes += [
        Node(package='nav2_controller', executable='controller_server', name='controller_server',
             output='screen', parameters=[configured], remappings=[('cmd_vel', 'cmd_vel_nav')]),
        Node(package='nav2_smoother', executable='smoother_server', name='smoother_server',
             output='screen', parameters=[configured]),
        Node(package='nav2_planner', executable='planner_server', name='planner_server',
             output='screen', parameters=[configured]),
        Node(package='nav2_behaviors', executable='behavior_server', name='behavior_server',
             output='screen', parameters=[configured]),
        Node(package='nav2_bt_navigator', executable='bt_navigator', name='bt_navigator',
             output='screen', parameters=[configured]),
        Node(package='nav2_velocity_smoother', executable='velocity_smoother', name='velocity_smoother',
             output='screen', parameters=[configured],
             remappings=[('cmd_vel', 'cmd_vel_nav'), ('cmd_vel_smoothed', 'cmd_vel')]),
        # configured FIRST so bond_timeout from the file reaches the manager
        # (the sim launch records the spurious teardown that happens otherwise).
        Node(package='nav2_lifecycle_manager', executable='lifecycle_manager',
             name='lifecycle_manager_navigation', output='screen',
             parameters=[configured, {'use_sim_time': use_sim_time,
                                      'autostart': autostart,
                                      'node_names': lifecycle_nodes}]),
    ]
    if gate:
        nodes.append(ExecuteProcess(
            cmd=['python3', os.path.join(LIB, 'pose_gate.py'), '--ros-args',
                 '-p', f'use_sim_time:={str(use_sim_time).lower()}',
                 '-p', f'database:={database}',
                 '-p', f'odom_frame:={cfg("odom_frame")}',
                 '-p', f'init_ignore_s:={cfg("init_ignore_s")}',
                 '-p', f'max_step:={cfg("max_step")}',
                 '-p', f'margin:={cfg("extent_margin")}',
                 '-p', f'manage_lifecycle:={str(not autostart).lower()}']
                # an EMPTY `-p log_file:=` is a parse error for rcl ("Couldn't parse
                # parameter override rule"), so the option is added only when set
                + (['-p', f'log_file:={cfg("gate_log")}'] if cfg('gate_log') else []),
            output='screen'))
    if bridge:
        nodes.append(ExecuteProcess(
            cmd=['python3', os.path.join(LIB, 'ackermann_bridge.py'), '--ros-args',
                 '-p', f'use_sim_time:={str(use_sim_time).lower()}',
                 '-p', f'dry_run:={str(bridge_dry).lower()}',
                 '-p', f'require_localization:={str(gate).lower()}',
                 '-p', f'wheel_base:={cfg("wheel_base")}',
                 '-p', f'max_speed:={cfg("max_speed")}'],
            output='screen'))
    return nodes


def LaunchConfiguration_perform(name, context):
    from launch.substitutions import LaunchConfiguration
    return LaunchConfiguration(name).perform(context)


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('params_file', default_value=os.path.join(SHARE, 'params', 'nav2_real.yaml')),
        DeclareLaunchArgument('use_sim_time', default_value='true',
                              description='true on bag replay (the bags record /clock), false live'),
        DeclareLaunchArgument('map_yaml', default_value='',
                              description='offline tests only: serve this map with map_server instead of RTAB-Map'),
        DeclareLaunchArgument('database', default_value='',
                              description='prior RTAB-Map database; gives the pose gate its map extent'),
        DeclareLaunchArgument('odom_frame', default_value='orbslam3_world'),
        DeclareLaunchArgument('autostart', default_value='false',
                              description='true starts Nav2 immediately (offline tests); false lets the pose gate start it'),
        DeclareLaunchArgument('pose_gate', default_value='true'),
        DeclareLaunchArgument('init_ignore_s', default_value='8.0'),
        DeclareLaunchArgument('max_step', default_value='1.0'),
        DeclareLaunchArgument('extent_margin', default_value='2.0'),
        DeclareLaunchArgument('gate_log', default_value='',
                              description='file receiving the gate transitions and lifecycle commands'),
        DeclareLaunchArgument('bridge', default_value='true'),
        DeclareLaunchArgument('bridge_dry_run', default_value='true',
                              description='true (default) logs the robot commands instead of publishing them. '
                                          'Set false only on the robot, after wheel_base and the topic units are verified.'),
        DeclareLaunchArgument('wheel_base', default_value='0.60'),
        DeclareLaunchArgument('max_speed', default_value='0.6'),
        OpaqueFunction(function=_setup),
    ])
