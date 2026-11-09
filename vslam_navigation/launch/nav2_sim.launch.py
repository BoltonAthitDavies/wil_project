"""Nav2 for the SIMULATED ackermann robot on a prior RTAB-Map map, plus the pose
gate that starts and pauses it on the soundness of the localisation.

The simulation counterpart of nav2_real.launch.py. Normally included by
navigate_sim.launch.py. On its own it is the OFFLINE test harness: with map_yaml
a map_server serves the grid and static transforms can stand in for the SLAM
chain, so planner, controller and behaviour tree are exercised with no Gazebo and
no estimator.

    ros2 launch vslam_navigation nav2_sim.launch.py \
        map_yaml:=/path/prior.yaml database:=map/<m>/<m>.db

NO ACKERMANN BRIDGE, and that is the one structural difference from the real rig.
The gz AckermannSteering plugin consumes geometry_msgs/Twist directly on
/model/<robot>/cmd_vel, and small_warehouse.launch.py's cmd_vel_bridge already
forwards /cmd_vel to it. The real rig instead needs /cmd_velocity and
/cmd_steering in its own units, which is what scripts/ackermann_bridge.py is for.
So the chain here stops at /cmd_vel:

    controller -> cmd_vel_nav -> velocity_smoother -> /cmd_vel -> gz bridge

Frame contract (what localization_rtabmap.launch.py must provide):
    map -> odom                 RTAB-Map localize mode (the live correction)
    odom -> vio_body            ORB-SLAM3 or VINS, never both
    vio_body -> base_footprint  static, from that launch
    base_footprint -> base_link robot_state_publisher
    /map                        RTAB-Map use_saved_map (latched), or map_server here
    /orbslam3/odometry          the front end (bt_navigator, velocity_smoother)

Note what is deliberately NOT included: localization_tf.launch.py. That is the
ground-truth chain, and it publishes the same map -> odom edge. Running both
would give two publishers for one edge, which is the failure this whole file
exists to avoid.
"""
import os

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from nav2_common.launch import RewrittenYaml

PKG = 'vslam_navigation'
SHARE = get_package_share_directory(PKG)
LIB = os.path.join(get_package_prefix(PKG), 'lib', PKG)


def _setup(context, *args, **kwargs):
    cfg = lambda k: LaunchConfiguration(k).perform(context)  # noqa: E731
    truthy = lambda k: cfg(k).lower() in ('true', '1', 'yes')  # noqa: E731
    params_file = cfg('params_file')
    use_sim_time = truthy('use_sim_time')
    map_yaml = cfg('map_yaml')
    autostart = truthy('autostart')
    gate = truthy('pose_gate')

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

    print(f'[nav2_sim] params     = {params_file}')
    print(f'[nav2_sim] map        = {"map_server " + map_yaml if map_yaml else "/map from RTAB-Map"}')
    print(f'[nav2_sim] lifecycle  = {"autostart" if autostart else "started by pose_gate" if gate else "MANUAL"}')
    print(f'[nav2_sim] cmd_vel    -> /cmd_vel (gz bridge; no ackermann_bridge in simulation)')

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
        # configured FIRST so bond_timeout from the file reaches the manager; the
        # sim launch records the spurious teardown that happens otherwise.
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
                 '-p', f'database:={cfg("database")}',
                 '-p', f'odom_frame:={cfg("odom_frame")}',
                 '-p', f'base_frame:={cfg("base_frame")}',
                 '-p', f'init_ignore_s:={cfg("init_ignore_s")}',
                 '-p', f'max_step:={cfg("max_step")}',
                 '-p', f'margin:={cfg("extent_margin")}',
                 '-p', f'manage_lifecycle:={str(not autostart).lower()}']
            # an EMPTY `-p log_file:=` is a parse error for rcl ("Couldn't parse
            # parameter override rule"), so the option is added only when set
            + (['-p', f'log_file:={cfg("gate_log")}'] if cfg('gate_log') else []),
            output='screen'))
    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('params_file', default_value=os.path.join(SHARE, 'params', 'nav2_sim.yaml')),
        DeclareLaunchArgument('use_sim_time', default_value='true',
                              description='true in simulation, always.'),
        DeclareLaunchArgument('map_yaml', default_value='',
                              description='offline tests only: serve this map with map_server instead of RTAB-Map'),
        DeclareLaunchArgument('database', default_value='',
                              description='prior RTAB-Map database; gives the pose gate its map extent'),
        DeclareLaunchArgument('odom_frame', default_value='odom',
                              description="what the front end broadcasts in simulation "
                                          "(the real rig uses orbslam3_world)."),
        DeclareLaunchArgument('base_frame', default_value='base_link'),
        DeclareLaunchArgument('autostart', default_value='false',
                              description='true starts Nav2 immediately (offline tests); '
                                          'false lets the pose gate start it'),
        DeclareLaunchArgument('pose_gate', default_value='true'),
        DeclareLaunchArgument('init_ignore_s', default_value='8.0'),
        DeclareLaunchArgument('max_step', default_value='1.0'),
        DeclareLaunchArgument('extent_margin', default_value='2.0'),
        DeclareLaunchArgument('gate_log', default_value='',
                              description='file receiving the gate transitions and lifecycle commands'),
        OpaqueFunction(function=_setup),
    ])
