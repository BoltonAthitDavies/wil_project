"""The full real-rig navigation stack in ONE command: ORB-SLAM3 + RTAB-Map
localize mode on a prior map (localize_real.launch.py) + Nav2 with the pose
gate and the Ackermann bridge (nav2_real.launch.py) + RViz as the GUI.

    ros2 launch vslam_navigation navigate_real.launch.py map_name:=realv7_001_prior
    # bag replay for a dry run (bridge_dry_run true by default), or the robot live:
    ros2 launch vslam_navigation navigate_real.launch.py map_name:=<prior> \
        use_sim_time:=false bridge_dry_run:=false

Order of events: ORB starts; RTAB-Map starts 5 s later and relocalises; the
pose gate waits out the init transient and the stability window, then STARTS
Nav2; a goal set in RViz (2D Goal Pose, /goal_pose) is planned on the prior
map and tracked; the bridge turns /cmd_vel into robot commands, zeroing them
whenever the gate drops. Every argument of both includes is passed explicitly
(launch configurations leak between includes otherwise -- see the 4 Nov notes
in rtabmap_ros/MEMORY_CLAUDE).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

WORKSPACE = os.path.expanduser('~/wil_project')
SHARE = get_package_share_directory('vslam_navigation')


def _setup(context, *args, **kwargs):
    cfg = lambda k: LaunchConfiguration(k).perform(context)  # noqa: E731
    map_name = cfg('map_name')
    database = os.path.join(WORKSPACE, 'map', map_name, f'{map_name}.db')
    run_name = cfg('run_name') or f'{map_name}_navigate'
    localize = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(SHARE, 'launch', 'localize_real.launch.py')),
        launch_arguments={
            'map_name': map_name,
            'run_name': run_name,
            'calibration': cfg('calibration'),
            'read_only': 'true',
            'use_viewer': 'false',
            'rviz': 'false',                      # one RViz, with the navigation config
            'rtabmap_delay': cfg('rtabmap_delay'),
        }.items())
    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(SHARE, 'launch', 'nav2_real.launch.py')),
        launch_arguments={
            'use_sim_time': cfg('use_sim_time'),
            'map_yaml': '',
            'database': database,
            'autostart': 'false',
            'pose_gate': 'true',
            'bridge': 'true',
            'bridge_dry_run': cfg('bridge_dry_run'),
            'wheel_base': cfg('wheel_base'),
            'max_speed': cfg('max_speed'),
            'gate_log': os.path.join(WORKSPACE, 'map', map_name, f'gate_{run_name}.csv'),
        }.items())
    acts = [localize, nav2]
    if cfg('rviz').lower() in ('true', '1', 'yes'):
        acts.append(Node(package='rviz2', executable='rviz2', name='nav_rviz', output='screen',
                         arguments=['-d', os.path.join(SHARE, 'rviz', 'navigation.rviz')],
                         parameters=[{'use_sim_time': cfg('use_sim_time').lower() in ('true', '1', 'yes')}]))
    return acts


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('map_name', description='prior map: map/<map_name>/<map_name>.db'),
        DeclareLaunchArgument('run_name', default_value=''),
        DeclareLaunchArgument('calibration', default_value='wil'),
        DeclareLaunchArgument('use_sim_time', default_value='true',
                              description='true for bag replay, false on the robot'),
        DeclareLaunchArgument('bridge_dry_run', default_value='true',
                              description='false only on the robot after the chassis values are verified'),
        DeclareLaunchArgument('wheel_base', default_value='0.46'),
        DeclareLaunchArgument('max_speed', default_value='0.6'),
        DeclareLaunchArgument('rtabmap_delay', default_value='5.0'),
        DeclareLaunchArgument('rviz', default_value='true'),
        OpaqueFunction(function=_setup),
    ])
