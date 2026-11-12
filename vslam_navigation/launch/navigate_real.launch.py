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
    if not os.path.isfile(database):
        raise RuntimeError(
            f'prior map {database} does not exist.\n'
            f'  Build one first:  ros2 launch vslam_navigation mapping_real.launch.py '
            f'map_name:={map_name} loop_closure:=false')

    print(f'[navigate_real] prior    = {database}')
    print(f'[navigate_real] run      = {run_name}')
    # There is no world file to read a spawn pose from: on the rig the start pose
    # is wherever the robot is physically placed, so the constraint is
    # operational rather than configured.
    if cfg('start_pose').strip().lower() == 'origin':
        print('[navigate_real] start_pose:=origin -- relocalisation is seeded at the '
              "prior's FIRST NODE,")
        print('[navigate_real]   i.e. the spot this prior was recorded from. Place the '
              'robot there before starting,')
        print('[navigate_real]   or pass start_pose:="x y z r p y" in map coordinates.')

    localize = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(SHARE, 'launch', 'localize_real.launch.py')),
        launch_arguments={
            'map_name': map_name,
            'run_name': run_name,
            'calibration': cfg('calibration'),
            'read_only': 'true',
            'use_viewer': 'false',
            'start_pose': cfg('start_pose'),
            'relocalize': cfg('relocalize'),
            'min_inliers': cfg('min_inliers'),
            'min_inliers_distribution': cfg('min_inliers_distribution'),
            'odom_cache_size': cfg('odom_cache_size'),
            'geometric_check': cfg('geometric_check'),
            'icp_correspondence_ratio': cfg('icp_correspondence_ratio'),
            'icp_max_rotation': cfg('icp_max_rotation'),
            'estimation_type': cfg('estimation_type'),
            'vis_inlier_distance': cfg('vis_inlier_distance'),
            'gen_scan': cfg('gen_scan'),
            'localize_delay_s': cfg('localize_delay_s'),
            'overwrite': cfg('overwrite'),
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
        DeclareLaunchArgument('wheel_base', default_value='0.60'),
        DeclareLaunchArgument('max_speed', default_value='0.6'),
        DeclareLaunchArgument('overwrite', default_value='false',
                              description='true replaces an existing run CSV of the same name.'),
        DeclareLaunchArgument('min_inliers', default_value='20',
                              description='Vis/MinInliers; try 35 against aisle aliasing.'),
        DeclareLaunchArgument('min_inliers_distribution', default_value='0.0',
                              description='Vis/MinInliersDistribution; 0 disables. Try 0.05-0.1: '
                                          'rejects a match bunched on repeating structure.'),
        DeclareLaunchArgument('odom_cache_size', default_value='10',
                              description='RGBD/MaxOdomCacheSize; anti-teleport context.'),
        DeclareLaunchArgument('relocalize', default_value='true',
                              description='false = dead reckon on the prior from the '
                                          'start_pose seed; no correction is applied and '
                                          'drift is unbounded.'),
        DeclareLaunchArgument('start_pose', default_value='origin',
                              description='passed to the localisation layer; the robot is '
                                          'assumed to start at the spawn point unless this '
                                          "is set to 'last' or to an explicit pose."),
        DeclareLaunchArgument('rtabmap_delay', default_value='5.0'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('geometric_check', default_value='false',
                              description='Reg/Strategy 2: verify an accepted loop closure by ICP against the 3-D '
                        'structure, not only the 2-D features that proposed it. The '
                        'one acceptance test outside the null space that defeats '
                        'OptimizeMaxError and MaxOdomCacheSize.'),
        DeclareLaunchArgument('icp_correspondence_ratio', default_value='0.3',
                              description='Icp/CorrespondenceRatio, the veto itself. geometric_check:=true only.'),
        DeclareLaunchArgument('icp_max_rotation', default_value='0.35',
                              description='Icp/MaxRotation in rad (0.35 = 20 deg). geometric_check:=true only.'),
        DeclareLaunchArgument('estimation_type', default_value='1',
                              description='Vis/EstimationType: 0 = 3D->3D sparse-geometry veto, '
                      '1 = PnP (RTAB-Map default).'),
        DeclareLaunchArgument('vis_inlier_distance', default_value='0.1',
                              description='Vis/InlierDistance (m); estimation_type:=0 only.'),
        DeclareLaunchArgument('gen_scan', default_value='false',
                              description='generate a laser scan from depth for the live frames; '
                      'geometric_check needs it.'),
        DeclareLaunchArgument('localize_delay_s', default_value='0.0',
                              description='hold RTAB-Map back this long so its first '
                                          'fix cannot latch on an unstable front-end '
                                          'orientation.'),
        OpaqueFunction(function=_setup),
    ])
