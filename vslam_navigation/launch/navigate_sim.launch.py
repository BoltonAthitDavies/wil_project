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


def _world_spawn_pose(world_path):
    """The ackermann robot's spawn pose as the world file sets it, or None.

    Worth reading at launch because start_pose:=origin asserts that the robot
    begins where the prior's first node is, and nothing downstream checks it. On
    11 Nov 2026 the U static world had been edited to spawn at 13.0, 14.0 while
    the prior recorded from that world begins at 5.25, 20.0 -- 9.8 m apart, which
    seeds relocalisation in the wrong aisle. The pose gate does catch it (the
    correction when the relocaliser recovers far exceeds max_step, so Nav2 never
    starts) but the symptom, a stack that silently never activates, looks nothing
    like a wrong spawn pose. Printing both makes it a two-second check.

    Only the ACTIVE pose counts: the edited world kept the original as an XML
    comment immediately above, so a naive search finds the wrong one.
    """
    try:
        text = open(world_path).read()
    except OSError:
        return None
    i = text.find('<name>ackermann_robot_001</name>')
    if i < 0:
        return None
    for line in text[i:].splitlines()[1:6]:
        line = line.strip()
        if line.startswith('<!--'):
            continue                      # the commented-out original
        if line.startswith('<pose>'):
            return line[len('<pose>'):line.index('</pose>')].strip()
    return None


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

    # The spawn pose the robot will actually have, beside the seeding assumption.
    world = cfg('world') or os.path.join(
        WAREHOUSE, 'worlds', 'small_warehouse', 'small_warehouse.world')
    spawn = _world_spawn_pose(world)
    print(f'[navigate_sim] world    = {os.path.basename(world)}')
    print(f'[navigate_sim] spawn    = {spawn if spawn else "NOT FOUND in the world file"}'
          f'   (x y z r p y, as the world sets it)')
    if cfg('start_pose').strip().lower() == 'origin':
        print('[navigate_sim] start_pose:=origin -- relocalisation is seeded at the '
              "prior's FIRST NODE.")
        print('[navigate_sim]   That is only correct if the spawn above is where '
              'this prior was recorded from.')
        print('[navigate_sim]   If it is not, pass the spawn pose in MAP coordinates '
              'as start_pose:="x y z r p y".')

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
                # Both are additive to what localize_sim already runs, because
                # neither lands on the topic it might collide with: the bridges
                # publish /model/<robot>/odometry and /model/<robot>/joint_state,
                # while localize_sim's joint_state_publisher publishes
                # /joint_states and the VSLAM odometry is its own topic. Nothing
                # here publishes a transform, so the one-publisher-per-edge
                # contract is untouched.
                'bridge_wheel_odom': cfg('bridge_wheel_odom'),
                'bridge_joint_states': cfg('bridge_joint_states'),
                # Opt-in, and it is the warehouse launch's own bridge rather than
                # one started beside it: gz model poses as a TFMessage whose
                # child_frame_id is the model name, on /world/default/pose/info.
                # It uses pose/info and NOT dynamic_pose/info deliberately -- the
                # dynamic props are <static>true</static> driven by
                # KinematicTrajectory, so they never appear in dynamic_pose/info
                # and a subscriber to that draws every prop frozen at its spawn
                # pose. Costs ~1.0 MB/s, which is why it is off unless asked for.
                'bridge_model_poses': cfg('bridge_model_poses'),
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
            'wait_for_imu_init': cfg('wait_for_imu_init'),
            'orb_output_path': cfg('orb_output_path'),
            'overwrite': cfg('overwrite'),
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
        DeclareLaunchArgument(
            'bridge_wheel_odom', default_value='false',
            description='publish the simulated wheel/steering odometry on '
                        '/model/<robot_name>/odometry. An independent dead-reckoning '
                        'signal to compare the VSLAM estimate against, and the sim '
                        'counterpart of the real wheel+gyro baseline. Publishes no '
                        'transform, so it never competes with the VSLAM chain.'),
        DeclareLaunchArgument(
            'bridge_joint_states', default_value='false',
            description='publish the TRUE wheel and steering joint angles on '
                        '/model/<robot_name>/joint_state, from the JointStatePublisher '
                        'in model.sdf. Not quantised encoder counts -- script/'
                        'encoder_sim.py adds the resolution. Distinct from '
                        '/joint_states, which localize_sim publishes for the robot '
                        'description, so both can run.'),
        DeclareLaunchArgument(
            'bridge_model_poses', default_value='false',
            description='publish every gz model pose on /world/default/pose/info as a '
                        'TFMessage (child_frame_id = model name). Needed to record '
                        'where the moving props and actors actually were, which is '
                        'ground truth for a dynamic world; pointless in a static one, '
                        'where the layout is in the .world file. NOT published on /tf.'),
        DeclareLaunchArgument('init_ignore_s', default_value='8.0',
                              description='seconds of localisation output the gate ignores while the '
                                          'front end initialises.'),
        DeclareLaunchArgument('max_step', default_value='1.0',
                              description='map->odom correction step above which the gate drops [m].'),
        DeclareLaunchArgument('overwrite', default_value='false',
                              description='true replaces an existing run CSV of the same name.'),
        DeclareLaunchArgument('wait_for_imu_init', default_value='true',
                              description='hold the front end back until its inertial state is '
                                          'valid; prevents a wrong correction being latched at '
                                          'startup.'),
        DeclareLaunchArgument('orb_output_path', default_value='',
                              description="directory for the front end's vio.csv."),
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
        DeclareLaunchArgument('extent_margin', default_value='2.0'),
        DeclareLaunchArgument('stack_delay', default_value='12.0',
                              description='seconds after Gazebo before the localisation stack starts.'),
        DeclareLaunchArgument('nav2_delay', default_value='8.0',
                              description='further seconds before Nav2 starts.'),
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
