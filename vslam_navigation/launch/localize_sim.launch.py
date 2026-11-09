"""Relocalise the simulated robot on a prior RTAB-Map map, and record how well it
went. The simulation counterpart of localize_real.launch.py.

    ros2 launch vslam_navigation localize_sim.launch.py map_name:=realistic_01_prior
    ros2 bag play "<a bag of the same world>" --rate 0.3        # replay, or
    ros2 launch aws_robomaker_small_warehouse_world small_warehouse.launch.py   # live

Writes map/<map_name>/localize_<run_name>.csv: one row per localisation pose and
per map->odom correction, with the step size and whether the pose is inside the
prior's extent, then a summary line. That file is the evidence a relocalisation
actually happened, and it is what the real rig's first successful relocalisation
was judged on.

WHAT THIS FILE ADDS over the warehouse package's localization_rtabmap.launch.py,
which it includes and does not replace:
  * map_name instead of a raw database_path, so the database and the recorder's
    CSV land together under map/<map_name>/ exactly as on the rig;
  * localization_recorder.py, which that launch has no notion of;
  * RViz, which that launch deliberately leaves to its caller.

Everything about the frame tree, the front-end renaming and the 0.338 / 0.192
imu_link offset is handled there and documented there. In short the chain is
map -> odom -> vio_body -> base_footprint -> base_link, with exactly one
publisher per edge, and ORB-SLAM3 or VINS is told to publish odom -> vio_body
directly rather than being bridged.

NOT included: localization_tf.launch.py. That is the ground-truth chain and it
publishes the same map -> odom edge; running both gives two publishers for one
edge and the pose visibly fights itself.

read_only defaults true: a localisation run must not write into the prior it is
being scored against.
"""
import os

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

WORKSPACE = os.path.expanduser('~/wil_project')
PKG = 'vslam_navigation'
SHARE = get_package_share_directory(PKG)
LIB = os.path.join(get_package_prefix(PKG), 'lib', PKG)


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
    run_name = cfg('run_name') or f'{map_name}_localize'
    csv = os.path.join(map_dir, f'localize_{run_name}.csv')
    # vslam_navigation spells ORB-SLAM3 'orb' everywhere -- rtabmap.launch.py's
    # frontend:=orb, mapping_sim, mapping_real. localization_rtabmap.launch.py
    # spells it 'orbslam3' and rejects anything else. Translate at the boundary
    # rather than make this package inconsistent with itself. Passing 'orb'
    # straight through aborted the entire launch with
    #     frontend must be vins|orbslam3, got 'orb'
    # and the aborts that follow it ("could not create service: rcl node's
    # context is invalid") are the delayed nodes being torn down mid-startup,
    # not faults of their own. Found on the first real run, 6 Nov 2026.
    ALIAS = {'orb': 'orbslam3', 'orbslam3': 'orbslam3', 'vins': 'vins'}
    frontend = cfg('frontend').lower()
    if frontend not in ALIAS:
        raise RuntimeError(f"frontend must be orb|vins, got {frontend!r}")
    frontend_child = ALIAS[frontend]

    print(f'[localize_sim] prior    = {database}')
    print(f'[localize_sim] frontend = {frontend} (as {frontend_child} to localization_rtabmap)')
    print(f'[localize_sim] record   = {csv}')

    loc = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('aws_robomaker_small_warehouse_world'),
            'launch', 'localization_rtabmap.launch.py')),
        launch_arguments={
            'use_sim_time': 'true',
            'frontend': frontend_child,
            'mode': 'localize',
            'database_path': database,
            'read_only': cfg('read_only'),
            'start_frontend': cfg('start_frontend'),
            'image_transport': cfg('image_transport'),
            'ground_height': cfg('ground_height'),
            'optimize_max_error': cfg('optimize_max_error'),
            'detection_rate': cfg('detection_rate'),
            'shim_rate': cfg('shim_rate'),
            'publish_joint_states': cfg('publish_joint_states'),
        }.items())

    rec = ExecuteProcess(
        cmd=['python3', os.path.join(LIB, 'localization_recorder.py'), '--ros-args',
             '-p', 'use_sim_time:=true',
             '-p', f'csv:={csv}',
             '-p', f'database:={database}',
             '-p', 'odom_frame:=odom',          # simulation; the rig uses orbslam3_world
             '-p', 'base_frame:=base_link',
             '-p', f'margin:={cfg("extent_margin")}'],
        output='screen')

    acts = [loc, rec]
    if truthy('rviz'):
        acts.append(Node(package='rviz2', executable='rviz2', name='localize_rviz',
                         output='screen',
                         arguments=['-d', os.path.join(SHARE, 'rviz', 'rtab_map.rviz')],
                         parameters=[{'use_sim_time': True}]))
    return acts


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('map_name', description='prior map: map/<map_name>/<map_name>.db'),
        DeclareLaunchArgument('run_name', default_value='',
                              description='names the CSV; empty = <map_name>_localize'),
        DeclareLaunchArgument('frontend', default_value='orb', description='orb | vins'),
        DeclareLaunchArgument('read_only', default_value='true',
                              description='true: the run cannot write into the prior it is scored against.'),
        DeclareLaunchArgument('start_frontend', default_value='true',
                              description='false reuses a front end already running in another terminal.'),
        DeclareLaunchArgument('image_transport', default_value='compressed',
                              description='the sim bags and the live bridge are both compressed.'),
        DeclareLaunchArgument('ground_height', default_value='0.29'),
        DeclareLaunchArgument('optimize_max_error', default_value='10.0'),
        DeclareLaunchArgument('detection_rate', default_value='1.0'),
        DeclareLaunchArgument('shim_rate', default_value='4.0'),
        DeclareLaunchArgument('publish_joint_states', default_value='true'),
        DeclareLaunchArgument('extent_margin', default_value='2.0',
                              description='metres outside the prior still counted as inside its extent.'),
        DeclareLaunchArgument('rviz', default_value='true'),
        OpaqueFunction(function=_setup),
    ])
