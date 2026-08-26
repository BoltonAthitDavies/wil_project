"""ORB-SLAM3 stereo-inertial on the simulated WiL rig (Gazebo Fortress).

Run the sim or a bag in a SECOND terminal -- as with every vins_fusion_ros2
launch file, this one starts the estimator only:

    ros2 launch aws_robomaker_small_warehouse_world small_warehouse.launch.py
  or
    ros2 bag play dataset/sim --clock

Then score the result against ground truth with the existing scripts, which need
no modification because the node writes VINS's CSV format:

    python3 script/plot_vio_vs_gt.py \\
        ~/output/wil_sim_orbslam3_stereo_imu/vio.csv \\
        ~/output/wil_sim_stereo_imu/ground_truth.csv \\
        ~/output/wil_sim_orbslam3_stereo_imu/orb_vs_gt
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# The vocabulary is ~145 MB of text built by ../build_orbslam3.sh. It is
# deliberately NOT installed into the package share dir -- colcon would copy it
# on every build, and disk on this machine is tight.
WORKSPACE = os.path.expanduser('~/wil_project')
DEFAULT_VOCAB = os.path.join(WORKSPACE, 'thirdparty', 'ORB_SLAM3', 'Vocabulary', 'ORBvoc.txt')

# Sibling of VINS's ~/output/wil_sim_stereo_imu/, so the two systems' vio.csv
# files never overwrite each other and can be plotted side by side.
DEFAULT_OUTPUT = os.path.expanduser('~/output/wil_sim_orbslam3_stereo_imu')


def generate_launch_description():
    config_file = os.path.join(
        get_package_share_directory('orbslam3_ros2'),
        'config', 'wil_sim', 'stereo_imu.yaml')

    args = [
        DeclareLaunchArgument(
            'vocabulary_file', default_value=DEFAULT_VOCAB,
            description='ORB vocabulary. Build it with ./build_orbslam3.sh.'),
        DeclareLaunchArgument(
            'output_path', default_value=DEFAULT_OUTPUT,
            description='Directory for vio.csv. Created if absent.'),
        DeclareLaunchArgument(
            'use_viewer', default_value='true',
            description='Pangolin map/feature window. Set false for headless runs.'),
    ]

    node = Node(
        package='orbslam3_ros2',
        executable='orbslam3_node',
        name='orbslam3',
        output='screen',
        emulate_tty=True,
        parameters=[{
            # Sim publishes /clock and every timestamp we consume is on it --
            # the same clock script/extract_gt.py sampled for ground_truth.csv,
            # which is what makes the two directly comparable.
            'use_sim_time': True,

            'config_file': config_file,
            'vocabulary_file': LaunchConfiguration('vocabulary_file'),
            'output_path': LaunchConfiguration('output_path'),
            'use_viewer': LaunchConfiguration('use_viewer'),

            'use_imu': True,
            # Simulated gravity is 9.8 against ORB-SLAM3's hardcoded 9.81 --
            # 0.1%, far below the noise floor. No correction needed here.
            # (The real rig is off by 5.6% and does need one.)
            'imu_accel_scale': 1.0,

            # gz bridges raw sensor_msgs/Image, so plain "raw" transport.
            'image0_topic': '/cam0/image_raw',
            'image1_topic': '/cam1/image_raw',
            'imu_topic': '/imu',
            'image_transport': 'raw',

            # ORB-SLAM3's world origin is its first keyframe, gravity-aligned
            # after IMU init -- NOT the sim's `odom`. Naming it distinctly keeps
            # that honest in the TF tree. viewer.py and plot_vio_vs_gt.py both
            # align on the first pose, so no extra work is needed downstream.
            'world_frame_id': 'orbslam3_world',
            'body_frame_id': 'base_footprint',
            'publish_tf': True,

            # Cameras nominally fire together, but the bag holds 804 cam0
            # frames against 805 cam1, so the stamps are close, not identical.
            # Half a frame interval at 30 Hz.
            'sync_slop': 0.02,
        }],
    )

    return LaunchDescription(args + [node])
