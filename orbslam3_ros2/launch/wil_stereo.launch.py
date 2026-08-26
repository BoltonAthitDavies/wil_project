"""ORB-SLAM3 STEREO (no IMU) on the real WiL rig -- the trustworthy real-rig path.

Replay in a SECOND terminal:
    ros2 bag play dataset/rosbag_realsense_imu_cambaseline95mm

No `image_transport republish` processes needed. The bags carry only
/camN/image_raw/compressed, and the node subscribes through image_transport with
transport="compressed", so it decompresses in-process. (VINS needs two extra
republish nodes for the same job -- see vins config/wil/stereo_imu.yaml:32-38.)

WHY STEREO AND NOT STEREO-INERTIAL IS THE DEFAULT HERE
    Two unresolved hardware problems make the real rig's IMU untrustworthy: the
    IMU->cam0 extrinsic is a zeroed placeholder with no Kalibr calibration, and
    the accelerometer has a measured 5.6% scale error. Neither can touch this
    config, because there is no IMU in it, and stereo is metric on its own from
    the calibrated 93.6 mm baseline. See wil_stereo_imu.launch.py to experiment.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

WORKSPACE = os.path.expanduser('~/wil_project')
DEFAULT_VOCAB = os.path.join(WORKSPACE, 'thirdparty', 'ORB_SLAM3', 'Vocabulary', 'ORBvoc.txt')
DEFAULT_OUTPUT = os.path.expanduser('~/output/wil_orbslam3_stereo')


def generate_launch_description():
    config_file = os.path.join(
        get_package_share_directory('orbslam3_ros2'),
        'config', 'wil', 'stereo.yaml')

    args = [
        DeclareLaunchArgument('vocabulary_file', default_value=DEFAULT_VOCAB),
        DeclareLaunchArgument('output_path', default_value=DEFAULT_OUTPUT),
        DeclareLaunchArgument('use_viewer', default_value='true'),
    ]

    node = Node(
        package='orbslam3_ros2',
        executable='orbslam3_node',
        name='orbslam3',
        output='screen',
        emulate_tty=True,
        parameters=[{
            # These bags carry no /clock -- `ros2 bag play` without --clock
            # republishes on the wall clock, so sim time would stall at 0.
            'use_sim_time': False,

            'config_file': config_file,
            'vocabulary_file': LaunchConfiguration('vocabulary_file'),
            'output_path': LaunchConfiguration('output_path'),
            'use_viewer': LaunchConfiguration('use_viewer'),

            'use_imu': False,
            'imu_accel_scale': 1.0,   # unused with use_imu False

            # Base topics WITHOUT the /compressed suffix: image_transport appends
            # it according to the transport below.
            'image0_topic': '/cam0/image_raw',
            'image1_topic': '/cam1/image_raw',
            'image_transport': 'compressed',

            # In pure stereo there is no separate body frame -- cam0 IS the body,
            # which is why config/wil/stereo.yaml carries no IMU.T_b_c1 and the
            # node falls back to identity for it.
            'world_frame_id': 'orbslam3_world',
            'body_frame_id': 'cam0',
            'publish_tf': True,

            # Real cameras are hardware-triggered but not perfectly aligned;
            # half a frame interval at 30 Hz.
            'sync_slop': 0.02,
        }],
    )

    return LaunchDescription(args + [node])
