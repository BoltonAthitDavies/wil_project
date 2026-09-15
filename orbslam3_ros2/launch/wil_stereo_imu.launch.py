"""ORB-SLAM3 STEREO-INERTIAL on the real WiL rig -- AN EXPERIMENT, NOT A BASELINE.

Use wil_stereo.launch.py for real-rig runs you intend to trust. Read the header
of config/wil/stereo_imu.yaml before interpreting anything this produces.

Two hardware problems, neither of them ORB-SLAM3's fault:

  1. The IMU->cam0 extrinsic is a zeroed PLACEHOLDER -- no Kalibr calibration
     exists for this rig. The real ~0.1 m lever arm is modelled as zero, and at
     the 1.29 rad/s peak yaw this rig reaches, the unmodelled centripetal term is
     a genuine acceleration error. Cannot be fixed in software.

  2. The accelerometer reads 9.2642 m/s^2 stationary against a true 9.81 -- a
     5.6% scale error. VINS hides this by setting g_norm to the OBSERVED value.
     ORB-SLAM3 hardcodes GRAVITY_VALUE = 9.81 in ImuTypes.h with no config key,
     so that escape hatch does not exist. Handled instead by imu_accel_scale
     below, which rescales the measurement to match the assumed gravity.

Replay in a SECOND terminal:
    ros2 bag play dataset/rosbag_realsense_imu_cambaseline95mm
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

WORKSPACE = os.path.expanduser('~/wil_project')
DEFAULT_VOCAB = os.path.join(WORKSPACE, 'thirdparty', 'ORB_SLAM3', 'Vocabulary', 'ORBvoc.txt')
DEFAULT_OUTPUT = os.path.expanduser('~/output/wil_orbslam3_stereo_imu')

# 9.81 / 9.2642. Set back to 1.0 once the accelerometer is recalibrated
# (rs-imu-calibration.py) -- leaving it on after a fix would double-correct.
ACCEL_SCALE = 9.81 / 9.2642

# YOLO weights for dynamic-object detection. Classes are bin/box/bucket -- these
# are warehouse-prop weights, so on real-rig bags expect fewer hits than in sim.
DEFAULT_WEIGHTS = os.path.join(WORKSPACE, 'weight', 'best.pt')


def generate_launch_description():
    config_file = os.path.join(
        get_package_share_directory('orbslam3_ros2'),
        'config', 'wil', 'stereo_imu.yaml')

    args = [
        DeclareLaunchArgument('vocabulary_file', default_value=DEFAULT_VOCAB),
        DeclareLaunchArgument('output_path', default_value=DEFAULT_OUTPUT),
        DeclareLaunchArgument('use_viewer', default_value='true'),
        DeclareLaunchArgument(
            'imu_accel_scale', default_value=str(ACCEL_SCALE),
            description='Accelerometer correction; 1.0 once the IMU is recalibrated.'),
        DeclareLaunchArgument(
            'filter', default_value='false',
            description='RY-SLAM dynamic-object filtering. When true, also starts '
                        'the YOLO detector node. Default false = stock tracker.'),
        DeclareLaunchArgument('weights_path', default_value=DEFAULT_WEIGHTS),
        DeclareLaunchArgument(
            'dynamic_classes', default_value="['bin', 'box', 'bucket']"),
        DeclareLaunchArgument('publish_debug_image', default_value='false'),
    ]

    node = Node(
        package='orbslam3_ros2',
        executable='orbslam3_node',
        name='orbslam3',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'use_sim_time': False,    # no /clock on these bags

            'config_file': config_file,
            'vocabulary_file': LaunchConfiguration('vocabulary_file'),
            'output_path': LaunchConfiguration('output_path'),
            'use_viewer': LaunchConfiguration('use_viewer'),

            'use_imu': True,
            'imu_accel_scale': LaunchConfiguration('imu_accel_scale'),

            'image0_topic': '/cam0/image_raw',
            'image1_topic': '/cam1/image_raw',
            # Legacy name: the topic is called /hwt101ct_yaw_publisher for
            # historical reasons but now carries a real 6-axis RealSense IMU at
            # 199.5 Hz. See vins config/wil/stereo_imu.yaml:12-18.
            'imu_topic': '/hwt101ct_yaw_publisher',
            'image_transport': 'compressed',

            # Body IS cam0 here, because IMU.T_b_c1 is the identity placeholder.
            # Renaming the frame would imply a calibration that does not exist.
            'world_frame_id': 'orbslam3_world',
            'body_frame_id': 'imu_link',
            'publish_tf': True,

            'sync_slop': 0.02,

            'filter': LaunchConfiguration('filter'),
        }],
    )

    detector = Node(
        package='orbslam3_ros2',
        executable='dynamic_detector_node.py',
        name='dynamic_detector',
        output='screen',
        emulate_tty=True,
        condition=IfCondition(LaunchConfiguration('filter')),
        parameters=[{
            'use_sim_time': False,
            'weights_path': LaunchConfiguration('weights_path'),
            'dynamic_classes': LaunchConfiguration('dynamic_classes'),
            'image_topic': '/cam0/image_raw',
            # These bags carry ONLY /camN/image_raw/compressed, so the detector
            # decodes JPEG itself -- to COLOUR, unlike the tracker, which wants
            # grayscale. Same blobs, different decode.
            'image_transport': 'compressed',
            'publish_debug_image': LaunchConfiguration('publish_debug_image'),
        }],
    )

    return LaunchDescription(args + [node, detector])
