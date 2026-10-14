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
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

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
        DeclareLaunchArgument('experiment_id', default_value=''),
        DeclareLaunchArgument('dataset_path', default_value=''),
        DeclareLaunchArgument('world_path', default_value=''),
        DeclareLaunchArgument('replay_rate', default_value='1.0'),
        DeclareLaunchArgument('run_command', default_value=''),
        DeclareLaunchArgument('run_notes', default_value=''),
        DeclareLaunchArgument('use_viewer', default_value='true'),
        DeclareLaunchArgument(
            'imu_accel_scale', default_value=str(ACCEL_SCALE),
            description='Accelerometer correction; 1.0 once the IMU is recalibrated.'),
        DeclareLaunchArgument(
            'filter', default_value='false',
            description='RY-SLAM dynamic-object filtering. When true, also starts '
                        'the YOLO detector node. Default false = stock tracker.'),
        DeclareLaunchArgument(
            'filter_mode', default_value='class',
            description="'class' masks every detected box (RY-SLAM). 'motion' "
                        'masks a box only after the tracker has seen it move: '
                        'frame k is scored after tracking and frame k+1 inherits '
                        'the verdict by IoU. Writes motion_votes.csv.'),
        DeclareLaunchArgument(
            'dets_csv', default_value='',
            description='Precomputed detections from script/yolo_precompute.py. '
                        'When set (with filter:=true) the live detector is NOT '
                        'started and every frame is masked from the file by its '
                        'own stamp, so repeat runs mask identical frames. This is '
                        'the only way a filter ON/OFF comparison is valid on a '
                        'machine that cannot run the detector in real time.'),
        DeclareLaunchArgument('weights_path', default_value=DEFAULT_WEIGHTS),
        DeclareLaunchArgument(
            # 'auto' = every class the chosen weight knows; see the detector node.
            'dynamic_classes', default_value="['auto']"),
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
            'experiment_id': ParameterValue(LaunchConfiguration('experiment_id'), value_type=str),
            'dataset_path': ParameterValue(LaunchConfiguration('dataset_path'), value_type=str),
            'world_path': ParameterValue(LaunchConfiguration('world_path'), value_type=str),
            'replay_rate': LaunchConfiguration('replay_rate'),
            'run_command': ParameterValue(LaunchConfiguration('run_command'), value_type=str),
            'run_notes': ParameterValue(LaunchConfiguration('run_notes'), value_type=str),
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
            'dets_csv': ParameterValue(LaunchConfiguration('dets_csv'), value_type=str),
            'filter_mode': LaunchConfiguration('filter_mode'),
        }],
    )

    detector = Node(
        package='orbslam3_ros2',
        executable='dynamic_detector_node.py',
        name='dynamic_detector',
        output='screen',
        emulate_tty=True,
        # Live detector only when filtering AND no precomputed file was given.
        condition=IfCondition(PythonExpression([
            "'", LaunchConfiguration('filter'), "'.lower() in ('true', '1') and '",
            LaunchConfiguration('dets_csv'), "' == ''"])),
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
