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
from launch.conditions import IfCondition
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

# YOLO weights for dynamic-object detection. Trained on the warehouse props, so
# its classes are bin/box/bucket -- NOT the COCO set the stock yolo26*.pt carry.
DEFAULT_WEIGHTS = os.path.join(WORKSPACE, 'weight', 'best.pt')


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
            'align_first_pose', default_value='false',
            description="Latch the inverse of the first tracked body pose so the "
                        "first published pose is identity and everything after is "
                        "in that initial BODY frame -- i.e. REP-103 (x forward, "
                        "y left, z up). With this FALSE, poses come out in "
                        "ORB-SLAM3's own internal world frame, which is not "
                        "REP-103: driving forward then renders as a turn, and the "
                        "trajectory will not line up with a map or with VINS. "
                        "Leave false only to reproduce an older run."),
        DeclareLaunchArgument(
            'use_viewer', default_value='true',
            description='Pangolin map/feature window. Set false for headless runs.'),
        DeclareLaunchArgument(
            'image_transport', default_value='raw',
            description='"raw" for the stock sim, which bridges plain '
                        'sensor_msgs/Image. Set "compressed" when the sim was '
                        'launched with compressed_images:=True, which adds '
                        '/camN/image_raw/compressed -- that is the transport the '
                        'REAL rig uses (see wil_stereo_imu.launch.py), so this is '
                        'how you exercise the identical path in sim.'),
        DeclareLaunchArgument(
            'filter', default_value='false',
            description='RY-SLAM dynamic-object filtering. When true, also starts '
                        'the YOLO detector node and ORB-SLAM3 discards keypoints '
                        'landing on detected movers. Default false, which is the '
                        'stock tracker bit for bit.'),
        DeclareLaunchArgument(
            'weights_path', default_value=DEFAULT_WEIGHTS,
            description='Ultralytics .pt for the detector. Only used with '
                        'filter:=true.'),
        DeclareLaunchArgument(
            'dynamic_classes', default_value="['bin', 'box', 'bucket']",
            description='Class NAMES treated as dynamic, resolved against the '
                        "model's own names dict at load. Names the model does not "
                        'know are reported at startup.'),
        DeclareLaunchArgument(
            'publish_debug_image', default_value='false',
            description='Publish /orbslam3/dynamic_debug with the boxes drawn on, '
                        'for rqt_image_view.'),
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
            'align_first_pose': LaunchConfiguration('align_first_pose'),

            'use_imu': True,
            # Simulated gravity is 9.8 against ORB-SLAM3's hardcoded 9.81 --
            # 0.1%, far below the noise floor. No correction needed here.
            # (The real rig is off by 5.6% and does need one.)
            'imu_accel_scale': 1.0,

            # The base topics, WITHOUT any /compressed suffix: image_transport
            # appends the suffix for the transport it is asked for, so these are
            # correct for both settings of the image_transport argument.
            'image0_topic': '/cam0/image_raw',
            'image1_topic': '/cam1/image_raw',
            'imu_topic': '/imu',
            'image_transport': LaunchConfiguration('image_transport'),

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

            # OFF by default: with filter=false the node creates no detection
            # subscription and feeds ORB-SLAM3 an empty mask, which its extractor
            # short-circuits -- so the recorded baseline stays reproducible.
            'filter': LaunchConfiguration('filter'),
        }],
    )

    # Only spawned when filtering is on. Loading ultralytics costs seconds and a
    # GPU context, so a default run should not pay for it.
    detector = Node(
        package='orbslam3_ros2',
        executable='dynamic_detector_node.py',
        name='dynamic_detector',
        output='screen',
        emulate_tty=True,
        condition=IfCondition(LaunchConfiguration('filter')),
        parameters=[{
            'use_sim_time': True,
            'weights_path': LaunchConfiguration('weights_path'),
            'dynamic_classes': LaunchConfiguration('dynamic_classes'),
            # Left camera only. In stereo, ComputeStereoMatches matches LEFT
            # keypoints against right candidates, so removing a left keypoint is
            # already enough to stop the map point being created -- and it halves
            # the inference cost.
            'image_topic': '/cam0/image_raw',
            'image_transport': LaunchConfiguration('image_transport'),
            'publish_debug_image': LaunchConfiguration('publish_debug_image'),
        }],
    )

    return LaunchDescription(args + [node, detector])
