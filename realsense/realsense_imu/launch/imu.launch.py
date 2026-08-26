from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    args = [
        # Defaults to /imu0 to match imu_topic in config/wil/mono.yaml.
        DeclareLaunchArgument('topic', default_value='/imu0'),
        DeclareLaunchArgument('frame_id', default_value='camera_imu_frame'),
        DeclareLaunchArgument('serial', default_value=''),
        # The topic rate equals the gyro rate: one Imu message per gyro sample.
        # 200 Hz is a common VINS-Fusion operating point and is advertised by
        # both the D455 (200/400) and the L515.
        DeclareLaunchArgument('gyro_fps', default_value='200'),
        # 0 = fastest the device advertises (250 Hz on a D455). Deliberately
        # left above the gyro rate: accel is interpolated onto gyro timestamps,
        # so a faster accel means a shorter gap to interpolate across. Dropping
        # it to the D455's other option, 63 Hz, would widen that gap from 4 ms
        # to 16 ms for no bandwidth saving that matters.
        DeclareLaunchArgument('accel_fps', default_value='0'),
    ]

    node = Node(
        package='realsense_imu',
        executable='imu_node',
        name='realsense_imu',
        output='screen',
        parameters=[{
            'topic': LaunchConfiguration('topic'),
            'frame_id': LaunchConfiguration('frame_id'),
            'serial': LaunchConfiguration('serial'),
            'accel_fps': LaunchConfiguration('accel_fps'),
            'gyro_fps': LaunchConfiguration('gyro_fps'),
        }],
    )

    return LaunchDescription(args + [node])
