#!/usr/bin/env python3
"""Print the motion profiles a connected RealSense advertises.

Useful before setting accel_fps / gyro_fps on the node, since the supported
rates differ per model and enable_stream rejects an unsupported value rather
than falling back to a nearby one.
"""

import pyrealsense2 as rs


def main(args=None):
    devices = list(rs.context().query_devices())
    if not devices:
        print('No RealSense device found. Check `lsusb -t`: the L515 needs a '
              '5000M / Class=Video link, not 480M / Vendor Specific Class.')
        return

    for dev in devices:
        print(f'{dev.get_info(rs.camera_info.name)}  '
              f'serial={dev.get_info(rs.camera_info.serial_number)}  '
              f'fw={dev.get_info(rs.camera_info.firmware_version)}')
        found = False
        for sensor in dev.query_sensors():
            for profile in sensor.get_stream_profiles():
                if profile.stream_type() in (rs.stream.accel, rs.stream.gyro):
                    print(f'  {profile.stream_name():>6}  {profile.format()}  '
                          f'{profile.fps()} Hz')
                    found = True
        if not found:
            print('  (no motion module -- this model has no IMU)')


if __name__ == '__main__':
    main()
