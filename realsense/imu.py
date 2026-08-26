#!/usr/bin/env python3
"""Read IMU (accel + gyro) from an Intel RealSense L515.

Requires librealsense <= 2.54.2 with python bindings -- L515 support was
removed from the SDK in 2.55.x, so newer builds report "No device detected".

Usage:
    python3 imu.py --list     # show the motion profiles the device advertises
    python3 imu.py            # stream accel + gyro
"""

import argparse

import pyrealsense2 as rs


def find_motion_sensor():
    """Return (device, motion_sensor) for the first connected RealSense."""
    ctx = rs.context()
    devices = list(ctx.query_devices())
    if not devices:
        raise RuntimeError(
            "No RealSense device found. If lsusb shows the L515 but this "
            "fails, the two usual causes are:\n"
            "  1. USB 2.0 connection -- the L515 needs USB 3.1 Gen 1. Check "
            "`lsusb -t`: it must show 5000M and Class=Video, not 480M and "
            "Class=Vendor Specific Class.\n"
            "  2. librealsense >= 2.55, which dropped L515 support."
        )

    dev = devices[0]
    name = dev.get_info(rs.camera_info.name)
    serial = dev.get_info(rs.camera_info.serial_number)
    fw = dev.get_info(rs.camera_info.firmware_version)
    print(f"Device: {name}  serial={serial}  fw={fw}")

    for sensor in dev.query_sensors():
        profiles = sensor.get_stream_profiles()
        if any(p.stream_type() in (rs.stream.accel, rs.stream.gyro) for p in profiles):
            return dev, sensor

    raise RuntimeError(f"{name} exposes no motion module (accel/gyro) streams.")


def list_profiles():
    _, sensor = find_motion_sensor()
    print(f"Motion module: {sensor.get_info(rs.camera_info.name)}")
    for p in sensor.get_stream_profiles():
        if p.stream_type() in (rs.stream.accel, rs.stream.gyro):
            print(f"  {p.stream_name():>6}  {p.format()}  {p.fps()} Hz")


def stream():
    # Confirm the motion module exists and pick the fastest advertised rate for
    # each stream, rather than hard-coding an fps the device may not support.
    _, sensor = find_motion_sensor()
    rates = {}
    for p in sensor.get_stream_profiles():
        st = p.stream_type()
        if st in (rs.stream.accel, rs.stream.gyro):
            rates[st] = max(rates.get(st, 0), p.fps())

    config = rs.config()
    for st in (rs.stream.accel, rs.stream.gyro):
        if st in rates:
            config.enable_stream(st, rs.format.motion_xyz32f, rates[st])
            print(f"Enabling {st} at {rates[st]} Hz")

    pipeline = rs.pipeline()
    pipeline.start(config)
    print("Streaming -- Ctrl-C to stop.\n")

    try:
        while True:
            # Motion frames arrive independently, so handle each frame in the
            # set rather than expecting a synchronised accel+gyro pair.
            for frame in pipeline.wait_for_frames():
                motion = frame.as_motion_frame()
                if not motion:
                    continue
                d = motion.get_motion_data()
                t = motion.get_timestamp() / 1000.0  # ms -> s
                label = motion.get_profile().stream_name()
                units = "m/s^2" if label.lower().startswith("accel") else "rad/s"
                print(
                    f"{t:.4f}  {label:>6}  "
                    f"x={d.x:+8.4f} y={d.y:+8.4f} z={d.z:+8.4f}  {units}"
                )
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        pipeline.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list", action="store_true", help="list motion profiles and exit"
    )
    args = parser.parse_args()

    list_profiles() if args.list else stream()
