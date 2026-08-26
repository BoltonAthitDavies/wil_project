# realsense_imu

Publishes accelerometer + gyroscope from an Intel RealSense camera as a single
`sensor_msgs/Imu` topic, ready for VINS-Fusion.

## Why this package exists

librealsense delivers accel and gyro as two independent streams at different
rates. VINS-Fusion needs both vectors inside one `sensor_msgs/Imu` message, so
this node interpolates the accelerometer onto each gyroscope timestamp and
publishes one message per gyro sample. That is the same thing realsense-ros
calls `unite_imu_method=linear_interpolation`.

Using this instead of the official realsense-ros wrapper avoids pulling in the
whole camera wrapper (and its `diagnostic_updater` dependency) when all you need
is the IMU.

## Prerequisites

`pyrealsense2` comes from the hand-built librealsense in `~/src/librealsense`,
**not** from apt. Version matters: librealsense 2.55 removed L500-series
support, and an L515 on a newer SDK fails with a misleading
`"No device detected. Is it plugged in?"`. This package was developed against
**2.54.2**, the last release that supports the L515.

It is on `PYTHONPATH` via `~/.bashrc`:

```bash
export LD_LIBRARY_PATH=$HOME/src/librealsense/build/Release:$LD_LIBRARY_PATH
export PYTHONPATH=$HOME/src/librealsense/build/Release:$PYTHONPATH
```

Because it is not installed system-wide, anything launching this node must
inherit those variables. A `systemd` unit or a bare `ros2 launch` from a
non-login shell will not see them.

## Build and run

```bash
cd ~/wil_project
colcon build --packages-select realsense_imu
source install/setup.bash

ros2 run realsense_imu list_profiles          # what rates does this camera offer?
ros2 launch realsense_imu imu.launch.py       # publish on /imu0
```

## Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `topic` | `/imu0` | Output topic. Matches `imu_topic` in `config/wil/mono.yaml`. |
| `frame_id` | `camera_imu_frame` | `header.frame_id` on published messages. |
| `serial` | `""` | Select a specific camera; empty means the first found. |
| `gyro_fps` | `200` | Sets the topic rate: one message per gyro sample. |
| `accel_fps` | `0` | `0` = fastest rate the device advertises (250 Hz on a D455). |
| `queue_depth` | `2000` | Publisher queue depth. |
| `global_time` | `true` | Ask the SDK to report timestamps in epoch time. |
| `max_timebase_skew` | `1.0` | Drop samples whose accel/gyro clocks disagree by more than this many seconds. |
| `linear_acceleration_stddev` | `0.0` | `0` publishes a zero covariance, meaning "unknown". |
| `angular_velocity_stddev` | `0.0` | As above. |

## Output rate

The topic publishes one `sensor_msgs/Imu` per gyroscope sample, so **`gyro_fps`
is the topic rate**. It defaults to 200 Hz. Only rates the device advertises are
accepted — `list_profiles` shows them, and an unsupported value is rejected
rather than silently rounded. A D455 offers gyro at 200/400 Hz and accel at
63/250 Hz.

Leave `accel_fps` at `0` (250 Hz). Accel is interpolated onto gyro timestamps,
so a faster accel narrows the gap being interpolated across: at 250 Hz that gap
is 4 ms, at 63 Hz it is 16 ms.

Configuring a rate is not the same as achieving it. The node measures its own
output and warns when it falls below 70% of the configured gyro rate:

```
[WARN] Publishing at 15.4 Hz but gyro is configured for 200 Hz.
       Check the USB link: `lsusb -t` should show 5000M, not 480M.
```

That warning is not hypothetical — see below.

## Design notes

**QoS is RELIABLE, not the sensor-data profile.** VINS-Fusion subscribes with
`rclcpp::QoS(rclcpp::KeepLast(2000))`, which defaults to RELIABLE. A BEST_EFFORT
publisher is QoS-incompatible with a RELIABLE subscriber: the topic appears in
`ros2 topic list`, the connection looks established, and not one message is ever
delivered. If you repoint this at a best-effort consumer, change both ends.

**Timestamps are mapped per stream, per domain.** A single device-to-ROS clock
offset is not sufficient. librealsense reports each stream in its own timestamp
domain and migrates them from `hardware_clock` (milliseconds since device boot)
to `global_time` (epoch milliseconds) independently, once its clock mapping
converges. Measured on a D455: accel was still reporting `hardware_clock`
621095 ms while gyro had already switched to `global_time` 1786607396477 ms.
Applying one offset across both overflows the int32 `sec` field in
`builtin_interfaces/Time` and, more insidiously, interpolates between two
unrelated clocks. `max_timebase_skew` drops samples during that crossover
window rather than emitting a plausible-looking but wrong message; expect a
`skipped_timebase_skew` count of 1 or 2 at startup.

**Exceptions in the frame callback are caught and logged.** librealsense invokes
the callback from its own thread and silently discards anything raised there, so
an uncaught error shows up only as a topic that never publishes.

**SIGTERM is handled.** Without it, `ros2 launch` shutdown or `timeout` kills the
process before the motion sensor is closed, leaving the device claimed — and the
next run segfaults inside the SDK while opening it.

## USB 2.0 will silently throttle you

Both cameras tested here were on USB 2.0 links, which is worth checking first
whenever rates look wrong:

```bash
lsusb -t     # want 5000M and Class=Video, not 480M
```

A D455 on USB 2.0 still enumerates and streams, but the IMU is throttled
severely. Measured on this rig, the delivered rate is roughly the same no matter
what is requested — the link, not the profile, is the limit:

| Requested gyro | Delivered |
|---|---|
| 400 Hz | ~26 Hz |
| 200 Hz | ~15-21 Hz |

Confirmed to be the camera rather than this node: a bare `pyrealsense2` script
with no ROS involved measured the same ~26 Hz. An L515 on USB 2.0
does not enumerate as a camera at all: it presents only
`Class=Vendor Specific Class` with no driver bound, and librealsense reports
`"No device detected"`.

The usual cause is a USB-C *charging* cable, which omits the SuperSpeed
differential pairs entirely. Use the cable that shipped with the camera, or one
explicitly rated USB 3.x / 5 Gbps.

## Before using this with VINS-Fusion

Publishing the IMU is the easy half. Two things still need doing:

- **IMU-camera extrinsics.** `body_T_cam0` in `config/wil/mono.yaml` must be the
  transform between this IMU and `cam0`. If the RealSense is only being used as
  an IMU source alongside a separate fisheye pair, the two are physically
  distinct and need real calibration (Kalibr). `estimate_extrinsic: 1` refines a
  good initial guess; it will not rescue a bad one.
- **Noise parameters.** `acc_n` / `gyr_n` in that config are BNO055 placeholders.
  They do not describe this IMU and should be re-measured (Allan variance).
