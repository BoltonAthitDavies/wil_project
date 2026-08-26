#!/usr/bin/env python3
"""Publish united accel+gyro from a RealSense camera as sensor_msgs/Imu.

librealsense delivers accelerometer and gyroscope as two independent streams at
different rates. VINS-Fusion (and most consumers) need both vectors inside a
single sensor_msgs/Imu message, so this node interpolates the accelerometer onto
each gyroscope timestamp and publishes one message per gyro sample -- the same
thing realsense-ros calls unite_imu_method=linear_interpolation.
"""

import signal
import threading
import traceback
from collections import deque

import pyrealsense2 as rs
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu

# sensor_msgs/Imu convention: covariance[0] = -1 means "this quantity is not
# reported at all". We never estimate orientation, only raw rates/accels.
UNKNOWN = -1.0


def _raise_keyboard_interrupt(signum, frame):
    raise KeyboardInterrupt


class RealsenseImuNode(Node):

    def __init__(self):
        super().__init__('realsense_imu')

        self.declare_parameter('topic', '/imu0')
        self.declare_parameter('frame_id', 'camera_imu_frame')
        self.declare_parameter('serial', '')
        self.declare_parameter('accel_fps', 0)
        self.declare_parameter('gyro_fps', 0)
        self.declare_parameter('queue_depth', 2000)
        self.declare_parameter('linear_acceleration_stddev', 0.0)
        self.declare_parameter('angular_velocity_stddev', 0.0)
        self.declare_parameter('global_time', True)
        self.declare_parameter('max_timebase_skew', 1.0)

        self._frame_id = self.get_parameter('frame_id').value
        self._max_skew = self.get_parameter('max_timebase_skew').value
        topic = self.get_parameter('topic').value
        depth = self.get_parameter('queue_depth').value

        # RELIABLE, not the best-effort sensor-data profile. VINS-Fusion
        # subscribes with rclcpp::QoS(KeepLast(2000)), which defaults to
        # RELIABLE; a BEST_EFFORT publisher is QoS-incompatible with that and
        # would connect but never deliver a single message.
        qos = QoSProfile(
            depth=depth,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self._pub = self.create_publisher(Imu, topic, qos)

        acc_sd = self.get_parameter('linear_acceleration_stddev').value
        gyr_sd = self.get_parameter('angular_velocity_stddev').value
        self._acc_cov = self._diag(acc_sd)
        self._gyr_cov = self._diag(gyr_sd)

        # Accelerometer history for interpolation, plus the lock guarding it:
        # librealsense invokes our callback from its own thread, and accel and
        # gyro frames can arrive concurrently.
        self._accel = deque(maxlen=2)
        self._lock = threading.Lock()

        # Device clock -> ROS clock offsets, keyed by (stream, timestamp domain).
        #
        # A single global offset is NOT enough. librealsense reports accel and
        # gyro in independent domains, and a stream migrates from
        # hardware_clock (milliseconds since device boot) to global_time (epoch
        # milliseconds) once the SDK has built its clock mapping -- the two
        # streams cross over at different moments. Measured on a D455: accel was
        # still at hardware_clock 621095 ms while gyro already reported
        # global_time 1786607396477 ms. Applying one offset across both
        # overflows the int32 'sec' field and makes the interpolation compare
        # two unrelated timebases.
        #
        # Stamping with ROS time at arrival instead would fold callback
        # scheduling jitter into the intervals between samples, which IMU
        # pre-integration is directly sensitive to.
        self._offsets = {}

        self._published = 0
        self._dropped = 0
        self._skewed = 0
        self._errors = 0

        # Rate tracking, so a device that accepts a profile but cannot actually
        # sustain it does not go unnoticed.
        self._expected_hz = None
        self._last_count = 0
        self._last_report = None
        self._warned_slow = False

        self._sensor = None
        self._start_stream(topic)
        self.create_timer(5.0, self._report)

    @staticmethod
    def _diag(stddev):
        """Covariance matrix from a scalar stddev; zeros mean 'unknown'."""
        if stddev <= 0.0:
            return [0.0] * 9
        var = stddev * stddev
        return [var, 0.0, 0.0, 0.0, var, 0.0, 0.0, 0.0, var]

    def _start_stream(self, topic):
        ctx = rs.context()
        devices = list(ctx.query_devices())
        if not devices:
            raise RuntimeError(
                'No RealSense device found. If lsusb lists the camera but this '
                'fails, check `lsusb -t`: an L515 on a USB 2.0 link shows 480M '
                'and Class=Vendor Specific Class, and librealsense will not '
                'detect it. It needs 5000M and Class=Video.'
            )

        wanted = self.get_parameter('serial').value
        dev = None
        for candidate in devices:
            serial = candidate.get_info(rs.camera_info.serial_number)
            if not wanted or serial == wanted:
                dev = candidate
                break
        if dev is None:
            raise RuntimeError(f'No RealSense with serial {wanted!r} found.')

        self.get_logger().info(
            f'Device: {dev.get_info(rs.camera_info.name)} '
            f'serial={dev.get_info(rs.camera_info.serial_number)} '
            f'fw={dev.get_info(rs.camera_info.firmware_version)}'
        )

        motion = None
        for sensor in dev.query_sensors():
            if any(p.stream_type() in (rs.stream.accel, rs.stream.gyro)
                   for p in sensor.get_stream_profiles()):
                motion = sensor
                break
        if motion is None:
            raise RuntimeError(
                f'{dev.get_info(rs.camera_info.name)} has no motion module. '
                'Not every RealSense model carries an IMU.'
            )

        # Ask the SDK to put both streams on epoch time. They still start out in
        # hardware_clock and converge separately, which is why _to_ros_time
        # handles the mixed case rather than assuming this succeeded.
        if motion.supports(rs.option.global_time_enabled):
            motion.set_option(
                rs.option.global_time_enabled,
                1.0 if self.get_parameter('global_time').value else 0.0,
            )
        else:
            self.get_logger().warn(
                'Motion module does not support global_time_enabled; '
                'falling back to arrival-time offset mapping.'
            )

        requested = {
            rs.stream.accel: self.get_parameter('accel_fps').value,
            rs.stream.gyro: self.get_parameter('gyro_fps').value,
        }
        chosen = []
        for stream, fps in requested.items():
            candidates = [p for p in motion.get_stream_profiles()
                          if p.stream_type() == stream]
            if not candidates:
                raise RuntimeError(f'Device does not offer a {stream} stream.')
            if fps:
                match = [p for p in candidates if p.fps() == fps]
                if not match:
                    offered = sorted({p.fps() for p in candidates})
                    raise RuntimeError(
                        f'{stream} does not support {fps} Hz; offered: {offered}'
                    )
                chosen.append(match[0])
            else:
                # 0 means "fastest advertised" -- rates differ per model, so
                # asking the device beats hard-coding a value it may reject.
                chosen.append(max(candidates, key=lambda p: p.fps()))

        for profile in chosen:
            self.get_logger().info(
                f'Enabling {profile.stream_name()} at {profile.fps()} Hz'
            )
            if profile.stream_type() == rs.stream.gyro:
                # The topic rate is one message per gyro sample, so this is the
                # rate the output is expected to hold.
                self._expected_hz = float(profile.fps())

        motion.open(chosen)
        motion.start(self._on_frame)
        self._sensor = motion
        self.get_logger().info(f'Publishing sensor_msgs/Imu on {topic}')

    def _on_frame(self, frame):
        # librealsense calls this from its own thread and discards any exception
        # we raise, so an error here would otherwise show up only as silence.
        try:
            self._handle_frame(frame)
        except Exception as exc:  # noqa: BLE001 - must not escape into the SDK
            self._errors += 1
            if self._errors <= 3:
                self.get_logger().error(
                    f'IMU callback failed: {type(exc).__name__}: {exc}\n'
                    f'{traceback.format_exc()}'
                )

    def _to_ros_time(self, motion, stream):
        """Map a frame's device timestamp onto the ROS clock, in seconds."""
        t = motion.get_timestamp() / 1000.0  # librealsense reports milliseconds
        domain = motion.get_frame_timestamp_domain()

        if domain == rs.timestamp_domain.global_time:
            # Already epoch time; librealsense has done the mapping for us and
            # its estimate is better than anything we can derive from arrival.
            return t

        key = (stream, domain)
        offset = self._offsets.get(key)
        if offset is None:
            offset = self.get_clock().now().nanoseconds * 1e-9 - t
            self._offsets[key] = offset
            self.get_logger().info(
                f'{motion.get_profile().stream_name()} is in domain {domain}; '
                f'mapping to ROS time with offset {offset:.6f} s'
            )
        return t + offset

    def _handle_frame(self, frame):
        motion = frame.as_motion_frame()
        if not motion:
            return

        stream = motion.get_profile().stream_type()
        if stream not in (rs.stream.accel, rs.stream.gyro):
            return

        t = self._to_ros_time(motion, stream)
        data = motion.get_motion_data()

        if stream == rs.stream.accel:
            with self._lock:
                self._accel.append((t, data.x, data.y, data.z))
            return

        with self._lock:
            if len(self._accel) < 2:
                # Gyro can lead the first two accel samples; nothing to
                # interpolate between yet.
                self._dropped += 1
                return
            (t0, ax0, ay0, az0), (t1, ax1, ay1, az1) = self._accel

        # Guard the window where the two streams have not yet converged onto the
        # same timebase: publishing then would silently pair a gyro sample with
        # an accel sample from a completely different clock, and the alpha clamp
        # below would hide it by holding the endpoint value.
        if abs(t - t1) > self._max_skew:
            self._skewed += 1
            return

        span = t1 - t0
        # Clamping rather than extrapolating: if accel stalls, holding the last
        # sample is a bounded error, whereas extrapolating a stale slope
        # diverges without limit.
        alpha = 0.0 if span <= 0.0 else min(max((t - t0) / span, 0.0), 1.0)

        msg = Imu()
        msg.header.stamp = rclpy.time.Time(seconds=t).to_msg()
        msg.header.frame_id = self._frame_id

        msg.linear_acceleration.x = ax0 + alpha * (ax1 - ax0)
        msg.linear_acceleration.y = ay0 + alpha * (ay1 - ay0)
        msg.linear_acceleration.z = az0 + alpha * (az1 - az0)
        msg.linear_acceleration_covariance = self._acc_cov

        msg.angular_velocity.x = data.x
        msg.angular_velocity.y = data.y
        msg.angular_velocity.z = data.z
        msg.angular_velocity_covariance = self._gyr_cov

        msg.orientation_covariance[0] = UNKNOWN

        self._pub.publish(msg)
        self._published += 1

    def _report(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        rate = None
        if self._last_report is not None:
            span = now - self._last_report
            if span > 0.0:
                rate = (self._published - self._last_count) / span
        self._last_report = now
        self._last_count = self._published

        if self._published:
            measured = f'{rate:.1f}' if rate is not None else '?'
            self.get_logger().info(
                f'published={self._published} rate={measured} Hz '
                f'dropped_before_accel_ready={self._dropped} '
                f'skipped_timebase_skew={self._skewed} '
                f'errors={self._errors}'
            )
            # A RealSense will happily accept a 200 Hz profile over USB 2.0 and
            # then deliver a fraction of it, with no error anywhere. Say so,
            # rather than letting a starved IMU quietly degrade VINS.
            if (rate is not None and self._expected_hz
                    and rate < 0.7 * self._expected_hz
                    and not self._warned_slow):
                self._warned_slow = True
                self.get_logger().warn(
                    f'Publishing at {rate:.1f} Hz but gyro is configured for '
                    f'{self._expected_hz:.0f} Hz. Check the USB link: '
                    f'`lsusb -t` should show 5000M, not 480M. A RealSense on '
                    f'USB 2.0 throttles the IMU severely.'
                )
        else:
            self.get_logger().warn(
                f'No IMU messages published yet (dropped={self._dropped}, '
                f'skew={self._skewed}, errors={self._errors}).'
            )

    def destroy_node(self):
        if self._sensor is not None:
            try:
                self._sensor.stop()
                self._sensor.close()
            except RuntimeError as exc:
                self.get_logger().warn(f'Error closing motion sensor: {exc}')
            self._sensor = None
        super().destroy_node()


def main(args=None):
    # Without this, SIGTERM (ros2 launch shutdown, `timeout`, systemd) kills the
    # process before the motion sensor is stopped and closed. librealsense then
    # leaves the device claimed, and the *next* run segfaults inside the SDK
    # while opening it. Routing SIGTERM through the same path as Ctrl-C lets
    # destroy_node() release the device properly.
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)

    rclpy.init(args=args)
    node = None
    try:
        node = RealsenseImuNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except RuntimeError as exc:
        print(f'realsense_imu: {exc}')
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
