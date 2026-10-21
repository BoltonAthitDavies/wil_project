# `dataset_realv3_003` — topic reference

Reference for the rosbag2 recording at `dataset/dataset_realv3_003/`. All figures below were
measured by parsing `dataset_realv3_003_0.db3` directly (CDR payloads read out of the SQLite
`messages` table), not taken from the launch configuration.

| Property | Value |
|---|---|
| Path | `dataset/dataset_realv3_003/` |
| Storage | `sqlite3`, serialization `cdr`, no compression |
| Size | 2,887,114,752 bytes (2.89 GB) |
| Duration | 114.16 s |
| Start | epoch 1792483894.830 |
| Messages | 63,818 across 9 topics |
| Verified | 2026-10-20 |

## Topics

| Topic | Type | Msgs | Rate | Contents |
|---|---|---|---|---|
| `/hwt101ct_yaw_publisher` | `sensor_msgs/Imu` | 22,775 | 199.5 Hz | 3-axis gyro (rad/s) + 3-axis accel (m/s²), `frame_id=imu_link`. **Despite the name, this is the RealSense motion module, not the HWT101CT.** No orientation |
| `/clock` | `rosgraph_msgs/Clock` | 22,767 | 199.4 Hz | Wall-clock published by the CAN node from its `SYSTEM_TIME` source |
| `/feedback/velocity` | `std_msgs/Float64` | 5,702 | 49.9 Hz | Longitudinal speed decoded from CAN `0x182`, bytes 0–3 |
| `/feedback/steering` | `std_msgs/Float64` | 5,702 | 49.9 Hz | Steering in degrees from CAN `0x182`, bytes 4–7. **Command echo, not a sensor** |
| `/cam0/image_raw/compressed` | `sensor_msgs/CompressedImage` | 3,424 | 30.0 Hz | 1920×1080, `yuv422; jpeg compressed mono8`, ~428 KB/frame, `frame_id=camera` |
| `/cam1/image_raw/compressed` | `sensor_msgs/CompressedImage` | 3,424 | 30.0 Hz | 1920×1080, same encoding, ~431 KB/frame, `frame_id=camera` |
| `/cmd_steering` | `std_msgs/Float64` | 22 | sparse | Steering commands issued during the run |
| `/cmd_velocity` | `std_msgs/Float64` | 2 | — | `0.0` then `1.25`; set once and held for 105 s |
| `/parameter_events` | `rcl_interfaces/ParameterEvent` | 0 | — | Empty |

Counts sum to exactly 63,818, so this is the complete topic list.

## Signal detail

### IMU — `/hwt101ct_yaw_publisher`

Published by `realsense_imu_publisher.py`, remapped onto this topic name at
`ros2_ws/src/robot_camera/launch/dual_camera.launch.py:54`. There is **no HWT101CT data in this
bag**.

| Field | State |
|---|---|
| `angular_velocity` | All three axes populated, rad/s |
| `angular_velocity_covariance` | diag `[0.01, 0.01, 0.0005]` |
| `linear_acceleration` | All three axes populated, m/s² |
| `orientation` | Identity (0,0,0,1) — **not** real data |
| `orientation_covariance[0]` | `-1.0`, the `sensor_msgs/Imu` flag for "no estimate" |

Measured statistics over the full run:

| Axis | accel mean (m/s²) | gyro std (°/s) | gyro range (°/s) |
|---|---|---|---|
| x | +0.178 | 1.518 | −13.40 … +14.50 |
| y | **+9.884** | 7.896 | −19.80 … +18.10 |
| z | +0.196 | 0.879 | −6.60 … +4.60 |

Mean `|accel|` = 9.915 m/s², consistent with a stationary-to-slow run under 1 g.

### Mounting orientation

Gravity reads **+9.884 on y**. An IMU at rest reads +g along the axis pointing *up*, so the
sensor's +y points up — the camera is mounted upside down.

To determine which axis the 180° flip is about, each accel axis was correlated against forward
acceleration derived from the wheel speed. Gravity alone cannot distinguish the candidates;
the forward axis can:

| Axis | r | slope |
|---|---|---|
| accel.x | −0.028 | −0.048 |
| accel.y | −0.066 | −0.029 |
| accel.z | **+0.618** | **+1.022** |

Sensor `z` carries forward acceleration with positive sign and near-unit scale. A 180° rotation
about **x** would point the optical axis backward and give slope ≈ −1, so the flip is 180° about
the **optical z axis**: upside down, still facing forward.

Resulting axis mapping, with `base_link` as REP-103 (x forward, y left, z up):

```
base x (forward) = sensor z
base y (left)    = sensor x
base z (up)      = sensor y
```

Static transform `base_link -> imu_link`:

```
quat (x, y, z, w) = (0.5, 0.5, 0.5, 0.5)
rpy               = (90°, 0°, 90°)
```

### Yaw rate

Yaw is on **`+gyro.y`**, not `gyro.z`. Correlated against the single-track model
`ω = -(v/L)·tan(δ)` with `L = 0.463 m`, `v` scaled by 1/3, and the CAN convention that positive
steering is a right turn:

| Axis | r | gyro / model |
|---|---|---|
| gyro.x | +0.076 | +0.010 |
| gyro.y | **+0.962** | **+0.590** |
| gyro.z | −0.015 | −0.001 |

The shape matches (r = 0.962) but the magnitude does not: measured yaw rate is 0.590× the model.
The combined error lives in the `×1/3` velocity scaling and `steering_ratio = 1.0`, which are off
by roughly 1.7× together. A scale-factor state in the estimator can identify this from data
rather than having it hard-coded.

Note that `robot_odometry/odometry_node.py:84` reads `angular_velocity.z`, which on this bag is
the near-silent axis (std 0.879 °/s of noise). Odometry run against this data integrates
essentially no yaw.

### Wheel speed

Range 0 → 1.29591, with only **14 distinct values** across 5,702 messages:

```
0.0, 0.213105, 0.215985, 0.426209, 0.429089, 0.642194, 0.645074,
0.647954, 0.858178, 0.863938, 1.074163, 1.079922, 1.290147, 1.295907
```

Step size ≈ 0.2131, i.e. ERPM quantisation from the VESC. Speed itself is usable; differentiating
it for acceleration is not — use the IMU accelerometer instead.

### Steering

`/feedback/steering` takes exactly 8 values across all 5,702 messages:

```
-30.0, -25.0, -20.0, 0.0, 1e-05, 5.0, 10.0, 30.0
```

These are precisely the 7 values `/cmd_steering` sent, plus the `1e-05` zero-sentinel injected by
`robot_can_interface_node.py:208`. No intermediate angles and no sensor noise, which no real
encoder could produce. **This channel is a command echo** — treat it as a control input, not as a
measurement.

## Timing

| Topic | mean Δt | max Δt |
|---|---|---|
| IMU | 5.01 ms | 35.9 ms |
| `/feedback/velocity` | 20.0 ms | 72.2 ms |
| `/feedback/steering` | 20.0 ms | 72.3 ms |
| `/cmd_steering` | 4876 ms | 14,773 ms |

Camera `header.stamp` tracks bag receive time to under 1 s, confirming that **all timestamps are
arrival-time, not measurement-time**. The slcan serial latency on the CAN path therefore appears
directly in the `/feedback/*` time deltas.

## Known issues

1. **IMU topic name is misleading.** `/hwt101ct_yaw_publisher` carries RealSense data. Any consumer
   assuming single-axis HWT101CT semantics (real `orientation`, yaw on `z`) will be wrong.
2. **Yaw is on `+gyro.y`.** Reading `angular_velocity.z` yields noise. Apply the static transform
   above rather than hard-coding an axis swap.
3. **`/feedback/steering` is not a measurement.** It is the setpoint echoed back by the NXP board.
4. **Both cameras share `frame_id='camera'`.** Stereo processing needs distinct left/right frames;
   nothing in the bag distinguishes them.
5. **`/clock` is not monotonic.** 15 backward steps, worst −2.48 ms, median magnitude 0.05 ms.
   Harmless for most uses but can trip TF lookups and message filters under
   `use_sim_time:=true`; prefer wall-time playback.
6. **Weak excitation.** Near-constant commanded speed (`/cmd_velocity` set twice), 22 steering
   changes, 7 discrete steering values, wheel-derived accel std 0.038 m/s². Adequate for
   developing an estimator, thin for identifying scale factors or gyro bias well.
7. **No ground truth.** Filter output can be sanity-checked but not quantitatively validated.
   VINS-Fusion on the stereo pair is the only available reference, and it is itself an estimate.

## Suitability for state estimation

Sufficient to build and run an EKF. Prediction from the 199.5 Hz IMU, body velocity from the
49.9 Hz wheel speed with steering as the control input, and absolute correction from the 30 Hz
stereo pair via VINS-Fusion. No required signal is missing and there are no data gaps.

Not sufficient for quantitative validation or clean calibration — see issues 6 and 7. A second
short run with varied speed and continuous steering sweeps would condition the scale factors
properly.
