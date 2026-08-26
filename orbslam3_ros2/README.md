# orbslam3_ros2

ORB-SLAM3 stereo and stereo-inertial for the WiL rigs, as a ROS 2 Humble node.

A second, independent SLAM system alongside `vins_fusion_ros2`, so trajectories can be
compared against each other and against the sim's ground truth on the same bags. It is
feature-based where VINS is optimisation-on-optical-flow, it models the real rig's
~180 degree lens natively as Kannala-Brandt, and it does loop closure, which this VINS
fork does not (there is no `loop_fusion` package here and `pose_graph_save_path` is inert).

Upstream ORB-SLAM3 ships a ROS 1 wrapper only, and the community ROS 2 forks are built
around offline dataset playback rather than live topics. So this package builds the
ORB-SLAM3 core out of tree (`../build_orbslam3.sh`) and wraps it in a purpose-written node.

## Status

| | |
| :-- | :-- |
| Sim stereo-inertial, `dataset/sim` | **verified** — ATE 0.275–0.34 m RMS across runs |
| Path length | 27.87 m against 28.35 m ground truth (1.7%) |
| `vio.csv` byte-compatible with VINS format | **verified** against `ground_truth.csv` |
| Real rig, `wil_stereo` | **verified** — full 101.5 s bag, 1578 poses, 0 map resets, 0 crashes |
| Real rig, `wil_stereo_imu` | **fails** — repeated map resets, never holds tracking. Blocked on calibration, not code; see Real-rig caveats |
| Pangolin viewer (`use_viewer:=true`) | **untested** — every run so far used `false` |

For reference, VINS scores ~26 cm ATE on the same bag, so the two are in the same league.

## Quick start

```bash
cd ~/wil_project
source /opt/ros/humble/setup.bash
source install/setup.bash
```

Two terminals. The launch file starts the estimator only — no launch file in this
workspace ever runs `ros2 bag play` for you, matching the `vins_fusion_ros2` convention.

```bash
# terminal 1
ros2 launch orbslam3_ros2 wil_sim_stereo_imu.launch.py

# terminal 2
ros2 bag play dataset/sim --clock --rate 0.5
```

`--rate 0.5` is not optional in spirit. Tracking 1280x720 does not keep up with
full-speed playback, and the node drops the stale frame rather than growing a queue. It
warns on shutdown if it dropped any, so a silently degraded run is visible.

Expect `waiting for IMU initialisation`, then `tracking OK -- publishing poses`.

Against the live sim instead of a bag:

```bash
ros2 launch aws_robomaker_small_warehouse_world small_warehouse.launch.py headless:=True
ros2 launch orbslam3_ros2 wil_sim_stereo_imu.launch.py
python3 drive.py        # or viewer.py, which drives too
```

## Launch files

| File | Rig | Mode | Notes |
| :--- | :-- | :--- | :---- |
| `wil_sim_stereo_imu.launch.py` | Gazebo sim | Stereo-Inertial | The verified path. `use_sim_time: True`. |
| `wil_stereo.launch.py` | Real RealSense | Stereo | **The real-rig default.** No IMU, so immune to both problems below. |
| `wil_stereo_imu.launch.py` | Real RealSense | Stereo-Inertial | An experiment. Read the caveats. |

```bash
ros2 launch orbslam3_ros2 wil_stereo.launch.py
ros2 bag play dataset/rosbag_realsense_imu_cambaseline95mm
```

No `image_transport republish` processes are needed. The real bags carry only
`/camN/image_raw/compressed`, and the node subscribes through `image_transport`, so it
decompresses in-process. VINS needs two extra terminals for the same job (see
`vins_fusion_ros2/config/wil/stereo_imu.yaml:32-38`).

### Arguments

All three take the same three; the real inertial launch adds a fourth.

| Argument | Default | Description |
| :------- | :------ | :---------- |
| `use_viewer` | `true` | Pangolin map/feature window. Set `false` for headless runs. Untested — see Status. |
| `output_path` | `~/output/wil_sim_orbslam3_stereo_imu` (varies per launch) | Directory for `vio.csv`. Created if absent. |
| `vocabulary_file` | `~/wil_project/thirdparty/ORB_SLAM3/Vocabulary/ORBvoc.txt` | 139 MB, built by `build_orbslam3.sh`. Deliberately not installed into the package share dir — colcon would copy it on every build. |
| `imu_accel_scale` | `1.0589149629757562` (`wil_stereo_imu` only) | Accelerometer correction; see below. |

```bash
ros2 launch orbslam3_ros2 wil_sim_stereo_imu.launch.py use_viewer:=false
```

## Output

| | |
| :-- | :-- |
| `/orbslam3/odometry` | `nav_msgs/Odometry`. Chosen so `viewer.py` picks it up unchanged. |
| `/orbslam3/pose` | `geometry_msgs/PoseStamped` |
| TF | `orbslam3_world` -> body frame |
| `<output_path>/vio.csv` | VINS-Fusion's exact CSV format |

The CSV is `t_ns,px,py,pz,qw,qx,qy,qz,vx,vy,vz` — comma-separated, integer nanoseconds,
quaternion **w first**, 11 columns. The node writes this directly rather than converting
after the fact: ORB-SLAM3's own `SaveTrajectoryEuRoC` emits *space*-separated
`t_ns tx ty tz qx qy qz qw` (`System.cc:762`), which is both the wrong delimiter and the
wrong quaternion order, and `SaveTrajectoryTUM` uses seconds. Writing the target format
directly means `plot_vio.py`, `plot_vio_vs_gt.py` and `viewer.py` all work **unmodified**.

Every row is flushed as it is written. At 30 Hz that is negligible I/O, and it means a
run that is Ctrl-C'd keeps every pose: without it the tail sits in the stream buffer and
a hard kill truncates the final line mid-field, which makes `numpy.loadtxt` throw on the
*whole* file and costs you the entire run.

## Scoring against ground truth

`plot_vio_vs_gt.py` takes **one run directory** containing both CSVs — not three file
paths.

```bash
OUT=~/output/wil_sim_orbslam3_stereo_imu
ln -sf ~/output/wil_sim_stereo_imu/ground_truth.csv "$OUT/ground_truth.csv"
python3 script/plot_vio_vs_gt.py "$OUT" "ORB-SLAM3 (sim)"
python3 script/plot_vio.py "$OUT/vio.csv"
```

**Read the ATE row. Ignore the yaw and error-x/y rows.**

ATE is a global SE(3) fit over all poses and is correct. The other columns anchor their
alignment on the *first* pose, and ORB-SLAM3 re-bases its map when the IMU initialises —
which happens right where that first pose comes from. One poisoned anchor rotates the
whole ground-truth overlay and reports a ~90 degree yaw error against a trajectory that
is actually sound. Measured directly: the body x axis aligns with the direction of travel
at +0.73 and quaternion yaw sits within ~3 degrees of path yaw, i.e. the published pose is
proper REP-103 and self-consistent. `plot_vio_vs_gt.py` was written for VINS, which does
not re-base.

Live, `viewer.py` needs no flags changed beyond the topic:

```bash
python3 viewer.py --vins-topic /orbslam3/odometry
```

## Configuration

OpenCV `FileStorage` format (`%YAML:1.0`, `!!opencv-matrix`), resolved through
`get_package_share_directory()` and passed down as the `config_file` parameter — the same
mechanism `vins_fusion_ros2.launch.py:9-14` uses.

Note the naming offset: **ORB-SLAM3 counts cameras from 1 and VINS from 0**, so
ORB-SLAM3's `Camera1` is VINS's `cam0`, and `IMU.T_b_c1` is VINS's `body_T_cam0` — same
quantity, same direction, different spelling.

| Config | Rig | Model |
| :----- | :-- | :---- |
| `config/wil_sim/stereo_imu.yaml` | Sim | PinHole 1280x720, fx=fy=640, cx=640, cy=360, zero distortion |
| `config/wil/stereo.yaml` | Real | KannalaBrandt8 1920x1080 |
| `config/wil/stereo_imu.yaml` | Real | as above + IMU |

Every number is derived from the same source of truth VINS uses, so a difference between
the two systems is a difference in the algorithms, not the calibration. The sim
extrinsics come from `model.sdf` geometry via
`inv(body_T_cam0) @ body_T_cam1`, giving a pure 0.0936 m baseline on cam0's optical +X —
the formal statement of "cam1 is the RIGHT camera". IMU noise maps 1:1 from VINS
(`NoiseGyro`==`gyr_n`, `NoiseAcc`==`acc_n`, `GyroWalk`==`gyr_w`, `AccWalk`==`acc_w`); the
conventions match, so no unit conversion is involved.

`Stereo.ThDepth` is 90, not EuRoC's stock 60: that value assumes a 0.11 m baseline, and at
our 0.0936 m it would put the close/far split at 5.6 m instead of a warehouse-appropriate
8.4 m.

For the real rig's fisheye, `fisheye_calibration.json` drops in unchanged —
`Camera*.k1..k4` **are** the raw OpenCV `cv::fisheye` D vector. Do not copy the numbers
out of VINS's `cam0_fisheye.yaml` instead: VINS fixes its own `k1=1` and shifts the
OpenCV terms into `k2..k5`, so those need re-indexing and ORB-SLAM3's do not.

## Real-rig caveats

Two hardware problems, neither of them ORB-SLAM3's fault, and the reason `wil_stereo` is
the default rather than `wil_stereo_imu`.

**The IMU-to-camera TRANSLATION is a placeholder — the rotation is fine.** Worth being
precise, because it changes what you have to do. The *rotation* in `IMU.T_b_c1` is
identity because that was **measured**: gravity in the stationary IMU frame sits at
`(+0.029, -0.997, -0.075)`, i.e. +Y points down, the RealSense convention, which is the
same as a camera optical frame. The *translation* is zero, and that is wrong — it asserts
the IMU sits at cam0's optical centre. The lever arm couples angular rate into measured
acceleration, and this rig reaches 1.29 rad/s of yaw.

This is why `wil_stereo_imu` fails outright here while VINS merely drifts: VINS runs
`estimate_extrinsic: 1` and refines the transform online, absorbing much of the error.
**ORB-SLAM3 has no online extrinsic refinement** and takes `IMU.T_b_c1` as truth, so the
inertial and visual constraints disagree and it resets the map instead.

Cheapest fix: tape-measure the IMU-to-cam0 offset **in the camera optical frame**
(x right, y down, z forward, metres) and write it into the last column of `IMU.T_b_c1`.
Since the rotation is already correct, that alone may be enough to get it tracking.
Kalibr is still the proper answer.

**The accelerometer has a 5.6% scale error.** It reads 9.2642 m/s^2 stationary against a
true 9.81. VINS hides this by setting `g_norm` to the *observed* value so its gravity
refinement stays self-consistent. That escape hatch does not exist here: ORB-SLAM3
hardcodes `GRAVITY_VALUE = 9.81` in `include/ImuTypes.h` with no config key. Handled
instead at the node — `imu_accel_scale` rescales the measurement to match the assumed
gravity, which is equivalent and keeps the patch out of ORB-SLAM3's source. Set it back to
`1.0` after recalibrating (`rs-imu-calibration.py`), or you will double-correct.

## Known issues

- **The IMU may not be initialising on the sim bag.** Gating pose output on
  `Frame::HasVelocity()` — which turns true exactly when ORB-SLAM3's inertial state
  becomes valid — produced *zero* poses, and the written velocities reach 7.2 m/s against
  the car's 4.0 m/s `TOP_SPEED`, which is the signature of finite-differenced position
  rather than an estimated velocity state. Taken together this suggests the run is
  effectively **stereo despite being launched as stereo-inertial**; ORB-SLAM3's IMU
  initialisation needs excitation a car on a flat floor may not provide. The ATE is still
  a valid number, but it is probably not an inertial one. Confirming it means patching
  `Tracking.h` to expose `Atlas::isImuInitialized()` (`Tracking::mpAtlas` is protected)
  and rebuilding the library.
- **Pose count varies between runs** (425 vs 534 on identical input), unexplained.
  Probably frame drops under load, but that is an assumption.
- **`wil_stereo_imu` on the real rig cycles through map resets** and never sustains
  tracking, so it writes no trajectory. This is the placeholder `IMU.T_b_c1` biting: a
  wrong lever arm makes the inertial and visual constraints disagree and ORB-SLAM3
  resolves that by resetting the map. `imu_accel_scale` corrects the scale error but
  cannot invent an extrinsic. Use `wil_stereo.launch.py` until a Kalibr run exists.
- **Real-rig stereo scale is unverified.** The 95 mm-baseline bag yields a 7.68 m path
  over 101.5 s (0.076 m/s mean, 2.94 m forward extent, returning within 0.16 m of the
  start). Self-consistent, but there is no ground truth for the real bags, so if the rig
  actually travelled further than that the first suspects are `overlappingBegin/End` set
  to full width and cam1 reusing cam0's intrinsics.
- In pure stereo there is no gravity alignment, so the world frame keeps the initial
  CAMERA optical axes (x-right, y-down, z-forward). Forward motion therefore shows up on
  z, not x. That is why `wil_stereo.launch.py` sets `body_frame_id: cam0`.
- The yaw / error-x / error-y columns from `plot_vio_vs_gt.py` are not meaningful here.
  See Scoring.

## Building

The ORB-SLAM3 core and Pangolin are built out of tree by `../build_orbslam3.sh` into
`../thirdparty/`, which carries a `COLCON_IGNORE`. That script is staged and resumable —
re-run it after any failure and it picks up where it stopped.

```bash
cd ~/wil_project
./build_orbslam3.sh                 # all stages that still need running
./build_progress.sh -w              # watch from any terminal; touches nothing
colcon build --packages-select orbslam3_ros2
```

Two things worth knowing if you ever rebuild from scratch:

**`Optimizer.cc` needs ~4.2 GB to compile on its own.** With 15 GB of RAM and no swap, it
gets OOM-killed when it lands in the same `-j` wave as the other heavy files, and gcc
reports only `Killed signal terminated program cc1plus` with no `error:` line, which looks
like a mystery failure. The script backs off `-j8 -> 2 -> 1` automatically.

**The out-of-tree libraries need DT_RPATH, not DT_RUNPATH.** `CMAKE_INSTALL_RPATH_USE_LINK_PATH`
alone is not enough here: it adds link *directories*, and `libORB_SLAM3.so` is linked by
full file path, contributing none. Worse, modern `ld` emits `DT_RUNPATH`, which — unlike
`DT_RPATH` — is not consulted when resolving a dependency's *own* dependencies, so
Pangolin's internal libraries go unfound. `CMakeLists.txt` sets an explicit RPATH list and
passes `-Wl,--disable-new-dtags`. Symptom if this regresses: the node links fine and dies
at launch with `cannot open shared object file`.
