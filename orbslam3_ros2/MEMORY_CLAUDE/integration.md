# ORB-SLAM3 bring-up, calibration and configuration — work log

How ORB-SLAM3 was brought into the WiL stack: the build, the package, and where
every calibration number in `config/` comes from. Written to support a later
write-up, so *root causes* and *measured numbers* are kept verbatim rather than
summarised, and the wrong turns are recorded too — several of them are the most
quotable part.

Date of the work: **2026-08-25**. Companion file: **[README.md](README.md)**, which
covers the later YOLO dynamic-object filtering work and assumes this file for
background.

---

## 1. Why ORB-SLAM3, and how it sits beside VINS

The workspace already ran one VIO system, the `vins_fusion_ros2` fork. ORB-SLAM3 is
added as a **second, independent front-end** so the two can be scored against each
other and against the sim's ground truth on identical bags. It is a genuinely
different algorithm — feature-based where VINS is optimisation-on-optical-flow — it
models the real rig's ~180° lens natively as Kannala-Brandt, and it does loop
closure, which this VINS fork does not (there is no `loop_fusion` package and
`pose_graph_save_path` is inert).

**Upstream ships a ROS 1 wrapper only.** The community ROS 2 forks
(`zang09/ORB_SLAM3_ROS2`, `Mechazo11/ros2_orb_slam3`) are built around offline
dataset playback rather than live topics. Rather than fight either, we build the
ORB-SLAM3 core out of tree and wrap it in a purpose-written Humble node — which
buys full control over topics, QoS, sim time, threading and the output format.

The single most consequential design decision: **the node writes VINS's CSV format
directly**, so `script/plot_vio.py`, `script/plot_vio_vs_gt.py` and `script/viewer.py`
work on ORB-SLAM3 output with **zero changes**. See §5.

---

## 2. Environment (reproducibility)

| | |
| :-- | :-- |
| OS / ROS | Ubuntu 22.04.5, ROS 2 Humble |
| Compiler | GCC 11.4.0, CMake 3.22.1 |
| OpenCV | 4.5.4 (system apt) |
| Machine | 16 cores, 15 GB RAM, **no swap** |
| `sudo` | **password-protected** — nothing was apt-installed for this work |

Two constraints shaped the build. **No sudo** meant everything had to land in a
user-writable prefix; it turned out no apt package was needed at all, since every
Pangolin dependency (`libglew-dev`, `libgl1-mesa-dev`, …) was already present.
**Disk was at 96% / 2.5 GB free** at planning time, which is why examples, tools and
tests are off everywhere and object trees are deleted after linking.

---

## 3. The third-party build

Everything lives in `thirdparty/` (carrying a `COLCON_IGNORE`, so colcon never tries
to build ORB-SLAM3 or Pangolin as workspace packages) and is driven by the
staged, resumable `build_orbslam3.sh` at the workspace root.

```
thirdparty/
├── COLCON_IGNORE
├── Pangolin/            v0.9.1, built and discarded
├── install/             Pangolin prefix (libpango_*.so)
└── ORB_SLAM3/           our fork; built IN TREE (upstream ships no install rules)
    ├── lib/libORB_SLAM3.so
    ├── Thirdparty/DBoW2/lib/libDBoW2.so
    ├── Thirdparty/g2o/lib/libg2o.so
    └── Vocabulary/ORBvoc.txt          139 MB
```

**Pangolin v0.9.1**, not the commonly-recommended v0.6. v0.6 dates from 2020 and
predates this toolchain. Every Pangolin symbol ORB-SLAM3's `Viewer.cc` touches
(`CreateWindowAndBind`, `OpenGlRenderState`, `Var`, `View`, `Attach`, `CreatePanel`,
`FinishFrame`) is stable across 0.6→0.9, so the upgrade is low-risk. **Cost: 0.9.1
defaults to C++17, which propagates to ORB-SLAM3** and is the root of patch (1).

**Sophus is never built.** `Thirdparty/Sophus/CMakeLists.txt` declares
`add_library(sophus INTERFACE)` — it is header-only and ORB-SLAM3 only puts it on
the include path. Upstream's `build.sh` builds it anyway, which turns its tests on;
those are a known failure and pure disk waste. This is the main reason we do not use
`build.sh`.

### 3.1 The fork and its four patches

Fork: **`BoltonAthitDavies/ORB_SLAM3`**, branch `wil`, pinned at commit
`5c2b00f`, forked from upstream `UZ-SLAMLab/ORB_SLAM3 @ 4452a3c`. Pinned to a commit
rather than a branch so a rebuild months later is byte-identical, and forked so that
upstream disappearing or force-pushing cannot break the build. The four patches are
committed on the fork, and `patch_orbslam3()` in `build_orbslam3.sh` keeps them as
grep-guarded no-ops that double as an assertion the checkout really is the patched
tree.

| # | File | Change | Why |
| :- | :--- | :----- | :-- |
| 1 | `src/LoopClosing.cc` ×3 | `mnFullBAIdx++` → `= true` | `++` on a `bool` was deprecated in C++11 and **removed in C++17**. `= true` is exactly what `bool++` did, so behaviour is unchanged and our build stays identical to every C++14 ORB-SLAM3 — which matters when the point is comparing against published results. |
| 2 | `CMakeLists.txt` | `-std=c++11` → `-std=c++17` | Pangolin 0.9's headers need it. Only the `set(...)` line is rewritten, so `add_definitions(-DCOMPILEDWITHC11)` survives — the sources use that ifdef to pick `std::chrono` over the monotonic clock, and dropping it silently changes which timing path compiles. |
| 3 | `CMakeLists.txt` | cut everything from `# Build examples` | We need only `libORB_SLAM3.so`. The ~20 example executables are disk, build time, and the place where OpenCV-4 incompatibilities (`CV_LOAD_IMAGE_*`) actually bite. |
| 4 | `include/System.h` | add `Tracking* GetTracker()` | `Frame::GetVelocity()` and `Tracking::mCurrentFrame` are public, but `System::mpTracker` is **private** (`System.h:218`), so the estimated IMU velocity is otherwise unreachable. Used to fill the velocity columns of `vio.csv`. |
| 5 | `src/Tracking.cc` | `setIntegrated()` on the `n==0` early return | **Upstream bug → SIGSEGV.** Added 2026-09-14; see §7.5. |

### 3.2 Upstream oddity, deliberately NOT changed

`mnFullBAIdx` is declared `bool` (`LoopClosing.h:226`) but used as a **generation
counter**: `int idx = mnFullBAIdx; … if(idx!=mnFullBAIdx)` at `LoopClosing.cc`
2300/2309. The name says "Idx". As a bool it saturates at `true`, so that check can
only ever fire once. Changing it to `int` would alter loop-closure behaviour, so we
left it. Worth a sentence in the report as an upstream bug we found but did not
"fix".

### 3.3 The widely-copied patch that is BACKWARDS

Search results for ORB-SLAM3 GCC-11 build failures recommend rewriting
`LoopClosing.h:51`'s `std::pair<KeyFrame* const, g2o::Sim3>` into
`std::pair<const KeyFrame*, …>`. **Do not.** For `std::map<Key,T>` the `value_type`
is `std::pair<const Key, T>`, and with `Key = KeyFrame*` that is `KeyFrame* const`
(a *const pointer*), not `const KeyFrame*` (a *pointer to const*). **Upstream is
already correct.** Applying the "fix" produces:

```
/usr/include/c++/11/bits/stl_map.h:123:71: error: static assertion failed:
    std::map must have the same value_type as its allocator
```

This was applied on the strength of a half-remembered "known fix" and cost a full
build cycle. It is now documented as a warning in `build_orbslam3.sh` so it cannot
creep back in.

### 3.4 Two build traps, each measured

**`Optimizer.cc` gets OOM-killed.** Peak `cc1plus` RSS on that one file is
**4231 MB** at `-O3 -march=native`. With 15 GB and no swap, several heavy files in
the same `-j8` wave exhaust memory and the kernel kills the compiler. GCC reports
only:

```
c++: fatal error: Killed signal terminated program cc1plus
```

— no `error:` line, so it looks like a mystery failure while `gmake` reports
`Error 1`. `build_orbslam3.sh` now retries `-j8 → 2 → 1`; make keeps completed
objects so each retry resumes. At `-j1` the file links in ~92 s.

**The out-of-tree libraries need `DT_RPATH`, not `DT_RUNPATH`.** Two separate
problems, and the first link produced a binary with **6 unresolvable libraries**:

1. `CMAKE_INSTALL_RPATH_USE_LINK_PATH` adds link *directories*, and
   `libORB_SLAM3.so` is linked by full file path, contributing none — so its folder
   was silently absent from the runpath.
2. Modern `ld` emits `DT_RUNPATH`, which — unlike `DT_RPATH` — is **not consulted
   when resolving a dependency's own dependencies**. Pangolin's
   `libpango_display.so` carries no RUNPATH of its own, so its siblings
   (`libpango_core`, `_windowing`, `_vars`, `_image`) went unfound even with
   `thirdparty/install/lib` in *our* runpath.

Fix: an explicit `CMAKE_INSTALL_RPATH` list plus `-Wl,--disable-new-dtags`.
Symptom if it regresses: links fine, dies at launch with `cannot open shared object
file`.

---

## 4. The ROS 2 package

```
orbslam3_ros2/
├── src/orbslam3_node.cpp        ROS layer: subs, sync, publishers, CSV
├── src/slam_wrapper.{hpp,cpp}   owns ORB_SLAM3::System, worker thread, IMU buffer
├── src/trajectory_writer.*      VINS-format CSV
├── config/wil_sim/stereo_imu.yaml
├── config/wil/{stereo,stereo_imu}.yaml
└── launch/{wil_sim_stereo_imu,wil_stereo,wil_stereo_imu}.launch.py
```

**Threading.** `TrackStereo()` blocks for tens of milliseconds. Called from a
subscription callback it would stall the executor and the middleware would drop IMU
messages at 200 Hz — starving the very preintegration inertial mode depends on. So
images go into a one-deep queue and a dedicated worker thread does the tracking,
dropping the *older* pending frame when it cannot keep up (counted and reported on
shutdown, so a silently degraded run is visible).

**One binary, both rigs, no republish.** `image_transport` for the sim's raw
`sensor_msgs/Image`; a direct `CompressedImage` subscription for the real rig
(see §7.2). VINS needs two extra `republish` processes for the same job — see
`vins_fusion_ros2/config/wil/stereo_imu.yaml:32-38`.

**Sync.** `message_filters` `ApproximateTime`, not `ExactTime`: the sim bag holds
804 cam0 frames against 805 cam1, so stamps are close but not identical. Slop 0.02 s,
about half a frame at 30 Hz.

**Timestamps** come from `msg->header.stamp`, never the node clock — the same clock
`script/extract_gt.py` samples for `ground_truth.csv`, which is what makes the two
directly comparable with no time alignment. `use_sim_time` therefore affects only
log throttling and cross-node agreement, not tracking.

---

## 5. Output format — the interoperability decision

The node writes `<output_path>/vio.csv` in **VINS-Fusion's exact format**:

```
t_ns,px,py,pz,qw,qx,qy,qz,vx,vy,vz
```

comma-separated, integer nanoseconds, quaternion **w first**, 11 columns.

This is deliberate. ORB-SLAM3's own `SaveTrajectoryEuRoC` (`System.cc:762`) emits
**space**-separated `t_ns tx ty tz qx qy qz qw` — wrong delimiter *and* wrong
quaternion order — and `SaveTrajectoryTUM` uses seconds. Converting after the fact
would be a second thing to get wrong. Writing the target format directly means the
existing analysis scripts need no modification, which is the whole basis of the
VINS-vs-ORB comparison. Verified byte-identical against a real `ground_truth.csv`
row.

`plot_vio.py` hard-requires columns 8-10 (`d[:, 8:11]`), so the velocity columns are
not optional padding.

### Published topics

| Topic | Type | Notes |
| :---- | :--- | :---- |
| `~/odometry` → `/orbslam3/odometry` | `nav_msgs/Odometry` | The one to use. `script/viewer.py`'s `_find_vins()` falls back to any Odometry publisher, so it picks this up unchanged — and VINS publishes Odometry too, which is what makes the two directly comparable. |
| `~/pose` → `/orbslam3/pose` | `geometry_msgs/PoseStamped` | A strict SUBSET: same header, same pose, published in the same callback so they can never disagree. Lacks `child_frame_id` and velocity. Convenience for tools that only accept PoseStamped. |
| TF | `orbslam3_world` → body frame | |

Two caveats for the report. **Velocity is world-frame in a field that is
conventionally body-frame** — `nav_msgs/Odometry` documents `twist` as being in
`child_frame_id`. Done deliberately so it matches what VINS writes to `vio.csv`, and
because `plot_vio_vs_gt.py` only compares speed *magnitude*, which is
frame-invariant. Feeding this to something that assumes the standard convention (a
`robot_localization` filter, say) would be a real mismatch. **Covariances are all
zero** — ORB-SLAM3 exposes no uncertainty through this path.

Publishing starts only once tracking reports OK, so nothing appears during IMU
initialisation.

**Every row is flushed as written.** At 30 Hz that is negligible I/O and it means a
Ctrl-C'd run keeps every pose. Without it the tail sits in the stream buffer and a
hard kill truncates the final line mid-field, which makes `numpy.loadtxt` throw on
the *whole* file — losing the entire run, not just the last pose. This was hit for
real.

---

## 6. Calibration — where every number comes from

Naming offset to keep straight: **ORB-SLAM3 counts cameras from 1, VINS from 0.**
ORB-SLAM3's `Camera1` is VINS's `cam0`; `IMU.T_b_c1` is VINS's `body_T_cam0` — same
quantity, same direction, different spelling.

Both configs derive from the *same source of truth* VINS uses, so a difference
between the two systems is a difference in the algorithms, not the calibration.

### 6.1 Sim rig — `config/wil_sim/stereo_imu.yaml`

Source: `aws-robomaker-small-warehouse-world/models/ackermann_robot/model.sdf`.

**Intrinsics are analytic, not calibrated.** A gz camera is an ideal pinhole with
square pixels and the principal point exactly at centre, so from `<image>` and
`<horizontal_fov>` alone:

```
fx = fy = (width/2) / tan(hfov/2) = 640 / tan(45°) = 640
cx, cy = 640, 360        all distortion = 0        1280×720
```

Valid only while `model.sdf` says 1280×720 at `horizontal_fov 1.5707963268`.
Both cameras are identical, legitimately — a simulated pinhole has no manufacturing
spread.

**Extrinsics are exact model geometry.** From the `base_footprint` poses:
`imu_link (0.338, 0, 0.192)`, `cam0 (0.496, 0.0936, 0.4395)`,
`cam1 (0.496, 0, 0.4395)`.

```
IMU.T_b_c1 = body_T_cam0 = [ 0  0  1  0.158 ]
                           [-1  0  0  0.0936]
                           [ 0 -1  0  0.2475]
                           [ 0  0  0  1     ]
```

The rotation block `[[0,0,1],[-1,0,0],[0,-1,0]]` is the fixed REP-103 body
(x-fwd, y-left, z-up) → REP-104 optical (x-right, y-down, z-fwd) transform. The
translation is a **real 0.25 m lever arm** — unlike the real rig.

`Stereo.T_c1_c2` computed as `inv(body_T_cam0) @ body_T_cam1`; since both cameras
carry the identical rotation it cancels exactly, leaving a pure baseline on cam0's
optical **+X**:

```
Stereo.T_c1_c2 = [1 0 0 0.0936]     ← +X ⇒ cam1 is the RIGHT camera
                 [0 1 0 0     ]
                 [0 0 1 0     ]
                 [0 0 0 1     ]
```

Verified numerically. `script/check_stereo_side.py` settles the left/right question
empirically from a bag if it is ever in doubt.

**IMU noise maps 1:1 from VINS** — both use the same continuous-density convention,
so no unit conversion: `NoiseGyro`=`gyr_n` 0.002, `NoiseAcc`=`acc_n` 0.02,
`GyroWalk`=`gyr_w` 0.0001, `AccWalk`=`acc_w` 0.001, `Frequency` 200. These are ~5×
looser than the ADIS16448 ground truth in `model.sdf` (gyro 0.0003394, accel 0.004),
deliberately, carried over from the VINS tuning so both estimators trust the IMU
equally.

Simulated gravity is exactly 9.8 against ORB-SLAM3's hardcoded 9.81 — 0.1%, far
below the noise floor, not worth correcting. Contrast the real rig at 5.6%.

### 6.2 Real rig — `config/wil/stereo.yaml` (the trustworthy one)

**Fisheye drops straight in.** ORB-SLAM3's `KannalaBrandt8` is the same model
OpenCV's `cv::fisheye` implements, so `vins_fusion_ros2/config/wil/fisheye_calibration.json`
is used unchanged — `Camera*.k1..k4` **are** the raw OpenCV `D` vector, in order:

```
fx 607.5108682653754   fy 609.9053834748036
cx 958.2558093864706   cy 539.7975091283472
k1..k4 = [-0.03645464928389926, -0.0008403394026266405,
           0.00017697497379393103, -0.000382711572282332]      1920×1080
```

**Do not copy these out of VINS's `cam0_fisheye.yaml` instead** — VINS fixes its own
`k1=1` and shifts the OpenCV terms into `k2..k5`, so those need re-indexing and
ORB-SLAM3's do not.

**cam1 reuses cam0's intrinsics.** Only one mono calibration exists for this rig.
This is a real approximation, not a sim coincidence: two physical lenses do differ.
**Calibrating cam1 separately is the cheapest available accuracy win.**

`Stereo.T_c1_c2` is VINS's `body_T_cam1` verbatim, which is exact rather than
convenient: VINS's `body_T_cam0` for this rig is identity, so `body_T_cam1` already
*is* the cam0→cam1 transform. Baseline 0.093604 m, 1.0854° relative rotation,
translation on camera +X.

`Stereo.ThDepth: 90`, not EuRoC's stock 60 — that value assumes a 0.11 m baseline;
at our 0.0936 m it would put the close/far split at 5.6 m instead of a
warehouse-appropriate 8.4 m. Same value used in the sim config.

`Camera*.overlappingBegin/End` set to full width (0–1919), which is what TUM-VI
does. **Untuned.** A ~180° pair may genuinely overlap over a narrower band, and this
is the first knob to try if fisheye stereo matches look poor.

### 6.3 Real rig IMU — be precise about which half is missing

**The ROTATION is measured and correct; only the TRANSLATION is a placeholder.**
This distinction was initially described too loosely and it changes what has to be
done about it.

*Rotation.* Averaging the stationary window of the 95 mm bag gives "up" in the IMU
frame as **(+0.029, −0.997, −0.075)** — gravity almost entirely on −Y, so the IMU's
+Y points **down**. That is the RealSense convention (x-right, y-down, z-forward),
which is the same convention as a camera optical frame. So IMU→cam0 really is
≈identity. Residual is ~4.3° pitch / ~1.7° roll of mounting tilt.

*Translation.* Zero, and wrong — it asserts the IMU sits at cam0's optical centre.
The lever arm couples angular rate into measured acceleration and this rig reaches
**1.29 rad/s** of yaw.

*Why this breaks ORB-SLAM3 but only degrades VINS:* VINS runs
`estimate_extrinsic: 1` and refines the transform online, absorbing much of the
error. **ORB-SLAM3 has no online extrinsic refinement** and takes `IMU.T_b_c1` as
truth, so the inertial and visual constraints disagree and it resets the map.

*Cheapest fix:* tape-measure the IMU→cam0 offset **in the camera optical frame**
(x right, y down, z forward, metres) into the last column of `IMU.T_b_c1`. Since
the rotation is already right, that alone may suffice. Kalibr remains the proper
answer.

**Note the rig topology:** `realsense/realsense_imu` publishes *only* the IMU. The
fisheye stereo pair is a **separate device**. So this is not a D435i where the SDK
can hand you a factory extrinsic — the SDK only knows the transform to its own
cameras, which are not the ones in use.

### 6.4 The 5.6% accelerometer scale error

Stationary, the real IMU reports **|a| = 9.2642 m/s² (std 0.030)** against a true
9.81 — steady across the whole bag (9.2642 start, 9.2610 end), so a calibration
error, not noise. RealSense IMUs ship uncalibrated.

VINS absorbs it by setting `g_norm` to the *observed* magnitude so its gravity
refinement stays self-consistent. **That escape hatch does not exist here:**
ORB-SLAM3 hardcodes `GRAVITY_VALUE = 9.81` in `include/ImuTypes.h` with no config
key. Handled instead at the node — `imu_accel_scale` (default **1.0589** =
9.81/9.2642 on the real inertial launch) rescales `linear_acceleration` before it
becomes an `IMU::Point`. Equivalent correction, and it keeps the patch out of
ORB-SLAM3's source. Set back to 1.0 after `rs-imu-calibration.py`, or you
double-correct.

### 6.5 ORB extractor parameters

Identical structure in all three configs; only `nFeatures` differs.

| Parameter | Sim | Real | Meaning |
| :-------- | --: | ---: | :------ |
| `nFeatures` | 1500 | 2000 | ORB corners per image |
| `scaleFactor` | 1.2 | 1.2 | pyramid ratio between levels |
| `nLevels` | 8 | 8 | pyramid levels |
| `iniThFAST` | 20 | 20 | FAST threshold, first pass |
| `minThFAST` | 7 | 7 | fallback where a grid cell yields nothing |

`nFeatures` was scaled by **frame area** against EuRoC's stereo-inertial default of
1200 at 752×480: the sim is 1280×720 (2.5× the area) → 1500; the real rig is
1920×1080 (5.7×) → 2000. Not proportional, because extraction cost is roughly linear
in feature count and has to fit a 30 Hz budget.

`scaleFactor` 1.2 with `nLevels` 8 are upstream defaults, untouched — together they
cover a 1.2⁷ ≈ 3.6× scale range. The two FAST thresholds are a two-pass scheme:
ORB-SLAM3 grids the image and tries `iniThFAST` per cell, retrying at `minThFAST`
where nothing is found, which keeps low-contrast regions (plain warehouse floor and
walls) from going empty.

**State this honestly in the report: these were chosen by area scaling, not
measured.** No sweep of `nFeatures` or the FAST thresholds against ATE was run. They
are clearly adequate on the sim (0.222 m, beating VINS), but whether 1200 or 2500
would be better is untested. The real rig is where tuning would most likely pay off,
since fisheye feature distribution differs substantially from the pinhole case these
defaults were designed around — and with no ground truth there, it would have to be
judged on tracking continuity rather than a number.

---

## 7. Bugs found, with root causes

Four, all found by verifying rather than assuming. Each is worth a line in the
report because each has a non-obvious root cause.

### 7.1 Missing `File.version` → segfault
`config/wil/stereo_imu.yaml` was generated by `sed`-deleting a header block out of
`stereo.yaml`, which took `File.version: "1.0"` with it. That key selects the parser
at `System.cc:78`: present → the modern `Settings` parser that understands
`Camera1.*`/`Camera2.*`; absent → the **legacy** parser, which wants flat
`Camera.fx`, finds none, prints `*Error with the camera parameters in the config
file*`, and then dereferences a null camera. Exit code −11.

The file was "validated" beforehand by confirming it parsed under OpenCV
FileStorage and that the intrinsics read back correctly — all of which passed,
because it *is* valid YAML with correct values. The one key that decides whether
ORB-SLAM3 reads any of them was never checked.

### 7.2 `yuv422; jpeg compressed mono8` → every frame dropped
The real bags' `CompressedImage.format` is **`"yuv422; jpeg compressed mono8"`**:
the original camera encoding is yuv422 while the JPEG payload is mono8.
`compressed_image_transport` decodes the payload to 1-channel mono (step = width)
but stamps the output `Image` with the *original* yuv422 encoding, which implies 2
channels. cv_bridge rightly rejects the inconsistent message:

```
Image is wrongly formed: step < width * byte_depth * num_channels
    or 1920 != 1920 * 1 * 2
```

Fixed by subscribing to `CompressedImage` and calling `cv::imdecode(…,
IMREAD_GRAYSCALE)` directly, skipping the plugin. Grayscale is what ORB-SLAM3 wants
anyway, and it is how `script/check_stereo_side.py` already reads these blobs.
**`image_transport republish` does not help** — it uses the same decoder, so the
mislabelling just moves to the raw topic.

### 7.3 Empty IMU vector → segfault
The node drained IMU samples `<= t_frame` and passed whatever came back to
`TrackStereo`, including an **empty vector**. ORB-SLAM3's `PreintegrateIMU()` prints
`Empty IMU measurements vector!!!`, leaves the frame un-preintegrated, then crashes
on it. Only the real rig triggered it: decoding a 1920×1080 JPEG *pair* is slow
enough that the 200 Hz IMU callback starves behind it. The bag was checked first and
is clean — IMU at 199.5 Hz with header stamps aligned to images within ~13 ms.

Fix: bounded 80 ms wait for the buffer to span the frame time, then **never** pass
an empty vector — skip the frame and count it. Losing a frame costs one image;
passing the empty vector costs the process.

### 7.5 `PreintegrateIMU()` forgets `setIntegrated()` → SIGSEGV  *(2026-09-14)*

**An upstream ORB-SLAM3 bug, not ours.** `Tracking::PreintegrateIMU()` has three
early returns. The first two (`non prev frame`, `Not IMU data in mlQueueImuData!!`)
both call `mCurrentFrame.setIntegrated()` before bailing. The third does not:

```cpp
const int n = mvImuFromLastFrame.size()-1;
if(n==0){
    cout << "Empty IMU measurements vector!!!\n";
    return;                       // <-- no setIntegrated()
}
```

The frame stays un-preintegrated, the caller reports `Not preintegrated
measurement`, and a null preintegration is dereferenced — **exit code −11**.

**The message is misleading and that cost time.** `n` is `size-1`, so the branch
fires when there is exactly **ONE** IMU sample between consecutive frames, not
zero. The node's existing guard rejected only an *empty* vector (§4), so it did not
cover this at all.

**Why this bag and not the earlier ones.** `dataset/dataset_real_000` is clean —
3497 images at 30.0 Hz with no duplicate or non-monotonic stamps, 23257 IMU samples
at 200.2 Hz, **median 7 samples per frame interval**. But the margin is thin: **three
intervals hold exactly two**, and the IMU has real dropouts (73 gaps >15 ms, 21
>20 ms, 2 >22 ms — longer than a whole frame). One lost message from a two-sample
interval lands on the bug.

**What was losing the messages: our own QoS.** The IMU subscription used
`rclcpp::SensorDataQoS()` — BEST_EFFORT, depth 5. At 200 Hz that is **25 ms of
buffer**, which a single ~60 ms `TrackStereo` call overruns, so the middleware was
free to drop samples under load. Fatal here rather than merely lossy.

**Fixed at both ends:**

* fork patch (5) — `setIntegrated()` before that return, so the branch degrades
  gracefully exactly as its two siblings already do;
* IMU subscription → `rclcpp::QoS(rclcpp::KeepLast(2000))` (RELIABLE). Safe for both
  sources here: `dataset_real_000` offers RELIABLE on every topic, and
  `realsense_imu/imu_node.py` publishes RELIABLE deliberately, for this same reason.
  **A BEST_EFFORT publisher would be QoS-incompatible and deliver nothing** — check
  the offered profile before pointing this at a new IMU source;
* node guard tightened from `imu.empty()` to `imu.size() < 2`.

**Result on the bag that was crashing:** 0 crashes, 0 `Empty IMU measurements`, 0
`Not preintegrated`, **1999 poses over 116.4 s**, path 13.67 m, no jumps >10 m/s.

### 7.4 Truncated CSV on hard kill
See §5. `numpy.loadtxt` throws on a ragged file, so one truncated final line
destroys the whole run's data.

---

## 8. Wrong turns worth recording

Three code changes made on unverified theories about frame conventions. All three
made things worse, and the measurement that settled it took one command. This is the
most transferable lesson in the file.

**The symptom:** `plot_vio_vs_gt.py` reported **ATE 0.309 m** alongside a **90° yaw
error and 11 m position error**. Not contradictory — diagnostic. A global SE(3) fit
recovering the trajectory to 31 cm means shape and scale are right, so the
discrepancy had to be frame convention.

1. **`align_first_pose`** — latched the inverse of the first tracked pose to
   re-express everything in the initial body frame. Made it *worse*: error moved
   into z, yaw stayed at 87°. ORB-SLAM3 re-bases its world to gravity during IMU
   initialisation, *after* that first pose, so the latched frame is rotated out from
   under it. Default is now `false`; raw output is already gravity-aligned and z-up
   (measured z error 0.041 m RMS).
2. **`wait_for_imu_init`** — gated publishing on `Frame::HasVelocity()`. Produced
   **zero poses**, which is itself the interesting result (§9). Default `false`.
3. Only then, the measurement:

```
body x axis vs travel direction: +0.729   (y −0.088, z +0.010)
quaternion yaw − path yaw: median +3.2°
```

Body x points along travel: the published pose is proper REP-103 and **self-consistent
with position**. The output was correct all along.

**The actual cause:** `plot_vio_vs_gt.py` anchors its alignment on the **first
pose**, and ORB-SLAM3 re-bases its map at IMU init — right where that first pose
comes from. One poisoned anchor rotates the entire ground-truth overlay, inflating
yaw and x/y, while ATE (fitted over all poses) stays correct. The script was written
for VINS, whose world frame is fixed from the start.

`script/plot_compare.py` was written for this reason: it aligns **every** trajectory
with a full SE(3) Umeyama fit over the whole overlap, scale fixed, which removes the
arbitrary choice of world frame while leaving drift and scale error visible.

---

## 9. Open questions

**Is the IMU actually initialising in the sim?** Gating on `Frame::HasVelocity()`
produced zero poses, and the written velocities reach **7.2 m/s** against the car's
`TOP_SPEED = 4.0` in `drive.py` — the signature of finite-differenced position
rather than an estimated velocity state. Together these suggest the sim runs are
effectively **stereo despite being launched as stereo-inertial**. The ATE is still a
valid number but probably not an inertial one. Confirming it means patching
`Tracking.h` to expose `Atlas::isImuInitialized()` (`Tracking::mpAtlas` is
protected) and rebuilding the library. **Resolve this before claiming a
stereo-inertial result in the report.**

**Real-rig scale is unverified.** The 95 mm bag yields a 7.68 m path over 101.5 s
(0.076 m/s mean, returning within 0.16 m of the start). Self-consistent, but there
is no ground truth for the real bags. If the rig actually travelled further, the
first suspects are the full-width `overlappingBegin/End` and cam1 reusing cam0's
intrinsics.

**Pose count varies between runs** (425 vs 534 on identical input). Probably frame
drops under load, but that is an assumption.

**CORRECTION (2026-09-14): real-rig map resets are an EXCITATION problem on at least
one bag, not the placeholder extrinsic.** §6.3 attributes the repeated
`Active map reset` on the real rig to the zeroed `IMU.T_b_c1` translation. That was
an inference, never checked against the logs. On `dataset_real_000` the logs are
unambiguous: **9 map resets, and all 9 are `Not enough motion for initializing`** —
the stationary-init check in §10, which has nothing to do with the extrinsic. That
run averages **0.117 m/s**, so it never banks the ~15 s of *cumulative motion*
`SetIniertialBA2()` needs.

So the tape measurement recommended in §6.3 may not be the blocker on this
recording; the recording itself is. The extrinsic remains wrong and worth fixing,
but **do not present it as the cause of the resets without grepping the log for
`Not enough motion`** — the two failure modes look identical from the outside and
produce the same `Active map reset` lines.

---

## 10. Behaviour notes worth knowing

**Stopping the robot resets the map — but only before IMU init completes.**
`LocalMapping.cc:129-144`, gated on `mbInertial`:

```cpp
if(dist>0.05)
    mTinit += <keyframe time delta>;      // counts only while MOVING >5 cm
if(!GetIniertialBA2())
    if((mTinit<10.f) && (dist<0.02))      // ~stationary
        → "Not enough motion for initializing. Reseting..."
```

`mTinit` is a **moving-time** counter, not wall clock. Accelerometer bias, gravity
direction and velocity are only observable under excitation, so ORB-SLAM3 discards
the map rather than converge on a wrong answer. The guard is `!GetIniertialBA2()`,
and `SetIniertialBA2()` fires at `mTinit > 15.0f` (line 218-220) — so the budget is
**~15 s of cumulative motion**, after which stopping is safe forever. Pure stereo is
completely immune (the block is behind `mbInertial`, and stereo has no scale to
estimate).

This is a second, independent candidate cause for the real-rig map resets besides
the placeholder extrinsic — and it is testable: grep the log for
`Not enough motion for initializing`.

**Tested on 2026-09-14, and it is the one that fires.** `dataset_real_000`:
9 `Active map reset`, 9 `Not enough motion for initializing`. Mean speed over the
whole run is 0.117 m/s. See the correction in §9.

**Practical recipe for `wil_stereo_imu`:** start moving immediately after launch and
keep moving for **15–20 s continuously**, with some turning — translation plus
rotation excites more of the state. After that, stop as much as you like; the check
is behind `!GetIniertialBA2()` and can never fire again.

**`use_sim_time`** — true iff something publishes `/clock`. It does *not* affect
SLAM timestamps (those come from message headers); it governs log throttling and
cross-node time agreement. `True` with no `/clock` leaves the node clock at 0 and
breaks throttled logging.

---

## 11. Results

Results live in parallel trees, one per system, so a dataset can be compared by name:

```
output/
├── output_vins/{simulation,real}/<dataset>/vio.csv   (+ ground_truth.csv for sim)
├── output_orb/ {simulation,real}/<dataset>/vio.csv
└── compare/    <dataset>/compare_{light,dark}.png
```

`script/plot_compare.py <dataset>` takes the dataset path relative to those two
trees, picks up ground truth from either side if present, and writes both themes.
Ground truth renders green, VINS blue, ORB-SLAM3 orange.

Three behaviours worth knowing, each added because a first version got it wrong:

* **SE(3) Umeyama alignment over the whole overlap, scale fixed** — not
  `plot_vio_vs_gt.py`'s first-pose anchor, for the reason in §8.
* **Jump detection.** Steps implying >10 m/s are counted separately and path length
  and ATE recomputed without them. This is what revealed that a "diverged" sim run
  was one 57 m teleport inside an otherwise excellent trajectory.
* **Reference-free plausibility check.** With no ground truth the reference is just
  whichever estimator came first; if *that* one has diverged, every "error vs
  reference" number blames the healthy run. Implied mean speed needs no reference
  and no common frame, so it settles which system to distrust. Without it the script
  flagged ORB-SLAM3 as diverging on the real bags when VINS was the broken one.

### Sim, against ground truth

| Dataset | System | poses | path m | GT m | ATE rms |
| :------ | :----- | ----: | -----: | ---: | ------: |
| `simulation/sim` | VINS-Fusion | 678 | 28.51 | 28.35 | 0.256 |
| `simulation/sim` | **ORB-SLAM3** | 471 | 29.48 | 28.35 | **0.222** |
| `simulation/sim_dynamic_warehouse` | VINS-Fusion | 755 | 63.16 | 28.96 | 13.744 |
| `simulation/sim_dynamic_warehouse` | ORB-SLAM3 | 362 | 59.66 | 28.68 | 5.045 |

**On the static sim ORB-SLAM3 is the more accurate of the two.** On the dynamic
warehouse both degrade, but differently: ORB's number is inflated by **6 tracking
jumps** — excluding them the path is 30.97 m against 28.68 m and its longest clean
6.1 s segment scores **0.046 m**. VINS has **no** jumps and still doubles the true
path length, so its 13.744 m is genuine divergence rather than an outlier artefact.
This is what motivated the YOLO work in [README.md](README.md).

*Earlier sim run, kept because it illustrates the failure mode:* 356 poses, 86.49 m
path, ATE 23.295 m — caused by **one 57.47 m teleport** at t=20.3 s. Path excluding
jumps was 28.25 m against GT 28.35 m (0.35% off) and the longest clean segment scored
0.190 m. The run was frame-starved (13.6 Hz against the bag's ~24 Hz camera).
**Replay sim bags with `--rate 0.5`.**

### Real, no ground truth

| Dataset | System | poses | path m | implied mean speed |
| :------ | :----- | ----: | -----: | -----------------: |
| `real/…95mm` | VINS-Fusion | 1648 | 3446.52 | **36.98 m/s** ⚠ |
| `real/…95mm` | ORB-SLAM3 | 536 | 6.41 | 0.07 m/s |
| `real/…95mm_00` | VINS-Fusion | 809 | 310.53 | **3.67 m/s** ⚠ |
| `real/…95mm_00` | ORB-SLAM3 | 448 | 12.37 | 0.15 m/s |

**VINS has diverged catastrophically on both real bags** — 37 m/s indoors, and
**1026 impossible-speed steps out of 1648 poses**. ORB-SLAM3's trajectories are the
plausible ones. This matters methodologically: with no ground truth, the first
version of `plot_compare.py` aligned everything to VINS and duly flagged *ORB* as
diverging. It now runs a **reference-free plausibility check** on implied mean
speed, so a broken reference cannot blame the healthy run.

`wil_stereo` on the real rig: full 101.5 s bag, 1578 poses, **0 map resets, 0
crashes**, start→end 0.16 m over a 7.68 m path.

`wil_stereo_imu` on the real rig **fails** — repeated map resets, tracking never
sustained, no trajectory written. Blocked on calibration (§6.3), not code.

---

## 12. Status summary

| | |
| :-- | :-- |
| Sim stereo-inertial | **verified**, ATE 0.222–0.34 m across runs |
| Real rig `wil_stereo` | **verified**, full bag, no resets |
| Real rig `wil_stereo_imu` | **runs** since 2026-09-14 — no crash, 1999 poses on `dataset_real_000`. Still resets 9× for lack of motion (§9 correction), and the IMU→cam translation is still a placeholder. |
| Pangolin viewer (`use_viewer:=true`) | **untested** — all runs used `false` |
| Whether the IMU truly initialises | **unresolved** — see §9 |
| Real-rig trajectories | **plausible, unverified** — no ground truth exists for these bags |

---

## 13. How it actually went — chronology

The topical sections above are organised for reading; this is the order things
happened, which is what a "lessons learned" paragraph needs. The shape of it: the
third-party build was slow but never in doubt, while **every genuine surprise came
from the interface between ORB-SLAM3 and this workspace's data** — never from
ORB-SLAM3's algorithms.

1. **Survey and plan.** Confirmed no sudo, 2.5 GB free, and that nothing SLAM-adjacent
   was vendored (`grep ORB_SLAM` returned zero hits workspace-wide). Verified the
   ORB-SLAM3 API and settings schema by fetching the upstream sources rather than
   working from memory — which is how the `SaveTrajectoryEuRoC` format was caught
   early enough to shape the CSV decision in §5.
2. **Pangolin** built clean first try.
3. **ORB-SLAM3 build, four attempts.** Killed twice by PC restarts, once by a session
   teardown, once by the OOM killer. This is why `build_orbslam3.sh` became
   stage-based and resumable, and why `build_progress.sh` exists — a progress check
   that reads only the filesystem and can be run from any terminal, so build state is
   never a mystery.
4. **The self-inflicted patch.** The backwards `LoopClosing.h` "fix" (§3.3) cost a
   full build cycle. It had been recorded during planning as "verified present" —
   true of the *string*, but the direction was never checked.
5. **First `colcon build` compiled first try**, then produced a binary with 6
   unresolvable libraries (§3.4). Caught only because linkage was checked rather than
   assumed from a successful build.
6. **Sim run worked immediately** — ATE 0.303 m on the first scored run.
7. **The 90° yaw detour** (§8): three code changes on unverified theories, all
   regressions, before one measurement settled it. **The single most transferable
   lesson here.**
8. **Real rig, three failures in a row**, each a different layer: `File.version`
   (config), the `yuv422` mislabelling (transport), the empty IMU vector (our node).
   Each was diagnosed from the data — reading the bag's `format` string, checking IMU
   header stamps against image stamps — rather than by guessing.
9. **Real stereo succeeded**; real stereo-inertial did not, and the cause is
   calibration (§6.3), not code.
10. **Comparison tooling** (§11) written last, and immediately exposed a
    methodological trap: with no ground truth, aligning to VINS blamed ORB-SLAM3 for
    VINS's divergence.

### If this were done again

* Verify the *direction* of any patch copied from a search result, not just that the
  pattern matches.
* Measure before changing code when the symptom is a frame convention. The
  diagnostic that settled §8 was one command; three edits preceded it.
* Check what a config key *does* to the consumer, not just that the file parses. The
  `File.version` segfault passed every validation that was actually run.
* On a machine with no swap, assume the biggest translation unit needs ~4 GB and cap
  `-j` accordingly.
* **Check free disk before any `FORCE=1` rebuild** — see the 2026-09-14 note below.

### Session 2: 2026-09-14 — the `PreintegrateIMU` segfault

`wil_stereo_imu` on the new `dataset_real_000` died with `exit -11` after two
`Empty IMU measurements vector!!!` lines. Sequence:

1. Checked the node's guard first — **still present and correct**, so the earlier
   theory (we pass an empty vector) was wrong.
2. Grepped the fork for the message and read `PreintegrateIMU()`. Found the missing
   `setIntegrated()` and, more importantly, that `n = size-1` means the branch fires
   at **one** sample, not zero — which is why the guard did not cover it (§7.5).
3. Checked the bag before blaming it: clean, median 7 IMU samples per frame
   interval, but **three intervals with exactly two**.
4. That pointed at message loss rather than data, which led to the BEST_EFFORT
   depth-5 IMU subscription.
5. Verified the bag's *offered* QoS before switching to RELIABLE — all topics offer
   RELIABLE, so the switch is compatible. Skipping this check is how you get a
   silent subscription.
6. Patched the fork, rebuilt, re-ran: 0 crashes, 1999 poses.
7. Read the logs rather than assuming, and found **9/9 resets are
   `Not enough motion`** — which corrects §6.3's attribution (§9).

**Mistake worth recording: a `FORCE=1` library rebuild filled the root filesystem.**
The `cleanup` stage had deleted the ~1 GB object tree, so recreating it exhausted the
disk to the point where *no command could run at all* — the shell could not create
its own temp directory. `require_space`'s 1500 MB gate is too low for a full `-j8`
rebuild and should be raised. The subsequent rebuild was run behind a watchdog that
kills the compilers below 300 MB free, so a repeat aborts cleanly instead of taking
the machine down.

**The pattern across both sessions:** every genuine surprise came from the
interface — config keys, transport encodings, QoS, upstream error handling — and
never from ORB-SLAM3's algorithms. And in each case the thing that resolved it was
reading the source or measuring the data, after at least one wrong guess.
