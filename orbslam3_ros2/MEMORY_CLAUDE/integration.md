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

### 3.1 The fork and its thirteen patches

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
| 6 | `src/Map.cc` | `PreSave()` iterates a snapshot of `mspMapPoints` | **Upstream bug → SIGSEGV at `SaveAtlas`.** The loop body erases from the set it is walking. Added 2026-10-14; see §7.6. |
| 7 | `src/LoopClosing.cc` | initialise `mpCurrentKF`; `!mbActiveLC` check moved below the queue pop | **Upstream bug → busy-spin, then SIGSEGV** with `loopClosing: 0`. Added 2026-10-15; see §7.7. |
| 8 | `src/System.cc` | `Shutdown()` stops the Viewer thread (bounded wait) | Upstream left it commented out → crash in GTK at exit with `use_viewer:=true`. Added 2026-10-15; see §7.8. |
| 9 | `src/Viewer.cc` | `DestroyWindow` on the viewer thread before `SetFinish()` | Companion to (8): Pangolin's window registry otherwise dies in `_dl_fini` with no GL context. Added 2026-10-15; see §7.8. |
| 10 | `include/ORBextractorGPU.h`, `src/ORBextractorGPU.cc` (new); hooks in `ORBextractor.{h,cc}`, `System.cc`, `CMakeLists.txt` | GPU front end for the ORB extractor behind `ORBextractor.useGPU` | **Performance, not a bug.** Committed in the fork, not applied by the script. Added 2026-10-28; see §3.5. |
| 10b | `GpuHamming` in the same two GPU files; hooks in `Frame.{h,cc}`, `System.cc` | GPU brute-force Hamming 2-NN for the fisheye stereo matcher (`Stereo.useGPUMatch`, absent = follow `useGPU`) | **Performance.** Bit-identical to `cv::BFMatcher`. Added 2026-10-29; see §3.5.1. |
| 11 | `src/ORBmatcher.cc` | `NLeft` guard on the keypoint-angle lookup in the relocalisation `SearchByProjection` (fisheye stereo read past `mvKeysUn`) | **Segfault after tracking loss on the real rig** (`exit code -11`). gdb-confirmed 2026-10-29; applied by the script; see §7.10. |
| 12 | `src/LocalMapping.cc`, `src/ORBmatcher.cc` (Fuse), `src/Optimizer.cc` (merge BA) | Five more right-camera index guards: `mvKeysRight[i]` → `[i - NLeft]` in `KeyFrameCulling`, `mvDepth`/`mvuRight` reads limited to left indices, `get<0> == -1` skipped | **Segfault while tracking normally on the real rig** (`-11` in the mapping thread). Kernel-log-confirmed 2026-11-02; applied by the script; see §7.12. |

Patches (6)–(9) keep the original upstream line as a comment next to the replacement, and
each is guarded in `build_orbslam3.sh` so it applies once and only to unpatched source. All
four were dry-run against the fork's committed files on 2026-10-15.

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

**Third trap, found 2026-10-14: the `require_space` gate is a flat 1500 MB.** It is sized for a
full `-j8` rebuild and refuses an incremental one that needs ~50 MB. With the object tree cached,
`cmake --build thirdparty/ORB_SLAM3/build -j2` on the same tree does the identical incremental
build without the gate — that is how patches (6)–(9) were built at ~1 GB free, each behind a
watchdog that kills `cc1plus` below 250 MB. The gate should be made proportional to the work.

### 3.5 The GPU front end (patch 10, 2026-10-28)

**Why.** On the real rig (1920×1080 fisheye, 2000 features, 8 levels) the tracker took
~85 ms per stereo pair, of which **~80 % was ORB extraction** (`tracking_frontend.csv`,
§7.9 runs) — one CPU thread doing FAST on an 8-level pyramid of two 2 MP images. A
30 Hz camera was therefore processed at ~10 Hz and the rest dropped (§7.9), which is
where the turn-time losses came from. The SLAM part (pose, local map) was under 10 ms.

**What moved to the GPU, and what did not.** `ORBextractorGPU` runs OpenCV's FAST-9
corner score (a literal port of `cornerScore<16>`), 3×3 NMS plus ORB-SLAM3's
two-threshold cell rule, the 7×7/σ2 Gaussian with `REFLECT_101`, `IC_Angle`, and the
rotated rBRIEF with `cvRound` semantics and OpenCV's `fastAtan2` polynomial. The
pyramid, the dynamic-mask veto, **`DistributeOctTree` (the uniform-coverage selection)**
and the mono/stereo ordering stay on the CPU, unchanged. The CPU code is intact beneath
a runtime switch, `ORBextractor.useGPU: 0/1` (int, absent = 0), read raw in `System.cc`
like `loopClosing`; the real-rig configs set 1, the sim configs stay at 0 so Experiments
1–4 remain comparable with stock ORB-SLAM3.

**No CUDA toolkit is needed to build.** The driver API (`libcuda.so.1`) and NVRTC
(`libnvrtc.so.12`, here from the torch pip wheels; override the directory with
`ORB_SLAM3_NVRTC_DIR`) are `dlopen`ed at first use and the kernels are JIT-compiled
(~90 ms once per process). If any step fails the extractor logs one line and runs the
CPU path. Two things learned the hard way: the pip `nvidia-cuda-nvcc-cu12` wheels ship
`ptxas` but not `nvcc`, so a compile-time CUDA build was never an option without sudo;
and installing a different `nvidia-cuda-runtime-cu12` than torch pins breaks the YOLO
detector's torch — it was restored to 12.1.105.

**Fidelity** (harness, four `realv5` frames, GPU vs stock CPU): per-level keypoint counts
within ±2; **90 % of GPU keypoints at the identical CPU position and octave**; on those,
angle difference 0.00°, descriptor Hamming distance 0.35 bits on average, 73 % bit-identical,
99 % within 2 bits. The other 10 % are NMS/octree tie-breaks between equal-score corners,
not a different definition. Per image: **39 → 9.3 ms** (×4.1–4.3).

**In the node** (stereo-only, `realv5`, loop closing off, same bag, viewer off):

| rate | arm | pairs processed | dropped | extraction stage | tracker total | inliers (median) | map points | losses / resets |
| :-- | :-- | --: | --: | --: | --: | --: | --: | --: |
| 0.5 | CPU | 2285 | 39 % | 67 ms | 110 ms | 948 | 4937 | 0 / 0 |
| 0.5 | **GPU** | **2803** | **25 %** | **30 ms** | **82 ms** | 923 | 4907 | 0 / 0 |
| 1.0 | CPU | 636 | 76 % | 131 ms | 206 ms | 806 | 4800 | 0 / 0 |
| 1.0 | **GPU** | **1043** | **58 %** | **44 ms** | **118 ms** | 830 | 5273 | 0 / 0 |

Both GPU runs logged the front end active twice — once per extractor thread — so the
two-thread left/right path is exercised. **The in-node gain is 1.3–1.75×, not the
harness's 4×:** with extraction cut to 30 ms the brute-force stereo matcher and
local-map tracking, both still on the CPU, become the larger share of the 82 ms. At
rate 1.0 the tracker is still above the 33 ms frame period, so frames still drop — fewer,
not none. The next levers, in order, are the stereo descriptor matching (a GPU Hamming
kNN is straightforward — **done as patch 10b, §3.5.1**), the local-map search, and input resolution.

**Caveats.** GPU extraction is not bit-reproducible run to run, so repeat runs of the
real rig are statistical, not exact. The GPU is shared with the YOLO detector in
Experiment 3; contention was not measured. Nothing in the simulation results uses this
path.

#### 3.5.1 The GPU stereo matcher (patch 10b, 2026-10-29)

**Why.** With extraction on the GPU, a stage-by-stage harness of the fisheye stereo step
(`output/compare/real/gpu-stereo-matcher-ab/stereo_stage_test.cc`) gave, per frame:
extraction 12 ms (two threads), **brute-force kNN 8–10 ms**, ratio test 0.1 ms,
triangulation 0.2 ms. The kNN is `cv::BFMatcher(NORM_HAMMING).knnMatch(k=2)` over *all*
≈2000×2000 descriptors, because the real-rig lapping area is the full 1920 px. It costs
8 ms only when OpenCV gets all 16 threads; single-threaded it is 38 ms, and inside the
node it competes with LocalMapping, the executor and the decoder, so its in-node share
was the larger part of the 30 ms "preprocessing + stereo" stage.

**What changed.** `GpuHamming::Knn2` (in `ORBextractorGPU.{h,cc}`, same dlopen'd
driver/NVRTC module, one extra kernel `knn2`: one thread per query descriptor, train
descriptors streamed through shared memory, `__popcll` on 4×u64) replaces the `knnMatch`
call in `Frame::ComputeStereoFishEyeMatches`; the CPU line is kept as a comment and as
the fallback. Strict `<` updates reproduce `cv::batchDistance`'s tie order (lower train
index wins), so the ratio test downstream sees **identical input: 2003/2003 and 2005/2005
queries identical in best index and both distances on the two harness pairs, identical
ratio-test sets; 9–10 ms → 0.44 ms.** Switch: it follows `ORBextractor.useGPU`;
`Stereo.useGPUMatch: 0/1` (int) overrides it (used for the A/B). The pinhole path
(`ComputeStereoMatches`, row-band search) is untouched, so the sim is unaffected.

**In the node — and a measurement lesson.** The first A/B (full bag, A then B back to
back) was unusable: on this laptop every run slows from ~60 to ~150 ms per frame as the
CPU package passes 90 °C and throttles, so whichever arm ran second looked worse. The
record is kept in `order_confounded/`. The valid measurement is an **ABBA sequence on the
first 60 s of `realv5`, each run started only below 72 °C**:

| rate | arm | start °C | pairs processed | dropped | extraction+stereo stage | pose pred. | local map | tracker total | inliers | map points | losses |
| :-- | :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| 1.0 | A1 CPU matcher | 73 | 840 | 47 % | 27.3 ms | 17.1 | 18.0 | 64.2 ms | 831 | 2498 | 0 |
| 1.0 | **B1 GPU matcher** | 74 | **1153** | **34 %** | **14.8 ms** | 14.1 | 14.2 | **44.7 ms** | 851 | 2500 | 0 |
| 1.0 | **B2 GPU matcher** | 73 | **1147** | **35 %** | **15.1 ms** | 14.0 | 14.3 | **44.5 ms** | 844 | 2444 | 0 |
| 1.0 | A2 CPU matcher | 73 | 866 | 45 % | 27.0 ms | 16.8 | 17.4 | 63.2 ms | 820 | 2711 | 0 |
| 0.5 | A1 CPU matcher | 73 | 1487 | 12 % | 26.1 ms | 18.1 | 18.4 | 64.9 ms | 878 | 2734 | 0 |
| 0.5 | **B2 GPU matcher** | 75 | **1660** | **7 %** | **14.4 ms** | 17.5 | 16.7 | **50.1 ms** | 917 | 2854 | 0 |
| 0.5 | A2 CPU matcher | 79 (hot) | 1088 | 29 % | 31.8 ms | 25.0 | 24.6 | 84.0 ms | 853 | 2883 | 0 |
| 0.5 | B1 GPU matcher | 95 (gate timed out) | 89 | — | — | — | — | — | — | — | crashed |

At rate 1.0 the two A runs and the two B runs agree to within 1 ms, which is the
repeatability the thermal gating buys: **tracker 64 → 45 ms (−30 %), stage 27 → 15 ms,
frames processed +35 %, drops 46 → 35 %**, inliers and map points at parity, 0 losses.
At rate 0.5 the one clean pair gives −23 % (65 → 50 ms) and drops 12 → 7 %. The
pose-prediction and local-map stages also got ~3 ms faster with the GPU matcher: the CPU
`knnMatch` spawned a 16-thread `parallel_for` on every frame that contended with the
other threads. r05_B1 crashed after 89 frames: it started at 95 °C with 190 ms frames,
lost tracking, logged "Relocalized!!" twice and segfaulted — the known stereo-only
loss→relocalise crash (§12, realv4), which the CPU-matcher arm `extract_r10` of the first
attempt also hit; neither is in the matcher path.

**Where the tracker time goes now** (rate 1.0, GPU both): stage 15 ms (≈12 ms of it the
two extractor threads, pyramid and uploads), pose prediction 14 ms, local map 14 ms,
total 45 ms — still above the 33 ms frame period at 30 Hz, so the remaining drops are
real. The next lever is the local-map search (`SearchByProjection` on the CPU), then
input resolution; both are larger changes than 10 and 10b.

**Operational.** The laptop throttles at ~90 °C and every long run drifts; any timing A/B
on this machine needs cool starts and ABBA order, and "frames processed" over a full bag
is not a reliable metric for effects under ~20 %.

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

**Extrinsics re-calibrated 2026-10-21** — `camera_parameters/211026camera1_rot_trans.dat`
(OpenCV `stereoCalibrate`, `X_cam1 = R·X_cam0 + T`, so the file is `cam1_T_cam0`). Baseline
**0.089723 m**, 0.9669° (was 0.093604 m, 1.0854°). After the 2026-10-19 remount cam1 is
physically **left** of cam0, and the user's rule is *cam0 = cam0, cam1 = cam1*: Camera1 := cam0,
no topic swap. So `Stereo.T_c1_c2` is the **inverse** of the file (cam1 on cam0's −X), unlike
the old value. That is legal for the KannalaBrandt8 path — `Settings.cc:347` takes
`b = |t|` and `TriangulateMatches` only requires depth > 0 in both cameras — so the realv3
"99 % of matches behind the cameras" was stale extrinsics contradicting the rig, not a
left-first rule; the `image0_topic:=/cam1` swap in the launch comments is retired. The same
`.dat` feeds VINS as `body_T_cam1 = body_T_cam0 · cam0_T_cam1` in both real-rig files, with
VINS `stereo.yaml`'s topics returned to cam0/cam1. Old blocks are kept as comments in all four
files. The `.dat` still records no RMS or image count. Every realv3 ORB result scored before
this date used the old (and, in `stereo_imu.yaml`, 2 mm-typo'd) baseline and should be re-run.

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

### 7.6 `Map::PreSave()` erases from the set it is iterating → SIGSEGV  *(2026-10-14)*

**Symptom:** `System.SaveAtlasToFile` set, run ends `exit -11` at shutdown with **no `.osa`**, but
only *after* every CSV and both trajectory files are written and `shutdown_status=graceful`
is logged — the node's own shutdown completes; the crash is inside `SaveAtlas()`.

**Cause (gdb):** `_Rb_tree_increment ← Map::PreSave ← Atlas::PreSave ← System::SaveAtlas`.
The first loop in `PreSave` range-fors over `mspMapPoints` and calls `EraseObservation()`,
which at `nObs<=2` calls `SetBadFlag()` → `Map::EraseMapPoint()` → `mspMapPoints.erase(pMP)`
— the element the iterator stands on. Data dependent: it fires only when some point actually
falls to ≤2 observations during the save, which is why `dataset_allsensor_realistic_001`
crashed every time while other sequences had written a healthy 2.7 MB atlas from the same build.

**Fix:** patch (6), iterate a snapshot. Verified: same dataset, exit 0, 36 MB `.osa`.

**Two things the fix does not change.** The `.osa` path is hardcoded `"./" + name + ".osa"`,
so it lands in the launch cwd, not `output_path`; and the save runs only on graceful shutdown —
a second Ctrl-C or `kill -9` skips it.

### 7.7 `loopClosing: 0` busy-spins, then reads uninitialised memory  *(2026-10-15)*

Two upstream defects behind the one YAML switch (§10 for what that switch actually controls):

* `LoopClosing::mpCurrentKF` is never initialised. Its only assignment is in
  `NewDetectCommonRegions()` *after* the `!mbActiveLC` early return, so with loop closing off
  `Run()` reads heap garbage on every iteration (the fork's `place_recognition` log line
  null-checks it, which is useless against garbage).
* The early return sat *before* the queue pop, so the keyframe was never dequeued,
  `CheckNewKeyFrames()` stayed true and the thread spun at 100 % CPU — **64 254
  `place_recognition` rows, all `current_keyframe_id = -1`**, on
  `dataset_allsensor_003/logging_20260916_02`. It also never added to `mpKeyFrameDB`, which
  starves `Tracking::Relocalization()` of candidates.

Harmless while the garbage happened to be zero. Fatal once a 64 MB `LoadAtlasFromFile` had
dirtied the heap before `new LoopClosing`: SIGSEGV in `LoopClosing::Run()` 0.2 s after
`tracking OK`. gdb: `Run()+271`, `mov 0x8(%rax),%rcx` with `rax = 0x60c6`.

**Fix:** patch (7) — initialise the pointer, and move the check below the pop with the same
add-to-DB / `SetErase` / `return false` body as the three gates that follow it. Verified:
`loop_closing.csv` now has one row per keyframe with real ids (157 rows, zero `-1`), and
`place_recognition_checks` is meaningful again.

### 7.8 Crashes at exit with `use_viewer:=true`  *(2026-10-15)*

Two layers, both only reachable once the run otherwise succeeds — every output file is already
on disk when they fire, so they are cosmetic for data but they break any wrapper that checks
the exit code.

1. `System::Shutdown()` never stops the Viewer thread: the upstream block that does so is
   commented out. `Viewer::Run()` is still inside `cv::waitKey()`'s GTK main loop when `main()`
   returns → SIGSEGV or SIGTRAP in `libgtk-3`/`libcanberra`. Patch (8) re-enables it with a
   bounded (≤2 s) wait so a wedged viewer can never hang shutdown.
2. With (8) in place, Pangolin's global `map<string, shared_ptr<PangolinGl>>` is then destroyed
   by the **main** thread in `_dl_fini` with no current GL context → `glDeleteTextures` faults
   (`GlFont::~GlFont ← PangolinGl::~PangolinGl ← __do_global_dtors_aux`). Patch (9) calls
   `cv::destroyAllWindows()` and `pangolin::DestroyWindow("ORB-SLAM3: Map Viewer")` on the
   viewer thread, before `SetFinish()` releases the main thread.

Verified under gdb with loaded atlas + `loopClosing: 0` + viewer + YOLO filter — the exact
configuration that crashed — `Inferior 1 exited normally`; viewer-off control exit 0.

**A note on method.** I attributed the §7.6 crash to the viewer before looking, and to
`loopClosing: 0` after that; both were wrong. Every one of (6)–(9) was then confirmed by
backtrace before touching source. The recipe is cheap: the launch leaves its params file in
`/tmp/launch_params_*`, so
`gdb -batch -ex 'handle SIGINT nostop noprint pass' -ex run -ex 'bt 30' --args install/orbslam3_ros2/lib/orbslam3_ros2/orbslam3_node --ros-args -r __node:=orbslam3 --params-file <it>`
with the bag played separately reproduces any of them in about a minute.

### 7.9 realv3: camera order reversed → map initialises with 0 points  *(2026-10-20)*

**Symptom as reported:** "ORB didn't extract image features at all" on `dataset_realv3_000`.
**Symptom as measured:** it extracted ~4000 features on every processed frame
(`tracking_frontend.csv` `features` ≈ 4003–4019), but `map_matches_inliers` was 0 on all 407
frames and `tracking_state` cycled `1→2→4` (NOT_INITIALIZED → OK → LOST) every three frames — 118
resets in 43 s, `last_reset_reason = tracking_lost_small_map`. Stereo init printed
`New Map created with 0 points` 211 times (1 point ×63, 2 points ×11) in the control run.

**Cause.** On this bag cam1 lies to cam0's **left** — `script/check_stereo_side.py`: median
disparity −33 px, every pair negative — the opposite of `dataset_real_000` (+19 px, 2026-09-14).
The real configs put Camera2 at +X of Camera1 (`Stereo.T_c1_c2` translation +0.0936 m), so with
the topics assigned as before every correct stereo correspondence triangulates *behind* the
cameras and `Frame::ComputeStereoFishEyeMatches` → `KannalaBrandt8::TriangulateMatches`
(`depth > 0.0001`) rejects it. Node-free check with the config's KB8 intrinsics and `T_c1_c2`
on 4308 real matches: **1 % in front of both cameras as-is, 99 % swapped** (median depth ~1 m).
Images are upright, so this is the topic↔side assignment, not an upside-down mount.

**Why the first swap attempt showed nothing.** `image0_topic`/`image1_topic` were hard-coded in
the launch files' params dict, not declared arguments, so `image0_topic:=...` on the command
line was silently ignored and the "swapped" run subscribed to the same topics. Both real launch
files now declare them (defaults unchanged); the install is symlinked, no rebuild.

**Verified in-node (params-file swap, viewer off, same bag):**

| run | init | inliers (median) | frames OK | losses | map points | path |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| stereo as-is | 0 points ×211 | 0 | 0 / 584 | 292 | 0 | 0.00 m |
| stereo swapped | 89 points, once | 242 | 536 / 536 | 0 | 1631 | 3.79 m |
| stereo-inertial swapped | 85 / 68 / 101 | 262 | 458 / 462 | 3 (early) | 3391, IMU initialised | 5.07 m |

**Procedure for every new real bag:** `python3 script/check_stereo_side.py <bag> 8` first. Negative
→ launch with `image0_topic:=/cam1/image_raw image1_topic:=/cam0/image_raw`. The VINS real
configs hard-code the same cam0=left mapping in their yaml and will fail identically.

**Still open from the same bag:** ~55 % of stereo pairs are dropped at `--rate 1.0`
(1920×1080 fisheye, 2000 features) — replay at 0.5 for comparable runs; and the IMU→camera
rotation `diag(1,-1,-1)` is consistent with gravity on +Y but gravity cannot distinguish an
X-flip from a Z-flip (yaw sign), so the inertial run is not yet trustworthy.

### 7.10 Stereo-fisheye relocalisation reads past `mvKeysUn` → SIGSEGV  *(2026-10-29)*

**Symptom.** Stereo-only (and stereo-inertial) runs of the real rig died with
`exit code -11` some seconds after a tracking loss: `Fail to track local map!`,
`tracking state 3`, one or two `Relocalized!!`, then the segfault (realv4 on 2026-10-27,
`realv5` on 10-28 and three times on 10-29). Never seen in the pinhole simulation.

**Evidence.** Caught under gdb on `realv5` at rate 1.0 (`output/compare/real/gpu-stereo-matcher-ab/`
is unrelated; the capture lives in the session scratch, backtrace copied below):

```
Thread 14 "orbslam3_node" received signal SIGSEGV
#0 ORB_SLAM3::ORBmatcher::SearchByProjection(Frame&, KeyFrame*, std::set<MapPoint*> const&, float, int) + 2828
#1 ORB_SLAM3::Tracking::Relocalization()
#2 ORB_SLAM3::Tracking::Track()
#3 ORB_SLAM3::Tracking::GrabImageStereo(...)
   rax = 0xffffffff80000000   (bin = INT_MIN)      r12 = 0x7e3 = 2019   (i, the map-point index)
   call roundf ; vcvttss2si %xmm0,%eax ; cmp $0x1e,%eax ; ... ; => mov -0x308(%rbp,%rcx,1),%rsi
```

The faulting instruction indexes the stack array `rotHist[bin]` with the result of
`round(rot*factor)`; `INT_MIN` is what the conversion returns for a NaN. `rot` is
`pKF->mvKeysUn[i].angle - CurrentFrame.mvKeysUn[bestIdx2].angle`. For a KannalaBrandt8
frame `mDistCoef` is all zeros, so `Frame::UndistortKeyPoints()` sets `mvKeysUn = mvKeys`,
which holds only the `NLeft` left-camera keypoints, while `GetMapPointMatches()` is
`NLeft + NRight` long. A map point first observed through the right camera has `i ≥ NLeft`
(here 2019 against ~2003), the read goes past the vector, the angle is garbage, and the
histogram index is wild. The `assert` is compiled out in Release. Every other
`SearchByProjection`/`SearchByBoW` variant on the tracking path selects
`mvKeys`/`mvKeysRight` by `NLeft`; this one, used only from `Tracking::Relocalization()`,
did not — which is why it only ever fired after a tracking loss on the fisheye rig.
It is nondeterministic because the overrun only faults when it crosses into unmapped
memory (one capture relocalised four times and survived).

**Fix (patch 11).** The same `NLeft`/`Nleft` selection as the other variants for both
angles, plus a `bin` range check so a NaN can never index the histogram. Original line
kept as a comment. Applied idempotently by `build_orbslam3.sh` (dry-run on a pristine
copy reproduces the fork file byte for byte).

**Verification.** Plain replays of `realv5` on the patched library no longer lose
tracking at all (the GPU matcher made the tracker fast enough), so relocalisation was
forced instead: build a map from the first 25 s, then restart playback at offsets 3, 40
and 8 s under gdb. Run 1 lost tracking, printed `Relocalized!!` and carried on; run 2 spent
48 frames in RECENTLY_LOST with repeated relocalisation attempts before opening a new map.
0 segfaults, graceful shutdown in both. Before the fix the same path crashed in four of
the six runs that reached it. Logs, scripts and the annotated backtrace:
`output/compare/real/relocalisation-crash-patch11/`.

### 7.11 Stereo-inertial on the real rig: stale accelerometer scale and IMU starved behind JPEG decode  *(2026-10-29)*

**Symptom.** `wil_stereo_imu.launch.py` on `dataset_realv5_000` (26855 IMU messages,
a clean 200 Hz stream covering the whole recording) never held tracking: 10 map
resets in 30 s, `imu_initialized 0`.

**Cause 1 — the accelerometer scale was for a different IMU.** The launch default
`ACCEL_SCALE = 9.81/9.2642` was measured on the RealSense IMU in September. The
rig's current IMU (`/hwt101ct_yaw_publisher`) reads **9.8858 m/s² at rest**
(stillest 2 s window of the bag, std 0.018), so the node fed ORB-SLAM3 10.47 m/s²
against its hard-coded 9.81 and the inertial initialisation never converged
(every reset `tracking_lost_before_inertial_ba2`). With `imu_accel_scale:=0.9923`
initialisation completed at 3.2 s. The launch default is now `9.81/9.8858`, the old
line commented. Gravity sits on +Y, (0.17, 9.88, 0.24), so `IMU.T_b_c1 = diag(1,-1,-1)`
still holds to about 1.5°.

**Cause 2 — the IMU arrived in bursts while the images flowed.** After the scale
fix, 49 % of frames were "IMU-starved" (fewer than two samples between frames even
after the wrapper's 80 ms wait): `performance.csv` showed zero IMU messages between
most frames, then bursts of up to 1063 samples (5 s of data), with IMU-free stretches
up to 15 s of wall time, while `stereo_received` advanced one frame per row. Only
83 % of the IMU messages ever arrived. Two layers:

- *Loss* — the Linux default socket buffers (`net.core.rmem_max` 208 KB) overflow under
  the two JPEG streams. `sysctl -w net.core.rmem_max=67108864 net.core.rmem_default=67108864
  net.core.wmem_max=67108864` raised delivery to 99.8 % but left the bursts.
- *Delay* — the IMU subscription lived in the node's default, mutually exclusive
  callback group, so every IMU callback queued behind a stereo JPEG decode on the
  executor, and a RELIABLE reader then released the backlog at once. Giving the IMU
  subscription its **own reentrant callback group** (`orbslam3_node.cpp`, old lines
  commented) removed it: starved frames 615 → 2, resets 19 → 2, map points
  177 → 9768, 2318 poses over the full 134 s. A `imu_best_effort:=true` launch option
  was added for the A/B; it starved 0 frames but processed fewer (more queue drops) and
  is not the default.

| run (rate 0.5, scale 0.9923) | IMU received | starved frames | resets | losses | map points |
| :-- | --: | --: | --: | --: | --: |
| default buffers, default callback group | 22409 (83 %) | 1063 | 19 | 73 | 3140 |
| 64 MB buffers | 26813 (99.8 %) | 615 | 19 | 78 | 177 |
| + IMU in own reentrant group (RELIABLE) | 26838 | **2** | **2** | 11 | 9768 |
| + IMU in own group, BEST_EFFORT | 26854 | 0 | 1 | 1 | 10068 |

**What remains.** In the fixed run the two resets are at 3.0 s and 5.6 s during
inertial initialisation (`local_mapping_bad_imu`, ORB-SLAM3's own IMU sanity check),
and one RECENTLY_LOST cluster at 59.6–61.9 s recovers every time (patch 11); from 62 s
to the end tracking is continuous. The remaining lever is the IMU→cam0 lever arm, still
zero and now on a relocated camera (PROBLEM 1 in the config header).

**Tools and rules.** `script/imu_arrival_probe.py` measures arrival gaps, bursts and
loss at a subscriber with the node's QoS. Make the sysctl permanent in
`/etc/sysctl.d/60-ros2-dds.conf`. On every new rig or IMU, re-measure |a| at rest
before trusting `imu_accel_scale`; a wrong gravity norm silently prevents
initialisation. Artefacts: `output/output_orb/real/realv5_imu_*`.

### 7.12 Mapping thread reads past the left-camera vectors on fisheye stereo → SIGSEGV  *(2026-11-02)*

**Symptom.** On `dataset_realv6_000` at replay rate 1.0 the node died with `-11`
20–50 s into the bag, five times in an afternoon, with tracking in the OK state and
no loss before the crash; the same binaries, bag and config had completed two runs
that morning. The kernel log (`journalctl -k`) gave the faulting address each time:

```
orbslam3_node[26065]: segfault at 712fd3e75030 ip 00007131209b2db7 ... in libORB_SLAM3.so[71312092c000+264000]
```

Three of five at the same offset, `0x12c000 + 0x86db7 = 0x1b2db7 = LocalMapping::KeyFrameCulling()+0x427`
(`nm` on the library; the executable segment's virtual address is `0x12c000`), one in
`CreateNewMapPoints()+0x6a0`. The disassembly at `+0x427` dereferences a reference
to a `cv::KeyPoint::octave` (`imul $0x1c` stride, `+0x14` field) built a few
instructions earlier from `mvKeysRight` with an index that was **not** reduced by
`NLeft`.

**Cause.** `LocalMapping.cc:1048` reads `pKF->mvKeysRight[i].octave` for `i ≥ NLeft`
(upstream typo: the other branches use `i - NLeft`). On a fisheye stereo keyframe
`mvKeysRight` has `NRight ≈ 2000` entries and `i` runs to `N ≈ 4000`, so every
right-camera-first map point reads past the vector; the fault occurs only when the
overrun crosses into unmapped memory, hence the morning/afternoon lottery. The same
audit found four more left-size-only accesses reachable with right indices:
`mvDepth[i]` in the same loop, `mvuRight[idx1/idx2]` in `CreateNewMapPoints`,
`mvuRight[idx]` in `ORBmatcher::Fuse` with a right-grid index, and `get<0> == -1`
observations indexed in the two merge-BA loops of `Optimizer.cc`. All five are
guarded (patch 12, old lines commented); nothing else in the library indexes those
vectors with a possibly-right index (`Tracking.cc` loops are already bounded by
`Nleft`, `MapPoint.cc`/`SearchForTriangulation` short-circuit on `mpCamera2`).

**Together with patch 11** this means stock ORB-SLAM3 has at least six out-of-bounds
reads on the KannalaBrandt8 stereo path with a full-width lapping area; none shows
in the pinhole simulation. Verification: the user re-runs `realv6` at rate 1.0
several times (the crash rate was 5 of 5 that afternoon).

### 7.13 The node and the library disagreed on class layout (ODR violation)  *(2026-11-02)*

**What.** `thirdparty/ORB_SLAM3/CMakeLists.txt` adds `-march=native` to its Release
flags; `orbslam3_ros2` did not. On this CPU that flag raises Eigen's maximum alignment
from 16 to 64 bytes, which changes the layout of library classes:

| compiled as | `EIGEN_MAX_ALIGN_BYTES` | `sizeof(ORB_SLAM3::Frame)` |
| :-- | --: | --: |
| the node (`-O3 -DNDEBUG`) | 16 | 149360 |
| the library (`-O3 -DNDEBUG -march=native`) | 64 | 149376 |

`sizeof(System)` is the same in both (512), so `make_unique<System>` was safe, but
`SlamWrapper` reads `tracker->mCurrentFrame` directly (`slam_wrapper.cpp:296, :338`),
computing the offset of a `Frame` inside `Tracking` with the wrong layout. That is an
ODR violation and undefined behaviour: the node read the velocity from the wrong
address of a 149 kB object.

**Found by** an AddressSanitizer build reporting a `new-delete-type-mismatch` on
`g2o::VertexSE3Expmap` (272 vs 288 bytes) — initially an artefact of the sanitiser
build's own flags, but the same question asked of the production pair gave the table
above.

**Fixed** by adding `add_compile_options(-march=native)` to
`orbslam3_ros2/CMakeLists.txt` with a comment tying the two files together; verified by
recompiling the size probe with the node's real flags (`Frame` 149376 on both sides).
If the build moves to another machine, the Jetson included, remove the flag in BOTH
files together.

**What it affects.** Poses come back as the return value of `System::TrackStereo`, so
trajectories, ATE and RPE are not touched by this. The **velocity columns of
`vio.csv`** were read through the bad offset and are suspect in every ORB-SLAM3 run
recorded before 2026-11-02; `script/vio_metrics.py` and
`script/analyse_relogged_runs.py` consume those columns. It is not established that
this caused any crash or the realv6 accuracy change (§7.12 and the open question
below): a wrong offset inside a 149 kB object yields garbage rather than a fault.

**Still open.** On `realv6`, two pre-reboot runs scored \SI{0.41}{\metre} while nine
later runs, across every combination of culling threshold, loop closing, GPU and CPU
extractor and replay rate, scored \SIrange{1.0}{4.6}{\metre} with a speed runaway at
100--130 s. Bag, IMU delivery, initialisation and throughput were matched; the cause is
not yet identified. Next: an AddressSanitizer run with matching layout, then repeat
runs to measure the run-to-run spread.

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

**Loop closing on the dynamic sets is an unresolved trade-off.** `loopClosing: 0` is
unambiguously right for `allsensor_003` (§10), but it also disables map merging, and the
dynamic no-floor-texture sets are exactly where Atlas spawns a second map (5 of 36 runs).
With closing off that second map can never rejoin the first; with it on, one false positive
can do what it did to `allsensor_003`. The 63-of-64 rejection rate says the gravity check
catches most candidates, not all. Decide per condition, and grep `maps` before trusting a run.

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

**`loopClosing` is the only loop-closing knob in the YAML, and it is not just loop closing.**
`System.cc` reads it off the raw `FileStorage` as an int (`0`/`1`, not `false`/`true`); absent
means on. Every threshold — 3 BoW candidates, 3 consecutive confirmations, the 30/50/100 match
gates, the 0.008 rad roll/pitch gravity check — is hard-coded in `LoopClosing.cc`. The flag
also disables **map merging** and, until patch (7), left the KeyFrameDatabase empty.

**Measured on `dataset_allsensor_003` (2026-09-18), loop closing ON vs OFF, same bag:**

| | ON | OFF |
| :-- | :-- | :-- |
| ATE rmse (SE(3), scale fixed, vs `/ground_truth/odometry`) | **6.481 m** | **0.515 m** |
| jumps > 1 m in `vio.csv` | 3, largest 29.99 m | 0 |
| path length | 190.0 m | 142.7 m |
| closures | 1 (18.69 m correction) | 0 |

527 place-recognition checks → 64 candidates → 63 rejected as `BAD LOOP!!!` by the gravity
check → **one accepted, and it was false**. Place recognition also costs ~51 ms per keyframe
even when it finds nothing. What *looks* like "loop detection firing constantly" in the console
is the 63 rejections, not closures.

**Atlas facts that the YAML does not tell you.**

* Atlas is unconditional (`mpAtlas = new Atlas(0)`); every run uses it. It creates a **second
  map with its own origin** whenever tracking is lost — no reset is logged, so this is
  invisible in `run_summary.csv`. 5 of 36 sim runs reached `maps = 2`, all dynamic
  no-floor-texture. `dynamic_nofloortexture_01_000/logging_20260916_01` glues two frames into
  one `vio.csv` (131 jumps, one of 452 m; 93 of 413 keyframes kept, no merge): any ATE from it
  is about frame mismatch, not drift. **Screen `performance.csv` column `maps` before computing
  ATE.**
* `System.SaveAtlasToFile` does **not** need loop closing: BoW vectors come from
  `LocalMapping::ProcessNewKeyFrame`, are serialised with the keyframe, and `Map::PostLoad`
  rebuilds the database on load.
* `System.LoadAtlasFromFile` **does**. After loading, `System` calls `mpAtlas->CreateNewMap()`,
  so the session starts in a fresh map; `Tracking::Relocalization` searches only
  `GetCurrentMap()`, and merging is the sole bridge to the loaded map — and merging is behind
  `mbActiveLC`. `LoadAtlasFromFile` + `loopClosing: 0` loads 64 MB and then ignores it.
* The saved map is whatever the run left: with closing off it carries the run's drift,
  uncorrected.

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
| Pangolin viewer (`use_viewer:=true`) | **works** since 2026-10-15 (patches 8+9; needs the NVIDIA module to match the userspace driver, else `__GLX_VENDOR_LIBRARY_NAME=mesa`) — but it is software rendering under Mesa and competes with SLAM for cores: look with it, record without it |
| Atlas save / load | **save verified** (patch 6); load runs (patch 7) but is only useful with `loopClosing: 1` — see §10 |
| Loop closing on `allsensor_003` | **OFF is 92 % better** (6.481 → 0.515 m ATE); dynamic sets undecided — see §9 |
| Real rig `dataset_realv3_000` | **tracks** with the image topics swapped (§7.9): 536/536 frames stereo, 458/462 stereo-inertial. Camera order must be checked per bag. |
| GPU front end (patch 10) | **validated** 2026-10-28 on `realv5`: parity in inliers/map points, 0 losses, tracker 110 → 82 ms; on by default in the real-rig configs, off in sim — see §3.5 |
| GPU stereo matcher (patch 10b) | **validated** 2026-10-29: bit-identical to `cv::BFMatcher`; ABBA on `realv5` rate 1.0: tracker 64 → 45 ms, frames processed +35 %, 0 losses; follows `ORBextractor.useGPU` — see §3.5.1 |
| Segfault after tracking loss on the real rig (`-11`) | **root-caused, fixed and verified** 2026-10-29 (patch 11, §7.10): relocalisation matcher read past `mvKeysUn` on fisheye stereo; forced relocalisations under gdb survive |
| Stereo-inertial on the real rig | **tracks continuously** on `realv5` since 2026-10-29 (§7.11): accelerometer scale re-measured (0.9923), IMU subscription in its own callback group, 64 MB socket buffers; 2 init-time resets remain (zero lever arm) |
| Mapping-thread segfault on the real rig (`-11` while tracking OK) | **root-caused and patched** 2026-11-02 (patch 12, §7.12): five right-index out-of-bounds reads in LocalMapping/Fuse/merge-BA; verification runs pending |
| Node/library class-layout mismatch | **fixed** 2026-11-02 (§7.13): `-march=native` added to `orbslam3_ros2`; velocity columns of earlier runs are suspect, trajectories are not |
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


### Session 3: 2026-09-18 → 2026-10-15 — loop closing, Atlas, and four crashes at the edges

1. "Loop closing fires too often" turned out to be 63 rejections and one false acceptance;
   the OFF run the user had already made gave the 92 % ATE improvement in §10. No code change,
   just `loopClosing: 0` in `wil_sim/stereo_imu.yaml` and a measurement.
2. Atlas questions led to `SaveAtlasToFile`, which segfaulted (§7.6). I blamed the viewer
   first — wrong, it crashed with the viewer off too. gdb found the iterator invalidation.
3. `LoadAtlasFromFile` then crashed 0.2 s into tracking (§7.7). Three variables were stacked
   (loaded atlas, viewer, YOLO filter); the backtrace said `LoopClosing::Run()`, i.e. none of
   them directly — the uninitialised pointer that `loopClosing: 0` exposes.
4. With that fixed, the viewer-on run still died at exit, twice, one layer at a time (§7.8).
   Each layer was confirmed by backtrace before patching; the A/B against viewer-off was what
   made the second one unambiguous.
5. Along the way: the build script's 1500 MB gate blocked a 50 MB incremental build (§3.4),
   and the NVIDIA driver was upgraded without a reboot, which is why Pangolin could not get a
   GL context in the first place (`glxinfo` failed identically; `nvidia-smi` reported a
   driver/library mismatch).

6. 2026-10-20, `dataset_realv3_000`: "no features" was really "no stereo matches". Features were
   fine; the camera order had flipped after the 10-19 remount. The first swap test proved nothing
   because the launch ignored the override — a `subscribed:` line caught that. The node-free
   triangulation count (1 % vs 99 %) then predicted the in-node A/B exactly (§7.9).

**The pattern held from sessions 1 and 2:** every surprise was at an interface — a YAML flag
that controls more than its name, a thread nobody stops, a counter that measures a spin loop —
and every wrong guess was replaced by reading the source or a backtrace, never by a better guess.

## 14. Evaluation logging status (2026-09-16/17)

The wrapper now creates a supplied `output_path` recursively. In addition to the
online `vio.csv`, it records `tracking_frontend.csv`, `local_mapping.csv`,
`loop_closing.csv`, `performance.csv`, `run_metadata.csv` and `run_summary.csv`.
On graceful shutdown the ORB-SLAM3 fork also exports
`final_frame_trajectory.txt` and `final_keyframe_trajectory.txt`. Together these
cover tracking, local mapping, place recognition, loop closure/map merge, global
BA, initialization, reset/loss events, queue pressure and process resource usage.
See `MEMORY_CLAUDE/README.md` §11 for the file-by-file interpretation.

Metadata caution: `replay_rate` and `world_path` are provenance labels, not runtime
controls. Dataset identity comes from the containing folder name. For the relogged
experiment only `dataset_dynamic_nofloortexture_01_000` and
`dataset_static_nofloortexture_001` are local; all other no-floor-texture bags are
on `/media/ambushee/32E4AAB1E4AA772F/dataset/`. The 66 raw-run metadata files were
audited on 2026-09-17 and 15 incorrect paths were repaired without modifying any
measurements.
