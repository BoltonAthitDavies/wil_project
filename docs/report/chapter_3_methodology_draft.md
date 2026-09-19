# Chapter 3: Methodology

> **SUPERSEDED as of 2026-09-18.** This Markdown draft has been replaced by
> `docs/report/minor_report/chapters/chapter_3_methodology.tex`, which is the
> maintained version and carries a later cutoff plus material this file never had
> (the Ackermann platform figure, the dataset registry, the non-visual baseline
> estimators, and relative pose error). It is kept only as a record of the
> 15 September state. Do not edit it and do not cite it; its implementation
> statuses are known stale — RPE in particular is listed here as planned and has
> since been implemented.

**Document type:** Minor-report draft for advisor progress review  
**Reporting cutoff:** 15 September 2026  
**Evidence basis:** Current repository source, configuration, launch files,
generated artifacts, and dated engineering records  
**Status:** Draft; implementation claims have been checked against the repository,
but literature citations and final thesis formatting remain to be completed

## 3.1 Progress scope and implemented system overview

This project investigates vision-based localization for an autonomous mobile robot
in a dynamic warehouse. At the reporting cutoff, the implemented work compares two
stereo-inertial front ends—ORB-SLAM3 and VINS-Fusion—using common simulated or
recorded sensor streams. A custom YOLO detector can be enabled to prevent image
features associated with selected warehouse-object classes from entering the pose
estimation process. RTAB-Map is used as a separate back end when persistent map
construction, loop closure, pose-graph correction, or localization against a saved
database is studied.

The implemented data flow is:

```text
left/right images + IMU
          |
          +-------------------------------+
          |                               |
          v                               v
 optional YOLO detector             unfiltered baseline
 (left image only)                        |
          |                               |
          +-------- detections -----------+
                          |
             +------------+------------+
             |                         |
             v                         v
      ORB-SLAM3 front end        VINS-Fusion front end
      ORB feature rejection      KLT-track culling and
                                 new-feature suppression
             |                         |
             +------ common `vio.csv`-+
                          |
                 trajectory evaluation

optional mapping path:
front-end odometry + stereo images -> RTAB-Map -> database, graph, map->odom
```

The current work therefore contains two related but separate evaluation paths.
The first measures front-end trajectory behavior with and without semantic
feature filtering. The second studies map reconstruction and external global
correction through RTAB-Map. RTAB-Map's `map -> odom` transform does not modify
the internal feature map, state history, or previously recorded trajectory of
ORB-SLAM3 or VINS-Fusion. A unified architecture for relocalization, global
optimization, and multi-session map merging is **planned**, not implemented.
Path planning is outside the present research scope.

Implementation status at the cutoff is summarized below.

| Component | Status | Scope of evidence |
|---|---|---|
| Warehouse simulation and controlled world variants | **Implemented and verified** | Static, dynamic, textured-floor, and featureless-floor world files and launch files are present |
| ORB-SLAM3 ROS 2 stereo-inertial wrapper | **Implemented and verified in simulation** | Live ROS topics, worker-thread tracking, TF, and common trajectory output are implemented |
| ORB-SLAM3 semantic feature filtering | **Implemented and mechanism-verified** | Detector messages reach the tracker and masked ORB candidates are rejected before octree distribution |
| VINS-Fusion stereo-inertial estimator | **Implemented** | Simulation and real-rig configurations, odometry publication, TF, and trajectory output are present |
| VINS-Fusion semantic feature filtering | **Implemented and mechanism-verified** | Existing KLT tracks are culled and new features are suppressed in detected regions |
| Offline YOLO evaluation | **Implemented and partially evaluated** | Ground-truth projection, IoU matching, per-frame CSVs, and validation overlays are present |
| RTAB-Map construction and saved-map localization | **Implemented and partially verified** | Construction and basic read-only localization have passed; arbitrary-start aliasing rejection has not |
| RTAB-Map extend/multi-session map merging | **Implemented at launch-parameter level; unverified experimentally** | Mode exists, but the planned Level 4 experiment has not been completed |
| Live VINS-Fusion and RTAB-Map co-execution | **Blocked on the present computer** | VINS diverges under the combined workload; the exact mechanism is unresolved |
| RPE and complete compute-performance evaluation | **Partially implemented** | Per-frame estimator timing, delivery/state counters, cumulative process CPU time, and resident memory are logged; RPE, external-process monitoring, and the repeated hardware matrix remain planned |

The repository state used for this draft is the superproject commit
`4425a415f624d6a45da1da4f3b2404cbf7914197`. The checked-out submodules are
VINS-Fusion `4a2223c9e54dfc5d2e94eb2ec1dfc8fd3b8045bb`, RTAB-Map ROS
`56064e4388f6f3ad6b994dbff98bcc0a41488102`, and the warehouse environment
`3df54c6e6ec267fad204e208db61c4ab491be410`. The ORB-SLAM3 core is pinned to
`5c2b00f732775f78836751dba13db25463a0aaa1`. The VINS-Fusion worktree also
contains an uncommitted output-directory creation fix at the reporting cutoff;
this must be committed or otherwise recorded before a reproducibility release.

## 3.2 Hardware, software, sensors, calibration, and coordinate frames

### 3.2.1 Computing and software environment

Development and recorded experiments use Ubuntu 22.04.5, ROS 2 Humble, GCC
11.4.0, CMake 3.22.1, and system OpenCV 4.5.4. The recorded development machine
has 16 logical CPU cores, 15 GB RAM, no swap, and an NVIDIA RTX 3070 Laptop GPU.
The detector environment recorded in the engineering log contains PyTorch
2.5.1+cu121, torchvision 0.20.1, Ultralytics 8.4.138, NumPy 1.26.4, and Python
OpenCV 4.11. ORB-SLAM3 is built out of tree with Pangolin 0.9.1. RTAB-Map uses
the apt core library `ros-humble-rtabmap` 0.23.7 with the project checkout of the
ROS wrappers; only the wrapper packages through `rtabmap_slam` are built.

These specifications describe the current laptop, not a completed cross-hardware
experiment. The requested Jetson-versus-laptop study has not yet been instrumented
or executed, so hardware comparison must not be inferred from the present data.

### 3.2.2 Simulated sensor rig

The simulated Ackermann robot provides two ideal pinhole cameras and an IMU. Both
front ends use the same camera and IMU definitions. Each camera produces
1280 × 720 images at a nominal 30 Hz with horizontal field of view
1.5707963268 rad. The analytic pinhole calibration is
`fx = fy = 640`, `cx = 640`, `cy = 360`, with zero distortion. The IMU is
configured at 200 Hz. The left camera is `cam0`; `cam1` is the right camera. The
camera separation is 0.0936 m, independently checked from the positive stereo
disparity in a recorded bag.

The body-to-camera transform is derived from the SDF geometry rather than from a
separate calibration process. Relative to the IMU body, the left camera
translation is `(0.158, 0.0936, 0.2475)` m and the right-camera translation is
`(0.158, 0, 0.2475)` m. The fixed rotation converts the ROS body convention
(forward, left, up) to the optical convention (right, down, forward). The VINS
simulation configuration uses fixed extrinsics (`estimate_extrinsic: 0`) and no
temporal-offset estimation (`estimate_td: 0`). Its configured inertial noise
densities are `acc_n = 0.02 m s^-2/sqrt(Hz)` and
`gyr_n = 0.002 rad s^-1/sqrt(Hz)`, with bias random walks `0.001` and
`0.0001` in the corresponding continuous-time units. The same values are mapped
to the ORB-SLAM3 configuration.

### 3.2.3 Real sensor rig and calibration limits

The real configuration uses a 1920 × 1080 fisheye stereo pair with an
approximately 0.093604 m baseline and a separate RealSense IMU stream published
on `/hwt101ct_yaw_publisher`. ORB-SLAM3 models the cameras with
`KannalaBrandt8`; VINS-Fusion uses the project fisheye calibration files. Only one
monocular intrinsic calibration is available and is reused for the second camera,
which is an approximation. VINS-Fusion enables online camera–IMU extrinsic and
time-offset estimation (`estimate_extrinsic: 1`, `estimate_td: 1`).

The real IMU-to-camera rotation has observational support, but its translation is
currently a zero placeholder. The stationary accelerometer magnitude is recorded
as 9.2642 m/s² rather than 9.81 m/s². VINS-Fusion uses the observed magnitude as
`g_norm`; the ORB-SLAM3 real inertial launch rescales acceleration by
`9.81/9.2642`. These treatments make the configurations internally consistent,
but they do not replace an independent camera–IMU calibration. Consequently,
real stereo trajectories can be inspected for plausibility and tracking
continuity, but metric trajectory accuracy cannot be claimed without both an
independent ground-truth trajectory and improved extrinsic calibration.

### 3.2.4 Coordinate-frame contract

Standalone VINS-Fusion publishes `world -> body`. Standalone ORB-SLAM3 publishes
an estimator-specific world frame. For RTAB-Map integration, both front ends are
instead configured to publish `odom -> vio_body`, and a measured static transform
connects `vio_body -> base_footprint` by `(-0.338, 0, -0.192)` m. RTAB-Map then
publishes the global correction `map -> odom`:

```text
map -> odom -> vio_body -> base_footprint -> base_link -> sensors
                  \-> cam0_optical
                  \-> cam1_optical
```

The two independent paths to `cam0_optical` provide a runtime calibration check:
`tf2_echo cam0_undis cam0_optical` must return zero translation and the identity
quaternion. This check has been measured successfully. It is required because a
previous OpenCV YAML parsing error produced a plausible translation but a
non-rigid camera rotation, invalidating an earlier map-quality campaign.

The standalone simulation ORB launch still labels the estimated IMU-body pose as
`base_footprint`. The composing RTAB-Map launch corrects this by using `vio_body`,
but standalone consumers must not assume that the default frame label represents
the chassis origin. Constant offsets are largely absorbed by rigid trajectory
alignment, but they are material for map placement and robot-footprint tests.

## 3.3 Warehouse environments, datasets, and controlled factors

### 3.3.1 Simulation environment

The principal large warehouse is a 41.2833 m square environment produced by
rescaling the original AWS warehouse to 1,704 m² while retaining a 9.020 m
interior height. Fourteen marked storage bays and five reduced-width walkways are
represented in the floor geometry. Static and dynamic families are stored under
`worlds/small_warehouse_static/` and `worlds/small_warehouse_dynamic/`. The
original tiny warehouse is a distinct environment under
`worlds/small_warehouse/`; results from the tiny and large layouts are not treated
as repeated trials of the same environment.

The controlled world factors currently available are:

- static versus scripted dynamic props;
- textured versus featureless floor;
- canonical, `_00`, `_01`, and `_02` prop/route arrangements;
- a fully stocked static condition; and
- a featureless-floor condition with the bays cleared and objects restricted to
  the wall region.

The featureless floor uses a separate ground model. Painted lines and concrete
texture are removed while mean floor albedo and collision geometry are preserved.
This design changes visible spatial texture without changing physics or the baked
occupancy truth. Dynamic props follow deterministic, analytically checked routes
using a kinematic Gazebo plugin. This maintains collision geometry and repeatable
motion at acceptable real-time factor, but it does not reproduce physical
collision response when the robot overlaps a moving prop. This is a simulation
limitation rather than a property of the SLAM algorithms.

World identity is recorded by the exact `.world` path. Launching with ROS requires
`colcon build --symlink-install`; a plain build can leave an old copied world in
`install/`, causing Gazebo and source-based tools to evaluate different scenes.
When object motion is required by the 2-D viewer, the simulator must also bridge
model poses.

### 3.3.2 Dataset registry

The raw workspace datasets present at the cutoff are:

- `dataset/dataset_dynamic_nofloortexture_01_000`;
- `dataset/dataset_map_tiny`;
- `dataset/dataset_real_000`;
- `dataset/dataset_static_nofloortexture_001`; and
- `dataset/dataset_static_nofloortexture_objonwallonly`.

Additional historical output folders exist under `output/`, but a generated plot
or `vio.csv` does not prove that the corresponding raw bag remains available or
valid. `dataset_real_001` and `dataset_real_002` are known broken datasets and are
excluded. Other named datasets must not be marked broken merely because their
source bag is currently stored outside the repository.

Before a new experiment, the selected bag's absolute path, topics, duration,
message counts, and storage health will be inspected. Each comparison will use the
same recorded bag for all treatments. Estimators will run separately to avoid
changing sensor delivery through resource contention. Environment, trajectory,
calibration, replay rate, image transport, initialization treatment, and software
revision are controlled within a paired comparison. A run is excluded if input
overlap, initialization, processed sample count, or sensor delivery differs
materially from its paired condition.

Recorded simulation bags used by the RTAB-Map work contain `/clock`. They are
played without rosbag's additional `--clock` option because two competing clocks
cause TF time jumps. Whether `--clock` is required must therefore be determined
from each bag's topic inventory rather than from a universal command template.

## 3.4 ORB-SLAM3 pipeline and semantic feature filtering

The project provides a purpose-built ROS 2 Humble wrapper around the pinned
ORB-SLAM3 core. Approximate-time synchronization pairs left and right images with
a 0.02 s slop. Raw and compressed transports are supported so the same executable
can process live simulation images and compressed real-rig bags. Image callbacks
place stereo pairs into a one-frame queue; a separate worker thread calls
`TrackStereo()` so blocking visual tracking does not starve the 200 Hz IMU
subscription. If the tracker falls behind, the older pending image is discarded
and the drop count is reported.

When `filter:=false`, the wrapper creates no detection subscription and sends an
empty mask through the patched interfaces. This is the baseline configuration.
When filtering is enabled, the wrapper searches a bounded detection buffer for the
message with the closest source-image timestamp, accepting it only when its age is
at most 0.15 s. Detected boxes are expanded by 8 pixels, clipped to the image, and
rasterized into an 8-bit mask where 255 means retain and 0 means reject. A safety
check discards a mask if it covers more than 0.8 of the frame.

The mask is passed through `System`, `Tracking`, and `Frame` to the ORB extractor.
Candidate keypoints are tested before octree distribution, allowing the requested
per-level feature count to be redistributed over surviving static regions. The
implemented boundary rule samples a 3 × 3 neighborhood at the appropriate pyramid
scale and removes a candidate when at least five samples are dynamic. Only the
left-camera feature set is masked because ORB-SLAM3 establishes stereo matches
from left keypoints to right candidates.

The current simulation configuration requests 1,000 ORB features, eight pyramid
levels, a 1.2 scale factor, initial FAST threshold 20, and fallback threshold 7.
An older engineering record states 1,500 simulated features, but the current YAML
file is authoritative. No controlled feature-count or FAST-threshold sweep has
been completed.

The wrapper publishes `/orbslam3/odometry`, `/orbslam3/pose`, and TF after tracking
is valid. It directly writes the common 11-column trajectory format described in
Section 3.7 and flushes every row to reduce loss after an interrupted run.

Repository evidence: `orbslam3_ros2/src/orbslam3_node.cpp`,
`orbslam3_ros2/src/slam_wrapper.cpp`, `orbslam3_ros2/src/trajectory_writer.cpp`,
`orbslam3_ros2/launch/wil_sim_stereo_imu.launch.py`, and
`orbslam3_ros2/config/wil_sim/stereo_imu.yaml`.

## 3.5 VINS-Fusion pipeline, persistent-feature filtering, and RTAB-Map back end

### 3.5.1 VINS-Fusion front end

VINS-Fusion uses stereo image features and inertial measurements in a sliding-window
nonlinear estimator. The ROS wrapper synchronizes the two raw image topics and
maintains separate callback groups for images, IMU, feature messages, and detector
messages. The current simulation configuration processes every image
(`image_skip: 1`), uses two cameras and IMU, fixed simulated extrinsics, a target
feature-publication frequency of 25 Hz, and 10-pixel keyframe parallax. Image queues
were deepened to 100 and the IMU queue to 2,000 to reduce losses during
initialization and temporary processing backlog.

The VINS filter cannot use only the ORB-SLAM3 strategy because VINS propagates
existing features with pyramidal KLT optical flow and detects new features only to
replenish the track set. A feature attached to an object before detection could
otherwise remain active after the object is masked. The implementation therefore
initializes `FeatureTracker::setMask()` from the YOLO mask. Existing tracked points
inside a zero-valued region fail the retention test, while the same mask is passed
to `goodFeaturesToTrack()` to block new points. The number of persistent tracks
dropped in dynamic regions is accumulated as a mechanism-level diagnostic.

VINS-Fusion writes the optimized state after each completed estimator update to
`vio.csv` and broadcasts `world -> body` using configurable frame names. The ROS
`output_path` parameter overrides the YAML path, recursively creates missing
directories, and fails with an explicit exception if `vio.csv` cannot be opened.
This output-path behavior is present as an uncommitted worktree change at the
reporting cutoff.

Repository evidence: `vins_fusion_ros2/src/vins_estimator.cpp`,
`vins_fusion_ros2/vins/src/featureTracker/feature_tracker.cpp`,
`vins_fusion_ros2/vins/src/estimator/estimator.cpp`, and
`vins_fusion_ros2/config/{wil_sim,wil}/stereo_imu.yaml`.

### 3.5.2 RTAB-Map mapping and localization back end

RTAB-Map receives front-end odometry through TF and receives stereo imagery through
`rtab_camera_shim.py`. The shim reconstructs `CameraInfo`, decompresses the stereo
pair, and publishes validated static camera transforms from the same simulation
calibration used by both front ends. TF was selected over an odometry topic because
including high-rate odometry in the multi-topic approximate synchronizer starved
the lower-rate image inputs. The cost is that odometry covariance is not supplied
to graph-edge weighting.

The back end uses stereo SGBM (`Stereo/DenseStrategy: 1`), height-based ground
separation, ray tracing, spatial noise filtering, and odometry-derived gravity
links. The currently measured default depth range is 10 m. Because the featureless
floor provides essentially no stereo ground points, ray tracing supplies free-space
evidence between the sensor and observed obstacles. These settings affect the
assembled occupancy representation; visual-word loop closure is a separate part
of RTAB-Map.

Three modes are implemented in one launch file:

- `construct` starts a new incremental database;
- `extend` opens an existing database with all prior nodes in working memory and
  permits new nodes; and
- `localize` disables incremental memory, loads all nodes, serves the saved map,
  and can enable `Mem/LocalizationReadOnly`.

Localization continues to extract features from current stereo images and match
them against stored RTAB-Map visual signatures. Read-only mode prevents database
modification; it does not disable perception. The output is an external
`map -> odom` correction and `/localization_pose`. It does not feed RTAB-Map
features or optimized poses back into ORB-SLAM3's Atlas or VINS-Fusion's sliding
window.

The composed localization launch raises `RGBD/OptimizeMaxError` from the upstream
default 3.0 to 10.0 because the default rejected all loop closures proposed with
the observed VINS drift. This change is effective but not yet proven safe against
perceptual aliasing in repeated warehouse bays. Occupancy-grid creation is disabled
in localization mode because the stored map is reused and dense stereo grid work
is otherwise discarded. Even with that reduction, live VINS-Fusion and RTAB-Map
co-execution remains blocked on the present machine. Recorded `vio.csv` can be
replayed as `odom -> vio_body` to isolate the localization back end deterministically,
but that method does not demonstrate real-time co-execution.

Repository evidence: `rtabmap_ros/rtab_sim.launch.py`,
`rtabmap_ros/rtab_camera_shim.py`,
`aws-robomaker-small-warehouse-world/launch/localization_rtabmap.launch.py`, and
`script/gt_tf_publisher.py`.

## 3.6 YOLO model, mask construction, and synchronization

The semantic front end is a standalone Python ROS 2 node using Ultralytics. The
project checkpoint is `weight/best.pt`, a custom three-class detector for `bin`,
`box`, and `bucket`, fine-tuned from `yolo26m.pt` using 640-pixel inputs. It is a
bounding-box detector, not a segmentation model. The stock `yolo26n.pt` and
`yolo26m.pt` files are not project-trained models and are not used as evidence of
custom training.

At startup the node resolves requested class names against the checkpoint metadata,
warms the model with three dummy images, and selects CUDA when available. Current
defaults are confidence 0.35, NMS IoU 0.5, and image size 640. Only the left image
is processed. Each `vision_msgs/Detection2DArray` retains the source image header,
so the downstream nearest-timestamp lookup measures sensor-time correspondence
rather than inference-completion time. Empty arrays are published to distinguish a
valid frame with no detections from a failed detector.

The detector is intentionally outside the stereo synchronizer. A three-input
image/image/detection synchronizer would discard stereo pairs whenever inference
lagged. The current design lets both estimators continue unfiltered when no recent
detection exists. For auditability, filtered runs write `yolo_mask.csv` with
timestamp, box count, coverage, match status, detection age, and rejection status.

Offline detector evaluation uses `script/yolo_eval.py`. Gazebo model poses and
collision-mesh vertices are transformed into the left-camera frame to form
projected ground-truth boxes. Object speed above 0.05 m/s defines the evaluation's
moving subset. Predictions are greedily matched by class at IoU 0.5. Heavily
occluded ground-truth boxes and predictions covered by known but unlabeled
box-like scenery are ignored according to stated thresholds. Outputs include
per-frame `yolo_eval.csv`, per-detection `yolo_detections.csv`, and visual overlays.
The overlay is inspected before accepting aggregate metrics because all-zero model
pose stamps, centimeter mesh units, or a malformed camera transform can each yield
plausible but invalid numerical results.

The present semantic rule assumes that all detections from the selected classes are
dynamic. No optical-flow, stereo-depth, or world-motion gate is implemented, and
axis-aligned boxes remove background pixels around each object. These are explicit
method limitations rather than hidden properties of the results.

## 3.7 Data collection, run validity, and stored artifacts

The sensor source, estimator, optional detector, and viewer are started as separate
processes. Launch files do not automatically play bags. Baseline and filtered
treatments are run separately on the same recording, with a replay rate low enough
for the estimator to consume a comparable input stream. Competing estimators are
not run simultaneously for a scientific A/B comparison.

The common front-end trajectory file has no header and uses:

```text
t_ns, px, py, pz, qw, qx, qy, qz, vx, vy, vz
```

Timestamps are integer nanoseconds from the source message header; quaternions use
`w,x,y,z` order. ORB-SLAM3 writes this format directly rather than exporting its
native EuRoC or TUM representation. VINS-Fusion writes the same estimator state
after a nonlinear update. Simulation ground truth is extracted into the same shape
by `script/extract_gt.py`. The filter-use record `yolo_mask.csv` is separate from
the offline detector score, because one reports what the tracker received and the
other reports detector accuracy against projected truth.

Principal stored artifacts are:

| Artifact | Purpose |
|---|---|
| `<run>/vio.csv` | Per-estimator trajectory and velocity |
| `<run>/ground_truth.csv` | Simulation reference trajectory when available |
| `<run>/yolo_mask.csv` | Actual mask matching, coverage, and rejection record |
| `<run>/run_metadata.csv` | Resolved estimator configuration and supplied run provenance |
| `<run>/performance.csv` | Per-frame timing, state, delivery, CPU, and memory measurements |
| `<run>/run_summary.csv` | Graceful-shutdown validity and robustness totals |
| `<run>/events.csv` | VINS initialization, reset-cause, and shutdown events |
| `output/yolo_eval/<experiment>/yolo_eval.csv` | Per-frame detector evaluation |
| `output/yolo_eval/<experiment>/yolo_detections.csv` | Per-detection matching record |
| `output/yolo_eval/<experiment>/overlay/` | Projection and matching validation images |
| `map/*.db` or `<run>/rtabmap.db` | Persistent RTAB-Map graph and visual database |
| `output/compare/<domain>/<experiment>/` | Derived figures and machine-readable summaries |

A valid trajectory comparison requires at least ten overlapping samples in the
current analysis code. Beyond that software minimum, the paired runs must cover
the same input interval, initialize successfully, retain comparable pose/frame
counts, and show no unaccounted sensor starvation. Steps implying more than
10 m/s are recorded as tracking jumps; an indoor estimate with error beyond 50 m
is treated as divergence rather than ordinary inaccuracy. For real data without
ground truth, path length and mean-speed plausibility are diagnostic checks, not
accuracy measurements.

The estimator now records resolved local configuration and optional experiment,
dataset, world, replay-rate, command, and note fields in `run_metadata.csv`.
However, it does not automatically determine bag health, repository revisions,
full dependency versions, external-process state, or the analyst's exclusion
decision. Empty optional metadata fields explicitly remain incomplete provenance.
Historical folders created before this instrumentation are preliminary unless
their provenance is recoverable from logs and engineering records.

## 3.8 Evaluation metrics and analysis procedures

### 3.8.1 Trajectory evaluation

`script/vio_metrics.py` associates each estimate over its common time interval
with the reference and interpolates the reference at estimator timestamps. A rigid
Umeyama SE(3) fit maps each estimated position sequence to the reference with scale
fixed at one. Fixed scale is appropriate for stereo-inertial estimators and
preserves scale error. Translational ATE RMSE, maximum and final error, rotational
geodesic error, path length, overlap duration, pose count, jump count, and the
longest jump-free segment are derived from the aligned data. An optional startup
trim is applied relative to each estimator's own first overlapping pose so unequal
initialization latency is not confused with steady-state behavior.

The older `plot_vio_vs_gt.py` also provides a first-pose-aligned diagnostic view,
but headline cross-estimator comparisons use the common SE(3) method. Real data
without independent ground truth is not assigned ATE. Relative pose error at a
stated time or distance interval is required by the intended evaluation design but
is **not currently implemented** in the shared metrics module; it must be added
before Chapter 4 claims complete trajectory evaluation.

### 3.8.2 Detection evaluation

Detector precision, recall, TP, FP, FN, moving/static recall, IoU, and image-area
coverage are computed from `yolo_eval.py` at explicitly recorded confidence,
matching-IoU, occlusion, minimum-size, and motion thresholds. Mask-use statistics
are joined to trajectories by `timestamp_ns`. The detector's semantic class label
is never treated as ground-truth motion.

### 3.8.3 Map and localization evaluation

RTAB-Map database grids are placed in the Gazebo world with one fixed rigid
world-to-map transform measured by `script/fit_map_pose.py`. This transform is
constant for a database; the live `map -> odom` transform is a drift correction
and must not be reused as world-to-map placement.

`script/validate_map.py` rasterizes world collision geometry and compares occupied
map cells at a stated spatial tolerance. It reports occupied-cell precision,
whole-world and observed-region surface recall, false obstacles under the true
0.83 × 0.50 m robot footprint, unmapped footprint poses, and clearance statistics.
Graph evaluation records nodes and link types, including neighbor, global closure,
local-space, and gravity links. Localization experiments additionally require time
to first fix, accepted/rejected matches, database checksum before and after
read-only operation, pose continuity, and bounds consistent with the known
warehouse. A finite covariance or a localization status message is insufficient
evidence of a correct pose.

### 3.8.4 Computational performance

The two estimator processes now write per-frame `performance.csv` files. These
record estimator processing time, delivery and backlog counters, tracking or
solver state, initialization and reset evidence, pose and keyframe accounting
where exposed, cumulative process CPU time, monotonic wall time, and resident
memory. CPU utilization and throughput can therefore be derived over explicit
wall-time intervals. The instrumentation is process-local: it does not measure
Gazebo, rosbag, RViz, RTAB-Map, or the separate YOLO process. Synchronized
whole-system monitoring, full per-module timing, ORB keyframe accounting, and the
repeated Jetson/laptop matrix remain **planned**. The desired aggregate figures
must be generated only after controlled runs populate these logs.

## 3.9 Reproducibility controls and planned-versus-implemented deviations

The following controls apply to future experiments and to the acceptance of
historical results:

1. Record the absolute bag and exact `.world` file, treatment, configuration,
   commands, replay clock/rate, hardware, dependency versions, and all relevant
   repository commits.
2. Use a unique output directory and retain failed runs with a failure label.
3. Run paired treatments independently on the same sensor data and compare only
   the common valid interval.
4. Record input and output counts, initialization, dropped frames, resets, tracking
   loss, warnings, and exclusion reasons.
5. Validate camera order, image transport, timestamps, metric scale, and every
   rigid transform before computing results.
6. Inspect representative detector and map overlays before trusting aggregate
   scores.
7. Repeat nondeterministic RTAB-Map processing and report the number of valid runs;
   do not interpret differences within observed run-to-run variation as algorithmic
   effects.
8. Keep static/dynamic, textured/featureless, simulation/real, filtered/unfiltered,
   and front-end/back-end factors separate unless an interaction is the declared
   independent variable.

Material deviations from the originally proposed methodology are:

- The implemented semantic model is a three-class YOLO bounding-box detector,
  whereas the reference RY-SLAM method uses segmentation. The current system
  rasterizes boxes and has no motion-consistency stage.
- ORB-SLAM3 rejects candidate ORB features before octree distribution; VINS-Fusion
  required a different intervention that also removes persistent KLT tracks.
- ORB-SLAM3 and VINS-Fusion share a common output representation, but they remain
  independent estimators rather than components of one fused estimator.
- RTAB-Map is a separate ROS process using external TF odometry. Its global
  correction is not injected into the internal VINS-Fusion or ORB-SLAM3 state.
- The real stereo-inertial ORB configuration remains calibration-limited, and real
  trajectory accuracy lacks independent ground truth.
- Complete RPE, resource, throughput, uncertainty, and run-manifest infrastructure
  is not yet implemented.
- Relocalization is only partially verified; robust arbitrary-start localization,
  perceptual-aliasing rejection, extend-mode validation, global optimization within
  the front end, and multi-session map merging remain future work.

These deviations define the actual experimental system at the reporting cutoff.
Chapter 4 should report only measurements that satisfy the validity rules above,
while retaining failed, excluded, and retracted runs in a clearly labeled
diagnostic record.

## Repository evidence used for this draft

- `README.md`
- `orbslam3_ros2/MEMORY_CLAUDE/README.md`
- `orbslam3_ros2/MEMORY_CLAUDE/integration.md`
- `vins_fusion_ros2/MEMORY_CLAUDE/README.md`
- `rtabmap_ros/MEMORY_CLAUDE/README.md`
- `aws-robomaker-small-warehouse-world/MEMORY_CLAUDE/README.md`
- `output/docs/report/project_limitations_and_map_benefits.md`
- Source, launch, and configuration files named within Sections 3.2–3.8
- `script/vio_metrics.py`, `script/extract_gt.py`, `script/yolo_eval.py`,
  `script/fit_map_pose.py`, and `script/validate_map.py`
