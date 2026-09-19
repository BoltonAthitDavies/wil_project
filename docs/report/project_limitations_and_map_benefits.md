# Project Limitations and Benefits of the Map Beyond Path Planning

**Document type:** Advisor progress note  
**Status:** Preliminary; not a final-project conclusion  
**Reporting date:** 2026-09-18 (supersedes 2026-09-15)

## Purpose and scope

This note summarizes the present limitations of the WiL semantic visual-SLAM
project and explains why a persistent map remains useful even when path planning
is excluded. Statements are separated into current evidence and possible future
benefits. A planned capability must not be reported as an implemented result.

The current project compares ORB-SLAM3 and VINS-Fusion on stereo-inertial data,
adds YOLO-based dynamic-feature filtering, and uses RTAB-Map for map reconstruction,
loop-closure experiments, and localization against a stored database. At present,
SLAM-performance evaluation and map reconstruction are separate processes. Their
deeper integration is future work.

## 1. Current project limitations

### 1.1 Dynamic-object perception

- Filtering is semantic rather than motion-aware. A detected `bin`, `box`, or
  `bucket` is masked even when it is stationary.
- The custom YOLO model produces axis-aligned bounding boxes rather than
  pixel-accurate segmentation masks.
- Measured moving-object recall is approximately 0.599 on
  `dataset_dynamic_nofloortexture_00_001`; therefore, about 40% of the moving
  objects represented by the ground-truth evaluation are not detected.
- ORB-SLAM3 and VINS-Fusion do not apply masks identically. ORB-SLAM3 rejects new
  feature extraction in masked regions, while VINS-Fusion must additionally cull
  persistent KLT tracks that enter those regions.
- The semantic classes used by the detector are not proof that an object is
  moving. The current system has no 3D motion gate.

### 1.2 Experimental evidence and comparison validity

- Some earlier A/B measurements are invalid because treatment runs processed
  substantially different numbers of poses over the same overlap interval or
  shared limited compute resources.
- Valid estimator comparisons require the same recorded input, common
  ground-truth overlap, equivalent initialization, and independently executed
  treatments.
- Live RTAB-Map runs can vary substantially with machine load. Differences in
  processed nodes or closures are not automatically algorithmic effects.
- RTAB-Map offline reprocessing is also nondeterministic because loop detection is
  re-executed. Small changes require repeated runs before interpretation.
- Real-world ATE or RPE cannot be claimed without an independent, synchronized
  ground-truth trajectory. This is a data limitation, not a tooling one: relative
  pose error was implemented on 2026-09-18 and applied to the whole simulation
  campaign without re-running any estimator.
- An earlier reading of the dynamic runs, that they remain "locally coherent"
  between tracking jumps, has been **withdrawn**. Measured directly, jump-free
  relative pose error at a one-second interval is 1.8-3.8 m, against 0.06-0.10 m
  in static scenes. The estimate misstates its own motion by more than the
  distance travelled, so the corruption is continuous as well as discontinuous.
  Clean-segment ATE looked reassuring because aligning a short segment absorbs
  much of that error.
- In static simulation a wheel+IMU EKF currently attains lower aligned ATE than
  either visual pipeline on four of five multi-sensor recordings. The visual
  system does not yet clear the dead-reckoning floor under these conditions.
- `dataset_real_001` and `dataset_real_002` are broken and cannot support valid
  experiments.

### 1.3 Localization and back-end integration

- RTAB-Map currently supplies a global `map -> odom` correction outside the visual
  front end. It does not update VINS-Fusion's or ORB-SLAM3's internal feature map,
  state history, or previously written trajectory.
- RTAB-Map extracts its own features from the live stereo images and matches them
  against visual signatures stored in its database. It does not reuse
  ORB-SLAM3's internal keypoints or map points.
- Read-only localization has been demonstrated at a basic level, but robust
  arbitrary-start relocalization has not been established.
- Repetitive warehouse bays create perceptual aliasing: visually similar but
  physically different places may produce an incorrect match.
- A finite localization covariance or a reported match is not proof that the pose
  is physically correct. Pose bounds, map consistency, or independent ground
  truth are required.
- Relocalization, global optimization, and multi-session map merging are future
  interests rather than completed project contributions.

### 1.4 Stereo reconstruction and occupancy-map quality

- The simulated stereo rig has a 93.6 mm baseline with `fx = 640`. Depth
  sensitivity increases quadratically with range; a one-pixel disparity error is
  about 0.42 m at 5 m and 1.67 m at 10 m.
- Low-texture walls and a featureless floor provide weak or absent stereo
  correspondence. Parameters cannot recover depth that was never observed.
- Ray tracing is needed to infer free space because the featureless floor yields
  essentially no reliable stereo ground points.
- Correcting the camera-transform parsing bug greatly improved map quality, but
  coverage remains incomplete and depends on the robot's route and field of view.
- Map placement in the Gazebo-world viewer requires a validated fixed transform.
  An incorrectly seated map can make correct geometry appear inaccurate.
- The present occupancy map should not be described as a fully validated map for
  autonomous navigation through narrow aisles.
- The drivability half of the map evaluation was scored against a route from a
  separate run, `dataset_static_tiny_repeat`, which is no longer present in the
  workspace or on the removable drive. The only surviving recording of this
  warehouse is the one that built the map, and scoring against it returns
  0.0% false obstacles by construction. Drivability is therefore a dated
  measurement that cannot currently be re-derived; the geometry half remains
  fully reproducible.

### 1.5 Real-rig compatibility

- The real camera configuration uses a wide-angle Kannala-Brandt fisheye model,
  while RTAB-Map's stereo route assumes rectified pinhole images.
- Rectification to a narrower virtual pinhole would discard peripheral image
  content that may be useful for place recognition.
- The real configuration estimates camera-IMU extrinsics online. RTAB-Map needs a
  fixed, validated transform before using a static TF.
- Simulation does not reproduce every real-camera effect, including calibration
  error, motion blur, exposure behavior, lighting variation, and image noise.

### 1.6 Computational and simulation limitations

- VINS-Fusion has diverged when co-run with RTAB-Map and the image republishing
  pipeline on the current machine. Lower bag playback speed did not resolve the
  problem, and the mechanism is not yet confirmed.
- Dense stereo map construction is computationally expensive and can compete with
  the front-end estimator for CPU resources.
- Deterministic replay of a previously recorded `vio.csv` as TF is useful for
  isolating localization behavior, but it does not prove live co-execution.
- Gazebo movers are driven kinematically for repeatability and real-time factor.
  They retain collision geometry but do not produce realistic dynamic collision
  responses against the robot.

## 2. Benefits of the map beyond path planning

### 2.1 Persistent spatial reference

A stored map establishes a coordinate frame that can persist across restarts and
experimental sessions. Robot poses, observations, detections, and evaluation
outputs can be compared in one frame instead of a new local odometry frame for
every run.

**Current status:** Available, provided that map-to-world placement and TF frame
conventions are validated.

### 2.2 Visual relocalization

An RTAB-Map database stores visual signatures and graph nodes in addition to an
occupancy grid. Current stereo observations can be matched against those stored
signatures to estimate the robot's location after startup or tracking loss.
Read-only mode prevents database modification; it does not disable live feature
extraction or matching.

**Current status:** Partially verified. Basic localization works, but arbitrary
starts and warehouse-aliasing rejection require further testing.

### 2.3 Global drift correction

A recognized revisit can add a loop-closure constraint and optimize the pose
graph. This can correct accumulated global drift and publish a corrected
`map -> odom` transform while leaving local odometry continuous.

**Current status:** Loop closures and graph corrections have been observed, but
their accuracy must be evaluated on controlled runs. External correction does not
currently modify the internal state of the visual front end.

### 2.4 Multi-session memory

A persistent database can retain observations from earlier sessions, support
localization against them, and provide the foundation for extending or merging
maps instead of reconstructing the environment from nothing after every restart.

**Current status:** Database persistence and construct/localize modes exist.
Robust extend-mode operation and multi-session merging are not yet demonstrated.

### 2.5 Environmental change analysis

Comparing present observations with a stored map could reveal moved objects,
newly occupied or cleared regions, obsolete geometry, or changes between
warehouse sessions.

**Current status:** Potential future benefit. A complete map-change detection and
update policy has not been implemented.

### 2.6 Semantic spatial memory

YOLO detections could be projected into the persistent map to record where object
classes were observed and to distinguish temporary objects from stable structure
over time.

**Current status:** Potential future benefit. Current detections and tracker masks
are image-based and are not maintained as persistent 3D semantic objects.

### 2.7 Quantitative research evaluation

The map provides an experimental output independent of trajectory-only metrics.
It enables measurements such as occupied-cell precision and recall, spatial
coverage, Chamfer distance, false obstacles under the true robot footprint,
clearance, graph-node counts, and loop-closure statistics.

**Current status:** Demonstrated in simulation against baked world geometry, with
the result remaining specific to the evaluated map, route, tolerance, and map
placement. Geometry metrics are independent of the trajectory and are sound:
96.6% occupied-cell precision at 0.25 m tolerance. Route-dependent metrics are
only as strong as the route's independence from the map, which is a property that
must be established rather than assumed.

### 2.8 System diagnosis and calibration validation

Structural map errors can expose problems that a single ATE value may conceal,
including incorrect camera transforms, stereo mismatches, gravity errors,
coordinate-frame mistakes, loop-closure failures, duplicated walls, and
insufficient route coverage. In this project, map distortion was important in
identifying a corrupted camera rotation caused by YAML parsing.

**Current status:** Demonstrated diagnostic value.

### 2.9 Visualization, communication, and auditability

A persistent map helps an operator or advisor inspect the observed area, robot
trajectory, localization failures, loop closures, and consistency between runs.
It also provides a visual artifact for explaining progress and limitations.

**Current status:** Demonstrated through RViz, the 2D viewer, validation overlays,
and advisor-facing result figures.

## 3. Appropriate current claim

The strongest defensible statement is:

> The reconstructed map currently provides persistent visual memory, a reference
> frame for relocalization and global correction, quantitative evaluation, system
> diagnosis, and progress visualization. Its geometry has been independently
> validated at 96.6% occupied-cell precision on the tiny warehouse; its
> drivability has not, because no independent route survives. Robust
> arbitrary-start localization, deep correction of the visual front end, semantic
> map updates, multi-session map merging, and planning-grade occupancy remain
> future or incompletely validated work.

A second claim now needs stating alongside it, because it bounds the thesis
rather than the map:

> In static simulation the visual pipelines do not yet outperform wheel+IMU dead
> reckoning, and in dynamic scenes neither pipeline produces a usable trajectory.
> The case for vision must therefore be made where dead reckoning fails -- long
> runs, wheel slip, relocalization after tracking loss, and moving obstacles --
> rather than on static-scene accuracy.

This statement should be revised when new controlled evidence becomes available.

## 4. Evidence locations

- `orbslam3_ros2/MEMORY_CLAUDE/README.md`
- `vins_fusion_ros2/MEMORY_CLAUDE/README.md`
- `rtabmap_ros/MEMORY_CLAUDE/README.md`
- `aws-robomaker-small-warehouse-world/MEMORY_CLAUDE/README.md`
- `output/yolo_eval/`
- `output/compare/`
- `map/`
- `script/validate_map.py`
- `script/vio_metrics.py` (now carries `rpe()`)
- `output/compare/relogged_20260916/rpe_metrics.csv`

