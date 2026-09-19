# SLAM run-logging schema

This document defines the files emitted by the instrumented ORB-SLAM3 and
VINS-Fusion paths. `vio.csv` is intentionally unchanged so existing trajectory
analysis remains compatible.

## Files common to both estimators

| File | Lifetime | Purpose |
|---|---|---|
| `vio.csv` | flushed per pose | Common 11-column trajectory |
| `run_metadata.csv` | written at startup | Resolved configuration and user-supplied provenance |
| `performance.csv` | flushed per processed frame | Timing, delivery, state, resource, and output accounting |
| `run_summary.csv` | written on graceful shutdown | Final cumulative validity and robustness counters |
| `yolo_mask.csv` | flushed per stereo pair when filtering is enabled | Mask matching, age, coverage, and rejection |

VINS-Fusion additionally emits `events.csv` for initialization, resets, and
shutdown, plus `filter_summary.csv`. ORB-SLAM3 records tracking-state transitions
and filter totals in `performance.csv` and `run_summary.csv`, and emits the
algorithm-level files below.

## ORB-SLAM3 algorithm-level files

| File | Unit of observation | Scientific use |
|---|---|---|
| `tracking_frontend.csv` | one processed stereo frame | Feature/stereo front end, IMU preintegration, pose prediction, local-map tracking, keyframe decision, inlier count, and total tracking latency |
| `local_mapping.csv` | one keyframe processed by local mapping | Map-point creation/fusion, local BA, initial inertial alignment, keyframe culling, later inertial refinements, BA problem size, map size, queue state, and reset cause |
| `loop_closing.csv` | one place-recognition, merge, loop-correction, or global-BA event | Candidate outcomes, accepted/rejected corrections, matched keyframe, correction magnitude, scale, and latency |
| `final_frame_trajectory.txt` | one frame not marked lost by ORB-SLAM3 after shutdown | EuRoC-format trajectory reconstructed from the final optimized map |
| `final_keyframe_trajectory.txt` | one retained keyframe after shutdown | EuRoC-format final optimized keyframe poses |

The two final trajectories are written only after local mapping, loop closing,
and any global BA have stopped. If tracking never creates a keyframe, both files
are deliberately empty; the failed run remains analyzable through its summary
and timing logs instead of crashing during shutdown.

`vio.csv` and `final_frame_trajectory.txt` intentionally have different online
validity semantics. The ROS wrapper writes `vio.csv` only in tracking state
`OK`; upstream trajectory reconstruction can also retain `RECENTLY_LOST` frames
that were not marked lost. Report their row counts separately and do not silently
substitute one for the other.

`tracking_frontend.csv` reports image conversion, ORB extraction, and stereo
matching together as `image_preprocessing_feature_stereo_ms`. This boundary is
intentional: the upstream fork does not expose reliable separate timers unless
its optional `REGISTER_TIMES` build mode is enabled. Do not interpret the field
as feature extraction alone.

`loop_closing.csv` distinguishes a proposed candidate from an accepted loop.
For example, `place_recognition,loop_candidate` followed by
`loop_closure,rejected_inertial_geometry` is not a loop closure. Summary counts
include accepted loops and merges only.

## VINS-Fusion algorithm-level files

| File | Unit of observation | Scientific use |
|---|---|---|
| `frontend.csv` | one image pair received by the feature tracker | Feature-tracking latency, mono/stereo feature counts, image-skip decision, feature backlog, and cumulative dynamic-feature rejection |
| `backend.csv` | one feature frame processed by the estimator | Queue and IMU wait, IMU propagation, visual update, initialization, triangulation, Ceres construction/solve/update, marginalization, outlier rejection, failure check, window slide, output time, problem size, costs, iterations, feature health, and state norms |
| `estimator_state.csv` | one processed feature frame | Full pose, velocity, accelerometer bias, gyroscope bias, and estimated camera--IMU time delay during both initialization and nonlinear operation |
| `events.csv` | one lifecycle or reset event | Startup, initialization completion, reset cause, and graceful shutdown |

`backend.csv` uses `external_feature_source=0` for features produced from the
node's subscribed images and `1` for frames injected through
`/feature_tracker/feature`. A controlled WiL run should not mix the two sources;
`run_summary.csv` reports the external-frame count as a validity check.

The VINS timing columns use wall-clock monotonic time. `backend_total_ms` is the
dequeue-to-completion envelope and `estimator_update_ms` is the `processImage()`
envelope. They contain several of the more detailed component fields and must not
be added to those fields. Similarly, `initialization_ms` contains any
triangulation, optimization, marginalization, and window-slide work performed by
the initialization branch.

`solver_termination_type` stores the Ceres termination enum: 0 convergence,
1 no convergence, 2 failure, 3 user success, and 4 user failure. Cost and solver
fields are `nan`, zero, or `-1` on frames that do not invoke Ceres. A frame may
contain more than one solver call during initialization; its durations and
iteration counts are summed, the initial cost is from the first call, and the
final cost and termination status are from the last call.

The current VINS fork has an unconditional early `return false` in
`failureDetection()`. Therefore `failure_detection_ms` measures the call but the
check cannot currently trigger a reset; `run_summary.csv` records
`failure_detection_enabled,0`. This is an algorithm limitation, not evidence
that no physically invalid state occurred.

`estimator_state.csv` records the optimized bias state but not a state
covariance. This VINS path does not maintain a directly exportable per-frame
covariance, and running a separate Ceres covariance computation on every frame
would materially change the real-time workload being measured. Do not infer
uncertainty from the bias norm, solver cost, or termination status.

## Provenance parameters

Both estimators accept these optional ROS parameters and copy them into
`run_metadata.csv`:

- `experiment_id`
- `dataset_path`
- `world_path`
- `replay_rate`
- `run_command`
- `run_notes`

They do not guess missing provenance. An empty value means it was not supplied
and the run is incomplete for a final reproducibility claim. ORB launch files
expose the parameters as launch arguments. VINS accepts them after `--ros-args`.

## Performance interpretation

ORB-SLAM3 records queue wait, `TrackStereo()` duration, total per-frame
processing time, tracking state, pose validity, IMU samples, received and
processed stereo pairs, queue drops, IMU starvation, pose count, process CPU
time, monotonic wall elapsed time, and resident memory.

VINS-Fusion's legacy `performance.csv` records estimator-thread processing time,
feature backlog, feature and IMU counts, initialization state, keyframe decision,
marginalization type, resets by cause, pose count, dynamic points removed,
process CPU time, monotonic wall elapsed time, and resident memory. Its
`processing_ms` still begins after feature tracking and IMU ingestion for
backward compatibility. Use `frontend.csv` and `backend.csv` for the scientific
front-end/back-end latency decomposition. This VINS estimator has no loop
closure; RTAB-Map loop closure and global correction must be logged and analyzed
as a separate component.

For adjacent rows, process CPU utilization relative to one logical core is:

```text
cpu_percent = 100 * delta(process_cpu_ms) / delta(wall_elapsed_ms)
```

Values above 100% are valid for a multithreaded process. Divide by the number of
logical cores only when reporting utilization as a percentage of the entire
machine. Do not use sensor timestamps for this calculation because rosbag replay
and Gazebo can run at rates other than wall time.

`resident_memory_kb` is current resident memory from `/proc/self/statm`. The
resource fields cover the estimator process, including its internal threads, but
not Gazebo, rosbag, RViz, RTAB-Map, or the separate YOLO process. Consequently,
they support estimator-process comparisons but are not system-wide measurements.

## Validity rules

- A missing `run_summary.csv` means shutdown was not graceful. Per-frame files
  remain usable up to their last complete row, but the run requires review.
- `images_received` or `stereo_received` counts callbacks delivered to the node;
  they cannot reveal samples dropped before DDS delivery.
- ORB state `-3` denotes insufficient IMU coverage for a dequeued image; it is a
  wrapper diagnostic, not an upstream ORB-SLAM3 tracking-state value.
- VINS reset totals are cumulative across reinitialization and are not cleared by
  `resetState()`.
- VINS keyframes are counted when nonlinear mode selects
  `MARGIN_OLD`; this is the implemented marginalization/keyframe decision.
- Compare paired treatments only when their input interval and delivery counters
  are materially equivalent.
