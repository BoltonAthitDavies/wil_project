# YOLO dynamic-object filtering in ORB-SLAM3 — engineering record

Working notes kept for the project report. Records WHY each decision was made,
what was measured, and what went wrong — the code already says what it does.
Companion files: **[integration.md](integration.md)** — how ORB-SLAM3 was built and
wrapped in the first place, and where every calibration number in `config/` comes
from (sim and real, stereo extrinsics, IMU noise, the fisheye mapping, the four
fork patches, and the bugs found along the way); and
`vins_fusion_ros2/MEMORY_CLAUDE/README.md` for the VINS side of the YOLO work.

Date: 2026-09-02/03. Last updated: 2026-09-17. Reference paper: *RY-SLAM: A robust YOLO-based real-time
visual SLAM solution for dynamic environments*, Chen et al., ICVRV 2025
(PDF at workspace root).


## 1. The problem

ORB-SLAM3 assumes a static world. In the dynamic warehouse sim the moving props
generate ORB features that get triangulated as static landmarks, and the run
diverges. `make_dynamic.py` re-arms a subset of props with the Ignition
VelocityControl plugin; the movers are `ClutteringA/C/D` (box piles),
`Bucket_01` and `TrashCanC_01` — which is where the detector classes
bin/box/bucket come from.

RY-SLAM's fix: run YOLO alongside the tracker and have `ORBextractor` discard
keypoints landing on dynamic regions. Two details from the paper we reproduced:

* the discard happens **before** the octree distributes the per-level feature
  quota, so the quota is refilled from surviving static keypoints rather than
  leaving a hole where the object was;
* **eq.(1)**, a neighbourhood rule for keypoints on a mask boundary — count
  dynamic pixels in a 3×3 neighbourhood, delete at ≥ 5, reserve otherwise.


## 2. Model provenance — state this precisely in the report

Three `.pt` files were present. Only one is ours:

| file | names | provenance |
|---|---|---|
| `yolo26n.pt` | 80 COCO classes | **STOCK** Ultralytics release. `data: cfg/datasets/coco.yaml`, `save_dir: /home/lq/codes/ultralytics/tune-yolo26n-objv1-coco/train31` |
| `yolo26m.pt` | 80 COCO classes | **STOCK** Ultralytics release. `save_dir: /home/lq/codes/ultralytics/YOLO26/YOLO26m-best-hyp-x-epochs80-cp03` |
| `weight/best.pt` | **{0: bin, 1: box, 2: bucket}** | **OURS.** `DetectionModel`, fine-tuned from `yolo26m.pt` on `/content/datasets/warehouse-1/data.yaml` (Colab), imgsz 640, 44 MB |

`/home/lq/` is an Ultralytics developer's machine — those two files are the public
release, not anything trained here. Both were initially mistaken for the project
model; the metadata inside the checkpoint pickle settled it.

**The model is a detector, not `-seg`.** RY-SLAM uses YOLOv8s-seg and gets
pixel-accurate masks; we get axis-aligned boxes. Everything downstream is written
against a generic `cv::Mat` mask, so swapping in a `-seg` model later needs no C++
change. Worth listing as a deliberate deviation from the paper.


## 3. Architecture, and why

```
dynamic_detector_node.py (rclpy + ultralytics, GPU)
    /cam0/image_raw ──► YOLO(best.pt) ──► /orbslam3/dynamic_dets
                                          vision_msgs/Detection2DArray
                                          (stamped with the SOURCE image stamp)
orbslam3_node.cpp
    onStereo ◄── unchanged 2-way ApproximateTime stereo sync
      └─ nearest-timestamp lookup in a detection ring buffer (non-blocking)
      └─ rasterize boxes ──► cv::Mat mask (255 = keep, 0 = dynamic)
      └─ pushStereo(t, left, right, mask)
SlamWrapper::workerLoop
      └─ TrackStereo(left, right, t, imu, "", maskLeft, maskRight)
thirdparty/ORB_SLAM3 (patched)
    System::TrackStereo ─► Tracking::GrabImageStereo ─► Frame ─► ExtractORB
      ─► ORBextractor::operator() ─► ComputeKeyPointsOctTree
```

**Why a separate Python process rather than in-process C++** (the paper uses ONNX
Runtime): the weights are a PyTorch `.pt`, so in-process would need an ONNX export
plus an ONNX Runtime dependency in CMakeLists, and system OpenCV 4.5.4 is too old
to import a YOLO26 ONNX graph via `cv::dnn`. One topic is cheaper, a slow or
crashed detector degrades SLAM to "unfiltered" instead of taking the tracker down,
and one detector can feed **both** ORB-SLAM3 and VINS with identical masks.

**Why detections are NOT in the message_filters sync:** a 3-way
`ApproximateTime<Image, Image, Detection2DArray>` would drop stereo pairs every
time inference falls behind frame rate, which costs SLAM far more than the dynamic
features do. Nearest-stamp lookup with a max-age bound degrades gracefully —
the same philosophy as the existing IMU-starvation guard in `slam_wrapper.cpp`.
This is why the detector stamps messages with the **source image's** stamp, never
`now()`.

**Why `filter` defaults to false:** with it off no subscription is created and an
empty mask travels down, which `ORBextractor` short-circuits — the stock tracker
bit for bit, so the recorded baseline stays reproducible.

**Why the left camera only:** in stereo, `ComputeStereoMatches` matches LEFT
keypoints against right candidates, so removing a left keypoint already stops the
map point being created. Halves inference cost. `maskRight` is threaded through
anyway so a second detector needs no further library patch.


## 4. The ORB-SLAM3 fork patch (138 insertions, 8 files)

Second local commit on the fork (the first was `5c2b00f`, C++17 + `GetTracker()`).
Mask convention: `CV_8UC1`, 255 = keep, 0 = dynamic, empty = disabled. All new
parameters appended **last** with defaults, so every existing call site compiles.

The filter itself, in `ComputeKeyPointsOctTree` where `vToDistributeKeys` is
filled — before `DistributeOctTree`:

```cpp
if(!mMask.empty()) {
    const float s  = mvScaleFactor[level];
    const float x0 = ((*vit).pt.x + minBorderX) * s;
    const float y0 = ((*vit).pt.y + minBorderY) * s;
    int nDynamic = 0;
    for(int dy=-1; dy<=1; ++dy) for(int dx=-1; dx<=1; ++dx) {
        const int xx = cvRound(x0 + dx*s), yy = cvRound(y0 + dy*s);
        if(xx<0||yy<0||xx>=mMask.cols||yy>=mMask.rows) continue;
        if(mMask.at<uchar>(yy,xx) == 0) ++nDynamic;
    }
    if(nDynamic >= 5) continue;   // RY-SLAM eq.(1)
}
```

One rule covers both cases the paper separates: a keypoint deep inside a dynamic
region scores 9 and goes; one just outside a boundary scores ≤ 4 and stays.

Two subtleties worth a sentence each in the report:

1. **Coordinate convention.** At that point `pt` is in LEVEL coordinates RELATIVE
   TO `(minBorderX, minBorderY)` — `j*wCell`/`i*hCell` are added just above, but
   `minBorderX/Y` only after `DistributeOctTree` returns. The mask is in level-0
   pixels, hence `+minBorder` then `*scale`. Getting this wrong misaligns the mask
   **silently** rather than failing.
2. **The neighbourhood is spaced by the level's scale factor**, not by 3 adjacent
   level-0 pixels. A coarse-pyramid keypoint's support in the source image really
   is larger, so it is tested over a correspondingly larger area.

`mMask` as a plain member is thread-safe: `Frame` extracts left and right in two
threads but through two *distinct* `ORBextractor` objects.

### Two build traps hit along the way (each cost a build cycle)

* **`System::TrackStereo` can remap or resize the images** (`needToRectify()` /
  `needToResize()`). The mask must get the identical transform or it drifts off
  the objects it describes. Applied with `INTER_NEAREST` (a mask is a label image;
  interpolating invents boundary values between 0 and 255) and `BORDER_CONSTANT`
  255 — pixels rectified in from outside the source carry no detection evidence
  and must not read as dynamic.
* **`std::thread` does NOT apply default arguments.** It forms the invoker from
  exactly the arguments given, so the fisheye `ExtractORB` thread sites at
  `Frame.cc:1064-1065` needed an explicit `cv::Mat()` even though they never mask.
  The error is an opaque `static assertion failed: std::thread arguments must be
  invocable after conversion to rvalues`.


## 5. `max_mask_fraction` — a measured default, not a guess

Safety valve: if the dilated boxes cover more than this fraction of the frame, the
mask is discarded for that frame rather than starving the tracker.

Measured over **all 791 frames** of `dataset/dynamic_dataset` (8 px dilation, 1280×720):

```
mean 43.5%   median 44.2%   p75 51%   p90 61%   p95 65%   p99 70%   max 72.8%

frames rejected at 0.50 : 237/791 (30%)
                   0.60 :  97/791 (12%)
                   0.70 :  10/791 ( 1%)
                   0.80 :   0/791 ( 0%)
```

The default was initially 0.6 (a guess) and is now **0.8**. Rationale: at 0.6 the
valve fires during entirely *normal* operation, which is the worst of both worlds —
those frames get no filtering while their neighbours do, so dynamic points leak
into the map anyway. 0.8 sits above the observed maximum, so it only catches a
genuinely runaway detection and normal operation stays consistently filtered.


## 6. Verification results

Bag: `dataset/dynamic_dataset` (790 stereo frames, compressed, dynamic warehouse).

```
detector : resolved 3/3 -- bin(0), box(1), bucket(2)
           11.7 ms/frame synthetic (86 FPS, RTX 3070); 19-33 ms in the live
           graph; 56 ms when the GPU is shared with VINS + republishers
           1460 boxes over 393 frames
tracker  : dynamic filter: 392 frames masked, 1 with no recent detection
           0 masks discarded
baseline : filter:=false -> "filter: off (stock ORB-SLAM3 feature extraction)",
           no subscription created (ros2 topic info shows 0 subscribers)
```

Timestamp matching is essentially perfect — 1 miss in 393 frames.

**A bug found and fixed during verification:** the detector's 10 s report timer ran
on **sim time**, so the bag's initial `/clock` jump made it fire once per missed
period — thousands of times, flooding the log and burying the tracker's output.
Now on `ClockType.STEADY_TIME`; the log went from thousands of lines to 106. Worth
a line in the report as a sim-time gotcha.


## 7. The finding that matters most for the report

**Mean mask coverage is 43.5% of every frame, and the filter is purely semantic —
it masks by CLASS and never tests whether an object is actually moving.** There is
no motion gate anywhere in the pipeline.

How much that costs, counted from the world files (`<include>` blocks carrying the
KinematicTrajectory plugin vs not):

| world | detector-class props | of those, actually moving |
|---|---|---|
| `small_warehouse_dynamic.world` (**the `dynamic_dataset` bag**) | 36 | **19 (53%)** |
| `small_warehouse_dynamic_00/_01/_02.world` | 39-40 | **30 (75-77%)** |

Every mover IS detector-class, so nothing that moves is missed *by the class
choice*. The cost is precision: in the recorded bag roughly **half** the masked
objects are static shelving scenery.

**How much this actually costs was later measured directly — see §8, and it is far
less than the 43.5% figure below suggests.** On the current
`dataset_dynamic_nofloortexture_00_001` the detector masks **11.4%** of the frame at
the same 8 px dilation, of which the moving share is ~8.4%. The gap a perfect motion
gate could recover is therefore only ~3 points of frame area, not ~20. Treat §5's
43.5% as specific to the retired `dataset/dynamic_dataset` bag and its much more
densely filled world.

Caveat on those counts: "detector-class" here means the model names
ClutteringA/C/D, Bucket, TrashCanC. The detector is trained on a custom dataset and
may also fire on shelves or desks, so the static share could be larger than the
table suggests. The 43.5% coverage figure is measured from real detections and is
not affected by this.

RY-SLAM gets away with "mask the whole class" because a person is never scenery.
That premise only partly transfers to a warehouse where the dynamic class also
furnishes the room.

The pipeline is correct and faithful to the paper. Run the A/B before drawing
conclusions. If it regresses, the candidate fix is **motion gating** — recover a 3D
position per box (stereo disparity on the box ROI, then `T_w_c`) and mask only boxes
whose WORLD position changes — which is what DS-SLAM and Blitz-SLAM add on top of
the semantic prior. Not implemented; nothing in the current filter is 3D-aware.

**But §8 weakens the case for it.** On current data the motion gate could recover
only ~3 points of frame area. The bigger lever is the detector's own **recall on
moving props, measured at 0.599** — the filter is currently missing ~40% of the
objects it exists to mask, which no amount of motion gating fixes.


## 8. Detector validation against Gazebo ground truth (2026-09-11)

`/world/default/pose/info` carries the TRUE world pose of every model, so YOLO can be
scored properly instead of eyeballed. Tooling (in `script/`, per project convention):

* `script/yolo_gt.py` — model-name -> class map, and collision-mesh vertices per prop
* `script/yolo_eval.py` — projects GT boxes into cam0, runs YOLO, matches by IoU,
  writes CSVs keyed by `timestamp_ns` **so they join straight onto `vio.csv`**

```bash
python3 script/yolo_eval.py dataset/dataset_dynamic_nofloortexture_00_001 \
    --out output/yolo_eval/dyn_00_001 --overlay 12
```

### Two traps that produce confident, plausible, completely wrong numbers

1. **`/world/default/pose/info` has ALL-ZERO header stamps.** Every one of the 5785
   messages claims `t=0`. Using them pins every prop *and the robot* to its
   start-of-run pose, giving GT boxes of the right size in the wrong place and making
   every prop look stationary. First run scored precision 0.08 / recall 0.13 /
   **zero** moving props and looked exactly like a botched extrinsic.
   Fix: rebuild sim time from `/clock` (payload = sim, receive stamp = wall). A
   constant wall->sim offset will NOT do — measured std 6.65 s over 100 s as the RTF
   wandered. **Anything else written against this topic will hit the same thing.**
2. **Prop meshes are in CENTIMETRES** (`<unit meter="0.01">`) with per-node `<matrix>`
   transforms that must be composed. Either mistake is silent.

Also note `cv2.FileStorage` mis-parses `body_T_cam0` out of the VINS yaml — it returns
~1e18 garbage for the rotation while leaving the translation plausible. Parse the
`data:` block by hand.

**Always render the overlay and LOOK at it before trusting a metric.** Both bugs above
were obvious in one glance at an image and invisible in the numbers.

### Results — `dataset_dynamic_nofloortexture_00_001`, 2236 frames, conf 0.35

```
GT boxes        17026   (moving 11143, static 5883, occluded->ignored 10271)
predictions     16256   (1430 ignored as unlabelled shelf boxes)
precision       0.704
recall (all)    0.597
recall (MOVING) 0.599   <- what the SLAM filter actually needs
```

| conf | TP | FP | precision |
|---:|---:|---:|---:|
| 0.35 | 10561 | 5695 | 0.650 |
| 0.50 | 9433 | 3909 | 0.707 |
| 0.70 | 7027 | 1976 | 0.781 |
| 0.90 | 2154 | 247 | 0.897 |

Frame-area coverage, mean over the run:

| | coverage |
|---|---|
| GT, all props | 11.6% |
| GT, **moving only** | 8.4% — ceiling for a motion-gated filter |
| YOLO predictions, undilated | 9.6% |
| YOLO predictions, **8 px dilation** (what the SLAM mask blanks) | **11.4%** |

### Both headline metrics are LOWER BOUNDS

* **Precision.** False positives were 84% class `box` with median 46 px side — not
  specks. They are the **loaded warehouse shelves**: `ShelfD/E_01` are modelled with
  cardboard boxes on them and YOLO is right to fire. Adding shelves, pallet jacks and
  desks as COCO-style *ignore* regions moved precision 0.641 -> 0.704, and the true
  figure is higher still because a shelf's projected rectangle does not tightly bound
  the boxes sitting on it.
* **Recall.** GT includes partially-occluded props the detector cannot reasonably
  see; the occlusion stencil only drops boxes >70% covered. Raising the GT size floor
  does NOT rescue recall (0.60 -> 0.70 from 12 px to 80 px) while destroying precision
  (0.65 -> 0.18), which proves the small boxes were being correctly detected — the
  misses are occlusion, not scale.

### What this says for the report

Recall on moving props is **0.599**: the filter misses roughly 40% of the objects it
exists to mask. That, not over-masking, is the thing limiting how much RY-SLAM-style
filtering can help here. Raising `conf` buys precision at the cost of that recall, so
there is an operating point to be found — sweep `conf` against ATE.

Caveat on dataset drift: `dataset/dynamic_dataset` (used for §5 and §6) has been
replaced by `dataset_dynamic_nofloortexture_*`. Coverage differs ~4x between the two
worlds at identical dilation, so do not mix numbers across them.


## 9. Environment (reproducibility)

```
torch 2.5.1+cu121   torchvision 0.20.1   ultralytics 8.4.138   numpy 1.26.4
OpenCV: system 4.5.4 for C++; pip opencv-python 4.11 for Python (they coexist)
GPU: RTX 3070 Laptop, driver 595, NO CUDA toolkit installed — the cu121 wheels
     bundle their own CUDA runtime, which is why no toolkit was needed
```

Install traps, all hit for real:

* `pip install ultralytics` auto-selects the **CUDA 13** wheel stack (~4.5 GB:
  2.6 GB `nvidia-*` + 690 MB triton). Pin the cu121 index instead (~2.5 GB).
* It also upgrades numpy to 2.x, which **breaks `cv_bridge`** with
  `AttributeError: _ARRAY_API not found` (binaries compiled against numpy 1.x).
  Always install with `"numpy<2"`.
* Needs ~5 GB free at peak — pip extracts wheels before installing, roughly
  doubling transient usage. Two installs failed with `[Errno 28]` before disk was
  freed.

Build order (the library must go first or `CMakeLists.txt:21` hard-fails):

```bash
./build_orbslam3.sh          # or: cmake --build thirdparty/ORB_SLAM3/build -j8
colcon build --packages-select orbslam3_ros2 --symlink-install   # FROM WORKSPACE ROOT
```

Running colcon from inside the package directory creates stray
`orbslam3_ros2/build|install|log` dirs — done twice by accident, both cleaned up.

### How to run

```bash
# baseline
ros2 launch orbslam3_ros2 wil_sim_stereo_imu.launch.py image_transport:=compressed

# with filtering (also starts the detector)
ros2 launch orbslam3_ros2 wil_sim_stereo_imu.launch.py filter:=true \
    image_transport:=compressed publish_debug_image:=true

ros2 bag play dataset/dynamic_dataset --clock -r 0.7
```


## 10. Files changed

`orbslam3_ros2` — 298 insertions, 7 files + `scripts/`:

* `scripts/dynamic_detector_node.py` (new) — the detector
* `src/orbslam3_node.cpp` — `filter` param, detection buffer, mask rasterization
* `src/slam_wrapper.cpp`, `include/orbslam3_ros2/slam_wrapper.hpp` — mask passthrough
* `launch/wil_sim_stereo_imu.launch.py`, `launch/wil_stereo_imu.launch.py` — `filter` arg
* `CMakeLists.txt`, `package.xml` — `vision_msgs`, install the script

`thirdparty/ORB_SLAM3` — 138 insertions, 8 files: `System`, `Tracking`, `Frame`,
`ORBextractor` (headers + sources).

Node parameters added: `filter` (false), `dynamic_dets_topic`, `mask_dilate_px`
(8), `det_max_age` (0.15 s), `max_mask_fraction` (0.8).

Validation tooling (2026-09-11), in `script/` per project convention:
* `script/yolo_gt.py` — class map + collision-mesh geometry per prop
* `script/yolo_eval.py` — GT projection, IoU matching, CSVs keyed by `timestamp_ns`

Note `max_mask_fraction` 0.8 was calibrated against the retired `dynamic_dataset`
world, whose coverage peaked at 72.8%. On the current datasets coverage averages
11.4%, so the valve now effectively never fires — harmless, but it is no longer
doing anything, and a future world could change that again.

Weights live at `weight/best.pt` (moved there 2026-09-03; defaults updated in the
detector and both launch files).


## 11. Structured evaluation logging and dataset provenance (2026-09-16/17)

The ORB-SLAM3 wrapper now creates `output_path` and missing parent directories and
records the complete observable pipeline, rather than only `vio.csv`:

| file | evidence recorded |
|---|---|
| `tracking_frontend.csv` | features, map inliers, frontend stage timings, tracking state and keyframe decisions |
| `local_mapping.csv` | keyframe insertion, point culling/creation, local BA, map size, queue state and reset requests |
| `loop_closing.csv` | place-recognition checks, accepted loop/map-merge events, correction magnitude/scale and duration |
| `performance.csv` | per-frame timing, queues/drops, IMU starvation/initialization, map counters, loop/global-BA counters, CPU and memory |
| `run_summary.csv` | tracking loss, initialization, map/local-BA/loop/merge/global-BA/reset totals and filter totals |
| `run_metadata.csv` | dataset/world provenance, topics, frames, settings, notes and replay-rate label |
| `final_frame_trajectory.txt`, `final_keyframe_trajectory.txt` | ORB-SLAM3 final optimized trajectories, written during clean shutdown |
| `yolo_mask.csv` | mask actually supplied to tracking; created only with `filter:=true` |
| `vio.csv` | online trajectory in the shared VINS-style comparison format |

Thus the logs reasonably cover the tracking frontend, local-mapping backend, place
recognition, loop closure, map merging and global bundle adjustment that are
observable from ORB-SLAM3. They do not contain every internal residual or optimizer
iteration. A forced kill can omit the final trajectories and a trustworthy
`run_summary.csv`; prefer a graceful shutdown.

`replay_rate` is descriptive metadata and must equal the rate actually passed to
`ros2 bag play`. `world_path` identifies the source Gazebo world; it is not an
ORB-SLAM map and is not used for localization or navigation.

For the relogged raw results, trust the containing dataset folder name over stale
metadata. The only datasets stored locally are
`dataset_dynamic_nofloortexture_01_000` and
`dataset_static_nofloortexture_001`, under
`/home/ambushee/wil_project/dataset/`. The other no-floor-texture datasets are
under `/media/ambushee/32E4AAB1E4AA772F/dataset/`. On 2026-09-17 all 66
`output/output_*/**/run_metadata.csv` files were audited and 15 wrong
`dataset_path` entries were corrected. All corrected paths existed; no measured
data or trajectories were edited.

## 12. Making filter runs deterministic — precomputed detections (2026-10-12)

### What went wrong

Two filter-ON runs of `dataset_allsensor_realistic_003`, same code, same config,
ten minutes apart:

| run | first-window `tracking_ms` | frames with a mask |
|---|---|---|
| 13:36 | 96.1 | **33 %** (289 / 875) |
| 13:46 | 57.4 | **91 %** (2419 / 2663) |

The detector never crashed (launch log: it died only on the user's Ctrl-C). It
ran a whole number of frames behind the tracker, and `det_max_age = 150 ms`
accepts four frames (132 ms) and refuses five (165 ms) — `det_age_ms` on matched
frames is quantised to 33/66/99/132 and never reaches 165. Whenever ORB's own
per-frame time rose (map growth, or a warm laptop: this machine's
`package_throttle_total_time` is 17 min and its idle clock 1.48 GHz against a
4.6 GHz max) the detector slipped to five frames and the mask vanished, writing
`matched=0, det_age_ms=-1` — the same row a dead detector writes. Mask
availability tracked `tracking_ms` directly: 98–100 % below 80 ms, 88 % at
100–120 ms, 68 % at 140–160 ms.

So "filter ON" was a different method every run, and depended on CPU
temperature. The ten-second `det_matched_ == 0` warning never fired because the
match was never zero. A partially masked run is the quiet failure.

A second, independent nondeterminism surfaced while testing the fix (see §12.3).

### 12.1 The fix: detections become a file

`script/yolo_precompute.py` runs the weight over every `/cam0` frame of a bag
once and writes `timestamp_ns,class,score,x0,y0,x1,y1`, one row per box, with a
row of empty class for frames that had none (so "nothing detected" and "frame
not in file" stay distinguishable). Provenance (`weights`, `conf`, `iou`,
`imgsz`, `classes`, `bag`) is a `#` line at the top. ~45 fps on the GPU; a
3547-frame bag takes 80 s. The class set follows the weight exactly as the live
node's `auto` does.

Both nodes gained `dets_csv`. When set with `filter=true`:
- no detection subscription is created and (ORB launch files) the detector
  process is not spawned — `IfCondition` on `filter and dets_csv == ''`;
- `maskFor()` looks the frame up by its **exact header stamp**; nearest-within-
  `det_max_age` is a fallback for a re-stamped replay and reports a nonzero age
  so a run that needed it is visible in `yolo_mask.csv`;
- everything downstream (dilation, `max_mask_fraction`, logging) is unchanged.

`run_summary.csv` gains `detection_match_rate` and `detection_source`
(`none|live|file`). At shutdown the node now warns when the match rate is below
98 % (the quiet failure above) and the dropped-frames warning states that the
run is not comparable with anything.

### 12.2 Acceptance test — passed

Two consecutive runs, `realistic_003`, `filter:=true dets_csv:=dets_best.csv
image_transport:=compressed`, bag at `--rate 0.3` for 200 s wall:

```
                          run 1     run 2
stereo_pairs_received      1128      1137
queue_dropped_frames          0         0
detection_match_rate     1.0000    1.0000
det_age_ms (every row)      0.0       0.0
common stamps              1128  ->  yolo_mask.csv rows that differ: 0
exclusive stamps              9  ->  all at the window edges (run 2 started earlier)
```

Identical frames, identical masks, no detector process. Rate 0.3 is what this
laptop needs for `queue_dropped = 0`; at 1.0 it dropped 39 % and 18 % of pairs in
the two runs above.

### 12.3 The second trap: stale republishers and the `raw` default

The first acceptance pair FAILED on the frame set — 919 vs 813 pairs over the
same window, `queue_dropped = 0` in both, 394 stamps present in only one run —
while the masks on common stamps were already identical. Cause: the sim bags
carry **only** `/camN/image_raw/compressed`, but `wil_sim_stereo_imu.launch.py`
defaults `image_transport:=raw`. The node was receiving raw images through two
`image_transport republish` processes left running from the user's 13:34 VINS
session. Raw 1280×720 rgb8 is 2.7 MB per message; over best-effort DDS loopback
that path loses frames nondeterministically, and a second republisher pair on the
same topics (a later VINS run) delivers every frame twice — the duplicate
`yolo_mask.csv` rows noted earlier in this record. None of this is visible from
inside the node.

Rule, now in the evaluation skill: replay into ORB with
`image_transport:=compressed` (the node decodes JPEG itself, no republisher at
all), and `pgrep -fa 'republis[h]'` before every run.

### 12.4 Files

| file | change |
|---|---|
| `script/yolo_precompute.py` | new |
| `src/orbslam3_node.cpp` | `dets_csv` param, `loadDetsCsv()`, `stampNs()`, file branch in `maskFor()`, match-rate warning, summary rows |
| `launch/wil_sim_stereo_imu.launch.py`, `launch/wil_stereo_imu.launch.py` | `dets_csv` arg; detector spawned only when live |
| `scripts/dynamic_detector_node.py` | `dynamic_classes` default `auto` = every class of the loaded weight (the sentinel is a word, not `[]`: launch_ros rejects an empty sequence and rclpy infers BYTE_ARRAY from an empty default -- both are startup crashes, explained in the node) |
| `.claude/skills/wil-thesis-evaluation/skill.md` | recipes use `dets_csv` + `image_transport:=compressed` |

Precomputed files for `realistic_003` live next to the bag:
`dets_best.csv` (3-class: box 11941, bucket 3056, bin 106) and `dets_runs1.csv`
(4-class: box 10381, bucket 5646, forklift 2893, bin 228).

## 13. Motion-gated masking — `filter_mode:=motion` (2026-10-12/14)

### Why

Class masking (§1–§7) removes every detected bin, box, bucket and forklift,
moving or not. In `realistic_000` nothing moves, so the filter can only delete
good landmarks; in `realistic_002/003` ten to twelve props move among ~150
static ones (see the auto-memory note `realistic-bags-dynamic-content`), so
most of what the filter removes is still static scenery. IQ-VIO (Zhang et al.
2023) goes the other way — keeps every feature and distrusts the EKF
measurement — which protects the fused output but not the map. Neither decides
per object. This mode does: a box is masked only after the tracker has seen it
move.

### How it works

The mask is applied inside `ORBextractor`, before a pose exists for the frame,
so the decision for frame *k* cannot gate frame *k* itself. It gates the next:

1. **Census (fork, `System::TrackStereo(..., voteBoxes)`).** After tracking,
   count for each box the left keypoints inside it and how many ended the frame
   as pose-optimisation inliers — a map point that survived `TrackLocalMap`
   **and has `Observations() > 0`**, i.e. is anchored at a keyframe. The same
   counts are taken for the background outside every box. Raw counts only;
   `GetBoxVotes()` returns them, the wrapper carries them in
   `TrackResult::box_votes`.
2. **Verdict (node, `applyVotes()`).** `ratio = n_inlier / n_kp`;
   `moving` if `ratio < motion_inlier_ratio × background_ratio`, else `static`;
   withheld (verdict inherited) if `n_kp < motion_min_points`.
3. **Association (node, `maskFor()`).** Each box of frame *k+1* inherits the
   verdict of the frame-*k* box with the highest IoU ≥ `motion_iou`; otherwise
   `unknown` (left unpainted unless `motion_unknown_mask`). Only `moving` boxes
   are painted.
4. **Re-vote probation.** A painted box has no keypoints, so its verdict could
   never be revisited. Every `motion_revote_frames` (15) frames of a chain the
   box is left unpainted for one frame so it is scored again.

Parameters, defaults: `motion_iou` 0.3, `motion_min_points` 8→**15**,
`motion_inlier_ratio` 0.5, `motion_unknown_mask` false, `motion_revote_frames`
15. `yolo_mask.csv` gains `n_moving,n_static,n_unknown`; `motion_votes.csv`
records every box-frame census and verdict; `run_summary.csv` gains
`filter_mode`, `boxes_voted_moving/static`.

### Calibration tooling

`script/motion_votes_eval.py <bag> motion_votes.csv` projects the true props
into cam0 at each row's stamp (yolo_eval.py machinery), labels the voted box's
prop MOVING/STATIC from `pose/info` speed, and prints the verdict-vs-truth
confusion, the box/background ratio distributions per truth class, and a
`min_points × ratio` sweep of recall vs false alarm on the same census.

### Run 1 (2026-10-12, defaults, no re-vote, temporal points counted) — FAILED

`realistic_003`, `dets_runs1.csv`, rate 0.3, 1720 frames, match rate 1.0.

```
truth / verdict   unknown   static   moving
moving                 22       37      680
static                 82     1107     5105     <- 81 % of static boxes masked
none                  128      215      683
census, voted rows only (n_kp >= 8): moving n=44, static n=1042
  box/bg inlier ratio   moving: p25 0.25 median 0.78 p75 1.57
                        static: p10 0.65 median 1.32 p75 1.78
sweep (recall / false alarm):  ratio 0.25: 0.25 / 0.052   0.5: 0.39 / 0.077   0.75: 0.48 / 0.124
```

Two faults, one of design and one of signal:

- **Runaway.** With ~5 static boxes per frame, a 5–8 % per-vote false alarm and
  no way to revisit a painted box, one bad vote spread through IoU association
  until motion mode had collapsed into class mode by t≈105 s. Fixed by the
  re-vote probation (step 4).
- **Weak signal.** The movers in this bag travel 0.17–0.27 m/s ≈ 2 px per frame
  at 3 m. ORB creates *temporal* map points from the previous frame's stereo
  depth every frame (`UpdateLastFrame`); 2 px of motion sits inside the
  projection search window, so a slow mover's temporal points match and count
  as inliers. Only map points anchored at keyframes — seconds of baseline —
  expose the motion. Fixed by counting only `Observations() > 0`, which is what
  `mnMatchesInliers` itself does.

Run 2 (both fixes, `min_points 15`) is recorded below.

### Run 2 (2026-10-14, `motion_20261014_02`: keyframe-anchored census + re-vote, `min_points 15`)

1721 frames, match rate 1.0, 0 drops, 1379 frames painted.

```
truth / verdict   unknown   static   moving
moving                 22       78      639
static                 79     2888     3328     <- 53 % of static boxes masked (was 81 %)
census, voted rows (n_kp >= 15): moving n=101, static n=2646
  box/bg inlier ratio   moving: p10 0.00 p25 0.00 median 0.54 p75 1.29
                        static: p10 0.58 median 1.24 p75 1.60
sweep (recall / false alarm), min_pts 15:  ratio 0.25 0.37/0.070   0.50 0.49/0.090   0.75 0.59/0.142
```

Better on both counts, still weak: recall 0.49 at 9 % false alarm against
VINS's 0.87 at 1.1 % (VINS record §10). A quarter of moving votes are a clean
0.00; the median is dragged up because local mapping inserts a keyframe every
~10 frames and re-anchors *fresh* map points on the mover — "keyframe-anchored"
is only ~0.3 s of baseline, and 0.3 s × 0.2 m/s is still inside the match
window. Run 3 adds a map-point age floor (`motion_min_point_age`, 30 frames =
1 s; `System::SetBoxVoteMinAge`), which is what VINS's solved-depth criterion
implies for free.

Incidental: run 2's first attempt died at launch because `run_notes` contained
a colon — `launch_ros` YAML-parses every string parameter, so `a: b` becomes a
dict and the launch aborts. Both launch files now wrap the free-text arguments
(`experiment_id`, `dataset_path`, `world_path`, `run_command`, `run_notes`,
`dets_csv`) in `ParameterValue(..., value_type=str)`.

### Run 3 (2026-10-14, `motion_20261014_03`: + map-point age floor 30 frames)

```
truth / verdict   unknown   static   moving
moving                 32       62      645
static               272     2093     3930     <- 62 % of static boxes masked (worse)
census, voted rows (n_kp >= 15): moving n=85, static n=1968
  box/bg inlier ratio   moving: p10 0.00 p25 0.00 median 0.00 p75 2.47
                        static: p10 0.00 p25 0.78 median 1.54 p75 3.23
sweep, min_pts 15:  ratio 0.25 0.58/0.128   0.50 0.62/0.154   0.75 0.67/0.237
```

The age floor did what it was meant to on movers — the median is now a clean
0.00 — and did the same to **static objects that had just come into view**:
all their map points are young too, so p10 of the static distribution is also
0.00 and the false alarm rate doubled to 15 %. The background ratio does not
compensate, because the background still contains long-seen regions. The
one-frame census cannot distinguish "moving" from "new"; the confound is scene
novelty, not signal quality.

The node can: a box's IoU association chain already measures how long it has
been in view. Run 4 adds `motion_min_box_age` (30 frames): a box is judged only
once its chain is a second old — by then a static box has old points and a
mover has none. `motion_votes.csv` gains a `box_age` column, and
`motion_votes_eval.py --min-box-age` restricts the sweep the same way.

### Run 4 (2026-10-14, `motion_20261014_04`: + box chain age 30) — current defaults

```
truth / verdict   unknown   static   moving
moving                267       59      413     <- 56 % of moving box-frames masked
static              1706     2675     1887     <- 30 % of static box-frames masked
census, voted rows (n_kp >= 15, box_age >= 30): moving n=72, static n=2502
  box/bg inlier ratio   moving: p10 0.00 p25 0.00 median 0.72 p75 3.32
                        static: p10 0.72 p25 1.15 median 2.02 p75 3.07
sweep, min_pts 15:  ratio 0.25 0.35/0.028   0.50 0.43/0.050   0.75 0.50/0.110
```

Progression of static box-frames wrongly masked: 81 % → 53 % → 62 % → **30 %**;
per-vote false alarm 7.7 % → 9.0 % → 15.4 % → **5.0 %**; per-vote recall
0.39 → 0.49 → 0.62 → **0.43**. The chain-age gate removed the novelty false
alarms the age floor had introduced, at the cost of the recall the age floor
had bought: a mover's box is only judged after a second in view, and by then
ORB has often re-matched it to young points that pass the floor.

**Where this leaves ORB-SLAM3 motion mode.** Against class mode (which masks
100 % of static and 100 % of detected moving boxes), run 4 keeps 70 % of the
static landmarks class mode throws away while still masking 56 % of the moving
box-frames. The per-vote discrimination is mediocre (0.43 / 5.0 %) and the
remaining static masking is inherited stickiness between re-vote probations.
VINS's census is far cleaner (§10 of its record: 0.62 / 0.8 %) because a solved
sliding-window depth is a stronger consistency test than a one-frame inlier
flag. Whether 30 %-of-static / 56 %-of-moving beats 100 % / 100 % for the
trajectory is exactly what the three-way A/B measures; it is not decidable
from the census.

Stopped iterating here (four runs). If it is resumed, the next lever is the
census itself, not the thresholds: an ORB analogue of VINS's criterion would
be the map point's reprojection residual over its *last N keyframe
observations* rather than this frame's inlier flag.

### Defaults as recorded

`filter_mode class` · `motion_iou 0.3` · `motion_min_points 15` ·
`motion_inlier_ratio 0.5` · `motion_unknown_mask false` ·
`motion_revote_frames 15` · `motion_min_point_age 30` (ORB only) ·
`motion_min_box_age 30`. Runs: `output/output_orb_yolo/simulation/
dataset_allsensor_realistic_003/motion_20261014_0{2,3,4}` and
`output/output_vins_yolo/.../motion_20261014_0{1,4}`, each with
`motion_votes.csv` and `calibration.txt`.
