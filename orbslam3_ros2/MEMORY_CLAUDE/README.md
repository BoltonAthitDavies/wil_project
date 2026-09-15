# YOLO dynamic-object filtering in ORB-SLAM3 — engineering record

Working notes kept for the project report. Records WHY each decision was made,
what was measured, and what went wrong — the code already says what it does.
Companion files: **[integration.md](integration.md)** — how ORB-SLAM3 was built and
wrapped in the first place, and where every calibration number in `config/` comes
from (sim and real, stereo extrinsics, IMU noise, the fisheye mapping, the four
fork patches, and the bugs found along the way); and
`vins_fusion_ros2/MEMORY_CLAUDE/README.md` for the VINS side of the YOLO work.

Date: 2026-09-02/03. Reference paper: *RY-SLAM: A robust YOLO-based real-time
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
