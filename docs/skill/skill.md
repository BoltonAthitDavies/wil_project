---
name: wil-thesis-evaluation
description: Design, instrument, analyze, and document reproducible experiments and LaTeX advisor progress reports for WiL semantic visual-SLAM, comparing ORB-SLAM3 and VINS/RTAB-Map in static, dynamic, and real warehouse conditions. Use when preparing data-collection scripts, evaluation plots, result tables, diagrams, or preliminary methodology and results updates for this workspace.
---

# WiL Thesis Evaluation

## 1. Research context

Support the thesis **Vision-Based Navigation Pipeline for AMRs in Dynamic
Warehouse**. The experimental comparison concerns two principal pipelines:

1. ORB-SLAM3 stereo-inertial SLAM, optionally with YOLO dynamic-feature masking.
2. VINS-Fusion stereo-inertial odometry, optionally with the same YOLO masking,
   with RTAB-Map used when loop closure, mapping, or long-term localization is
   under study.

The scientific objective is to determine how pipeline choice, dynamic-object
filtering, environment dynamics, floor texture, illumination, and compute load
affect localization accuracy, map quality, robustness, and real-time performance.
Do not reduce the work to producing attractive plots: each result must answer a
defined research question under controlled conditions.

### 1.1 Future research direction: integrated localization and mapping

At the current reporting stage, SLAM performance evaluation and map
reconstruction operate separately. The user's future interest is to integrate
them into one localization-and-mapping pipeline that can improve the robot's
navigation state and map. **Path planning is outside this scope.**

The planned topics are:

- relocalization against a previously built map after startup or tracking loss;
- global pose-graph optimization using loop-closure constraints;
- alignment and merging of maps or sessions into a consistent representation;
- evaluation of whether the integrated estimate and reconstructed map provide a
  more reliable navigation input than the current separate processes.

This direction is **planned and not yet fully specified or implemented**. Do not
describe it as a completed contribution, choose an architecture prematurely, or
assume that integration will improve performance. Before proposing an
implementation or experiment, read and analyze:

`/home/ambushee/wil_project/docs/references/My next plan is Relocalization  Global Optimization and Map Merging VSLAM.pdf`

Use that paper as the starting reference, then map its assumptions, state
representation, data flow, optimization variables, map-merging conditions, and
evaluation method onto the actual ORB-SLAM3, VINS-Fusion, and RTAB-Map interfaces
in this repository. Identify conceptual gaps and ask for decisions when multiple
architectures remain plausible.

## 2. Authoritative project sources

Before designing or interpreting an experiment, read only the sources relevant
to that experiment:

- Thesis and proposal: `/home/ambushee/wil_project/docs/MyThesisInfo/`
- Main literature: `/home/ambushee/wil_project/docs/references/`
- Desired performance figures and tables:
  `/home/ambushee/wil_project/docs/skill/wil result collection info.pdf`
- Advisor note on current limitations and non-planning map benefits:
  `/home/ambushee/wil_project/docs/report/project_limitations_and_map_benefits.md`
- Workspace overview: `/home/ambushee/wil_project/README.md`
- Engineering records:
  - `vins_fusion_ros2/MEMORY_CLAUDE/README.md`
  - `orbslam3_ros2/MEMORY_CLAUDE/README.md`
  - `aws-robomaker-small-warehouse-world/MEMORY_CLAUDE/README.md`
  - `rtabmap_ros/MEMORY_CLAUDE/README.md`
- Existing evaluation tools: `/home/ambushee/wil_project/script/`

Treat the thesis proposal as intended methodology, not proof that an experiment
was completed. Treat engineering-memory entries as dated evidence: preserve
their caveats, dataset identity, invalid comparisons, and retractions. Current
code, configuration, raw data, and generated manifests take precedence when they
demonstrate that an older note is stale.

The existing Chapter 3 in the major report was written as a **planned method
before implementation**. Use it only for the original intent and research scope.
Do not copy planned components, procedures, parameters, or capabilities into the
minor report as though they were implemented. Reconstruct the final methodology
from repository evidence and explicitly identify material differences between
the plan and the implemented system.

## 3. Data and environment registry

### 3.1 Storage locations

- Flash drive root: `/media/ambushee/`
- Real-world datasets: `/media/ambushee/`
- Simulation datasets on flash drive: `/media/ambushee/dataset/`
- Workspace datasets: `/home/ambushee/wil_project/dataset/`
- Raw estimator results: `/home/ambushee/wil_project/output/output_*`
- Offline YOLO evaluation: `/home/ambushee/wil_project/output/yolo_eval/`
- Analysis products: `/home/ambushee/wil_project/output/compare/`
- Minor scientific reports: `/home/ambushee/wil_project/docs/report/`

### 3.2 Known dataset status

The following datasets are broken and must not be selected for experiments:

- `dataset_real_001`
- `dataset_real_002`

The other named datasets are located in
`/home/ambushee/wil_project/dataset/`, including:

- `dataset_dynamic_nofloortexture_01_000`
- `dataset_static_nofloortexture_001`
- `dataset_real_000`

Do not mark these workspace copies as broken. Before preparing a run, report the
resolved absolute path and inspect its topic inventory, duration, message counts,
and storage health. Never silently replace a requested dataset with another run.

### 3.3 World naming

- `aws-robomaker-small-warehouse-world/worlds/small_warehouse` is the original
  **tiny** map.
- `worlds/small_warehouse_dynamic` and `worlds/small_warehouse_static` are the
  rescaled **large** warehouse families, **with one exception**:
  `worlds/small_warehouse_static/small_warehouse_static_tinymap.world` carries
  tiny geometry despite living in the static family directory. It is the world the
  RTAB-Map dense map (`map/map_tiny.db`) was built and evaluated against. The
  directory name does not determine the scale; the file does.

Always record the exact `.world` file, not only “small warehouse.” Do not compare
metrics across different world variants as if they were repeated trials. In
particular, tiny-map and large-map results are never repeats of one condition:
they differ in extent, aisle width and route length.

## 4. Scientific experimental design

For every requested result, define the experiment before changing a script.
Record:

1. Research question and falsifiable hypothesis.
2. Independent variable: change one primary factor at a time.
3. Dependent metrics and their units.
4. Controlled variables: dataset/bag, world, trajectory, estimator configuration,
   calibration, image transport, replay rate, hardware, software revision, and
   alignment method.
5. Experimental unit and number of valid repeated runs (`n`).
6. Inclusion, exclusion, initialization, and failure criteria decided before
   examining the result.
7. Expected raw files, plots, tables, and logs.

Prefer paired comparisons: run both treatments on the same recorded sensor data.
Do not run competing estimators simultaneously when resource contention can alter
frame or IMU delivery. A comparison is invalid when overlap duration, processed
frame count, initialization state, or input data differs materially; report the
run as invalid rather than presenting its metric as a result.

Keep these factors distinct unless the experiment explicitly studies their
interaction:

- estimator: ORB-SLAM3 vs VINS-Fusion;
- back end: native/no RTAB-Map vs RTAB-Map;
- dynamic filter: off vs on;
- environment: static vs dynamic;
- floor: textured vs featureless;
- domain: simulation vs real robot;
- illumination, replay rate, and compute platform.

Never mix measurements from retired and current datasets in one aggregate.

## 5. Required provenance and quality checks

Each run must be traceable through a manifest or structured log containing:

- experiment ID, timestamp, and run status;
- absolute bag and world paths;
- pipeline and filter state;
- complete launch/run command;
- relevant ROS parameters and calibration paths;
- git commit for the superproject and affected submodules;
- hardware, OS/ROS, detector weight, and important dependency versions;
- replay rate, use of `/clock`, initialization interval, and shutdown method;
- input message counts and output pose/keyframe counts;
- warnings, dropped data, estimator resets, tracking loss, and exclusion reason.

Check timestamps and frames before calculating metrics. In particular:

- Some simulation bags already record `/clock`. RTAB-Map experiments must not add
  a conflicting player clock; inspect the bag before deciding whether to use
  `ros2 bag play --clock`.
- Compare only the common ground-truth/estimate time interval and report its
  duration and number of matched samples.
- Validate every calibration transform as rigid: orthonormal rotation and
  determinant +1. A plausible translation does not validate the rotation.
- Confirm camera order, coordinate frames, time basis, and metric scale.
- Inspect a small visual overlay before trusting an aggregate map or detection
  score.

Never overwrite a valid prior run. Use a unique output directory, and preserve
failed runs with an explicit status when they contain diagnostic evidence.

## 6. Output artifacts and metrics

Current primary artifacts are:

| Artifact | Meaning | Typical consumer |
|---|---|---|
| `vio.csv` | Timestamped estimated trajectory | trajectory metrics and plots |
| `yolo_mask.csv` | Masks actually used by the tracker | filtering coverage and timing analysis |
| `output/yolo_eval/*/yolo_eval.csv` | YOLO predictions matched against projected Gazebo ground truth | precision, recall, IoU, and moving/static analysis |
| `output/yolo_eval/*/yolo_detections.csv` | Per-detection offline results | threshold, class, and error analysis |
| `output/yolo_eval/*/overlay/` | Visual GT/prediction overlays | qualitative validation of projections and matching |
| `map_tiny.db` or another `.db` | RTAB-Map graph/database | map, graph, and localization analysis |

Extend the scripts to record any missing variables required by the stated
hypothesis. Do not infer unavailable timing, CPU, dropped-frame, covariance, or
keyframe data from unrelated logs.

### 6.1 Localization and robustness

Report at minimum when valid ground truth exists:

- ATE RMSE after a stated timestamp association and rigid SE(3) alignment;
- RPE translation and rotation at a stated time or distance interval;
- matched duration, matched sample count, trajectory completeness, and alignment
  method;
- initialization time, tracking-loss count/duration, resets, and successful-run
  fraction.

Stereo-inertial trajectories are metric-scale; do not apply scale correction
unless the experiment explicitly justifies it. Do not report ATE for real data
without a valid independent ground-truth source.

### 6.2 Object detection and dynamic filtering

When Gazebo ground truth or annotated data is available, report:

- IoU threshold and confidence threshold;
- TP, FP, FN, precision, recall, F1, AP when supported, and moving-object recall;
- mask coverage, matched/missed detector frames, detection age, rejected masks,
  and tracked features removed;
- separate moving and static-object results where ground truth permits.

State that the current model produces bounding boxes rather than segmentation
masks and that semantic class membership is not proof of motion.

### 6.3 Map quality and lifelong mapping

Read `docs/report/project_limitations_and_map_benefits.md` when summarizing
the role of the map. Preserve its status labels so demonstrated benefits are not
combined with future possibilities.

Report the coordinate placement and observed region used for comparison, then
measure as applicable:

- occupied-cell precision, recall, and F1 at a stated spatial tolerance;
- Chamfer-distance summaries and map coverage;
- false obstacles under the true robot footprint and clearance statistics;
- RTAB-Map node, neighbor-link, loop-closure, local-space-link, and gravity-link
  counts;
- relocalization success, time to first fix, pose sanity bounds, database growth,
  and memory behavior.

Localization covariance alone is not a correctness criterion. Require a spatial
sanity check against the known environment or independent ground truth.

### 6.4 Real-time and computational performance

When the necessary instrumentation exists, reproduce the figure families defined
in `wil result collection info.pdf`:

1. Per-module latency distributions, separated by pipeline, hardware, and input
   rate.
2. Capacity tables containing input frequency, achieved throughput, and duration
   as mean ± standard deviation.
3. CPU usage over time, average CPU usage with uncertainty, and processed/dropped
   frame and keyframe counts.
4. Covisibility graph diagrams with clearly defined nodes and edges.
5. Hardware-stress heatmaps showing the metric difference between platforms over
   datasets and input rates.

Use consistent units and axes. Show raw run values or distributions alongside
summaries. Report `n`, mean and standard deviation for approximately symmetric
repeated measurements; otherwise use median and IQR. Add confidence intervals or
effect sizes when the number and design of repeats support them. Never fabricate
uncertainty from a single run.

## 7. Working protocol with the user

This is a supervised, manually executed workflow.

1. Inspect existing source, metadata, and small text outputs as needed.
2. Explain the proposed experiment, controls, logging additions, expected outputs,
   and validity checks.
3. Create or modify the required collection/analysis script under `script/`.
4. **Do not execute the created or modified collection/analysis script.** The user
   runs it to avoid wasting compute and interaction budget.
5. Give exact copy-paste commands for every terminal, in start order, including
   environment setup and absolute output paths.
6. State what files and console signals should appear, how long the run should
   continue, and the criterion for stopping it.
7. Stop and wait for the user to return the artifacts or logs. Analyze only after
   confirming that the run passed the predefined validity checks.

Do not launch Gazebo, play a rosbag, start an estimator, or run a full evaluation
unless the user explicitly overrides this protocol for that run.

## 8. Command-selection rules

Choose exact world, dataset, configuration, filter state, and output directory
from the experimental design; never leave `{YOU CHOOSE}` in the final command.
Use the following command families as templates, after sourcing ROS 2 Humble and
the workspace install.

### 8.1 Simulation or bag source

```bash
ros2 launch aws_robomaker_small_warehouse_world <selected-launch>.launch.py \
  headless:=False bridge_model_poses:=True

ros2 bag play <absolute-bag-path> <validated-clock-and-rate-options>
```

### 8.2 Viewer

```bash
python3 /home/ambushee/wil_project/script/viewer.py --rotate 180 \
  --world <absolute-world-file> --live-poses --map ros \
  --map-pose <validated-x,y,yaw>
```

Do not assume `1.8,9.0,-90` fits every database. Use a measured fixed placement
when placing an RTAB-Map grid in the Gazebo-world viewer.

### 8.3 ORB-SLAM3

```bash
# Simulation
ros2 launch orbslam3_ros2 wil_sim_stereo_imu.launch.py \
  output_path:=/home/ambushee/wil_project/output/output_orb/simulation/<run-id>

# Dynamic-filter treatment
ros2 launch orbslam3_ros2 wil_sim_stereo_imu.launch.py filter:=true \
  output_path:=/home/ambushee/wil_project/output/output_orb_yolo/simulation/<run-id>

# Real rig
ros2 launch orbslam3_ros2 wil_stereo_imu.launch.py \
  output_path:=/home/ambushee/wil_project/output/output_orb/real/<run-id>
```

### 8.4 VINS-Fusion

```bash
# Simulation baseline
ros2 run vins_fusion_ros2 vins_fusion_ros2_node --ros-args \
  -p config_file:=/home/ambushee/wil_project/vins_fusion_ros2/config/wil_sim/stereo_imu.yaml \
  -p output_path:=/home/ambushee/wil_project/output/output_vins/simulation/<run-id> \
  -p use_sim_time:=true

# Simulation dynamic-filter treatment
ros2 run vins_fusion_ros2 vins_fusion_ros2_node --ros-args \
  -p config_file:=/home/ambushee/wil_project/vins_fusion_ros2/config/wil_sim/stereo_imu.yaml \
  -p output_path:=/home/ambushee/wil_project/output/output_vins_yolo/simulation/<run-id> \
  -p use_sim_time:=true -p filter:=true

# Real rig
ros2 run vins_fusion_ros2 vins_fusion_ros2_node --ros-args \
  -p config_file:=/home/ambushee/wil_project/vins_fusion_ros2/config/wil/stereo_imu.yaml \
  -p output_path:=/home/ambushee/wil_project/output/output_vins/real/<run-id>
```

VINS requires raw images. If a bag contains only compressed images, include the
two explicit compressed-to-raw republish terminals documented in the VINS
engineering record.

### 8.5 RTAB-Map

```bash
ros2 launch /home/ambushee/wil_project/rtabmap_ros/rtab_sim.launch.py \
  rviz:=true grid_range_max:=10.0 database_path:=<absolute-db-path> \
  detection_rate:=4.0 shim_rate:=4.0
```

Use `grid_range_max:=10.0` as the current measured default unless the experiment
is explicitly a range sweep. Select `construct`, `extend`, or `localize` mode and
front-end according to the RTAB-Map engineering record. Never delete or replace a
map database without an explicit experiment-specific instruction and backup.

## 9. Analysis products

Write plots, tables, machine-readable summaries, and validation images beneath:

```text
/home/ambushee/wil_project/output/compare/<domain>/<experiment-id>/
```

Every figure must have a descriptive filename, labeled axes, units, sample count,
legend, and caption-ready explanation. Every table must define its population,
aggregation, and missing/failed runs. Save the numerical data used to render each
figure so the plot is auditable.

### 9.1 LaTeX report artifacts

Write report prose as LaTeX source, not Markdown. When an existing thesis or
faculty LaTeX template is available, preserve its document class, package choices,
bibliography system, naming conventions, and chapter inclusion structure. If no
template exists, make a requested full report self-contained and compilable; for
a requested chapter alone, create an includeable `.tex` chapter and clearly state
which packages or commands its parent document requires.

Use this default organization unless the repository already establishes another:

```text
docs/report/minor_report/
|-- main.tex
|-- chapters/
|   |-- chapter_3_methodology.tex
|   `-- chapter_4_results.tex
|-- figures/
|-- tables/
`-- references.bib
```

Use native LaTeX structures: `\chapter`/`\section`, `table` with `booktabs`,
`figure`, `equation`, `\label`, and `\ref` or `\autoref`. Escape repository paths,
underscores, percent signs, and other LaTeX-special characters correctly. Give
every figure and table a descriptive caption and stable label. Place literature
in the selected bibliography system; identify repository evidence in prose,
footnotes, or a dedicated implementation-evidence table rather than inventing
bibliographic entries for source files.

When a diagram would materially clarify a design, create it as part of the LaTeX
deliverable. Good candidates include the system architecture, sensor/data flow,
semantic-mask insertion points, coordinate-frame tree, RTAB-Map correction path,
and controlled-experiment layout. Prefer reproducible vector diagrams made with
TikZ/PGF (or PGFPlots for plotted data) so labels and typography match the report.
Use a conventional block diagram or flowchart with an explicit legend when line
styles or colors carry meaning. A diagram must reflect the implemented repository
interfaces and must visually distinguish implemented, external, blocked, and
planned components when those statuses coexist. Do not draw a planned feedback
connection as though it currently exists. If TikZ would make a simple relationship
less readable, use a repository-owned vector PDF instead and retain its editable
source.

Compile or syntax-check changed LaTeX when a compatible local toolchain is
available. Report compilation warnings that affect correctness, cross-references,
citations, or figure placement. Do not silently change scientific content merely
to make compilation succeed.

## 10. Minor scientific progress report

Write the minor report as LaTeX source beneath
`/home/ambushee/wil_project/docs/report/`, following Section 9.1. Its
primary purpose is to update
the advisor on the current state of the work. It is **not the final project
report**, does not need to demonstrate that the proposed system succeeded, and
must not present preliminary evidence as a final conclusion. It may later inform
the major report, but its immediate value is an accurate record of implementation
progress, measured behavior, unresolved problems, and the next research steps.

Use a clear reporting cutoff date. At that cutoff, distinguish components and
experiments as **implemented**, **verified**, **partially verified**, **blocked**,
**failed**, or **planned**. The report must contain these two main chapters:

### Chapter 3: Methodology

Describe **what was actually implemented by the reporting cutoff**, not what was
originally planned or what is expected to work later.
Build this chapter from the current repository, configurations, launch files,
scripts, generated artifacts, and dated engineering records. Use the major
report's existing Chapter 3 only to explain the original intent or a
planned-versus-implemented deviation.

Recommended Chapter 3 structure:

1. `3.1` Progress scope, reporting cutoff, and implemented system overview
2. `3.2` Hardware, software, sensors, calibration, and coordinate frames
3. `3.3` Warehouse simulation, real environment, datasets, and controlled factors
4. `3.4` ORB-SLAM3 pipeline and implemented dynamic-feature filtering
5. `3.5` VINS-Fusion pipeline, persistent-feature filtering, and RTAB-Map back end
6. `3.6` YOLO model, detector integration, mask construction, and synchronization
7. `3.7` Data-collection protocol, run validity checks, and stored artifacts
8. `3.8` Evaluation metrics, timestamp association, trajectory alignment, map
   comparison, and computational measurements
9. `3.9` Reproducibility controls and planned-versus-implemented deviations

For implementation claims, cite precise repository evidence such as a source or
configuration path, parameter name, algorithmic insertion point, or commit. Keep
literature citations for the scientific origin of methods, while repository
evidence establishes how this project implemented them. Do not describe an
unimplemented proposal feature in the present tense.

### Chapter 4: Experiments and Research Results

Present the experiments attempted so far, validated measurements, failed runs,
and appropriately cautious interpretation. Weak, negative, or disappointing
results are valid progress-report findings and must be shown honestly; do not
select, soften, or omit them to make the system appear successful. At the same
time, an expectation that results will be poor is not evidence: conclusions must
still come from measured artifacts.

Use validated results for numerical comparisons. Keep invalid or retracted
measurements out of headline comparison tables, but document them in a clearly
labeled diagnostic or excluded-results subsection when they explain a discovered
bug, experimental limitation, or methodological correction.

Recommended Chapter 4 structure, including only subsections supported by actual
data:

1. `4.1` Experiment progress matrix: attempted, completed, valid, failed, blocked,
   and remaining runs
2. `4.2` Dynamic-object detector and mask-filtering results
3. `4.3` Localization accuracy and robustness results: ATE, RPE, completeness,
   initialization, and tracking failures
4. `4.4` Current separate mapping/SLAM results and any completed loop-closure,
   relocalization, or lifelong-mapping experiments
5. `4.5` Real-time performance, latency, throughput, CPU, memory, frames, and
   keyframes
6. `4.6` Simulation-to-real or integrated-system results, when valid reference
   data exists
7. `4.7` Cross-experiment discussion, comparison with literature, threats to
   validity, limitations, and implications for the thesis objectives
8. `4.8` Current conclusions, unresolved questions, corrective actions, and next
   experiments

Each experiment subsection should follow the same internal order: question and
hypothesis; conditions and controls; validity/sample count; numerical results;
figures/tables; interpretation; limitations. This keeps method choices out of the
results narrative except where a result requires clarification.

Make the advisor update easy to act on. State explicitly:

- what changed or was implemented since the previous update;
- what evidence now exists and how strong it is;
- which outcomes were poor, failed, or inconclusive;
- confirmed root causes versus untested explanations;
- blockers, technical debt, and threats to validity;
- the next experiment or engineering action, its purpose, and any decision or
  guidance needed from the advisor.

List integrated relocalization, global optimization, and map merging as a future
direction until repository code and validated experiments demonstrate otherwise.
Explain that its intended contribution is a unified navigation-state and map
input, while path planning itself is excluded.

Separate observations from interpretations. Use cautious causal language unless
the experiment isolates the cause. Report negative, null, failed, and excluded
runs transparently. Do not reuse a numerical result without its dataset, units,
conditions, sample count, and uncertainty. Cite primary literature for theoretical
claims and project artifacts for local measurements.
