---
name: wil-thesis-evaluation
description: Design, instrument, analyze, and document reproducible experiments and the LaTeX final project report for WiL semantic visual-SLAM, comparing ORB-SLAM3 and VINS/RTAB-Map in static, dynamic, and real warehouse conditions across four fixed experiments run in simulation and on the real robot. Use when preparing data-collection scripts, evaluation plots, result tables, diagrams, or writing chapters 3 to 5 of the final report for this workspace.
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

The work is organised as **exactly four experiments**, fixed by the 23 September
2026 progress update (`docs/MyThesisInfo/WiL Update Progress-1.pdf`), each located
on a numbered block of the system-architecture diagram and each run in **two
domains, simulation and the real robot**, giving eight result slots in total:

| # | Experiment | Architecture block | Question it answers |
|---|---|---|---|
| 1 | Non-Visual Baseline and Estimator Comparison | #4 robot odometry / SLAM output | Do the visual pipelines genuinely outperform wheels, IMU, and a wheel+IMU EKF? |
| 2 | Offline YOLO Detector Performance | #2 object detection | How reliably does the detector find and name warehouse objects, independent of SLAM, and how much of the frame could masking remove? |
| 3 | Visual-Inertial Odometry With Dynamic Environment Filtering Evaluation | #1 SLAM front end | Does masking dynamic-object features improve localization, at what compute cost, and does tracking survive it? |
| 4 | Dense Map Reconstruction and Map-Quality Evaluation | #3 SLAM back end / map | Does the reconstructed map reflect the physical environment and is it usable by a route planner? |

Do not add a fifth experiment, merge two, or rename these. New evidence goes into
one of the eight slots. The project's stated targets are ATE $\le$ 0.05 m and
RPE($\Delta t$ = 1 s) $\le$ 1.5 m at a robot speed of 1.5 m/s; the real site is
the Gensurv building first floor, a 10 m by 7 m scope with test Zones A and B on
a 5-degree slope and scheduled lighting changes.

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
- **Storytelling reference** (chapter order, the four experiments, per-experiment
  internal order): `docs/MyThesisInfo/WiL Update Progress-1.pdf`
- **Content source for chapters 1--3**: `docs/MyThesisInfo/final_report.pdf`
  (Thai faculty template with the written chapters; figures in
  `docs/MyThesisInfo/figure/`). **Format source**: the Faculty of Engineering
  English template, `docs/MyThesisInfo/27.8.67 Template เล่มโครงงาน (Eng Ver.)/`
  (Section 9.1).
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

The existing Chapter 3 in `final_report.pdf` was written as a **planned method
before implementation**. Use it only for the original intent and research scope.
Do not copy planned components, procedures, parameters, or capabilities into the
final report as though they were implemented. Reconstruct the final methodology
from repository evidence and explicitly identify material differences between
the plan and the implemented system. Chapters 1 and 2 of that document are
finished prose: port them, do not rewrite their science.

## 3. Data and environment registry

### 3.1 Storage locations

- Flash drive root: `/media/ambushee/`
- Real-world datasets: `/media/ambushee/`
- Simulation datasets on flash drive: `/media/ambushee/dataset/`
- Workspace datasets: `/home/ambushee/wil_project/dataset/`
- Raw estimator results: `/home/ambushee/wil_project/output/output_*`
- Offline YOLO evaluation: `/home/ambushee/wil_project/output/yolo_eval/`
- Analysis products: `/home/ambushee/wil_project/output/compare/`
- Final report LaTeX: `/home/ambushee/wil_project/docs/report/final_report/`

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

The deliverable is the **final project report**, not a progress report. Its
FORMAT is the Faculty of Engineering English template,
`docs/MyThesisInfo/27.8.67 Template เล่มโครงงาน (Eng Ver.)/` (manual:
`เอกสารประกอบ_template ENG.pdf`): outer cover with the KMUTT emblem and
all-capitals text, title/approval page ending "Copyright reserved", abstract
page opening with the Project Title / Credits / Candidate / Advisor / Program /
Field of Study / Faculty / Academic Year block, a THAI ABSTRACT page, ACKNOWLEDGEMENTS,
CONTENTS with a PAGE column and a CHAPTER group line ("1. INTRODUCTION"),
LIST OF TABLES / FIGURES with TABLE|FIGURE ... PAGE headers and "(Cont'd)"
running heads, LIST OF SYMBOLS (symbol, meaning, UNIT), LIST OF TECHNICAL
VOCABULARY AND ABBREVIATIONS, chapters "CHAPTER n TITLE" 15 pt / 14 / 13 pt
bold, Times 12 pt, margins 4/2/3/2 cm (left/right/top/bottom), page number
top right and none on a chapter's first page, "Table x.y" above and "Figure
x.y" below, REFERENCES in the faculty number system (Author, A.B., year,
"title," **Journal**, Vol., No., pp.) ordered by first citation, and a
CURRICULUM VITAE page. Its CONTENT (chapters 1-2) comes from the Thai template
`docs/MyThesisInfo/final_report.pdf`. Use this organization:

```text
docs/report/final_report/
|-- main.tex                      % Faculty of Engineering English format (above)
|-- frontmatter/
|   |-- cover.tex, approval.tex, abstract.tex, acknowledgements.tex, symbols.tex
|   `-- thai_abstract.fodt        % Thai abstract, typeset by LibreOffice -> .pdf
|-- chapters/
|   |-- chapter_1_introduction.tex   % English rendering of final_report.pdf ch. 1
|   |-- chapter_2_theory.tex         % English rendering of final_report.pdf ch. 2
|   |-- chapter_3_methodology.tex    % template hardware/site/procedure + fragments
|   |-- chapter_4_experiments.tex    % four experiments x two domains, eight slots
|   |-- chapter_5_conclusion.tex
|   |-- references.tex               % hand-written thebibliography, faculty format,
|   |                                % first-citation order (references.bib = record)
|   |-- curriculum_vitae.tex
|   `-- fragments/                   % VERBATIM cuts of the retired minor report,
|                                    % headers say the source line range; edit the
|                                    % source of a number, never the fragment
|-- figures/                      % minor-report figures + figures/template/ (PDF images,
|                                 % kmutt_emblem.png from the template's cover .docx)
`-- tables/                       % generated tables (block_performance_*, rpe_*, ...)
```

Build with `script/build_final_report.sh`: it (1) converts the Thai abstract
with `soffice --headless` (Norasi 16 pt stands in for AngsanaUPC; pdflatex has
no Thai font here, LuaLaTeX's font database is broken and `xelatex` is absent),
(2) runs `latexmk -pdf main.tex` (pdflatex + `mathptmx`), and (3) regenerates
`docs/report/concerns/concerns.tex` with `script/extract_concerns.py` and
builds it. `\flag`, `\moved`, `queried` and the eight
`\slotstatus{<label>}{...}` boxes are SUPPRESSED in the report and moved into
the concerns document (fragments expanded in place); keep the markup in the
source, it is the single record of what is queried. fancyhdr here is 3.x: page
styles must set every field themselves, so the ordinary style is `report`,
never `fancy`, and chapter openers use the empty `plain`. TOC chapter titles are
upper-cased through `\chapter[UPPER]{Title}`; the CHAPTER group line is emitted
from inside `chapter_1_introduction.tex`, not `main.tex`.

**Write the report in English.** Chapters 1 and 2 are translated from the Thai
template prose, keeping their content, citations and figure references; chapters
3--5 are written in English from repository evidence. The only Thai page is the
THAI ABSTRACT (`frontmatter/thai_abstract.fodt`, one page, Norasi 13 pt on a
0.58 cm pitch -- Norasi's body is far larger than AngsanaUPC's, so 16 pt ran to
three pages); the cover and approval pages carry no Thai, as in the English
template. Keep the Thai abstract's wording the author's: it was drafted by the
assistant and must be read as one's own abstract before submission.

`docs/report/minor_report/` is retired. Do not extend it; reuse its verified
figures, tables and generated files by copying or by `\input@path`, never by
re-deriving numbers.

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

## 10. Final project report

Write the final report as English LaTeX source beneath
`/home/ambushee/wil_project/docs/report/final_report/`, following Section 9.1.
It replaces the minor progress report. It is the document the project is
examined on, so it must be complete in structure from the first draft, and
**honest about which of its eight result slots are filled**: the user has not
finished collecting results, and an unfilled slot is reported as such, never
omitted, never padded, and never filled with an expectation of how it will go.

Use a clear reporting cutoff date. At that cutoff label every component and every
result slot as one of **completed**, **partial**, **untested**, **blocked**,
**failed**, or **planned**, and keep the label attached wherever the result is
quoted.

### Chapter 3: Methodology

Describe what was actually implemented by the cutoff, in the storytelling order
of the progress update, so that the four experiments arrive as the natural
consequence of the design rather than as a list:

1. `3.1` Problem, requirements and scope: the Gensurv site, the 10 m by 7 m test
   area, the 1.5 m/s speed, the ATE and RPE targets, the lighting-change
   conditions, and what is excluded (path planning).
2. `3.2` System architecture: the block diagram with its data rates (control
   20 Hz, pose 20 Hz, image 30 Hz simulated / 20 Hz real, bounding boxes
   30 Hz simulated / 10 Hz real, IMU 200 Hz, wheel odometry 20 Hz), with the four
   experiment blocks #1--#4 marked on it.
3. `3.3` The two pipelines as implemented: ORB-SLAM3 with YOLO feature rejection,
   and VINS-Fusion with persistent-track removal and the RTAB-Map back end, each
   with its parameter table (the tables on slides 11 and 13 of the progress
   update are the current values; verify against the config files).
4. `3.4` Simulation platform: sensor rig, platform geometry, obstacle and floor
   families, tiny and large maps, and the dataset registry.
5. `3.5` Real platform: AC-IMX390-H190 stereo cameras on the NRU-51V (Jetson
   Xavier NX), calibration status and its limits, Zones A and B, and the
   real-dataset registry including which recordings are unusable and why.
6. `3.6` The four experiments, one subsection each, in the fixed order of
   Section 1: objective, process, evaluation metrics, constraints, and the
   validity rule that decides whether a run counts. State here, once, that every
   experiment is run in both domains.
7. `3.7` Evaluation: timestamp association, rigid SE(3) alignment, ATE and RPE
   definitions, detector matching rules, map scoring, computational measurement,
   and the run-manifest provenance rules of Section 5.
8. `3.8` Planned-versus-implemented deviations from the template's original
   Chapter 3, as a table.

Cite precise repository evidence for every implementation claim. Do not describe
an unimplemented feature in the present tense.

### Chapter 4: Experiments and results

Exactly four sections, in the fixed order, and inside each **two domain
subsections** -- eight result slots:

```text
4.1 Experiment 1: Non-Visual Baseline and Estimator Comparison
    4.1.1 Simulation
    4.1.2 Real robot
4.2 Experiment 2: Offline YOLO Detector Performance
    4.2.1 Simulation
    4.2.2 Real robot
4.3 Experiment 3: Visual-Inertial Odometry With Dynamic Environment Filtering
    4.3.1 Simulation
    4.3.2 Real robot
4.4 Experiment 4: Dense Map Reconstruction and Map-Quality Evaluation
    4.4.1 Simulation
    4.4.2 Real robot
4.5 Cross-experiment discussion, simulation-to-real comparison, threats to
    validity, and limitations
```

Every slot follows the same internal order, which is the order of the progress
update's per-experiment slides: **status label; objective restated in one
sentence; process; dataset information (registry table); results as tables and
figures; analysis; summary; future plan.** A slot with no valid data carries its
status label, the reason (not yet recorded, recorded but invalid, blocked by
hardware or calibration), and the exact next action -- and nothing else. Do not
let an empty slot borrow the other domain's numbers, and do not compare the two
domains of an experiment until both slots hold valid results.

Status of the eight slots at the 25 September 2026 cutoff, which the report must
state and the user updates as data arrives:

| Experiment | Simulation | Real robot |
|---|---|---|
| 1 Baseline / estimators | completed, 5 `allsensor` bags, 4 worlds | planned |
| 2 YOLO detector | completed, 4 dynamic bags, 9,860 frames | planned |
| 3 VIO + dynamic filtering | **untested** -- VINS filter never fired, ORB confounded by frame loss; runs exist, hypothesis has no valid test | planned |
| 4 Dense map | **partial** -- tiny map scored (occupied-cell P 96.6 %, observed-surface R 89.5 %, whole-building R 67.2 %), large map failed through tracking loss with no artifact retained | planned |

"Untested" and "failed" are reported as findings, with cause where confirmed and
untested explanations labelled as such. Weak, negative, or disappointing results
are valid and must not be softened or omitted. Keep invalid or retracted
measurements out of headline tables but document them in a clearly labelled
diagnostic subsection when they explain a bug or a methodological correction.

### Chapter 5: Conclusions and recommendations

1. Conclusions per experiment, each tied to its research question and to the
   ATE / RPE targets of Section 1, stated only as strongly as the filled slots
   allow.
2. Limitations and threats to validity, carried over from `4.5` without
   softening.
3. Recommendations and next steps: the remaining plan items (integrated live test
   in both domains at 1.5 m/s under lighting change; metric collection; report
   completion), and the future direction of Section 1.1 -- integrated
   relocalization, global optimization and map merging -- listed as future until
   repository code and validated experiments show otherwise, with path planning
   excluded.

### Back matter

- **References**: hand-written `chapters/references.tex` in the faculty number
  system and format, listed in order of first citation (figure-caption citations
  count, and the List of Figures is read before the chapters); the template's
  two uncited entries come last. `references.bib` is kept only as the record.
  Add a primary source for every algorithm the text relies on.
- **No appendix.** The report carries none (user decision, 25 September 2026);
  file-level provenance of the numbers stays in the run manifests and the
  `output/compare/` trees, cited inline where a value is quoted.
- **Curriculum Vitae** (`chapters/curriculum_vitae.tex`, the template's layout:
  NAME, Date of Birth, EDUCATIONAL RECORD, SCHOLARSHIP/RESEARCH GRANT,
  EMPLOYMENT RECORD, PUBLICATION): name, degree and placement only; personal
  details are the author's to fill and are flagged, never invented.

Separate observations from interpretations. Use cautious causal language unless
the experiment isolates the cause. Do not reuse a numerical result without its
dataset, units, conditions, sample count, and uncertainty. Cite primary
literature for theoretical claims and project artifacts for local measurements.
