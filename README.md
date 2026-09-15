# wil_project

A ROS 2 Humble workspace for comparing two visual-inertial SLAM systems —
**VINS-Fusion** and **ORB-SLAM3** — on the same stereo-inertial data, in a Gazebo
warehouse simulation and on a real RealSense rig.

Both estimators write an identically-shaped `vio.csv`, so a single set of plotting
scripts can put them on one axis against the simulator's ground truth.

## Layout

| Path | What it is |
| :-- | :-- |
| [`aws-robomaker-small-warehouse-world/`](aws-robomaker-small-warehouse-world/) | **submodule** (`gz-sim-fortress`) — the warehouse world, static and dynamic variants, nav2 params |
| [`vins_fusion_ros2/`](vins_fusion_ros2/) | **submodule** (`wil`) — VINS-Fusion port to ROS 2, with `wil` / `wil_sim` configs |
| [`orbslam3_ros2/`](orbslam3_ros2/) | ORB-SLAM3 stereo + stereo-inertial node, written for this workspace. See its [README](orbslam3_ros2/README.md) |
| [`realsense/realsense_imu/`](realsense/realsense_imu/) | IMU publisher for the real D435i rig |
| [`script/`](script/) | world authoring, trajectory extraction, plotting and comparison tools |
| [`output/`](output/) | committed results — `vio.csv` per run, plus comparison plots |
| `thirdparty/` | *not committed* — ORB-SLAM3 (our fork, `wil`) and Pangolin sources, built by `build_orbslam3.sh` |
| `dataset/` | *not committed* — recorded rosbags, see [Data](#data) |

Tools:

- [`build_orbslam3.sh`](build_orbslam3.sh) — build the ORB-SLAM3 core and Pangolin out of tree
- [`build_progress.sh`](build_progress.sh) — read-only progress check on that build
- [`script/viewer.py`](script/viewer.py) — 2D top-down viewer for the warehouse sim; also drives
- [`script/drive.py`](script/drive.py) — car-style keyboard driving for the ackermann robot
- [`script/bake_map.py`](script/bake_map.py) — bake a nav2 occupancy map from the world's collision meshes

## First-time setup

Clone **with submodules** — the two package repos are not vendored:

```bash
git clone --recurse-submodules <this-repo-url> wil_project
cd wil_project
```

Already cloned without them? `git submodule update --init --recursive`

Build the ORB-SLAM3 core first. It is not a colcon package; it is fetched and built
out of tree into `thirdparty/` by a script that pins ORB-SLAM3 to commit `5c2b00f` of
[our fork's `wil` branch](https://github.com/BoltonAthitDavies/ORB_SLAM3/tree/wil) --
which carries the C++17 / Pangolin-0.9 build patches -- and Pangolin to `v0.9.1`, so
the build reproduces months from now and does not depend on upstream:

```bash
./build_orbslam3.sh          # ~1 GB, needs 1.5 GB free; ./build_progress.sh -w to watch
```

Then the workspace itself:

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## Running

No launch file in this workspace ever plays a bag for you — start the estimator in
one terminal and the data source in another. Per-estimator usage lives in the
package READMEs; the shortest path on the sim bag is:

```bash
# terminal 1
ros2 launch orbslam3_ros2 wil_sim_stereo_imu.launch.py
# terminal 2
ros2 bag play dataset/sim --clock --rate 0.5
```

Against the live simulator instead of a bag:

```bash
ros2 launch aws_robomaker_small_warehouse_world small_warehouse.launch.py headless:=True
python3 script/viewer.py     # watch and drive
```

Then compare the two estimators on a finished run:

```bash
python3 script/plot_compare.py       # writes into output/compare/
```

## Data

Recorded rosbags are **not in this repository** — the two sim bags alone are 8.2 GB.
`dataset/` is gitignored apart from [`rviz_config.rviz`](dataset/rviz_config.rviz).
Runs referenced by `output/` expect bags at:

```
dataset/sim/                          sim, static warehouse
dataset/sim_dynamic_warehouse/        sim, moving obstacles
dataset/rosbag_realsense_imu_cambaseline95mm*/   real rig, 95 mm stereo baseline
```

The sim bags can be re-recorded by running the simulator and `ros2 bag record`;
the real-rig bags cannot, and are kept outside the repo.

## Submodules

Both submodules track a branch rather than `main`, recorded in
[`.gitmodules`](.gitmodules): `vins_fusion_ros2` → `wil`,
`aws-robomaker-small-warehouse-world` → `gz-sim-fortress`.

```bash
git submodule update --remote      # pull each submodule's tracked branch
git add <submodule-path> && git commit   # record the new pin in the superproject
```
