# wil_project in Docker

One image holds the whole stack the host runs today: Ubuntu 22.04, ROS 2
Humble, Gazebo Fortress, nav2, RTAB-Map core 0.23.x from apt with the fork's
five `rtabmap_ros` packages built from source, Ceres 2.2.0 built from source
for VINS-Fusion, ORB-SLAM3 (pinned fork) + Pangolin 0.9.1 built by
`build_orbslam3.sh`, the `KinematicTrajectory` gz-sim plugin, and CUDA
PyTorch + ultralytics for the YOLO detector node.

Everything lives at the **same paths as on the host** (`/home/ambushee/wil_project`,
user `ambushee`, uid 1000) because the launch files, VINS YAML configs and
scripts hard-code them. The commands in `wil_cmd_v2` therefore run unchanged
inside the container.

| Stays outside the image (bind-mounted) | Baked into the image |
|---|---|
| `dataset/`, `/media/ambushee` (bags, read-only flash) | source of all four workspace packages + `rtabmap_ros/` (5 packages) |
| `output/`, `map/`, `trash/`, `gif/` (results) | `install_docker/` colcon install of those packages |
| `weight/` (host copy overrides the baked `best.pt`) | `/opt/wil/thirdparty` (ORB-SLAM3, Pangolin, vocabulary), `/opt/wil/gz_plugins` |
| `.git`, `docs/`, host `build/ install/ log/ thirdparty/` | `docker/build_info.txt` (git HEAD + submodule pins at build time) |

Files:

| File | Role |
|---|---|
| `Dockerfile` | stages `deps` → `thirdparty` → `full` |
| `compose.yml` | services `wil` (baked) and `wil-dev` (host checkout mounted, profile `dev`) |
| `compose.gpu.yml` | NVIDIA GPU + GL override, needs nvidia-container-toolkit |
| `build.sh` | records provenance to `build_info.txt`, then `docker compose build` |
| `entrypoint.sh` | sources ROS + `install_docker/`, links `thirdparty/`, execs the command |
| `ws_build` | in-container colcon build into `build_docker/` + `install_docker/` |
| `requirements.txt` | pinned Python stack (torch cu121, ultralytics 8.4.138, numpy 1.26.4, …) |

## 1. Host prerequisites (state of this laptop on 2026-09-24)

### 1.1 Disk: the blocker

`/` (ext4, 67 GB) has **about 2 GB free**. The finished image is roughly
12 GB and the build needs ~25 GB free while intermediate layers and the
build cache exist. Docker's data root is `/var/lib/docker` on `/`, and the
other partitions are NTFS (Windows), which overlay2 cannot use; the flash
drive is not ext4 either.

Options, pick one:

1. **Free ≥ 25 GB on `/`.** Known reclaimable now (≈ 8 GB): the unused
   `ghcr.io/fixposition/fixposition-sdk` image (3.5 GB, `docker rmi …`),
   `~/Downloads` (2.4 GB), host `build/` (0.9 GB, `colcon build` recreates it),
   `~/.cache` (0.55 GB), `trash/` (0.24 GB, check first), apt cache (0.16 GB,
   `sudo apt clean`). That is not enough on its own; shrinking a Windows
   partition from Windows Disk Management and growing `/` with GParted is the
   realistic route on this machine.
2. **Attach an ext4 SSD and move Docker's data root** to it:
   ```bash
   sudo mkdir -p /mnt/ssd/docker
   printf '{\n  "data-root": "/mnt/ssd/docker"\n}\n' | sudo tee /etc/docker/daemon.json
   sudo systemctl restart docker && docker info | grep 'Docker Root Dir'
   ```
3. **Build on another machine** (lab PC). The context is only ~115 MB; copy
   the checkout (with submodules) there, and use the same commands.

### 1.2 Docker plugins

The Ubuntu `docker.io` package (29.1.3) ships without buildx and compose;
the Dockerfile needs BuildKit (heredocs, `COPY --chmod`). Both plugins are in
the Ubuntu archive:

```bash
sudo apt update && sudo apt install docker-buildx docker-compose-v2
docker buildx version && docker compose version
```

### 1.3 NVIDIA container toolkit (GPU for YOLO and for Gazebo/rviz rendering)

Not installed yet and no NVIDIA apt source is configured. Driver 595 on the
host is new enough for the cu121 wheels.

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi   # smoke test
```

Without the toolkit, drop `-f docker/compose.gpu.yml` from every command
below: the detector node falls back to CPU automatically, and Gazebo/rviz use
Mesa through `/dev/dri` (add `LIBGL_ALWAYS_SOFTWARE=1` if the Intel iGPU
path fails).

## 2. Build

From the repository root, with submodules checked out:

```bash
git submodule update --init --recursive
./docker/build.sh                # ~45-90 min first time; ORB-SLAM3 and Ceres compile from source
```

`build.sh` writes `docker/build_info.txt` (HEAD, describe, submodule pins,
dirty files) and passes the describe string as the image label
`org.opencontainers.image.revision`. Rebuild after changing source with the
same command; Docker reuses the `deps` and `thirdparty` layers.

Memory: `build_orbslam3.sh` caps at `-j8` and retries at `-j2`/`-j1` when
the compiler is OOM-killed, exactly as on the host.

## 3. Run

One long-lived container; one `exec` shell per terminal of the protocol.

```bash
xhost +local:                                            # once per X session, lets the container open windows
docker compose -f docker/compose.yml -f docker/compose.gpu.yml up -d wil
docker compose -f docker/compose.yml exec wil bash      # repeat in every terminal
```

Inside each shell the environment is already sourced (`~/.bashrc`), the
working directory is `/home/ambushee/wil_project`, and the usual commands
apply verbatim, e.g.

```bash
# terminal 1
ros2 launch aws_robomaker_small_warehouse_world small_warehouse_static_tinymap.launch.py headless:=True bridge_model_poses:=True
# terminal 2
python3 script/viewer.py --rotate 180 --world /home/ambushee/wil_project/aws-robomaker-small-warehouse-world/worlds/small_warehouse_static/small_warehouse_static_01.world --live-poses
# terminal 3
ros2 launch orbslam3_ros2 wil_sim_stereo_imu.launch.py output_path:=/home/ambushee/wil_project/output/output_orb/simulation/<run-id> filter:=true
```

Results land in the host's `output/`, `map/`, `trash/` through the bind
mounts, owned by uid 1000. Stop with
`docker compose -f docker/compose.yml down`.

Wayland sessions: `xhost +local:` still works through XWayland; if a window
does not appear, check `echo $DISPLAY` on the host and pass it explicitly
(`DISPLAY=:1 docker compose … up -d wil`).

## 4. Developing against the host checkout

`wil-dev` mounts the entire repository over the workspace path so edits on
the host are visible immediately. Builds go to `build_docker/` and
`install_docker/` (gitignored) so the host's own `build/`/`install/` stay
untouched; those link against host-only paths (`~/local/lib/libceres.so.4`)
and would not load in the container.

```bash
docker compose -f docker/compose.yml -f docker/compose.gpu.yml --profile dev up -d wil-dev
docker compose -f docker/compose.yml --profile dev exec wil-dev bash
ws_build                                  # full workspace, symlink install
ws_build --packages-select orbslam3_ros2  # one package
```

If the host has no `thirdparty/` (fresh clone), the entrypoint links it to
the image's `/opt/wil/thirdparty`. If the host built it with
`build_orbslam3.sh`, that copy is used: same Ubuntu, same OpenCV 4.5.4, same
Pangolin prefix path.

## 5. Verify the image before trusting a run

```bash
docker compose -f docker/compose.yml -f docker/compose.gpu.yml exec wil bash -lc '
  cat docker/build_info.txt | head -5
  ros2 pkg list | grep -E "^(orbslam3_ros2|vins_fusion_ros2|rtabmap_slam|aws_robomaker_small_warehouse_world|realsense_imu)$"
  ign gazebo --version | head -1
  ls ~/.ignition/gazebo/plugins/libKinematicTrajectory.so
  ldd install_docker/vins_fusion_ros2/lib/vins_fusion_ros2/vins_fusion_ros2_node | grep -E "ceres|not found"
  ldd install_docker/orbslam3_ros2/lib/orbslam3_ros2/*node* | grep -E "ORB_SLAM3|pango_display|not found"
  ls -l thirdparty/ORB_SLAM3/Vocabulary/ORBvoc.txt
  python3 -c "import torch, ultralytics; print(torch.__version__, torch.cuda.is_available(), ultralytics.__version__)"
  which rtabmap-databaseViewer
'
```

Expected: five packages listed, Gazebo 6.x, `libceres.so.4` resolved from
`/usr/local/lib`, no `not found`, a 139 MB vocabulary, `True` for CUDA under
the GPU override, and the database viewer on PATH.

## 6. What differs from the host

- Ceres 2.2.0 is in `/usr/local` instead of `~/local`; VINS's CMake falls
  back to it because `libceres-dev` (2.0) is deliberately absent.
- ORB-SLAM3 and Pangolin live in `/opt/wil/thirdparty`; the workspace's
  `thirdparty/` is a symlink to it.
- The gz plugin is installed both in `~/.ignition/gazebo/plugins` and on
  `IGN_GAZEBO_SYSTEM_PLUGIN_PATH`.
- `pyrealsense2` and `ros-humble-realsense2-camera` are not installed: the
  real rig is recorded on the host, the container replays bags.
- Only the five `rtabmap_ros` packages up to `rtabmap_slam` are built, as
  on the host; `rtabmap_odom`, `rtabmap_viz` and the demos are excluded from
  the build context entirely.
- `.git` is excluded; provenance comes from `docker/build_info.txt` and the
  image label. Run manifests that print a git commit will show none.
