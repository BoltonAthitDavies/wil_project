# syntax=docker/dockerfile:1.6
#
# wil_project container image.
#
# Stages
#   deps        Ubuntu 22.04 + ROS 2 Humble desktop + Gazebo Fortress + nav2 +
#               RTAB-Map core (apt 0.23.x) + Ceres 2.2.0 (source) + CUDA torch +
#               ultralytics. No project code. Reusable for the dev service.
#   thirdparty  ORB-SLAM3 (pinned fork) + Pangolin 0.9.1 via build_orbslam3.sh,
#               and the KinematicTrajectory gz-sim plugin. Lives in /opt/wil.
#   full        Workspace copied to /home/ambushee/wil_project and colcon-built.
#
# The workspace path, user name and home are kept identical to the host
# (/home/ambushee/wil_project) because launch files, VINS configs and scripts
# hard-code them (orbslam3_ros2/launch/*.py, rtabmap_ros/rtab_sim.launch.py,
# vins_fusion_ros2/config/*/*.yaml).
#
# Build from the repository root (see docker/README.md):
#   ./docker/build.sh
#
ARG ROS_DISTRO=humble

# ============================================================================
FROM osrf/ros:${ROS_DISTRO}-desktop AS deps
ARG ROS_DISTRO
ARG DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 ROS_DISTRO=${ROS_DISTRO}
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# --- OSRF Gazebo apt repository (Fortress = ignition-gazebo6, what ros-gz on
#     Humble targets and what the host runs: libignition-gazebo6 6.18.0) -------
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg lsb-release \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://packages.osrfoundation.org/gazebo.gpg \
        -o /etc/apt/keyrings/gazebo-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/gazebo-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
        > /etc/apt/sources.list.d/gazebo-stable.list \
    && rm -rf /var/lib/apt/lists/*

# --- System, ROS and simulator packages --------------------------------------
# Everything the four workspace packages and the five rtabmap_ros packages
# (rtabmap_msgs conversions sync util slam) need, resolved from their
# package.xml files and from the engineering records:
#   * ros-humble-rtabmap  = the CORE library from apt (0.23.x).
#     ros-humble-rtabmap-ros must NOT be installed: the fork in rtabmap_ros/
#     replaces it (rtabmap_ros/MEMORY_CLAUDE/README.md).
#   * libceres-dev must NOT be installed: apt ships 2.0.0, VINS needs the
#     Manifold API from >= 2.1, so Ceres 2.2.0 is built from source below.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git pkg-config sudo wget unzip nano less \
        python3-pip python3-dev \
        python3-colcon-common-extensions python3-rosdep python3-vcstool \
        python3-pyqt5 python3-yaml python3-pil \
        # Gazebo Fortress runtime + dev headers for gz_kinematic_trajectory
        ignition-fortress libignition-gazebo6-dev libignition-plugin-dev \
        ros-${ROS_DISTRO}-ros-gz \
        # nav2 stack (exec deps of aws_robomaker_small_warehouse_world)
        ros-${ROS_DISTRO}-navigation2 ros-${ROS_DISTRO}-nav2-bringup \
        ros-${ROS_DISTRO}-robot-state-publisher ros-${ROS_DISTRO}-joint-state-publisher \
        ros-${ROS_DISTRO}-teleop-twist-keyboard \
        # perception / geometry deps shared by orbslam3_ros2, vins_fusion_ros2, rtabmap_*
        ros-${ROS_DISTRO}-cv-bridge ros-${ROS_DISTRO}-image-transport \
        ros-${ROS_DISTRO}-image-transport-plugins ros-${ROS_DISTRO}-image-geometry \
        ros-${ROS_DISTRO}-vision-msgs ros-${ROS_DISTRO}-stereo-msgs \
        ros-${ROS_DISTRO}-tf2-geometry-msgs ros-${ROS_DISTRO}-tf2-eigen \
        ros-${ROS_DISTRO}-pcl-ros ros-${ROS_DISTRO}-pcl-conversions \
        ros-${ROS_DISTRO}-laser-geometry ros-${ROS_DISTRO}-diagnostic-updater \
        ros-${ROS_DISTRO}-grid-map-ros ros-${ROS_DISTRO}-octomap-msgs \
        ros-${ROS_DISTRO}-apriltag-msgs ros-${ROS_DISTRO}-aruco-msgs \
        ros-${ROS_DISTRO}-aruco-opencv-msgs ros-${ROS_DISTRO}-imu-filter-madgwick \
        ros-${ROS_DISTRO}-rosbag2-storage-mcap \
        # RTAB-Map core library + tools (rtabmap-databaseViewer)
        ros-${ROS_DISTRO}-rtabmap \
        # native libraries: OpenCV 4.5.4 (what cv_bridge links), Eigen, Boost,
        # OpenGL/GLEW/EGL/X11 for Pangolin, OpenSSL for ORB-SLAM3,
        # glog/gflags/SuiteSparse/METIS/TBB for Ceres 2.2
        libopencv-dev libeigen3-dev \
        libboost-serialization-dev libboost-system-dev libboost-filesystem-dev \
        libssl-dev libglew-dev libegl1-mesa-dev libgl1-mesa-dev \
        libwayland-dev libxkbcommon-dev libx11-dev libepoxy-dev \
        libgoogle-glog-dev libgflags-dev libatlas-base-dev \
        libsuitesparse-dev libmetis-dev libtbb-dev \
        # runtime GL/X11 for GUIs inside the container
        libgl1-mesa-dri libglx-mesa0 mesa-utils x11-apps \
    && rm -rf /var/lib/apt/lists/*

# --- Ceres 2.2.0 from source into /usr/local ---------------------------------
# vins_fusion_ros2/CMakeLists.txt prefers $HOME/local/lib/cmake/Ceres when it
# exists and otherwise falls back to find_package(Ceres); with libceres-dev
# absent, that fallback resolves to this install.
RUN git clone --depth 1 --branch 2.2.0 https://github.com/ceres-solver/ceres-solver.git /tmp/ceres \
    && cmake -S /tmp/ceres -B /tmp/ceres/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF \
        -DUSE_CUDA=OFF \
    && cmake --build /tmp/ceres/build -j"$(nproc)" \
    && cmake --install /tmp/ceres/build \
    && ldconfig \
    && rm -rf /tmp/ceres

# --- Python: CUDA torch + ultralytics + analysis stack ------------------------
# Pinned to the versions the host validated (pip freeze, 2026-09-24). numpy is
# pinned to 1.x because Humble's cv_bridge / rclpy bindings break on numpy 2.
# The base image has apt's distutils-installed sympy 1.9, which pip cannot
# uninstall; torch 2.5.1 needs sympy==1.13.1, so install it alongside
# (--ignore-installed) into /usr/local, which shadows the apt copy.
COPY docker/requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir --ignore-installed sympy==1.13.1 \
    && python3 -m pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# --- Non-root user matching the host so bind-mounted output stays writable ---
ARG USER=ambushee
ARG USER_UID=1000
ARG USER_GID=1000
RUN (getent group "${USER_GID}" >/dev/null || groupadd -g "${USER_GID}" "${USER}") \
    && useradd -m -u "${USER_UID}" -g "${USER_GID}" -G video,sudo -s /bin/bash "${USER}" \
    && echo "${USER} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-${USER} \
    && chmod 0440 /etc/sudoers.d/90-${USER} \
    && mkdir -p /opt/wil && chown "${USER_UID}:${USER_GID}" /opt/wil
ENV WIL_USER=${USER} WIL_WS=/home/${USER}/wil_project

COPY --chmod=755 docker/entrypoint.sh /usr/local/bin/wil_entrypoint
COPY --chmod=755 docker/ws_build      /usr/local/bin/ws_build
RUN cat >> /home/${USER}/.bashrc <<'BASHRC'

# --- wil_project ---------------------------------------------------------------
source /opt/ros/humble/setup.bash
if [ -f "$HOME/wil_project/install_docker/setup.bash" ]; then
    source "$HOME/wil_project/install_docker/setup.bash"
else
    echo "[wil] no install_docker/ in $HOME/wil_project -- run: ws_build" >&2
fi
export IGN_GAZEBO_SYSTEM_PLUGIN_PATH="/opt/wil/gz_plugins${IGN_GAZEBO_SYSTEM_PLUGIN_PATH:+:$IGN_GAZEBO_SYSTEM_PLUGIN_PATH}"
export GZ_SIM_SYSTEM_PLUGIN_PATH="$IGN_GAZEBO_SYSTEM_PLUGIN_PATH"
cd "$HOME/wil_project" 2>/dev/null || true
BASHRC

USER ${USER}
WORKDIR /home/${USER}
ENTRYPOINT ["/usr/local/bin/wil_entrypoint"]
CMD ["bash"]

# ============================================================================
FROM deps AS thirdparty
ARG USER=ambushee

# ORB-SLAM3 (fork BoltonAthitDavies/ORB_SLAM3 @ 5c2b00f) + Pangolin v0.9.1,
# built exactly as on the host by the repository's own script. The script
# builds into <its own dir>/thirdparty, so it is copied to /opt/wil and the
# result lands in /opt/wil/thirdparty. The workspace gets a symlink to it.
COPY --chown=${USER} build_orbslam3.sh /opt/wil/build_orbslam3.sh
RUN cd /opt/wil && ./build_orbslam3.sh \
    && rm -rf /opt/wil/thirdparty/ORB_SLAM3/.git /opt/wil/thirdparty/Pangolin

# KinematicTrajectory gz-sim system plugin (moving props in the dynamic worlds).
# gz-sim's SystemLoader always searches ~/.ignition/gazebo/plugins; the copy in
# /opt/wil/gz_plugins is also exported on IGN_GAZEBO_SYSTEM_PLUGIN_PATH.
COPY --chown=${USER} gz_kinematic_trajectory/CMakeLists.txt gz_kinematic_trajectory/KinematicTrajectory.cc /opt/wil/gz_kinematic_trajectory/
RUN cmake -S /opt/wil/gz_kinematic_trajectory -B /opt/wil/gz_kinematic_trajectory/build \
        -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /opt/wil/gz_kinematic_trajectory/build -j"$(nproc)" \
    && mkdir -p /opt/wil/gz_plugins "/home/${USER}/.ignition/gazebo/plugins" \
    && cp /opt/wil/gz_kinematic_trajectory/build/libKinematicTrajectory.so /opt/wil/gz_plugins/ \
    && cp /opt/wil/gz_kinematic_trajectory/build/libKinematicTrajectory.so "/home/${USER}/.ignition/gazebo/plugins/" \
    && rm -rf /opt/wil/gz_kinematic_trajectory/build

# ============================================================================
FROM thirdparty AS full
ARG USER=ambushee
ARG USER_UID=1000
ARG USER_GID=1000
ARG ROS_DISTRO

# Provenance written by docker/build.sh (git HEAD + submodule pins of the
# checkout that was copied in). Harmless placeholder if built without it.
ARG WIL_GIT_DESCRIBE=unknown
LABEL org.opencontainers.image.title="wil_project" \
      org.opencontainers.image.revision="${WIL_GIT_DESCRIBE}"

WORKDIR /home/${USER}/wil_project
# .dockerignore keeps datasets, outputs, host build trees and thirdparty out.
COPY --chown=${USER_UID}:${USER_GID} . .
RUN ln -sfn /opt/wil/thirdparty thirdparty

# Safety net for any dependency the explicit apt list above missed. Scoped to
# the packages that are actually built; the rtabmap packages that are NOT
# built (odom, viz, examples, demos, costmap_plugins) are excluded so their
# heavy optional deps are not pulled in.
#
# rosdep's database is fetched with git from github.com rather than from
# raw.githubusercontent.com, which the KMUTT campus firewall blocks (it serves
# a self-signed "Web Page Blocked" page -> CERTIFICATE_VERIFY_FAILED).
ENV ROSDISTRO_INDEX_URL=file:///opt/rosdistro/index-v4.yaml
RUN sudo git clone --depth 1 https://github.com/ros/rosdistro /opt/rosdistro \
    && sudo mkdir -p /etc/ros/rosdep/sources.list.d \
    && printf 'yaml file:///opt/rosdistro/rosdep/%s.yaml\n' base python ruby \
        | sudo tee /etc/ros/rosdep/sources.list.d/20-default.list \
    && sudo apt-get update \
    && rosdep update --rosdistro "${ROS_DISTRO}" \
    && rosdep install --rosdistro "${ROS_DISTRO}" --ignore-src -y -r \
        --from-paths orbslam3_ros2 vins_fusion_ros2 aws-robomaker-small-warehouse-world realsense/realsense_imu \
                     rtabmap_ros/rtabmap_msgs rtabmap_ros/rtabmap_conversions rtabmap_ros/rtabmap_sync \
                     rtabmap_ros/rtabmap_util rtabmap_ros/rtabmap_slam \
        --skip-keys "velodyne realsense2_camera pyrealsense2" \
    && sudo rm -rf /var/lib/apt/lists/*

# Same package set and build/install bases as `ws_build`, so the dev service
# (host workspace bind-mounted) and this baked image behave identically.
# Plain (non-symlink) install so the build tree can be dropped afterwards.
RUN WS_SYMLINK=0 ws_build && rm -rf build_docker log

ENV WIL_IMAGE_STAGE=full
