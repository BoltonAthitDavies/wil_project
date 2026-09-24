#!/usr/bin/env bash
# Container entrypoint for wil_project.
#
# Sources ROS 2 and the workspace overlay, makes sure the ORB-SLAM3 third-party
# tree is reachable at the path the launch files expect, then execs the command.
# `docker exec` bypasses this file; interactive shells get the same environment
# from ~/.bashrc, which the Dockerfile appends.
set -e

source /opt/ros/humble/setup.bash

WS="${WIL_WS:-$HOME/wil_project}"

# In the dev service the host workspace is bind-mounted over $WS. Its
# thirdparty/ (if the host ever ran build_orbslam3.sh) is ABI-compatible and
# takes precedence; otherwise point the workspace at the image's copy.
if [ -d "$WS" ] && [ ! -e "$WS/thirdparty/ORB_SLAM3/lib/libORB_SLAM3.so" ]; then
    if [ ! -e "$WS/thirdparty" ] || [ -L "$WS/thirdparty" ]; then
        ln -sfn /opt/wil/thirdparty "$WS/thirdparty" 2>/dev/null || true
    fi
fi

if [ -f "$WS/install_docker/setup.bash" ]; then
    source "$WS/install_docker/setup.bash"
else
    echo "[wil] no $WS/install_docker/setup.bash -- run 'ws_build' inside the container" >&2
fi

export IGN_GAZEBO_SYSTEM_PLUGIN_PATH="/opt/wil/gz_plugins${IGN_GAZEBO_SYSTEM_PLUGIN_PATH:+:$IGN_GAZEBO_SYSTEM_PLUGIN_PATH}"
export GZ_SIM_SYSTEM_PLUGIN_PATH="$IGN_GAZEBO_SYSTEM_PLUGIN_PATH"

mkdir -p "${XDG_RUNTIME_DIR:-/tmp/runtime-wil}" 2>/dev/null || true

cd "$WS" 2>/dev/null || true
exec "$@"
