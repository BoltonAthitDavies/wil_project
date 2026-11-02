#!/usr/bin/env bash
# Reproduce the stereo-only RECENTLY_LOST segfault under gdb: live real-rig config (GPU extractor + GPU matcher), realv5 at rate 1.0.
G=/tmp/claude-1000/-home-ambushee-wil-project/88846a24-8e90-425d-beed-b9cdd3dcb9c4/scratchpad/crash_rv5
Y=/tmp/claude-1000/-home-ambushee-wil-project/88846a24-8e90-425d-beed-b9cdd3dcb9c4/scratchpad/gpumatch/ab2/stereo_match.yaml
source /opt/ros/humble/setup.bash; source /home/ambushee/wil_project/install/setup.bash
cd /home/ambushee/wil_project; O=$G/gdb_run; rm -rf $O; mkdir -p $O; echo $$ > $G/pids.txt; : > $G/progress.txt
gdb -q -batch -ex "set pagination off" -ex "handle SIGINT pass nostop noprint" -ex run -ex "echo \n=== CRASH BACKTRACE ===\n" -ex "bt 12" -ex "info symbol \$pc" -ex "p/x \$pc" -ex "info registers" -ex "x/12i \$pc-30" -ex "echo \n=== ALL THREADS ===\n" -ex "thread apply all bt 6" \
  --args install/orbslam3_ros2/lib/orbslam3_ros2/orbslam3_node --ros-args -r __node:=orbslam3 \
    -p config_file:=$Y -p vocabulary_file:=/home/ambushee/wil_project/thirdparty/ORB_SLAM3/Vocabulary/ORBvoc.txt \
    -p output_path:=$O -p experiment_id:=gdb_repro -p dataset_path:=/media/ambushee/32E4AAB1E4AA772F/dataset_realv5_000 \
    -p use_imu:=false -p use_viewer:=false -p image_transport:=compressed -p use_sim_time:=false \
    -p image0_topic:=/cam0/image_raw -p image1_topic:=/cam1/image_raw -p imu_topic:=/hwt101ct_yaw_publisher \
    -p filter:=false -p publish_tf:=false -p run_notes:="gdb repro RECENTLY_LOST segfault" > $O/gdb.log 2>&1 &
P=$!; echo $P >> $G/pids.txt
for i in $(seq 1 180); do grep -q "subscribed:" $O/gdb.log 2>/dev/null && break; sleep 1; done
ros2 bag play /media/ambushee/32E4AAB1E4AA772F/dataset_realv5_000 --rate 1.0 > /dev/null 2>&1 &
B=$!; echo $B >> $G/pids.txt; wait $B; sleep 3
N=$(pgrep -x orbslam3_node); [ -n "$N" ] && kill -INT $N; for i in $(seq 1 90); do kill -0 $P 2>/dev/null || break; sleep 1; done; kill -9 $P 2>/dev/null
echo "done $(date +%T) crash_lines=$(grep -c 'CRASH BACKTRACE' $O/gdb.log) sigsegv=$(grep -c 'SIGSEGV' $O/gdb.log)" >> $G/progress.txt
