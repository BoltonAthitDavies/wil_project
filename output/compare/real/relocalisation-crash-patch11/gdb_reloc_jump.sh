#!/usr/bin/env bash
# Exercise Tracking::Relocalization() deterministically on the patched library: build a map from the first 25 s of realv5,
# then restart playback near the start (--start-offset) so the pose no longer matches and the node must relocalise. Under gdb.
G=/tmp/claude-1000/-home-ambushee-wil-project/88846a24-8e90-425d-beed-b9cdd3dcb9c4/scratchpad/crash_rv5
Y=/tmp/claude-1000/-home-ambushee-wil-project/88846a24-8e90-425d-beed-b9cdd3dcb9c4/scratchpad/gpumatch/ab2/stereo_match.yaml
BAG=/media/ambushee/32E4AAB1E4AA772F/dataset_realv5_000
source /opt/ros/humble/setup.bash; source /home/ambushee/wil_project/install/setup.bash
cd /home/ambushee/wil_project; : > $G/jump_progress.txt
for k in 1 2; do
  O=$G/jump_run$k; rm -rf $O; mkdir -p $O; echo $$ > $G/pids.txt
  gdb -q -batch -ex "set pagination off" -ex "handle SIGINT pass nostop noprint" -ex run -ex "echo \n=== CRASH BACKTRACE ===\n" -ex "bt 12" -ex "info symbol \$pc" -ex "info registers rax r12 rip" \
    --args install/orbslam3_ros2/lib/orbslam3_ros2/orbslam3_node --ros-args -r __node:=orbslam3 \
      -p config_file:=$Y -p vocabulary_file:=/home/ambushee/wil_project/thirdparty/ORB_SLAM3/Vocabulary/ORBvoc.txt \
      -p output_path:=$O -p experiment_id:=jump$k -p dataset_path:=$BAG -p use_imu:=false -p use_viewer:=false \
      -p image_transport:=compressed -p use_sim_time:=false -p image0_topic:=/cam0/image_raw -p image1_topic:=/cam1/image_raw \
      -p imu_topic:=/hwt101ct_yaw_publisher -p filter:=false -p publish_tf:=false -p run_notes:="patch 11 verification - relocalisation via playback jump" > $O/gdb.log 2>&1 &
  P=$!; echo $P >> $G/pids.txt
  for i in $(seq 1 180); do grep -q "subscribed:" $O/gdb.log 2>/dev/null && break; sleep 1; done
  timeout -s INT 25 ros2 bag play $BAG --rate 1.0 > /dev/null 2>&1; sleep 2
  for off in 3 40 8; do timeout -s INT 30 ros2 bag play $BAG --rate 1.0 --start-offset $off > /dev/null 2>&1; sleep 2; done
  N=$(pgrep -x orbslam3_node); [ -n "$N" ] && kill -INT $N; for i in $(seq 1 90); do kill -0 $P 2>/dev/null || break; sleep 1; done; kill -9 $P 2>/dev/null
  echo "run $k sigsegv=$(grep -c SIGSEGV $O/gdb.log) relocalized=$(grep -c 'Relocalized' $O/gdb.log) state3_lines=$(grep -c 'tracking state 3' $O/gdb.log) lost_frames=$(awk -F, 'NR>1&&($3==3||$3==4)' $O/tracking_frontend.csv | wc -l) $(grep -o 'wrote [0-9]* poses' $O/gdb.log) $(date +%T)" >> $G/jump_progress.txt
done
echo "jump done" >> $G/jump_progress.txt
