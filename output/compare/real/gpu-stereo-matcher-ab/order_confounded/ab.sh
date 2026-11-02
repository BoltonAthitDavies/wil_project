#!/usr/bin/env bash
# Node A/B for patch (10b): GPU extractor + CPU BF stereo matcher ("extract") vs GPU extractor + GPU Hamming matcher ("match").
S=/tmp/claude-1000/-home-ambushee-wil-project/88846a24-8e90-425d-beed-b9cdd3dcb9c4/scratchpad/gpumatch/ab
source /opt/ros/humble/setup.bash; source /home/ambushee/wil_project/install/setup.bash
cd /home/ambushee/wil_project; echo $$ > $S/pids.txt; : > $S/progress.txt
run() { # label config rate
  O=$S/$1; rm -rf $O; mkdir -p $O
  install/orbslam3_ros2/lib/orbslam3_ros2/orbslam3_node --ros-args -r __node:=orbslam3 \
    -p config_file:=$2 -p vocabulary_file:=/home/ambushee/wil_project/thirdparty/ORB_SLAM3/Vocabulary/ORBvoc.txt \
    -p output_path:=$O -p experiment_id:=$1 -p dataset_path:=/media/ambushee/32E4AAB1E4AA772F/dataset_realv5_000 \
    -p use_imu:=false -p use_viewer:=false -p image_transport:=compressed -p use_sim_time:=false \
    -p image0_topic:=/cam0/image_raw -p image1_topic:=/cam1/image_raw -p imu_topic:=/hwt101ct_yaw_publisher \
    -p filter:=false -p publish_tf:=false -p run_notes:="GPU stereo matcher A/B (patch 10b)" > $O/node.log 2>&1 &
  P=$!; echo $P >> $S/pids.txt
  for i in $(seq 1 120); do grep -q "subscribed:" $O/node.log 2>/dev/null && break; sleep 1; done
  ros2 bag play /media/ambushee/32E4AAB1E4AA772F/dataset_realv5_000 --rate $3 > /dev/null 2>&1 &
  B=$!; echo $B >> $S/pids.txt; wait $B; sleep 3
  kill -INT $P 2>/dev/null; for i in $(seq 1 60); do kill -0 $P 2>/dev/null || break; sleep 1; done; kill -9 $P 2>/dev/null
  echo "done $1 $(date +%T)" >> $S/progress.txt
}
run extract_r05 $S/stereo_extract.yaml 0.5
run match_r05   $S/stereo_match.yaml   0.5
run extract_r10 $S/stereo_extract.yaml 1.0
run match_r10   $S/stereo_match.yaml   1.0
echo "all done $(date +%T)" >> $S/progress.txt
