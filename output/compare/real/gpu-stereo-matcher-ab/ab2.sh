#!/usr/bin/env bash
# ABBA node A/B for patch (10b) on the first 60 s of realv5, temperature-gated to remove the order/throttling bias seen in ab/.
S=/tmp/claude-1000/-home-ambushee-wil-project/88846a24-8e90-425d-beed-b9cdd3dcb9c4/scratchpad/gpumatch/ab2
source /opt/ros/humble/setup.bash; source /home/ambushee/wil_project/install/setup.bash
cd /home/ambushee/wil_project; echo $$ > $S/pids.txt; : > $S/progress.txt
temp() { for z in /sys/class/thermal/thermal_zone*; do if [ "$(cat $z/type)" = x86_pkg_temp ]; then echo $(($(cat $z/temp)/1000)); fi; done; }
cool() { for i in $(seq 1 30); do t=$(temp); [ "$t" -le 72 ] && break; sleep 10; done; echo "  cooled to ${t}C after $((i*10)) s" >> $S/progress.txt; }
run() { # label config rate wallseconds
  cool; O=$S/$1; rm -rf $O; mkdir -p $O; t0=$(temp)
  install/orbslam3_ros2/lib/orbslam3_ros2/orbslam3_node --ros-args -r __node:=orbslam3 \
    -p config_file:=$2 -p vocabulary_file:=/home/ambushee/wil_project/thirdparty/ORB_SLAM3/Vocabulary/ORBvoc.txt \
    -p output_path:=$O -p experiment_id:=$1 -p dataset_path:=/media/ambushee/32E4AAB1E4AA772F/dataset_realv5_000 \
    -p use_imu:=false -p use_viewer:=false -p image_transport:=compressed -p use_sim_time:=false \
    -p image0_topic:=/cam0/image_raw -p image1_topic:=/cam1/image_raw -p imu_topic:=/hwt101ct_yaw_publisher \
    -p filter:=false -p publish_tf:=false -p run_notes:="GPU stereo matcher ABBA (patch 10b), first 60 s" > $O/node.log 2>&1 &
  P=$!; echo $P >> $S/pids.txt
  for i in $(seq 1 120); do grep -q "subscribed:" $O/node.log 2>/dev/null && break; sleep 1; done
  timeout -s INT $4 ros2 bag play /media/ambushee/32E4AAB1E4AA772F/dataset_realv5_000 --rate $3 > /dev/null 2>&1 &
  B=$!; echo $B >> $S/pids.txt; wait $B; sleep 3
  kill -INT $P 2>/dev/null; for i in $(seq 1 60); do kill -0 $P 2>/dev/null || break; sleep 1; done; kill -9 $P 2>/dev/null
  echo "done $1 $(date +%T) temp ${t0}C -> $(temp)C" >> $S/progress.txt
}
run r05_A1 $S/stereo_extract.yaml 0.5 120
run r05_B1 $S/stereo_match.yaml   0.5 120
run r05_B2 $S/stereo_match.yaml   0.5 120
run r05_A2 $S/stereo_extract.yaml 0.5 120
run r10_A1 $S/stereo_extract.yaml 1.0 60
run r10_B1 $S/stereo_match.yaml   1.0 60
run r10_B2 $S/stereo_match.yaml   1.0 60
run r10_A2 $S/stereo_extract.yaml 1.0 60
echo "all done $(date +%T)" >> $S/progress.txt
