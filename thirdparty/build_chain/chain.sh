#!/usr/bin/env bash
B=/home/ambushee/wil_project/thirdparty/build_chain; O=/home/ambushee/wil_project/thirdparty/ORB_SLAM3; A=/home/ambushee/wil_project/thirdparty/orb_asan_build
echo "normal build start $(date +%T)" > $B/progress.txt
cd $O && cmake --build build --target ORB_SLAM3 -j3 > $B/normal.log 2>&1; rc=$?
echo "normal build exit $rc errors $(grep -c 'error:' $B/normal.log) $(date +%T) lib $(stat -c '%y' $O/lib/libORB_SLAM3.so | cut -c1-19)" >> $B/progress.txt
[ $rc -ne 0 ] && { echo "NORMAL BUILD FAILED" >> $B/progress.txt; exit 1; }
echo "asan build start $(date +%T)" >> $B/progress.txt
cd $A/build && cmake --build . --target ORB_SLAM3 -j1 > $A/build.log 2>&1; rc=$?
echo "asan build exit $rc errors $(grep -c 'error:' $A/build.log) $(date +%T) lib $(stat -c '%y' $A/lib/libORB_SLAM3.so | cut -c1-19)" >> $B/progress.txt
echo "chain done" >> $B/progress.txt
