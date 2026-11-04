cmake_minimum_required(VERSION 3.8)
project(vslam_navigation)

find_package(ament_cmake REQUIRED)

# Nothing is compiled: launch files, configs and two Python nodes.
install(
  DIRECTORY launch rviz params urdf behavior_trees
  DESTINATION share/${PROJECT_NAME})

install(
  PROGRAMS
    scripts/rtab_camera_shim.py
    scripts/rtab_trajectory_writer.py
    scripts/localization_recorder.py
    scripts/pose_gate.py
    scripts/ackermann_bridge.py
  DESTINATION lib/${PROJECT_NAME})

ament_package()
