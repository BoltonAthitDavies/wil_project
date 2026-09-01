cmake_minimum_required(VERSION 3.10)
project(gz_kinematic_trajectory)
set(CMAKE_CXX_STANDARD 17)
find_package(ignition-gazebo6 REQUIRED)
find_package(ignition-plugin1 REQUIRED COMPONENTS register)
add_library(KinematicTrajectory SHARED KinematicTrajectory.cc)
target_link_libraries(KinematicTrajectory
  PRIVATE ignition-gazebo6::ignition-gazebo6 ignition-plugin1::register)

# gz-sim's SystemLoader always searches ~/.ignition/gazebo/plugins, so installing
# there makes the plugin resolvable with no IGN_GAZEBO_SYSTEM_PLUGIN_PATH and no
# sudo. `cmake --build build --target install-local` refreshes it after a rebuild.
add_custom_target(install-local
  COMMAND ${CMAKE_COMMAND} -E make_directory $ENV{HOME}/.ignition/gazebo/plugins
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:KinematicTrajectory>
          $ENV{HOME}/.ignition/gazebo/plugins/
  DEPENDS KinematicTrajectory)
