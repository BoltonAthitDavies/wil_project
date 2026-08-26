cmake_minimum_required(VERSION 3.5)
project(orbslam3_ros2)

# C++17, not the 14 vins_fusion_ros2 uses. Not a preference: Pangolin 0.9's
# headers are C++17, ORB-SLAM3 is patched to -std=c++17 to match, and its
# headers reach us transitively. All three have to agree.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-Wextra -Wpedantic)

set(CMAKE_BUILD_TYPE Release)

# ORB-SLAM3 is built out of tree by ../build_orbslam3.sh and ships no install
# rules, so we point at its source tree. Override with
#   colcon build --cmake-args -DORB_SLAM3_ROOT=/somewhere/else
set(ORB_SLAM3_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/ORB_SLAM3"
    CACHE PATH "ORB-SLAM3 source tree containing lib/libORB_SLAM3.so")
set(THIRDPARTY_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/install"
    CACHE PATH "Prefix that Pangolin was installed into")

if(NOT EXISTS "${ORB_SLAM3_ROOT}/lib/libORB_SLAM3.so")
  message(FATAL_ERROR
    "No libORB_SLAM3.so under ${ORB_SLAM3_ROOT}.\n"
    "Build it first:  cd ${CMAKE_CURRENT_SOURCE_DIR}/.. && ./build_orbslam3.sh")
endif()

list(APPEND CMAKE_PREFIX_PATH "${THIRDPARTY_PREFIX}")

# libORB_SLAM3.so, libDBoW2.so, libg2o.so and Pangolin all live outside the
# linker's default search path, so their directories must be baked into the
# installed binary or it links fine and dies at launch with "cannot open shared
# object file". vins_fusion_ros2 needs the same thing for its ~/local Ceres
# (its CMakeLists.txt:32), but one line is not enough here -- two extra problems:
#
#   1. USE_LINK_PATH adds link DIRECTORIES. We link libORB_SLAM3.so by full file
#      path, which contributes no directory, so its folder was silently absent
#      and only libORB_SLAM3.so itself failed to resolve. Hence the explicit list.
#
#   2. Modern ld defaults to DT_RUNPATH, which -- unlike the older DT_RPATH -- is
#      NOT used when resolving a dependency's own dependencies. Pangolin's
#      libpango_display.so carries no RUNPATH of its own, so its siblings
#      (libpango_core, _windowing, _vars, _image) went unfound even with
#      thirdparty/install/lib in OUR runpath. --disable-new-dtags emits DT_RPATH,
#      which IS inherited down the chain, and fixes all of them at once.
set(CMAKE_INSTALL_RPATH
  "${ORB_SLAM3_ROOT}/lib"
  "${ORB_SLAM3_ROOT}/Thirdparty/DBoW2/lib"
  "${ORB_SLAM3_ROOT}/Thirdparty/g2o/lib"
  "${THIRDPARTY_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(message_filters REQUIRED)
find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(Threads REQUIRED)

add_executable(orbslam3_node
  src/orbslam3_node.cpp
  src/slam_wrapper.cpp
  src/trajectory_writer.cpp)

target_include_directories(orbslam3_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include)

# SYSTEM, so -Wextra -Wpedantic does not bury our own warnings under thousands
# from ORB-SLAM3, g2o and Pangolin headers. Same trick vins_fusion_ros2 uses.
target_include_directories(orbslam3_node SYSTEM PRIVATE
  ${ORB_SLAM3_ROOT}
  ${ORB_SLAM3_ROOT}/include
  ${ORB_SLAM3_ROOT}/include/CameraModels
  ${ORB_SLAM3_ROOT}/Thirdparty/Sophus
  ${EIGEN3_INCLUDE_DIR}
  ${Pangolin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS})

target_link_libraries(orbslam3_node
  ${ORB_SLAM3_ROOT}/lib/libORB_SLAM3.so
  ${ORB_SLAM3_ROOT}/Thirdparty/DBoW2/lib/libDBoW2.so
  ${ORB_SLAM3_ROOT}/Thirdparty/g2o/lib/libg2o.so
  ${OpenCV_LIBS}
  ${Pangolin_LIBRARIES}
  Threads::Threads
  -lboost_serialization
  -lcrypto)

# See the RPATH note above: emits DT_RPATH instead of DT_RUNPATH so the search
# path reaches Pangolin's transitive dependencies.
target_link_options(orbslam3_node PRIVATE "-Wl,--disable-new-dtags")

ament_target_dependencies(orbslam3_node
  rclcpp sensor_msgs nav_msgs geometry_msgs
  tf2 tf2_ros tf2_geometry_msgs
  cv_bridge image_transport message_filters)

install(
  DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME})

install(
  TARGETS orbslam3_node
  DESTINATION lib/${PROJECT_NAME})

ament_package()
