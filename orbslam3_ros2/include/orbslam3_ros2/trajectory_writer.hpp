// Writes trajectories in the CSV format VINS-Fusion produces, so the existing
// analysis scripts read ORB-SLAM3 output with no changes.
//
// The format is not TUM and not ORB-SLAM3's own. It is what
// vins/src/estimator/estimator.cpp:528-543 emits and what script/plot_vio.py and
// script/plot_vio_vs_gt.py parse:
//
//     t_ns,px,py,pz,qw,qx,qy,qz,vx,vy,vz
//
// comma-separated, integer nanoseconds, quaternion W FIRST, 11 columns.
//
// ORB-SLAM3's SaveTrajectoryEuRoC (System.cc:762) writes SPACE-separated
// `t_ns tx ty tz qx qy qz qw` -- wrong delimiter and wrong quaternion order --
// and SaveTrajectoryTUM uses seconds. Converting after the fact would be a
// second thing to get wrong, so we write the target format directly.
//
// plot_vio.py hard-requires columns 8-10 (it does d[:, 8:11]), so the velocity
// columns are not optional padding.

#ifndef ORBSLAM3_ROS2__TRAJECTORY_WRITER_HPP_
#define ORBSLAM3_ROS2__TRAJECTORY_WRITER_HPP_

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <fstream>
#include <string>

namespace orbslam3_ros2
{

class TrajectoryWriter
{
public:
  /// Creates parent directories if needed and truncates any existing file.
  /// VINS deliberately does not mkdir and that has bitten before, so we do.
  /// Throws std::runtime_error if the file cannot be opened.
  explicit TrajectoryWriter(const std::string & path);

  /// One row. `t_sec` is the ROS message stamp in seconds (sim clock when
  /// use_sim_time is on), which is the same clock script/extract_gt.py samples
  /// for ground_truth.csv -- that is what makes the two directly comparable
  /// with no time alignment.
  void write(
    double t_sec,
    const Eigen::Vector3f & position,
    const Eigen::Quaternionf & orientation,
    const Eigen::Vector3f & velocity);

  void flush() { out_.flush(); }
  std::size_t rows() const { return rows_; }
  const std::string & path() const { return path_; }

private:
  std::string path_;
  std::ofstream out_;
  std::size_t rows_{0};
};

}  // namespace orbslam3_ros2

#endif  // ORBSLAM3_ROS2__TRAJECTORY_WRITER_HPP_
