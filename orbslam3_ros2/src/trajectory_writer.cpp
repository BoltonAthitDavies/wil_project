#include "orbslam3_ros2/trajectory_writer.hpp"

#include <filesystem>
#include <iomanip>
#include <stdexcept>

namespace orbslam3_ros2
{

TrajectoryWriter::TrajectoryWriter(const std::string & path)
: path_(path)
{
  const std::filesystem::path p(path);
  if (p.has_parent_path()) {
    std::error_code ec;
    std::filesystem::create_directories(p.parent_path(), ec);
    if (ec) {
      throw std::runtime_error("cannot create output directory " +
              p.parent_path().string() + ": " + ec.message());
    }
  }
  out_.open(path, std::ios::out | std::ios::trunc);
  if (!out_.is_open()) {
    throw std::runtime_error("cannot open trajectory file for writing: " + path);
  }
  out_.setf(std::ios::fixed, std::ios::floatfield);
}

void TrajectoryWriter::write(
  double t_sec,
  const Eigen::Vector3f & position,
  const Eigen::Quaternionf & orientation,
  const Eigen::Vector3f & velocity)
{
  if (!out_.is_open()) {
    return;
  }
  // precision(0) on a fixed-format double prints the integer nanoseconds with
  // no decimal point, matching VINS byte for byte. numpy.loadtxt would accept
  // a float here, but staying identical means one less thing to explain.
  out_.precision(0);
  out_ << t_sec * 1e9 << ',';
  out_.precision(5);
  out_ << position.x() << ',' << position.y() << ',' << position.z() << ','
       << orientation.w() << ',' << orientation.x() << ','
       << orientation.y() << ',' << orientation.z() << ','
       << velocity.x() << ',' << velocity.y() << ',' << velocity.z() << '\n';
  ++rows_;
  // Flush every row. At ~30 Hz this is negligible I/O, and it buys something
  // real: a run that is Ctrl-C'd or SIGKILLed keeps every pose it had written.
  // Without it the tail sits in the stream buffer and a hard kill truncates the
  // final line mid-field, which then makes numpy.loadtxt throw on the WHOLE
  // file -- losing the entire run, not just the last pose.
  out_.flush();
}

}  // namespace orbslam3_ros2
