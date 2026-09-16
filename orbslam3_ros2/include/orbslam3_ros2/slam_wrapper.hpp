// Owns the ORB_SLAM3::System and keeps it off the ROS executor's threads.
//
// WHY A WORKER THREAD
//   TrackStereo() blocks for tens of milliseconds per frame. Called straight
//   from a subscription callback it would stall the executor, and IMU messages
//   arriving at 200 Hz during that window would be dropped by the middleware --
//   which silently starves the very preintegration the inertial mode depends on.
//   So images go into a small queue and a dedicated thread does the tracking.
//
// DROP POLICY
//   The queue holds one pending frame. If tracking cannot keep up with playback
//   the OLDEST pending frame is discarded and a counter incremented, because a
//   stale frame is worth less than the fresh one behind it. The node logs the
//   counter; a run that quietly dropped half its frames should not look healthy.

#ifndef ORBSLAM3_ROS2__SLAM_WRAPPER_HPP_
#define ORBSLAM3_ROS2__SLAM_WRAPPER_HPP_

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <opencv2/core.hpp>
#include <string>
#include <thread>

namespace ORB_SLAM3 {class System;}

namespace orbslam3_ros2
{

struct SlamConfig
{
  std::string vocabulary_path;
  std::string settings_path;
  bool use_imu{true};
  bool use_viewer{true};
  /// Multiplies linear acceleration before it reaches ORB-SLAM3. Exists for the
  /// real rig, whose accelerometer reads a stationary magnitude of 9.2642 m/s^2
  /// against a true 9.81 -- a 5.6% scale error. VINS absorbs this with
  /// `g_norm: 9.2642`, but ORB-SLAM3 hardcodes GRAVITY_VALUE=9.81 in
  /// ImuTypes.h with no config key, so the measurement has to be corrected
  /// instead: 9.81/9.2642 = 1.0589. Leave at 1.0 for the sim.
  double accel_scale{1.0};
  /// Re-express poses in the body frame of the FIRST tracked pose, so the
  /// published world frame has REP-103 axes (x-forward, y-left, z-up) like
  /// `odom`, instead of ORB-SLAM3's own world.
  ///
  /// ORB-SLAM3's world origin is its first keyframe, so its AXES are the initial
  /// camera optical frame (x-right, y-down, z-forward). Publishing that raw is
  /// what made plot_vio_vs_gt.py report a ~90 deg yaw error and an 11 m position
  /// error while the SE(3)-fitted ATE was a healthy 0.31 m -- the shape was
  /// right, the frame convention was not, and a yaw-only alignment cannot undo
  /// an axis permutation.
  ///
  /// DEFAULT OFF, after measuring. Latching the transform at the first tracked
  /// pose is wrong for inertial mode: ORB-SLAM3 re-bases its world to gravity
  /// during IMU initialisation, which happens AFTER that first pose, so the
  /// latched frame gets rotated out from under us and the trajectory ends up
  /// with the vertical on x (measured: vio spanned x=0.45 y=5.57 z=12.49 against
  /// ground truth's x=5.21 y=12.12 z=0.00).
  ///
  /// Left raw, ORB-SLAM3's world is already gravity-aligned and z-up -- the
  /// measured z error was 0.041 m RMS -- and only the heading is arbitrary,
  /// which is exactly what the SE(3)-fitted ATE in plot_vio_vs_gt.py and
  /// viewer.py's --vins-align both absorb.
  bool align_first_pose{false};
  /// In inertial mode, suppress poses until ORB-SLAM3 has initialised the IMU.
  ///
  /// Not cosmetic. ORB-SLAM3 RE-BASES its map when the IMU initialises, which
  /// happens shortly after tracking first reports OK. Poses emitted before that
  /// belong to a frame the estimator is about to rotate away. That does not hurt
  /// ATE (a global SE(3) fit absorbs it), but plot_vio_vs_gt.py anchors its
  /// yaw-and-translation alignment on the FIRST pose, so one bad anchor rotated
  /// the whole ground-truth overlay and reported a ~90 deg yaw error against a
  /// trajectory that was actually sound (measured: body x aligned with travel at
  /// +0.73, quaternion yaw within ~3 deg of path yaw).
  ///
  /// Frame::HasVelocity() is the signal: it turns true exactly when the inertial
  /// state becomes valid, and needs no patch to ORB-SLAM3.
  bool wait_for_imu_init{false};   // see note: HasVelocity() never fired on this bag
};

/// One tracked pose, already converted out of ORB-SLAM3's camera convention.
struct TrackResult
{
  double stamp{0.0};
  bool tracking_ok{false};
  int state{-1};
  /// Body (IMU) pose in ORB-SLAM3's world frame. That world is the first
  /// keyframe, gravity-aligned once the IMU initialises -- it is NOT the sim's
  /// `odom`. Downstream alignment is deliberately left to the existing tools:
  /// plot_vio_vs_gt.py does a yaw-only fit on the first overlapping pose and
  /// viewer.py has --vins-align first.
  Eigen::Vector3f position{Eigen::Vector3f::Zero()};
  Eigen::Quaternionf orientation{Eigen::Quaternionf::Identity()};
  /// World-frame velocity. Real estimate from Frame::GetVelocity() in inertial
  /// modes; finite-differenced from position otherwise. See slam_wrapper.cpp.
  Eigen::Vector3f velocity{Eigen::Vector3f::Zero()};
  bool velocity_is_estimated{false};
  /// Wall-clock instrumentation. queue_wait_ms measures time waiting behind the
  /// current frame, tracking_ms is the TrackStereo call alone, and processing_ms
  /// includes the bounded IMU wait plus tracking and result conversion.
  double queue_wait_ms{0.0};
  double tracking_ms{0.0};
  double processing_ms{0.0};
  std::size_t imu_samples{0};
};

class SlamWrapper
{
public:
  using ResultCallback = std::function<void (const TrackResult &)>;

  explicit SlamWrapper(const SlamConfig & cfg);
  ~SlamWrapper();

  SlamWrapper(const SlamWrapper &) = delete;
  SlamWrapper & operator=(const SlamWrapper &) = delete;

  /// Called from the IMU subscription thread. Cheap: buffers only.
  void pushImu(double t, const Eigen::Vector3d & accel, const Eigen::Vector3d & gyro);

  /// Called from the stereo sync callback. Never blocks; may drop (see above).
  ///
  /// mask_left / mask_right are OPTIONAL dynamic-object masks (RY-SLAM style):
  /// CV_8UC1, same size as the image, 255 = static/keep, 0 = dynamic. They reach
  /// ORB-SLAM3's ORBextractor, which drops the keypoints landing on dynamic
  /// regions. Empty (the default) means no filtering at all -- not "filter
  /// nothing", but the stock upstream code path, bit for bit.
  void pushStereo(
    double t, const cv::Mat & left, const cv::Mat & right,
    const cv::Mat & mask_left = cv::Mat(), const cv::Mat & mask_right = cv::Mat());

  /// Invoked on the worker thread once per tracked frame. Set before starting.
  void setResultCallback(ResultCallback cb) {callback_ = std::move(cb);}

  /// Joins the worker and calls ORB_SLAM3::System::Shutdown(). Idempotent.
  void shutdown();

  std::size_t droppedFrames() const {return dropped_.load();}
  /// Frames skipped because no IMU data covered them (inertial mode only).
  std::size_t starvedFrames() const {return imu_starved_.load();}

  /// body <- cam0, parsed from the settings file's IMU.T_b_c1. Identity for
  /// non-inertial configs, where cam0 IS the body frame.
  const Eigen::Isometry3f & bodyFromCam0() const {return T_b_c0_;}

private:
  void workerLoop();
  /// Waits (briefly) until the IMU buffer actually spans `t`, then pops every
  /// sample at or before it. ORB-SLAM3 wants the measurements spanning the
  /// previous frame to this one, and it CRASHES on an empty vector -- see
  /// workerLoop.
  std::vector<std::pair<double, Eigen::Matrix<double, 6, 1>>> drainImuUpTo(double t);

  SlamConfig cfg_;
  std::unique_ptr<ORB_SLAM3::System> slam_;
  Eigen::Isometry3f T_b_c0_{Eigen::Isometry3f::Identity()};
  ResultCallback callback_;

  struct PendingFrame
  {
    double t{0.0};
    cv::Mat left, right;
    /// Dynamic-object masks; empty when filtering is off or no detection matched
    /// this frame. cv::Mat copies are refcounted headers, so carrying them
    /// through the queue costs nothing even when a frame is dropped.
    cv::Mat mask_left, mask_right;
    std::chrono::steady_clock::time_point enqueued_at;
  };

  std::mutex imu_mu_;
  std::condition_variable imu_cv_;
  std::deque<std::pair<double, Eigen::Matrix<double, 6, 1>>> imu_buf_;

  std::mutex frame_mu_;
  std::condition_variable frame_cv_;
  std::deque<PendingFrame> frame_q_;
  std::atomic<std::size_t> dropped_{0};
  std::atomic<std::size_t> imu_starved_{0};

  std::thread worker_;
  bool running_{false};

  bool imu_ready_{false};
  bool have_align_{false};
  Eigen::Isometry3f T_align_{Eigen::Isometry3f::Identity()};

  bool have_prev_{false};
  double prev_t_{0.0};
  Eigen::Vector3f prev_p_{Eigen::Vector3f::Zero()};
};

}  // namespace orbslam3_ros2

#endif  // ORBSLAM3_ROS2__SLAM_WRAPPER_HPP_
