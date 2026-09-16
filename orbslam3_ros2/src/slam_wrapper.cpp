#include "orbslam3_ros2/slam_wrapper.hpp"

#include <chrono>
#include <opencv2/core/eigen.hpp>
#include <sophus/se3.hpp>
#include <stdexcept>
#include <utility>

#include "Frame.h"
#include "System.h"
#include "Tracking.h"

namespace orbslam3_ros2
{
namespace
{

/// Reads a 4x4 !!opencv-matrix out of an ORB-SLAM3 settings file.
/// Returns identity if the key is absent, which is the correct answer for
/// non-inertial configs: with no IMU there is no separate body frame, so cam0
/// is the body.
Eigen::Isometry3f readTransform(const std::string & settings_path, const std::string & key)
{
  cv::FileStorage fs(settings_path, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    throw std::runtime_error("cannot open settings file: " + settings_path);
  }
  cv::FileNode node = fs[key];
  if (node.empty()) {
    return Eigen::Isometry3f::Identity();
  }
  cv::Mat m;
  node >> m;
  if (m.rows != 4 || m.cols != 4) {
    throw std::runtime_error(key + " in " + settings_path + " is not 4x4");
  }
  cv::Mat m32;
  m.convertTo(m32, CV_32F);
  Eigen::Matrix4f e;
  cv::cv2eigen(m32, e);
  Eigen::Isometry3f t = Eigen::Isometry3f::Identity();
  t.matrix() = e;
  return t;
}

}  // namespace

SlamWrapper::SlamWrapper(const SlamConfig & cfg)
: cfg_(cfg)
{
  // IMU.T_b_c1 is ORB-SLAM3's name for what VINS calls body_T_cam0 -- the same
  // quantity, same direction. Upstream's own comment on the key in
  // Examples/Stereo-Inertial/EuRoC.yaml reads "Transformation from camera 0 to
  // body-frame (imu)".
  T_b_c0_ = readTransform(cfg_.settings_path, "IMU.T_b_c1");

  const auto sensor = cfg_.use_imu ? ORB_SLAM3::System::IMU_STEREO
                                   : ORB_SLAM3::System::STEREO;
  slam_ = std::make_unique<ORB_SLAM3::System>(
    cfg_.vocabulary_path, cfg_.settings_path, sensor, cfg_.use_viewer);

  running_ = true;
  worker_ = std::thread(&SlamWrapper::workerLoop, this);
}

SlamWrapper::~SlamWrapper()
{
  shutdown();
}

void SlamWrapper::shutdown()
{
  {
    std::lock_guard<std::mutex> lk(frame_mu_);
    if (!running_) {
      return;
    }
    running_ = false;
  }
  frame_cv_.notify_all();
  if (worker_.joinable()) {
    worker_.join();
  }
  if (slam_) {
    slam_->Shutdown();
  }
}

void SlamWrapper::pushImu(double t, const Eigen::Vector3d & accel, const Eigen::Vector3d & gyro)
{
  Eigen::Matrix<double, 6, 1> s;
  // accel_scale corrects the real rig's accelerometer scale error before the
  // sample is ever seen by the preintegrator -- see SlamConfig::accel_scale.
  s.head<3>() = accel * cfg_.accel_scale;
  s.tail<3>() = gyro;
  {
    std::lock_guard<std::mutex> lk(imu_mu_);
    imu_buf_.emplace_back(t, s);
  }
  imu_cv_.notify_all();
}

std::vector<std::pair<double, Eigen::Matrix<double, 6, 1>>>
SlamWrapper::drainImuUpTo(double t)
{
  std::vector<std::pair<double, Eigen::Matrix<double, 6, 1>>> out;
  std::unique_lock<std::mutex> lk(imu_mu_);
  // Wait until the buffer actually reaches past the frame time. Decoding a
  // 1920x1080 JPEG pair takes long enough that the 200 Hz IMU callback can fall
  // behind, and draining then returns nothing at all. Bounded, so a genuinely
  // dead IMU topic degrades to "skip the frame" rather than hanging forever.
  imu_cv_.wait_for(
    lk, std::chrono::milliseconds(80),
    [&] {return !running_ || (!imu_buf_.empty() && imu_buf_.back().first >= t);});
  while (!imu_buf_.empty() && imu_buf_.front().first <= t) {
    out.push_back(imu_buf_.front());
    imu_buf_.pop_front();
  }
  return out;
}

void SlamWrapper::pushStereo(
  double t, const cv::Mat & left, const cv::Mat & right,
  const cv::Mat & mask_left, const cv::Mat & mask_right)
{
  {
    std::lock_guard<std::mutex> lk(frame_mu_);
    if (!running_) {
      return;
    }
    // Keep at most one pending frame. Dropping the stale one and keeping the
    // fresh one degrades more gracefully than letting a queue grow unbounded
    // and tracking fall further and further behind the clock.
    while (frame_q_.size() >= 1) {
      frame_q_.pop_front();
      ++dropped_;
    }
    frame_q_.push_back(
      PendingFrame{t, left, right, mask_left, mask_right, std::chrono::steady_clock::now()});
  }
  frame_cv_.notify_one();
}

void SlamWrapper::workerLoop()
{
  for (;;) {
    PendingFrame frame;
    {
      std::unique_lock<std::mutex> lk(frame_mu_);
      frame_cv_.wait(lk, [this] {return !running_ || !frame_q_.empty();});
      if (!running_ && frame_q_.empty()) {
        return;
      }
      frame = std::move(frame_q_.front());
      frame_q_.pop_front();
    }

    const auto processing_start = std::chrono::steady_clock::now();
    TrackResult r;
    r.stamp = frame.t;
    r.queue_wait_ms = std::chrono::duration<double, std::milli>(
      processing_start - frame.enqueued_at).count();

    std::vector<ORB_SLAM3::IMU::Point> imu;
    if (cfg_.use_imu) {
      for (const auto & [t, s] : drainImuUpTo(frame.t)) {
        imu.emplace_back(
          static_cast<float>(s(0)), static_cast<float>(s(1)), static_cast<float>(s(2)),
          static_cast<float>(s(3)), static_cast<float>(s(4)), static_cast<float>(s(5)), t);
      }
    }

    // ORB-SLAM3 needs at least TWO IMU samples spanning the interval, not one.
    // PreintegrateIMU() computes n = mvImuFromLastFrame.size()-1 and bails when
    // n==0 -- so a single sample is as fatal as none, despite the message
    // reading "Empty IMU measurements vector!!!". Upstream forgets to call
    // setIntegrated() on that path, so the frame stays un-preintegrated and the
    // caller dereferences a null preintegration (patch (5) in
    // ../build_orbslam3.sh fixes that end of it). Guard both ends: skipping a
    // frame costs one image, passing a thin vector costs the process.
    if (cfg_.use_imu && imu.size() < 2) {
      ++imu_starved_;
      r.state = -3;  // insufficient IMU coverage; distinct from SLAM states
      r.imu_samples = imu.size();
      r.processing_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - processing_start).count();
      if (callback_) {
        callback_(r);
      }
      continue;
    }

    // The empty filename is upstream's own default; the masks follow it because
    // they were appended last, so every unpatched call site still compiles.
    const auto tracking_start = std::chrono::steady_clock::now();
    const Sophus::SE3f Tcw = slam_->TrackStereo(
      frame.left, frame.right, frame.t, imu, "", frame.mask_left, frame.mask_right);
    const auto tracking_end = std::chrono::steady_clock::now();

    r.imu_samples = imu.size();
    r.tracking_ms = std::chrono::duration<double, std::milli>(
      tracking_end - tracking_start).count();
    r.state = slam_->GetTrackingState();
    // 2 == Tracking::OK. Anything else (NOT_INITIALIZED, RECENTLY_LOST, LOST)
    // carries a pose that is either meaningless or about to be revised, and
    // must not reach the trajectory CSV.
    r.tracking_ok = (r.state == 2);

    // Hold everything back until the inertial state is valid; see
    // SlamConfig::wait_for_imu_init.
    if (r.tracking_ok && cfg_.use_imu && cfg_.wait_for_imu_init && !imu_ready_) {
      auto * tk = slam_->GetTracker();
      if (tk && tk->mCurrentFrame.HasVelocity()) {
        imu_ready_ = true;
      } else {
        r.tracking_ok = false;
        r.state = -2;          // reported as "waiting for IMU init"
      }
    }

    if (r.tracking_ok) {
      // TrackStereo returns T_cam_world. The camera pose in world is its
      // inverse; the body pose is that composed with cam0 <- body.
      const Sophus::SE3f Twc = Tcw.inverse();
      Eigen::Isometry3f T_w_c = Eigen::Isometry3f::Identity();
      T_w_c.matrix() = Twc.matrix();
      const Eigen::Isometry3f T_w_b = T_w_c * T_b_c0_.inverse();

      // Latch the inverse of the first tracked body pose, then left-multiply
      // every pose by it. The first published pose becomes identity and all
      // later ones are expressed in that initial BODY frame -- which is
      // REP-103, because the body frame is the IMU. See align_first_pose.
      Eigen::Isometry3f T_out = T_w_b;
      if (cfg_.align_first_pose) {
        if (!have_align_) {
          T_align_ = T_w_b.inverse();
          have_align_ = true;
        }
        T_out = T_align_ * T_w_b;
      }

      r.position = T_out.translation();
      r.orientation = Eigen::Quaternionf(T_out.rotation());
      r.orientation.normalize();

      // Prefer the filter's own velocity state. It exists only in inertial
      // modes and only once the IMU has initialised, hence the HasVelocity()
      // guard; GetTracker() is the accessor patched into System.h, since
      // System keeps mpTracker private.
      bool got_velocity = false;
      if (cfg_.use_imu) {
        if (auto * tracker = slam_->GetTracker()) {
          if (tracker->mCurrentFrame.HasVelocity()) {
            // A world-frame vector, so it takes the alignment's ROTATION only.
            r.velocity = T_align_.rotation() * tracker->mCurrentFrame.GetVelocity();
            r.velocity_is_estimated = true;
            got_velocity = true;
          }
        }
      }
      if (!got_velocity && have_prev_) {
        const double dt = frame.t - prev_t_;
        if (dt > 1e-6) {
          r.velocity = (r.position - prev_p_) / static_cast<float>(dt);
        }
      }

      have_prev_ = true;
      prev_t_ = frame.t;
      prev_p_ = r.position;
    }

    if (callback_) {
      r.processing_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - processing_start).count();
      callback_(r);
    }
  }
}

}  // namespace orbslam3_ros2
