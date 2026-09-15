// ORB-SLAM3 stereo / stereo-inertial node for the WiL rigs.
//
//   ros2 launch orbslam3_ros2 wil_sim_stereo_imu.launch.py
//   ros2 bag play dataset/sim --clock          # second terminal, as VINS expects
//
// Publishes nav_msgs/Odometry so script/viewer.py picks it up with no changes
// (its _find_vins() falls back to any Odometry topic), and writes vio.csv in
// VINS-Fusion's exact column format so script/plot_vio_vs_gt.py scores it
// against ground_truth.csv with no changes either.

#include <cv_bridge/cv_bridge.h>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <image_transport/subscriber_filter.hpp>
#include <memory>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <opencv2/imgcodecs.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <string>
#include <tf2_ros/transform_broadcaster.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <deque>
#include <mutex>
#include <utility>
#include <vector>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <opencv2/imgproc.hpp>

#include "orbslam3_ros2/slam_wrapper.hpp"
#include "orbslam3_ros2/trajectory_writer.hpp"

namespace orbslam3_ros2
{

class Orbslam3Node : public rclcpp::Node
{
public:
  Orbslam3Node()
  : Node("orbslam3_node")
  {
    // --- parameters ---------------------------------------------------------
    // config_file follows the vins_fusion_ros2 convention exactly: the launch
    // file resolves it through get_package_share_directory() and passes the
    // absolute path down as a parameter.
    const auto config_file = declare_parameter<std::string>("config_file", "");
    const auto voc_file = declare_parameter<std::string>("vocabulary_file", "");
    const auto output_path = declare_parameter<std::string>("output_path", "");
    use_imu_ = declare_parameter<bool>("use_imu", true);
    const auto use_viewer = declare_parameter<bool>("use_viewer", true);
    const auto accel_scale = declare_parameter<double>("imu_accel_scale", 1.0);

    const auto image0_topic = declare_parameter<std::string>("image0_topic", "/cam0/image_raw");
    const auto image1_topic = declare_parameter<std::string>("image1_topic", "/cam1/image_raw");
    const auto imu_topic = declare_parameter<std::string>("imu_topic", "/imu");
    // "raw" for the sim (gz bridges sensor_msgs/Image), "compressed" for the
    // real rig, whose bags carry ONLY /camN/image_raw/compressed. This is what
    // lets one binary serve both rigs -- VINS needs two extra republish
    // processes for the same job (see config/wil/stereo_imu.yaml:32-38).
    const auto transport = declare_parameter<std::string>("image_transport", "raw");

    // --- dynamic object detection (RY-SLAM) ---------------------------------
    // OFF BY DEFAULT. With filter=false no detection subscription is created and
    // an empty mask travels all the way down, where ORBextractor short-circuits
    // on it -- so a default run is the stock tracker, bit for bit, and the
    // recorded sim baseline stays reproducible.
    filter_ = declare_parameter<bool>("filter", false);
    const auto dets_topic =
      declare_parameter<std::string>("dynamic_dets_topic", "/orbslam3/dynamic_dets");
    // Detection boxes clip their objects slightly, and a feature ON the silhouette
    // of a moving crate is exactly the one that corrupts the map. Grow them.
    mask_dilate_px_ = declare_parameter<int>("mask_dilate_px", 8);
    // Older than this and the object has moved on; masking with it would blank a
    // static region and leave the real one uncovered.
    det_max_age_ = declare_parameter<double>("det_max_age", 0.15);
    // Safety valve. One false positive spanning the frame would starve the tracker
    // of features and lose the map, which is far worse than not filtering.
    //
    // 0.8 is measured, not guessed. Over all 791 frames of dataset/dynamic_dataset
    // the dilated boxes cover mean 43.5% of the frame, p90 61%, max 72.8% -- the
    // warehouse is FULL of bins and boxes, most of them scenery. A 0.6 limit
    // therefore fires on 12% of frames during entirely normal operation, which is
    // the worst of both worlds: those frames get no filtering while their
    // neighbours do, so the dynamic points leak into the map anyway. 0.8 sits
    // above the observed maximum, so it only catches a genuinely runaway
    // detection while leaving normal operation consistently filtered.
    max_mask_fraction_ = declare_parameter<double>("max_mask_fraction", 0.8);

    world_frame_ = declare_parameter<std::string>("world_frame_id", "orbslam3_world");
    body_frame_ = declare_parameter<std::string>("body_frame_id", "base_footprint");
    publish_tf_ = declare_parameter<bool>("publish_tf", true);
    const auto sync_slop = declare_parameter<double>("sync_slop", 0.02);

    if (config_file.empty()) {
      throw std::runtime_error("parameter 'config_file' is required");
    }
    if (voc_file.empty()) {
      throw std::runtime_error("parameter 'vocabulary_file' is required");
    }

    RCLCPP_INFO(get_logger(), "settings   : %s", config_file.c_str());
    RCLCPP_INFO(get_logger(), "vocabulary : %s", voc_file.c_str());
    RCLCPP_INFO(get_logger(), "mode       : %s", use_imu_ ? "STEREO-INERTIAL" : "STEREO");
    RCLCPP_INFO(get_logger(), "transport  : %s", transport.c_str());
    if (use_imu_ && accel_scale != 1.0) {
      RCLCPP_WARN(
        get_logger(), "imu_accel_scale = %.4f -- accelerometer is being rescaled", accel_scale);
    }

    // --- SLAM ---------------------------------------------------------------
    SlamConfig cfg;
    cfg.vocabulary_path = voc_file;
    cfg.settings_path = config_file;
    cfg.use_imu = use_imu_;
    cfg.use_viewer = use_viewer;
    cfg.accel_scale = accel_scale;
    cfg.align_first_pose = declare_parameter<bool>("align_first_pose", false);
    cfg.wait_for_imu_init = declare_parameter<bool>("wait_for_imu_init", false);
    slam_ = std::make_unique<SlamWrapper>(cfg);
    slam_->setResultCallback([this](const TrackResult & r) {onTrackResult(r);});

    if (!output_path.empty()) {
      writer_ = std::make_unique<TrajectoryWriter>(output_path + "/vio.csv");
      RCLCPP_INFO(get_logger(), "trajectory : %s", writer_->path().c_str());
      // What the filter ACTUALLY did this run, one row per frame, keyed by the
      // same timestamp vio.csv uses so the two join on column 0. Not the same
      // thing as scoring the detector offline with script/yolo_eval.py: this
      // records the masks the tracker really saw, including the frames where no
      // detection was recent enough and the ones whose mask was rejected.
      if (filter_) {
        mask_log_.open(output_path + "/yolo_mask.csv");
        if (mask_log_) {
          mask_log_ << "timestamp_ns,n_boxes,mask_coverage,matched,det_age_ms,rejected\n";
          RCLCPP_INFO(get_logger(), "mask log   : %s/yolo_mask.csv", output_path.c_str());
        } else {
          RCLCPP_WARN(get_logger(), "could not open %s/yolo_mask.csv",
                      output_path.c_str());
        }
      }
    }

    // --- publishers ---------------------------------------------------------
    // Reliable, which a BEST_EFFORT subscriber (viewer.py asks for best-effort)
    // is still compatible with. The reverse would not be.
    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("~/odometry", 10);
    pose_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>("~/pose", 10);
    if (publish_tf_) {
      tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    }

    // --- subscriptions ------------------------------------------------------
    // SensorDataQoS (BEST_EFFORT) is compatible with both reliable and
    // best-effort publishers; a RELIABLE subscriber would silently receive
    // nothing from a best-effort one.
    const auto qos = rclcpp::SensorDataQoS().get_rmw_qos_profile();

    // ApproximateTime rather than ExactTime throughout: the sim bag holds 804
    // cam0 frames against 805 cam1 frames, so stamps are close but not identical.
    if (transport == "compressed") {
      // Subscribe to CompressedImage and decode with cv::imdecode ourselves,
      // rather than going through image_transport's compressed plugin.
      //
      // WHY. The real rig's bags carry format "yuv422; jpeg compressed mono8":
      // the ORIGINAL camera encoding is yuv422 while the JPEG payload is mono8.
      // The plugin decodes the payload to 1-channel mono (step = width) but
      // stamps the output Image with the original yuv422 encoding, which implies
      // 2 channels. cv_bridge then rightly rejects the message:
      //   "Image is wrongly formed: step < width * byte_depth * num_channels
      //    or 1920 != 1920 * 1 * 2"
      // and every single frame is dropped. Decoding straight to grayscale
      // sidesteps the mislabelled encoding entirely -- and grayscale is what
      // ORB-SLAM3 wants anyway. script/check_stereo_side.py reads these same
      // blobs the same way.
      cleft_sub_.subscribe(this, image0_topic + "/compressed", qos);
      cright_sub_.subscribe(this, image1_topic + "/compressed", qos);
      csync_ = std::make_unique<CompressedSync>(
        CompressedSyncPolicy(10), cleft_sub_, cright_sub_);
      csync_->setMaxIntervalDuration(rclcpp::Duration::from_seconds(sync_slop));
      csync_->registerCallback(
        std::bind(
          &Orbslam3Node::onStereoCompressed, this,
          std::placeholders::_1, std::placeholders::_2));
    } else {
      left_sub_.subscribe(this, image0_topic, transport, qos);
      right_sub_.subscribe(this, image1_topic, transport, qos);
      sync_ = std::make_unique<Sync>(SyncPolicy(10), left_sub_, right_sub_);
      sync_->setMaxIntervalDuration(rclcpp::Duration::from_seconds(sync_slop));
      sync_->registerCallback(
        std::bind(&Orbslam3Node::onStereo, this, std::placeholders::_1, std::placeholders::_2));
    }

    if (filter_) {
      dets_sub_ = create_subscription<vision_msgs::msg::Detection2DArray>(
        dets_topic, rclcpp::SensorDataQoS(),
        std::bind(&Orbslam3Node::onDetections, this, std::placeholders::_1));
      RCLCPP_INFO(
        get_logger(), "filter     : ON -- masking dynamic objects from %s "
        "(dilate %d px, max age %.0f ms)",
        dets_topic.c_str(), mask_dilate_px_, det_max_age_ * 1e3);
    } else {
      RCLCPP_INFO(get_logger(), "filter     : off (stock ORB-SLAM3 feature extraction)");
    }

    if (use_imu_) {
      // RELIABLE with a deep queue, NOT SensorDataQoS.
      //
      // SensorDataQoS is BEST_EFFORT with depth 5 -- about 25 ms of buffer at
      // 200 Hz, which a single 60 ms TrackStereo call overruns, so IMU samples
      // were being dropped exactly when the node was busiest. That matters more
      // than it sounds: ORB-SLAM3's PreintegrateIMU() dies (SIGSEGV) when only
      // ONE sample falls between consecutive frames, and dataset_real_000 has
      // three frame intervals holding exactly two -- one dropped message from
      // the cliff edge.
      //
      // Safe here because every producer offers RELIABLE: `ros2 bag play`
      // replays the recorded profile (all three topics in that bag are
      // RELIABLE/VOLATILE) and realsense_imu/imu_node.py publishes RELIABLE
      // deliberately, for this same reason. If you ever point this at a
      // BEST_EFFORT publisher the subscription will match nothing and go
      // silent -- that is the QoS trap viewer.py's docstring warns about.
      imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
        imu_topic, rclcpp::QoS(rclcpp::KeepLast(2000)),
        std::bind(&Orbslam3Node::onImu, this, std::placeholders::_1));
    }

    RCLCPP_INFO(
      get_logger(), "subscribed: %s | %s | %s",
      image0_topic.c_str(), image1_topic.c_str(), use_imu_ ? imu_topic.c_str() : "(no imu)");
  }

  /// Stops tracking and flushes the CSV. Called explicitly from main() before
  /// rclcpp::shutdown(), because the Pangolin viewer thread must be joined
  /// while the node is still alive.
  void finish()
  {
    if (finished_) {
      return;
    }
    finished_ = true;
    if (slam_) {
      slam_->shutdown();
    }
    if (writer_) {
      writer_->flush();
      RCLCPP_INFO(
        get_logger(), "wrote %zu poses to %s", writer_->rows(), writer_->path().c_str());
    }
    if (slam_ && slam_->starvedFrames() > 0) {
      RCLCPP_WARN(
        get_logger(),
        "skipped %zu frames with no IMU data covering them. A few at startup is "
        "normal; many means the IMU topic is lagging or absent.",
        slam_->starvedFrames());
    }
    if (filter_) {
      RCLCPP_INFO(
        get_logger(), "dynamic filter: %zu frames masked, %zu with no recent detection",
        det_matched_, det_missed_);
      if (det_matched_ == 0) {
        RCLCPP_WARN(
          get_logger(),
          "filter was ON but NOT ONE frame matched a detection. Is the detector node "
          "running, and is det_max_age (%.0f ms) long enough for its inference time?",
          det_max_age_ * 1e3);
      }
      if (mask_rejected_ > 0) {
        RCLCPP_WARN(
          get_logger(), "discarded %zu masks for exceeding max_mask_fraction",
          mask_rejected_);
      }
    }
    if (slam_ && slam_->droppedFrames() > 0) {
      RCLCPP_WARN(
        get_logger(),
        "dropped %zu stereo frames -- tracking could not keep up. "
        "Replay with `ros2 bag play --rate 0.5` for a clean run.",
        slam_->droppedFrames());
    }
  }

private:
  using SyncPolicy = message_filters::sync_policies::ApproximateTime<
    sensor_msgs::msg::Image, sensor_msgs::msg::Image>;
  using Sync = message_filters::Synchronizer<SyncPolicy>;
  using CompressedSyncPolicy = message_filters::sync_policies::ApproximateTime<
    sensor_msgs::msg::CompressedImage, sensor_msgs::msg::CompressedImage>;
  using CompressedSync = message_filters::Synchronizer<CompressedSyncPolicy>;

  static double stampSeconds(const builtin_interfaces::msg::Time & t)
  {
    return static_cast<double>(t.sec) + static_cast<double>(t.nanosec) * 1e-9;
  }

  /// Buffers detections by their SOURCE IMAGE stamp, which the detector copies
  /// verbatim. Deliberately NOT part of the stereo message_filters sync: a 3-way
  /// ApproximateTime would drop stereo pairs whenever inference falls behind the
  /// camera, and losing frames hurts tracking far more than an unfiltered frame
  /// does. Nearest-stamp lookup with a max-age bound degrades gracefully instead.
  void onDetections(const vision_msgs::msg::Detection2DArray::ConstSharedPtr msg)
  {
    std::vector<cv::Rect2f> boxes;
    boxes.reserve(msg->detections.size());
    for (const auto & det : msg->detections) {
      const float w = static_cast<float>(det.bbox.size_x);
      const float h = static_cast<float>(det.bbox.size_y);
      if (w <= 0.0f || h <= 0.0f) {
        continue;
      }
      boxes.emplace_back(
        static_cast<float>(det.bbox.center.position.x) - 0.5f * w,
        static_cast<float>(det.bbox.center.position.y) - 0.5f * h, w, h);
    }
    std::lock_guard<std::mutex> lk(dets_mu_);
    dets_.emplace_back(stampSeconds(msg->header.stamp), std::move(boxes));
    // A couple of seconds of history at camera rate. Anything older than
    // det_max_age is unusable anyway, so the cap only bounds memory.
    while (dets_.size() > 60) {
      dets_.pop_front();
    }
  }

  /// Builds the CV_8UC1 mask for a frame: 255 = static/keep, 0 = dynamic.
  /// Returns an empty Mat when filtering is off, when no detection is recent
  /// enough, or when the result would blank too much of the image.
  cv::Mat maskFor(double t, const cv::Size & size)
  {
    if (!filter_) {
      return cv::Mat();
    }

    std::vector<cv::Rect2f> boxes;
    {
      std::lock_guard<std::mutex> lk(dets_mu_);
      double best = det_max_age_;
      const std::vector<cv::Rect2f> * pick = nullptr;
      for (const auto & [t_det, b] : dets_) {
        const double age = std::fabs(t_det - t);
        if (age <= best) {
          best = age;
          pick = &b;
        }
      }
      if (!pick) {
        ++det_missed_;
        last_ = LogRow{0, 0.0, false, -1.0, false};
        return cv::Mat();
      }
      boxes = *pick;
      last_age_ = best;
    }
    ++det_matched_;

    if (boxes.empty()) {
      // A real "nothing dynamic in this frame" answer. Returning an empty Mat is
      // both correct and cheaper than an all-255 one -- ORBextractor skips the
      // per-keypoint test entirely.
      last_ = LogRow{0, 0.0, true, last_age_, false};
      return cv::Mat();
    }

    cv::Mat mask(size, CV_8UC1, cv::Scalar(255));
    const int d = std::max(0, mask_dilate_px_);
    double dynamic_area = 0.0;
    for (const auto & b : boxes) {
      cv::Rect r(
        static_cast<int>(std::floor(b.x)) - d, static_cast<int>(std::floor(b.y)) - d,
        static_cast<int>(std::ceil(b.width)) + 2 * d,
        static_cast<int>(std::ceil(b.height)) + 2 * d);
      r &= cv::Rect(0, 0, size.width, size.height);
      if (r.area() <= 0) {
        continue;
      }
      // Count before painting so overlapping boxes are not double-counted.
      dynamic_area += cv::countNonZero(mask(r));
      mask(r).setTo(0);
    }

    const double fraction = dynamic_area / static_cast<double>(size.area());
    if (fraction > max_mask_fraction_) {
      ++mask_rejected_;
      last_ = LogRow{boxes.size(), fraction, true, last_age_, true};
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "dynamic mask would cover %.0f%% of the frame (limit %.0f%%) -- ignoring it "
        "for this frame rather than starving the tracker",
        fraction * 100.0, max_mask_fraction_ * 100.0);
      return cv::Mat();
    }
    last_ = LogRow{boxes.size(), fraction, true, last_age_, false};
    return mask;
  }

  /// One row per frame into yolo_mask.csv. Timestamp is written as integer
  /// nanoseconds, exactly as TrajectoryWriter does for vio.csv, so a join on
  /// column 0 lines the two files up with no tolerance fudging.
  void logMask(const builtin_interfaces::msg::Time & stamp)
  {
    if (!mask_log_) {
      return;
    }
    const int64_t ns = static_cast<int64_t>(stamp.sec) * 1000000000LL +
      static_cast<int64_t>(stamp.nanosec);
    mask_log_ << ns << ',' << last_.n_boxes << ','
              << std::fixed << std::setprecision(4) << last_.coverage << ','
              << (last_.matched ? 1 : 0) << ','
              << std::setprecision(1) << (last_.age_s < 0 ? -1.0 : last_.age_s * 1e3)
              << ',' << (last_.rejected ? 1 : 0) << '\n';
    // Flush every row. A run is normally ended with Ctrl+C or a kill, neither of
    // which is guaranteed to run this object's destructor, and an unflushed
    // stream loses its tail -- which showed up as a half-written final line that
    // broke CSV parsing. One flush per frame at camera rate costs nothing.
    mask_log_.flush();
  }

  void onImu(const sensor_msgs::msg::Imu::ConstSharedPtr msg)
  {
    slam_->pushImu(
      stampSeconds(msg->header.stamp),
      {msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z},
      {msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z});
  }

  void onStereo(
    const sensor_msgs::msg::Image::ConstSharedPtr left,
    const sensor_msgs::msg::Image::ConstSharedPtr right)
  {
    cv_bridge::CvImageConstPtr l, r;
    try {
      // Convert to grayscale here rather than letting ORB-SLAM3 do it. That
      // makes the settings file's Camera.RGB irrelevant and sidesteps the
      // difference between the sim (rgb8) and the real rig (JPEG -> bgr8).
      l = cv_bridge::toCvShare(left, sensor_msgs::image_encodings::MONO8);
      r = cv_bridge::toCvShare(right, sensor_msgs::image_encodings::MONO8);
    } catch (const cv_bridge::Exception & e) {
      RCLCPP_ERROR(get_logger(), "cv_bridge: %s", e.what());
      return;
    }
    // The LEFT stamp is the frame time, and it is the same clock
    // script/extract_gt.py samples for ground_truth.csv, so no time alignment
    // is needed when comparing the two.
    const double t = stampSeconds(left->header.stamp);
    const cv::Mat mask = maskFor(t, l->image.size());
    logMask(left->header.stamp);
    slam_->pushStereo(t, l->image, r->image, mask);
  }

  void onStereoCompressed(
    const sensor_msgs::msg::CompressedImage::ConstSharedPtr left,
    const sensor_msgs::msg::CompressedImage::ConstSharedPtr right)
  {
    const cv::Mat l = cv::imdecode(cv::Mat(left->data), cv::IMREAD_GRAYSCALE);
    const cv::Mat r = cv::imdecode(cv::Mat(right->data), cv::IMREAD_GRAYSCALE);
    if (l.empty() || r.empty()) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "could not decode a compressed frame (format \"%s\")", left->format.c_str());
      return;
    }
    const double t = stampSeconds(left->header.stamp);
    const cv::Mat mask = maskFor(t, l.size());
    logMask(left->header.stamp);
    slam_->pushStereo(t, l, r, mask);
  }

  void onTrackResult(const TrackResult & r)
  {
    if (!r.tracking_ok) {
      // Throttled, because NOT_INITIALIZED fires every frame for the first few
      // seconds of every inertial run and would otherwise bury the log.
      if (r.state == -2) {
        RCLCPP_INFO_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "waiting for IMU initialisation (drive the robot; it needs motion)");
      } else {
        RCLCPP_INFO_THROTTLE(
          get_logger(), *get_clock(), 2000, "tracking state %d (no pose published)", r.state);
      }
      return;
    }
    if (!first_pose_seen_) {
      first_pose_seen_ = true;
      RCLCPP_INFO(get_logger(), "tracking OK -- publishing poses");
    }

    rclcpp::Time stamp(static_cast<int64_t>(r.stamp * 1e9), RCL_ROS_TIME);

    nav_msgs::msg::Odometry odom;
    odom.header.stamp = stamp;
    odom.header.frame_id = world_frame_;
    odom.child_frame_id = body_frame_;
    odom.pose.pose.position.x = r.position.x();
    odom.pose.pose.position.y = r.position.y();
    odom.pose.pose.position.z = r.position.z();
    odom.pose.pose.orientation.w = r.orientation.w();
    odom.pose.pose.orientation.x = r.orientation.x();
    odom.pose.pose.orientation.y = r.orientation.y();
    odom.pose.pose.orientation.z = r.orientation.z();
    // World-frame velocity in a twist field that is conventionally body-frame.
    // Deliberate: it matches what VINS writes to vio.csv, and plot_vio_vs_gt.py
    // only ever compares speed magnitude, which is frame-invariant.
    odom.twist.twist.linear.x = r.velocity.x();
    odom.twist.twist.linear.y = r.velocity.y();
    odom.twist.twist.linear.z = r.velocity.z();
    odom_pub_->publish(odom);

    geometry_msgs::msg::PoseStamped pose;
    pose.header = odom.header;
    pose.pose = odom.pose.pose;
    pose_pub_->publish(pose);

    if (tf_broadcaster_) {
      geometry_msgs::msg::TransformStamped tf;
      tf.header = odom.header;
      tf.child_frame_id = body_frame_;
      tf.transform.translation.x = r.position.x();
      tf.transform.translation.y = r.position.y();
      tf.transform.translation.z = r.position.z();
      tf.transform.rotation = odom.pose.pose.orientation;
      tf_broadcaster_->sendTransform(tf);
    }

    if (writer_) {
      writer_->write(r.stamp, r.position, r.orientation, r.velocity);
    }
  }

  bool use_imu_{true};
  bool publish_tf_{true};
  bool filter_{false};
  int mask_dilate_px_{8};
  double det_max_age_{0.15};
  double max_mask_fraction_{0.8};
  bool finished_{false};
  bool first_pose_seen_{false};
  std::string world_frame_, body_frame_;

  std::unique_ptr<SlamWrapper> slam_;
  std::unique_ptr<TrajectoryWriter> writer_;

  image_transport::SubscriberFilter left_sub_, right_sub_;
  std::unique_ptr<Sync> sync_;
  message_filters::Subscriber<sensor_msgs::msg::CompressedImage> cleft_sub_, cright_sub_;
  std::unique_ptr<CompressedSync> csync_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;

  rclcpp::Subscription<vision_msgs::msg::Detection2DArray>::SharedPtr dets_sub_;
  std::mutex dets_mu_;
  std::deque<std::pair<double, std::vector<cv::Rect2f>>> dets_;
  /// What maskFor() decided about the frame currently being pushed.
  struct LogRow
  {
    std::size_t n_boxes{0};
    double coverage{0.0};
    bool matched{false};
    double age_s{-1.0};
    bool rejected{false};
  };
  LogRow last_;
  double last_age_{-1.0};
  std::ofstream mask_log_;

  std::size_t det_matched_{0};
  std::size_t det_missed_{0};
  std::size_t mask_rejected_{0};

  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};

}  // namespace orbslam3_ros2

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  int rc = 0;
  try {
    auto node = std::make_shared<orbslam3_ros2::Orbslam3Node>();
    // Multi-threaded so the 200 Hz IMU subscription keeps draining while the
    // stereo callback is busy. The heavy TrackStereo call is on SlamWrapper's
    // own thread and never occupies an executor thread at all.
    rclcpp::executors::MultiThreadedExecutor exec;
    exec.add_node(node);
    exec.spin();
    node->finish();
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("orbslam3_node"), "%s", e.what());
    rc = 1;
  }
  rclcpp::shutdown();
  return rc;
}
