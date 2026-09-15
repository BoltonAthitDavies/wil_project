#!/usr/bin/env python3
"""YOLO dynamic-object detector for orbslam3_ros2 (RY-SLAM style).

Runs an Ultralytics detector on the LEFT camera and publishes the boxes of every
class considered dynamic. The C++ node rasterizes them into a mask and hands it to
ORB-SLAM3, whose ORBextractor drops the keypoints that land on them -- see
ORBextractor::ComputeKeyPointsOctTree in the patched thirdparty/ORB_SLAM3.

WHY THIS IS A SEPARATE PROCESS
    The tracker is C++ and the model is a PyTorch .pt. Loading it in-process would
    mean an ONNX export plus an ONNX Runtime dependency in CMakeLists; here it costs
    one topic. It also means a slow or crashed detector degrades SLAM to "unfiltered"
    rather than taking the tracker down with it.

WHY THE DETECTIONS ARE NOT IN THE STEREO SYNC
    The C++ side deliberately does NOT put this topic in its message_filters sync. A
    3-way ApproximateTime would drop stereo pairs every time inference falls behind
    frame rate, which costs SLAM far more than the dynamic features do. It looks up
    the nearest detection by timestamp instead, with a max-age bound.

    That only works because every message here is stamped with the SOURCE IMAGE's
    stamp, never now(). Do not "fix" that.

WEIGHT-AGNOSTIC BY DESIGN
    dynamic_classes is a list of class NAMES, resolved against the model's own
    names dict at load. Point weights_path at a different .pt and the node adapts;
    names that the model does not know are reported loudly at startup rather than
    silently never firing.
"""

from __future__ import annotations

import time
from typing import Dict, List

import cv2
import numpy as np
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image
from vision_msgs.msg import (
    BoundingBox2D,
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
)


def stamp_to_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def to_bgr(message: Image) -> np.ndarray:
    """sensor_msgs/Image -> HxWx3 BGR, without a cv_bridge dependency.

    The model has channels: 3, so a mono source has to be replicated up. Detection
    on a grayscale-replicated image is weaker than on true colour but is exactly
    what the sim's left camera provides in some bridge configurations, so it must
    work rather than throw.
    """
    encoding = message.encoding.lower()
    buf = np.frombuffer(message.data, dtype=np.uint8)

    if encoding in ("bgr8", "rgb8"):
        img = buf.reshape(message.height, message.width, 3)
        return img[:, :, ::-1] if encoding == "rgb8" else img
    if encoding in ("bgra8", "rgba8"):
        img = buf.reshape(message.height, message.width, 4)
        code = cv2.COLOR_BGRA2BGR if encoding == "bgra8" else cv2.COLOR_RGBA2BGR
        return cv2.cvtColor(img, code)
    if encoding in ("mono8", "8uc1"):
        img = buf.reshape(message.height, message.width)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if encoding in ("mono16", "16uc1"):
        img = buf.view(np.uint16).reshape(message.height, message.width)
        img = cv2.convertScaleAbs(img, alpha=255.0 / 65535.0)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    raise ValueError(f"unsupported image encoding: {message.encoding}")


class DynamicDetectorNode(Node):

    def __init__(self) -> None:
        super().__init__("dynamic_detector")

        weights = self.declare_parameter(
            "weights_path", "/home/ambushee/wil_project/weight/best.pt").value
        image_topic = self.declare_parameter("image_topic", "/cam0/image_raw").value
        # Mirrors the C++ node's convention: "raw" for the sim (gz bridges
        # sensor_msgs/Image), "compressed" for the real rig, whose bags carry only
        # /camN/image_raw/compressed.
        transport = self.declare_parameter("image_transport", "raw").value
        dets_topic = self.declare_parameter(
            "detections_topic", "/orbslam3/dynamic_dets").value
        wanted = list(self.declare_parameter(
            "dynamic_classes", ["bin", "box", "bucket"]).value)
        self.conf = float(self.declare_parameter("conf", 0.35).value)
        self.iou = float(self.declare_parameter("iou", 0.5).value)
        self.imgsz = int(self.declare_parameter("imgsz", 640).value)
        device = self.declare_parameter("device", "").value
        # 0 = run on every frame. Raise the floor between inferences if the GPU
        # cannot keep up; the C++ side reuses the most recent detection within
        # det_max_age, so a modest throttle costs accuracy, not stability.
        self.max_rate_hz = float(self.declare_parameter("max_rate_hz", 0.0).value)
        debug = bool(self.declare_parameter("publish_debug_image", False).value)

        # Imported here, not at module scope: ultralytics drags in torch and takes
        # seconds to load, and a missing install should produce a legible ROS error
        # rather than an ImportError traceback before the node even exists.
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is not installed -- run "
                "`python3 -m pip install --user ultralytics`") from exc

        if not device:
            try:
                import torch
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        self.device = device

        self.get_logger().info(f"weights    : {weights}")
        self.model = YOLO(weights)
        names: Dict[int, str] = dict(self.model.names)

        # Resolve requested class NAMES to the ids this particular checkpoint uses.
        by_name = {str(v).lower(): int(k) for k, v in names.items()}
        self.keep_ids: List[int] = []
        missing: List[str] = []
        for name in wanted:
            key = str(name).strip().lower()
            if key in by_name:
                self.keep_ids.append(by_name[key])
            else:
                missing.append(str(name))
        self.names = names

        resolved = ", ".join(f"{names[i]}({i})" for i in self.keep_ids) or "none"
        self.get_logger().info(
            f"classes    : resolved {len(self.keep_ids)}/{len(wanted)} -- {resolved}")
        if missing:
            self.get_logger().warn(
                f"these dynamic_classes are NOT in the model: {', '.join(missing)} "
                f"(model knows {len(names)} classes)")
        if not self.keep_ids:
            # Not fatal on purpose: the node still publishes empty arrays, so the
            # C++ side sees a live detector and simply filters nothing. That is a
            # much clearer failure mode than a node that exits at startup.
            self.get_logger().error(
                "NO dynamic classes resolved -- every frame will publish an empty "
                "detection array and filtering will be a no-op. Check weights_path.")

        self.get_logger().info(f"device     : {self.device}")
        self.get_logger().info(f"transport  : {transport}")
        self.get_logger().info(f"detections : {dets_topic}")

        # WarmUp, as RY-SLAM section III-B describes: push a few dummy tensors
        # through so CUDA context creation and cuDNN autotuning happen now instead
        # of stalling the first real frame by a second or more.
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        t0 = time.monotonic()
        for _ in range(3):
            self.model.predict(dummy, imgsz=self.imgsz, device=self.device,
                               verbose=False)
        self.get_logger().info(f"warmup     : {1e3 * (time.monotonic() - t0):.0f} ms")

        self.pub = self.create_publisher(Detection2DArray, dets_topic, 10)
        self.debug_pub = (
            self.create_publisher(Image, "/orbslam3/dynamic_debug", 1) if debug else None)

        if transport == "compressed":
            self.sub = self.create_subscription(
                CompressedImage, image_topic + "/compressed",
                self.on_compressed, qos_profile_sensor_data)
            self.get_logger().info(f"image      : {image_topic}/compressed")
        else:
            self.sub = self.create_subscription(
                Image, image_topic, self.on_image, qos_profile_sensor_data)
            self.get_logger().info(f"image      : {image_topic}")

        self.last_infer = 0.0
        self.n_frames = 0
        self.n_boxes = 0
        self.infer_ms = 0.0
        # STEADY_TIME, not the node clock. Under use_sim_time the clock jumps from
        # 0 to the bag's start the instant /clock appears, and a sim-time timer
        # tries to "catch up" by firing once per missed period -- thousands of
        # times, flooding the log and burying the tracker's own output. This is a
        # wall-clock progress report, so a wall clock is also the correct one.
        self.create_timer(10.0, self.report, clock=Clock(clock_type=ClockType.STEADY_TIME))

    # -- callbacks ----------------------------------------------------------

    def on_image(self, msg: Image) -> None:
        if self.throttled():
            return
        try:
            bgr = to_bgr(msg)
        except ValueError as exc:
            self.get_logger().warn(str(exc), throttle_duration_sec=5.0)
            return
        self.detect_and_publish(bgr, msg.header)

    def on_compressed(self, msg: CompressedImage) -> None:
        if self.throttled():
            return
        # IMREAD_COLOR, not GRAYSCALE: unlike the tracker, the model wants colour.
        bgr = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            self.get_logger().warn("failed to decode compressed image",
                                   throttle_duration_sec=5.0)
            return
        self.detect_and_publish(bgr, msg.header)

    def throttled(self) -> bool:
        if self.max_rate_hz <= 0.0:
            return False
        now = time.monotonic()
        if now - self.last_infer < 1.0 / self.max_rate_hz:
            return True
        self.last_infer = now
        return False

    # -- inference ----------------------------------------------------------

    def detect_and_publish(self, bgr: np.ndarray, header) -> None:
        t0 = time.monotonic()
        # classes=self.keep_ids filters inside ultralytics, before NMS, so nothing
        # downstream has to know about class ids at all. An empty list would mean
        # "no filter" to ultralytics, i.e. the exact opposite of what we want, so
        # that case is short-circuited here.
        if self.keep_ids:
            results = self.model.predict(
                bgr, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                classes=self.keep_ids, device=self.device, verbose=False)
        else:
            results = []
        self.infer_ms += 1e3 * (time.monotonic() - t0)

        out = Detection2DArray()
        # The source image's stamp, NOT now(). The C++ side matches detections to
        # frames by this value; using now() would offset every mask by the
        # inference latency and drag the masks behind the objects.
        out.header = header

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), score, cls in zip(xyxy, confs, clss):
                det = Detection2D()
                det.header = header
                bbox = BoundingBox2D()
                bbox.center.position.x = float((x1 + x2) * 0.5)
                bbox.center.position.y = float((y1 + y2) * 0.5)
                bbox.center.theta = 0.0
                bbox.size_x = float(x2 - x1)
                bbox.size_y = float(y2 - y1)
                det.bbox = bbox
                hyp = ObjectHypothesisWithPose()
                # class_id carries the NAME, not the index: the index is meaningless
                # to anything that did not load this exact checkpoint.
                hyp.hypothesis.class_id = str(self.names.get(int(cls), int(cls)))
                hyp.hypothesis.score = float(score)
                det.results.append(hyp)
                out.detections.append(det)

        # Published even when empty. An empty array is the positive statement "this
        # frame had no dynamic objects", which is what lets the C++ side tell a
        # clean frame apart from a dead detector.
        self.pub.publish(out)
        self.n_frames += 1
        self.n_boxes += len(out.detections)

        if self.debug_pub is not None:
            self.publish_debug(bgr, out, header)

    def publish_debug(self, bgr: np.ndarray, dets: Detection2DArray, header) -> None:
        canvas = bgr.copy()
        for det in dets.detections:
            cx, cy = det.bbox.center.position.x, det.bbox.center.position.y
            hw, hh = det.bbox.size_x * 0.5, det.bbox.size_y * 0.5
            p1 = (int(cx - hw), int(cy - hh))
            p2 = (int(cx + hw), int(cy + hh))
            cv2.rectangle(canvas, p1, p2, (0, 0, 255), 2)
            if det.results:
                label = f"{det.results[0].hypothesis.class_id} {det.results[0].hypothesis.score:.2f}"
                cv2.putText(canvas, label, (p1[0], max(0, p1[1] - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        msg = Image()
        msg.header = header
        msg.height, msg.width = canvas.shape[:2]
        msg.encoding = "bgr8"
        msg.is_bigendian = 0
        msg.step = 3 * msg.width
        msg.data = canvas.tobytes()
        self.debug_pub.publish(msg)

    def report(self) -> None:
        if not self.n_frames:
            self.get_logger().warn("no images received yet")
            return
        self.get_logger().info(
            f"{self.n_frames} frames, {self.n_boxes} boxes, "
            f"{self.infer_ms / self.n_frames:.1f} ms/frame")


def main() -> None:
    rclpy.init()
    node = None
    try:
        node = DynamicDetectorNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
