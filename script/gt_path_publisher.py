#!/usr/bin/env python3
"""Republish /ground_truth/odometry as a Path in the map frame, for eyeballing maps.

    python3 script/gt_path_publisher.py --ros-args -p use_sim_time:=true

WHY THIS EXISTS -- the frame collision
    The bags stamp ground truth `frame_id: "odom"`, and the SLAM stack repurposes
    `odom` for the FRONT-END's frame (localization_rtabmap.launch.py renames VINS /
    ORB-SLAM3 to publish odom -> vio_body). Those are different frames with
    different origins: ground truth starts at the Gazebo spawn (1.80, 8.99) while a
    front-end starts at (0, 0). So adding an Odometry display on
    /ground_truth/odometry in RViz draws the true route in the WRONG PLACE, silently
    and plausibly -- the shape looks right and the position is metres off.

    This node strips the collision by republishing into an explicit frame of its
    own, so what RViz draws is unambiguous.

WHEN THE OVERLAY IS MEANINGFUL
    Only when the map frame really is the Gazebo world frame, i.e. when the map was
    built with ground-truth odometry (script/gt_path_publisher.py's sibling,
    gt_path... see gt_tf_publisher.py). Then map == world and the true route lands
    exactly on the map, so any mismatch you see IS map error. That is the useful
    case for "simple map evaluation".

    If the map was built from VINS or ORB-SLAM3 with no ground truth, its frame is
    anchored wherever that front-end initialised, and it is NOT the Gazebo world
    frame. The overlay is then offset by an unknown rigid transform and comparing
    by eye is meaningless. Pass --frame odom_gt in that case: the path appears in
    its own disconnected frame, RViz says so plainly, and you are not fooled.
"""
import sys

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped


class GtPath(Node):
    def __init__(self):
        super().__init__('gt_path_publisher')
        self.declare_parameter('frame', 'map')
        self.declare_parameter('in_topic', '/ground_truth/odometry')
        self.declare_parameter('out_topic', '/ground_truth/path')
        # One pose every N metres. The raw feed is 40 Hz; a Path holding every
        # sample of a 90 s run is ~3600 poses re-serialised on every update, which
        # is a lot of bandwidth for a line you are looking at.
        self.declare_parameter('min_step', 0.05)

        self.frame = self.get_parameter('frame').value
        self.min_step = float(self.get_parameter('min_step').value)
        self.path = Path()
        self.path.header.frame_id = self.frame
        self.last = None

        self.pub = self.create_publisher(
            Path, self.get_parameter('out_topic').value, 1)
        self.create_subscription(
            Odometry, self.get_parameter('in_topic').value, self.cb, 20)
        self.get_logger().info(
            f"ground truth -> {self.get_parameter('out_topic').value} "
            f"in frame '{self.frame}'")
        if self.frame == 'map':
            self.get_logger().warn(
                'frame=map assumes the map was built with GROUND-TRUTH odometry. '
                'If it was built by VINS/ORB-SLAM3 the frames differ by an unknown '
                'transform and this overlay is not comparable -- use --frame odom_gt.')

    def cb(self, msg):
        p = msg.pose.pose.position
        if self.last is not None:
            dx, dy = p.x - self.last[0], p.y - self.last[1]
            if dx * dx + dy * dy < self.min_step * self.min_step:
                return
        self.last = (p.x, p.y)

        ps = PoseStamped()
        ps.header.stamp = msg.header.stamp
        ps.header.frame_id = self.frame
        ps.pose = msg.pose.pose
        self.path.poses.append(ps)
        self.path.header.stamp = msg.header.stamp
        self.pub.publish(self.path)


def main():
    rclpy.init()
    node = GtPath()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
