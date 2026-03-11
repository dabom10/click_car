#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.node import Node
from turtlebot4_navigation.turtlebot4_navigator import (
    TurtleBot4Directions,
    TurtleBot4Navigator,
)


# ================================
# 설정 상수
# ================================
ROBOT_NAMESPACE = "robot3"

# INITIAL_POSITION = [0.142463, 1.76936]
# INITIAL_DIRECTION = -5.43
INITIAL_POSITION = [0.0, 0.0]
INITIAL_DIRECTION = TurtleBot4Directions.NORTH

WAYPOINTS = [
    ([-0.725, -0.2], TurtleBot4Directions.WEST),
    ([-0.725, 1.9], TurtleBot4Directions.SOUTH),
    ([-2.1, 1.95], TurtleBot4Directions.EAST),
    ([-2.15, -0.3], TurtleBot4Directions.EAST),
    ([-2.3, -2.0], TurtleBot4Directions.EAST),
    ([-2.4, -4.0], TurtleBot4Directions.NORTH),
    ([2.1, -4.0], TurtleBot4Directions.WEST),
    ([1.97, -2.5], TurtleBot4Directions.SOUTH),
    ([-1.5, -2.2], TurtleBot4Directions.WEST),
]

# 구역 정의
# 형식:
# "구역명": {
#     "waypoints": [포함 웨이포인트 번호],
#     "min_x": 값,
#     "max_x": 값,
#     "min_y": 값,
#     "max_y": 값,
# }
ZONES = {
    "1": {
        "waypoints": [1, 2],
        "min_x": -1.67,
        "max_x": 0.38,
        "min_y": -1.37,
        "max_y": 2.55,
    },
    "2": {
        "waypoints": [3, 4],
        "min_x": -2.91,
        "max_x": -1.67,
        "min_y": -1.37,
        "max_y": 2.55,
    },
    "3": {
        "waypoints": [5],
        "min_x": -2.91,
        "max_x": -1.70,
        "min_y": -3.23,
        "max_y": -1.37,
    },
    "4": {
        "waypoints": [6, 7],
        "min_x": -3.00,
        "max_x": 2.60,
        "min_y": -5.15,
        "max_y": -3.23,
    },
    "5": {
        "waypoints": [8, 9],
        "min_x": -1.67,
        "max_x": 2.70,
        "min_y": -3.23,
        "max_y": -1.37,
    },
}


class AmclPoseMonitor(Node):

    def __init__(self):
        super().__init__(
            "amr2_pose_monitor",
            namespace=ROBOT_NAMESPACE,
        )

        self.current_x = None
        self.current_y = None

        self.create_subscription(
            PoseWithCovarianceStamped,
            "amcl_pose",
            self.amcl_pose_cb,
            10,
        )

    def amcl_pose_cb(self, msg: PoseWithCovarianceStamped):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

    def get_current_zone(self) -> str:
        if self.current_x is None or self.current_y is None:
            return "UNKNOWN"

        x = self.current_x
        y = self.current_y

        for zone_name, zone_info in ZONES.items():
            if (
                zone_info["min_x"] <= x <= zone_info["max_x"]
                and zone_info["min_y"] <= y <= zone_info["max_y"]
            ):
                return zone_name

        return "OUT_OF_ZONE"


def format_waypoint_label(index_zero_based):
    if index_zero_based is None:
        return "NONE"

    waypoint_number = index_zero_based + 1

    if waypoint_number < 1 or waypoint_number > len(WAYPOINTS):
        return "NONE"

    position, _ = WAYPOINTS[index_zero_based]
    return f"WP{waypoint_number}{tuple(position)}"


def main():
    rclpy.init()

    pose_monitor = AmclPoseMonitor()
    navigator = TurtleBot4Navigator(namespace=ROBOT_NAMESPACE)

    # 도킹 상태가 아니면 먼저 도킹해서 시작 상태를 맞춤
    if not navigator.getDockedStatus():
        navigator.info("Docking before initialising pose")
        navigator.dock()

    # 초기 위치 설정
    initial_pose = navigator.getPoseStamped(INITIAL_POSITION, INITIAL_DIRECTION)
    navigator.setInitialPose(initial_pose)

    # Nav2 활성화 대기
    navigator.waitUntilNav2Active()

    # 웨이포인트 생성
    goal_pose_list = []
    for position, direction in WAYPOINTS:
        goal_pose_list.append(navigator.getPoseStamped(position, direction))

    # 도킹 해제
    navigator.undock()

    # 웨이포인트 순차 이동 시작
    navigator.startFollowWaypoints(goal_pose_list)

    last_log_text = ""

    # 작업 완료까지 대기
    while not navigator.isTaskComplete():
        rclpy.spin_once(pose_monitor, timeout_sec=0.1)

        feedback = navigator.getFeedback()
        if feedback is None:
            continue

        current_goal_index = feedback.current_waypoint
        recent_arrived_index = current_goal_index - 1

        current_goal_text = format_waypoint_label(current_goal_index)
        recent_arrived_text = format_waypoint_label(recent_arrived_index)
        current_zone_text = pose_monitor.get_current_zone()

        log_text = (
            f"target={current_goal_text} | "
            f"recent={recent_arrived_text} | "
            f"zone={current_zone_text}"
        )

        if log_text != last_log_text:
            navigator.info(log_text)
            last_log_text = log_text

    # 마지막 상태 한 번 더 출력
    rclpy.spin_once(pose_monitor, timeout_sec=0.1)
    navigator.info(
        f"target=DONE | "
        f"recent={format_waypoint_label(len(WAYPOINTS) - 1)} | "
        f"zone={pose_monitor.get_current_zone()}"
    )

    # 완료 후 다시 dock
    navigator.dock()

    pose_monitor.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()