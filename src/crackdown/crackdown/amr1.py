#!/usr/bin/env python3
"""
patrol_node.py
--------------
TurtleBot4Navigator 기반 웨이포인트 순환 순찰

참고: turtlebot4_navigation 예제 코드 기반
"""

import math
import rclpy
from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Directions, TurtleBot4Navigator
from nav_msgs.msg import Odometry


def quaternion_to_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

def get_current_yaw():
    node = rclpy.create_node('yaw_reader')
    yaw = [None]

    def odom_cb(msg):
        yaw[0] = quaternion_to_yaw(msg.pose.pose.orientation)

    node.create_subscription(Odometry, '/robot3/odom', odom_cb, 1)
    while yaw[0] is None:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    return yaw[0]

# ── 웨이포인트 정의 ──────────────────────────────────────
WAYPOINTS = [
    ([-0.725, -0.2],  TurtleBot4Directions.WEST),
    ([-0.725,  1.9],  TurtleBot4Directions.SOUTH),
    ([-2.1,   1.95],  TurtleBot4Directions.EAST),
    ([-2.15, -0.3],   TurtleBot4Directions.EAST),
    ([-2.3,  -2.0],   TurtleBot4Directions.EAST),
    ([-2.4,  -4.0],   TurtleBot4Directions.NORTH),
    ([ 2.1,  -4.0],   TurtleBot4Directions.WEST),
    ([ 1.97, -2.5],   TurtleBot4Directions.SOUTH),
    ([-1.5,  -2.2],   TurtleBot4Directions.WEST),
    # ([-0.63, -0.20],  TurtleBot4Directions.NORTH),    # dock
]

# INITIAL_POSITION = [0.0,0.0]    # 1 기준
INITIAL_POSITION = [0.1424, 1.769]    # 2 기준

# DOCKING_POSITION1 = [-0.63, -0.20]
DOCKING_POSITION2 = [-2.5, -1.7]

def main():
    rclpy.init()

    navigator = TurtleBot4Navigator()

    # ── 도킹 상태 확인 후 초기 포즈 설정 ────────────────
    if not navigator.getDockedStatus():
        navigator.info('도킹 상태 아님 → 도킹 후 초기 포즈 설정')
        navigator.dock()

    actual_yaw = get_current_yaw()
    navigator.info(f'측정된 yaw: {actual_yaw:.3f} rad')
    initial_pose = navigator.getPoseStamped(INITIAL_POSITION, actual_yaw)
    navigator.setInitialPose(initial_pose)

    # ── Nav2 준비 대기 ────────────────────────────────
    navigator.waitUntilNav2Active()

    # ── 언독 ─────────────────────────────────────────
    navigator.undock()

    # ── 웨이포인트 리스트 생성 ────────────────────────
    goal_pose = []
    for (pos, direction) in WAYPOINTS:
        goal_pose.append(navigator.getPoseStamped(pos, direction))

    navigator.info(f'웨이포인트 {len(goal_pose)}개 등록 완료. 순찰 시작!')

    # ── 무한 순환 루프 ────────────────────────────────
    try:
        while True:
            navigator.startFollowWaypoints(goal_pose)
    except KeyboardInterrupt:
        navigator.cancelTask()  # Nav2 목표 취소
        navigator.startFollowWaypoints(DOCKING_POSITION2, TurtleBot4Directions.NORTH)
        navigator.dock()

    # ── 이후 탈출 조건 생기면 여기 실행됨 ────────────
    # navigator.info('단속 대상 탐지 → 순찰 중단')

    # navigator.dock()

    rclpy.shutdown()

if __name__ == '__main__':
    main()