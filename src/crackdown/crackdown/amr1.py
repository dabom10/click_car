#!/usr/bin/env python3
"""
patrol_node.py
--------------
TurtleBot4Navigator 기반 웨이포인트 순환 순찰
+ 탐지 신호 실시간 구독 (스레딩)

참고: turtlebot4_navigation 예제 코드 기반
"""

import math
import threading
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Directions, TurtleBot4Navigator
from nav_msgs.msg import Odometry
from std_msgs.msg import String


def quaternion_to_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

def get_current_yaw(robot_ns):
    node = rclpy.create_node('yaw_reader')
    yaw = [None]

    def odom_cb(msg):
        yaw[0] = quaternion_to_yaw(msg.pose.pose.orientation)

    node.create_subscription(Odometry, f'/{robot_ns}/odom', odom_cb, 1)
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
]

# ── robot3 (AMR 1번) 기준 ──────────────────────────
INITIAL_POSITION  = [0.0, 0.0]      # dock 1 기준
# INITIAL_POSITION  = [0.1424, 1.769]   # dock 2 기준

DOCKING_POSITION = [-0.63, -0.20]   # dock 1
# DOCKING_POSITION  = [-2.5, -1.7]      # dock 2

class PatrolNode(Node):
    def __init__(self, robot_ns='robot3'):
        super().__init__('patrol_node')

        self.robot_ns = robot_ns

        # ── 탐지 플래그 ──────────────────────────────
        self.detected    = False
        self.detected_x  = None
        self.detected_y  = None

        # ── Navigator ────────────────────────────────
        self.navigator = TurtleBot4Navigator()

        # ── 탐지 토픽 구독 ───────────────────────────
        self.create_subscription(
            String,
            f'/{robot_ns}/detection_result',
            self.detection_callback,
            10
        )
        self.get_logger().info(f'[{robot_ns}] 탐지 신호 구독 시작')

    def detection_callback(self, msg):
        self.get_logger().info(f'탐지 신호 수신: {msg.data}')

        if msg.data.startswith('detected:'):
            coords = msg.data.split(':')[1]
            x, y = map(float, coords.split(','))
            self.detected_x = x
            self.detected_y = y
            self.detected   = True

            # ── 순찰 즉시 중단 ───────────────────────
            self.navigator.cancelTask()
            self.get_logger().info(f'차량 탐지! x:{x} y:{y} → 순찰 중단')

        elif msg.data == 'cleared':
            self.detected   = False
            self.detected_x = None
            self.detected_y = None
            self.get_logger().info('차량 사라짐 → 플래그 초기화')

def main():
    rclpy.init()

    node = PatrolNode(robot_ns='robot3')

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    navigator = node.navigator

    # ── 도킹 상태 확인 후 초기 포즈 설정 ────────────────
    if not navigator.getDockedStatus():
        navigator.info('도킹 상태 아님 → 도킹 후 초기 포즈 설정')
        navigator.dock()

    actual_yaw = get_current_yaw('robot3')
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
        while rclpy.ok():
            navigator.info('== 순찰 한 바퀴 시작 ==')
            navigator.startFollowWaypoints(goal_pose)

            # 탐지 신호로 cancelTask() 된 경우
            if node.detected:
                navigator.info(f'단속 대상 탐지 → 순찰 중단')
                navigator.info(f'목표 좌표: x={node.detected_x}, y={node.detected_y}')
                # ── 다음 단계: 여기서 탐지 좌표로 이동 로직 추가 예정 ──
                break
            navigator.info('== 한 바퀴 완료, 다시 시작 ==')

    except KeyboardInterrupt:
        navigator.info('Ctrl+C → 순찰 중단')
        navigator.cancelTask()

    finally:
        navigator.info('도킹 위치로 이동 중...')
        dock_pose = navigator.getPoseStamped(DOCKING_POSITION, TurtleBot4Directions.NORTH)
        navigator.startToPose(dock_pose)
        navigator.info('도킹 시작')
        navigator.dock()
        rclpy.shutdown()

if __name__ == '__main__':
    main()