#!/usr/bin/env python3
"""
amr1.py
-------
robot3 - 출발 신호 수신 후 순찰 시작
구역 기반 이동 + 단속 + 도킹 + 재시작 대기

실행:
  ros2 run crackdown amr1 --ros-args -r __ns:=/robot3

토픽 송신:
  출발: ros2 topic pub /robot2/patrol_command std_msgs/String "{data: 'start'}" --once
  이동: ros2 topic pub /robot2/goto_target std_msgs/String "{data: '0.5,-4.7'}" --once
  정지: ros2 topic pub /robot2/patrol_command std_msgs/String "{data: 'stop'}" --once
"""

import math
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Directions, TurtleBot4Navigator
from nav_msgs.msg import Odometry
from std_msgs.msg import String

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

ZONES = [
    {
        "zone_id": 1,
        "waypoint_indices": [0, 1],
        "corner1": ( 0.38,  2.55),
        "corner2": (-1.67, -1.37),
    },
    {
        "zone_id": 2,
        "waypoint_indices": [2, 3],
        "corner1": (-1.67,  2.55),
        "corner2": (-2.91, -1.37),
    },
    {
        "zone_id": 3,
        "waypoint_indices": [4],
        "corner1": (-2.91, -1.37),
        "corner2": (-1.7,  -3.23),
    },
    {
        "zone_id": 4,
        "waypoint_indices": [5, 6],
        "corner1": (-3.0,  -3.23),
        "corner2": ( 2.6,  -5.15),
    },
    {
        "zone_id": 5,
        "waypoint_indices": [7, 8],
        "corner1": (-1.67, -1.37),
        "corner2": ( 2.7,  -3.23),
    },
]
PATROL_ORDER  = [0, 1, 2, 3, 4, 5, 6, 7, 8]
PATROL_LENGTH = len(PATROL_ORDER)
FINAL_WAYPOINT_INDEX = 0

# ── robot3 (AMR 1번) 기준 ──────────────────────────
INITIAL_POSITION  = [0.0, 0.0]      # dock 1 기준
# INITIAL_POSITION  = [0.1424, 1.769]   # dock 2 기준

PRE_DOCK_POSITION = [-0.63, -0.20]   # dock 1
# PRE_DOCK_POSITION  = [-2.5, -1.7]      # dock 2
PRE_DOCK_DIRECTION = TurtleBot4Directions.NORTH

ENFORCEMENT_WAIT   = 10.0  # 단속 대기 시간 (초) - 파라미터로 조정 가능
TASK_POLL_PERIOD_SEC = 0.1

# ================================
# 모드 상수
# ================================
MODE_PATROL        = "PATROL"
MODE_ROUTE_TO_ZONE = "ROUTE_TO_ZONE"
MODE_WAIT_RESUME   = "WAIT_RESUME"

# ================================
# 유틸 함수
# ================================
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

def point_in_zone(x, y, zone):
    x1, y1 = zone["corner1"]
    x2, y2 = zone["corner2"]
    xmin, xmax = min(x1, x2), max(x1, x2)
    ymin, ymax = min(y1, y2), max(y1, y2)
    return xmin <= x <= xmax and ymin <= y <= ymax

def find_zone_by_point(x, y):
    for zone in ZONES:
        if point_in_zone(x, y, zone):
            return zone
    return None

def waypoint_index_to_patrol_pos(waypoint_index):
    return PATROL_ORDER.index(waypoint_index)

def patrol_pos_to_waypoint_index(patrol_pos):
    return PATROL_ORDER[patrol_pos % PATROL_LENGTH]

def next_patrol_pos(patrol_pos):
    return (patrol_pos + 1) % PATROL_LENGTH

def move_to_waypoint(navigator, waypoint_index, log_prefix):
    pos, direction = WAYPOINTS[waypoint_index]
    navigator.info(f'{log_prefix}: waypoint {waypoint_index+1} {pos} 이동 시작')
    pose = navigator.getPoseStamped(pos, direction)
    navigator.goToPose(pose)
    while not navigator.isTaskComplete():
        time.sleep(TASK_POLL_PERIOD_SEC)
    navigator.info(f'{log_prefix}: waypoint {waypoint_index+1} 도착 완료')

def move_to_pre_dock_and_dock(navigator):
    navigator.info('Pre-dock 위치로 이동 중...')
    pre_dock_pose = navigator.getPoseStamped(PRE_DOCK_POSITION, PRE_DOCK_DIRECTION)
    navigator.goToPose(pre_dock_pose)
    while not navigator.isTaskComplete():
        time.sleep(TASK_POLL_PERIOD_SEC)
    navigator.info('Pre-dock 도착 → 도킹 실행')
    navigator.dock()

def do_enforcement(navigator, target_x, target_y, wait_sec=ENFORCEMENT_WAIT):
    """목표 좌표로 이동 → 단속 대기"""
    navigator.info(f'[단속] 목표 좌표 ({target_x}, {target_y}) 로 이동 시작')
    goal_pose = navigator.getPoseStamped(
        [target_x, target_y],
        TurtleBot4Directions.NORTH
    )
    navigator.goToPose(goal_pose)
    while not navigator.isTaskComplete():
        time.sleep(TASK_POLL_PERIOD_SEC)

    navigator.info(f'[단속] 도착 완료 → {wait_sec}초 단속 대기')
    time.sleep(wait_sec)
    navigator.info('[단속] 단속 완료!')

class AMR2Node(Node):
    def __init__(self, robot_ns='robot3'):
        super().__init__('amr1_node')

        self.robot_ns = robot_ns
        self.state_lock = threading.Lock()

        # ── 탐지 플래그 ──────────────────────────────
        self.start_patrol   = False
        self.goto_requested = False
        self.stop_requested = False
        self.target_x       = None
        self.target_y       = None

        # ── Navigator ────────────────────────────────
        self.navigator = TurtleBot4Navigator()

        # ── 탐지 토픽 구독 ───────────────────────────
        self.create_subscription(
            String,
            f'/{robot_ns}/patrol_command',
            self.patrol_command_callback,
            10
        )

        self.create_subscription(
            String,
            f'/{robot_ns}/goto_target',
            self.goto_target_callback,
            10
        )

        self.get_logger().info(f'[{robot_ns}] 초기화 완료')

    def patrol_command_callback(self, msg):
        with self.state_lock:
            if msg.data == 'start':
                self.start_patrol = True
                self.get_logger().info('출발 신호 수신 → 순찰 시작!')
            elif msg.data == 'stop':
                self.stop_requested = True
                self.get_logger().info('정지 신호 수신 → waypoint 1까지 완료 후 도킹')

    def goto_target_callback(self, msg):
        try:
            x, y = map(float, msg.data.split(','))
            with self.state_lock:
                self.target_x       = x
                self.target_y       = y
                self.goto_requested = True
            self.get_logger().info(f'목표 좌표 수신: x={x}, y={y}')
        except Exception as e:
            self.get_logger().error(f'좌표 파싱 실패: {msg.data} → {e}')

# ================================
# 순찰 한 사이클
# ================================
def patrol_cycle(node: AMR2Node):
    navigator = node.navigator

    # ── 출발 신호 대기 ────────────────────────────────
    navigator.info('[robot3] 출발 신호 대기 중...')
    with node.state_lock:
        node.start_patrol = False  # 이전 신호 초기화

    while rclpy.ok():
        with node.state_lock:
            if node.start_patrol:
                break
        time.sleep(0.1)

    # ── 초기 설정 ─────────────────────────────────────
    if not navigator.getDockedStatus():
        navigator.info('도킹 상태 아님 → 도킹 후 초기 포즈 설정')
        navigator.dock()

    actual_yaw = get_current_yaw('robot3')
    navigator.info(f'측정된 yaw: {actual_yaw:.3f} rad')
    initial_pose = navigator.getPoseStamped(INITIAL_POSITION, actual_yaw)
    navigator.setInitialPose(initial_pose)
    navigator.waitUntilNav2Active()
    navigator.undock()
    navigator.info('순찰 시작!')

    current_mode = MODE_PATROL
    patrol_pos   = 0

    target_zone_id                = None
    target_zone_first_index       = None
    target_zone_first_patrol_pos  = None
    target_zone_resume_patrol_pos = None
    enforcement_x                 = None
    enforcement_y                 = None

    while rclpy.ok():

        with node.state_lock:
            local_goto     = node.goto_requested
            local_stop     = node.stop_requested
            local_target_x = node.target_x
            local_target_y = node.target_y

        # ── WAIT_RESUME 상태 ─────────────────────────
        if current_mode == MODE_WAIT_RESUME:
            if local_stop:
                # waypoint 1까지 순찰 후 도킹
                current_mode = MODE_PATROL
                if target_zone_resume_patrol_pos is not None:
                    patrol_pos = target_zone_resume_patrol_pos
                navigator.info(
                    f'WAIT_RESUME 중 stop → '
                    f'waypoint {patrol_pos_to_waypoint_index(patrol_pos)+1} 부터 '
                    f'waypoint 1까지 완료 후 도킹'
                )
                continue

            # ── 목표 좌표로 이동 + 단속 ──────────────
            do_enforcement(navigator, enforcement_x, enforcement_y, wait_sec=ENFORCEMENT_WAIT)

            # ── 단속 완료 → 순찰 재개 ────────────────
            current_mode = MODE_PATROL
            if target_zone_resume_patrol_pos is not None:
                patrol_pos = target_zone_resume_patrol_pos
            navigator.info(
                f'단속 완료 → waypoint {patrol_pos_to_waypoint_index(patrol_pos)+1} 부터 순찰 재개'
            )
            continue

        # ── 목표 좌표 요청 처리 ──────────────────────
        if local_goto and current_mode != MODE_ROUTE_TO_ZONE:
            zone = find_zone_by_point(local_target_x, local_target_y)
            if zone is None:
                navigator.info(f'좌표 ({local_target_x}, {local_target_y}) → 어떤 구역에도 속하지 않음')
            else:
                target_zone_id                = zone["zone_id"]
                target_zone_first_index       = zone["waypoint_indices"][0]
                target_zone_first_patrol_pos  = waypoint_index_to_patrol_pos(target_zone_first_index)
                target_zone_resume_patrol_pos = next_patrol_pos(target_zone_first_patrol_pos)
                enforcement_x                 = local_target_x
                enforcement_y                 = local_target_y
                current_mode = MODE_ROUTE_TO_ZONE
                navigator.info(
                    f'목표 ({local_target_x}, {local_target_y}) → '
                    f'구역 {target_zone_id}, '
                    f'첫 waypoint {target_zone_first_index+1} 으로 이동 시작'
                )
            with node.state_lock:
                node.goto_requested = False

        # ── stop 조건: waypoint 1 지난 후 도킹 ───────
        if local_stop and patrol_pos == 0:
            navigator.info('마지막 waypoint 1 완료 → Pre-dock 이동 후 도킹')
            break

        # ── 현재 웨이포인트 이동 ─────────────────────
        current_waypoint_index = patrol_pos_to_waypoint_index(patrol_pos)

        if current_mode == MODE_ROUTE_TO_ZONE:
            log = f'구역 {target_zone_id}로 이동 중'
        elif local_stop:
            log = '종료 전 이동'
        else:
            log = '순찰'

        move_to_waypoint(navigator, current_waypoint_index, log)

        arrived_waypoint_index = current_waypoint_index
        patrol_pos = next_patrol_pos(patrol_pos)

        # ── 목표 구역 첫 waypoint 도착 처리 ──────────
        if (
            current_mode == MODE_ROUTE_TO_ZONE
            and arrived_waypoint_index == target_zone_first_index
        ):
            if local_stop:
                current_mode = MODE_PATROL
                navigator.info(f'구역 {target_zone_id} 도착 but stop 요청 → 계속 진행')
            else:
                current_mode = MODE_WAIT_RESUME
                navigator.info(f'구역 {target_zone_id} 첫 waypoint 도착 → 단속 시작')

    # ── Pre-dock 이동 후 도킹 ─────────────────────────
    move_to_pre_dock_and_dock(navigator)
    navigator.info('도킹 완료 → 다음 출발 신호 대기')

def main():
    rclpy.init()

    node = AMR2Node(robot_ns='robot3')

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    try:
        # ── 무한 사이클: 도킹 후 다음 start 신호 대기 ──
        while rclpy.ok():
            patrol_cycle(node)

    except KeyboardInterrupt:
        node.navigator.info('Ctrl+C → 순찰 중단')
        node.navigator.cancelTask()
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()