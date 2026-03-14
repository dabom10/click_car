#!/usr/bin/env python3
"""
amr1.py
-------
robot2 기준 테스트 - 출발 신호 수신 후 순찰 시작
구역 기반 이동 + 단속 + waypoint 9까지 복귀 + 도킹 + 재시작 대기

[전체 동작 흐름]
  1. start 신호 대기
  2. 현재의 yaw 값을 추출하고 initial_pose 선정 -> 언독 후 waypoint 순서대로 순찰 (시계 반대 방향)
  3. goto_target 신호 수신 시 해당 구역 첫 waypoint로 이동
  4. 구역 첫 waypoint 도착 → 목표 좌표로 이동 → ENFORCEMENT_WAIT초 단속 대기
  5. 단속 완료 → 남은 waypoint 순서대로 waypoint 9까지 이동
  6. Pre-dock 위치로 이동 → 도킹
  7. 다시 1번으로 (무한 반복)

[모드 전환]
  MODE_PATROL
    └─ goto_target 수신 ──→ MODE_ROUTE_TO_ZONE
                                └─ 구역 첫 waypoint 도착 ──→ MODE_ENFORCEMENT
                                                                └─ 단속 완료 ──→ MODE_PATROL (stop ON)

실행:
  ros2 run crackdown amr2 --ros-args -r __ns:=/robot2

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
from sensor_msgs.msg import BatteryState


# ── 웨이포인트 정의 ──────────────────────────────────────
WAYPOINTS = [
    ([-0.725, -0.2],  TurtleBot4Directions.WEST),
    ([-0.725,  1.9],  TurtleBot4Directions.SOUTH),
    ([-2.2,   2.25],  TurtleBot4Directions.EAST),
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
        "corner1": (0.38, 2.55),
        "corner2": (-1.37, -1.37),
    },
    {
        "zone_id": 2,
        "waypoint_indices": [2, 3],
        "corner1": (-1.37, 2.55),
        "corner2": (-2.91, -1.37),
    },
    {
        "zone_id": 3,
        "waypoint_indices": [4],
        "corner1": (-2.91, -1.37),
        "corner2": (-1.7, -3.23),
    },
    {
        "zone_id": 4,
        "waypoint_indices": [5, 6],
        "corner1": (-3.0, -3.23),
        "corner2": (2.6, -5.15),
    },
    {
        "zone_id": 5,
        "waypoint_indices": [7, 8],
        "corner1": (-1.67, -1.37),
        "corner2": (2.7, -3.23),
    },
]

PATROL_ORDER = [0, 1, 2, 3, 4, 5, 6, 7, 8]
PATROL_LENGTH = len(PATROL_ORDER)

# ── robot2 (AMR 2번) 기준 ──────────────────────────
INITIAL_POSITION = [0.1424, 1.769]  

PRE_DOCK_POSITION = [-0.15, 1.83]  
PRE_DOCK_DIRECTION = TurtleBot4Directions.NORTH

ENFORCEMENT_WAIT = 10.0
TASK_POLL_PERIOD_SEC = 0.1
BATTERY_LOW_THRESHOLD = 0.25

# ── 상태 publish 설정 ─────────────────────────────────
STATUS_PUBLISH_PERIOD_SEC = 0.5

STATUS_PATROL = "patrol"
STATUS_ENFORCE = "enforce"
STATUS_RETURNING = "returning"
STATUS_CHARGING = "charging"

# ================================
# 모드 상수
# ================================
MODE_PATROL = "PATROL"
MODE_ROUTE_TO_ZONE = "ROUTE_TO_ZONE"
MODE_WAIT_RESUME = "WAIT_RESUME"
MODE_ENFORCEMENT = "ENFORCEMENT"


# ================================
# 유틸 함수
# ================================
def quaternion_to_yaw(q):
    """
    쿼터니언(q.x, q.y, q.z, q.w)을 yaw 각도(라디안)로 변환.
    """
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def get_current_yaw(robot_ns):
    """
    Odometry 토픽에서 현재 로봇의 yaw 값을 한 번 읽어서 반환.
    """
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
    """
    좌표 (x, y)가 zone의 직사각형 영역 안에 있는지 확인.
    """
    x1, y1 = zone["corner1"]
    x2, y2 = zone["corner2"]
    xmin, xmax = min(x1, x2), max(x1, x2)
    ymin, ymax = min(y1, y2), max(y1, y2)
    return xmin <= x <= xmax and ymin <= y <= ymax


def find_zone_by_point(x, y):
    """
    좌표 (x, y)가 속하는 구역을 ZONES에서 찾아 반환.
    """
    for zone in ZONES:
        if point_in_zone(x, y, zone):
            return zone
    return None


def waypoint_index_to_patrol_pos(waypoint_index):
    """
    WAYPOINTS index → PATROL_ORDER 내 위치(patrol_pos)로 변환.
    """
    return PATROL_ORDER.index(waypoint_index)


def patrol_pos_to_waypoint_index(patrol_pos):
    """
    patrol_pos(0~8) → WAYPOINTS index로 변환.
    """
    return PATROL_ORDER[patrol_pos % PATROL_LENGTH]


def next_patrol_pos(patrol_pos):
    """
    현재 patrol_pos의 다음 순찰 위치를 반환.
    """
    return (patrol_pos + 1) % PATROL_LENGTH


def move_to_waypoint(navigator, waypoint_index, log_prefix):
    """
    WAYPOINTS[waypoint_index]로 이동하고 완료될 때까지 대기.
    """
    pos, direction = WAYPOINTS[waypoint_index]
    navigator.info(f'{log_prefix}: waypoint {waypoint_index+1} {pos} 이동 시작')
    pose = navigator.getPoseStamped(pos, direction)
    navigator.goToPose(pose)

    while not navigator.isTaskComplete():
        time.sleep(TASK_POLL_PERIOD_SEC)

    navigator.info(f'{log_prefix}: waypoint {waypoint_index+1} 도착 완료')


def move_to_pre_dock_and_dock(navigator):
    """
    Pre-dock 위치로 이동 후 도킹 실행.
    """
    navigator.info('Pre-dock 위치로 이동 중...')
    pre_dock_pose = navigator.getPoseStamped(PRE_DOCK_POSITION, PRE_DOCK_DIRECTION)
    navigator.goToPose(pre_dock_pose)

    while not navigator.isTaskComplete():
        time.sleep(TASK_POLL_PERIOD_SEC)

    navigator.info('Pre-dock 도착 → 도킹 실행')
    navigator.dock()


def do_enforcement(navigator, target_x, target_y, wait_sec=ENFORCEMENT_WAIT):
    """
    단속 실행: 목표 좌표로 이동 후 wait_sec초 대기.
    """
    navigator.info(f'[단속] 목표 좌표 ({target_x}, {target_y}) 로 이동 시작')
    goal_pose = navigator.getPoseStamped(
        [target_x, target_y],
        TurtleBot4Directions.NORTH
    )
    navigator.goToPose(goal_pose)

    while not navigator.isTaskComplete():
        time.sleep(TASK_POLL_PERIOD_SEC)
    AMR2Node.set_status(STATUS_ENFORCE)
    navigator.info(f'[단속] 도착 완료 → {wait_sec}초 단속 대기')
    time.sleep(wait_sec)
    navigator.info('[단속] 단속 완료!')


class AMR2Node(Node):
    """
    순찰 명령 수신 및 상태 관리 노드.
    """
    def __init__(self, robot_ns='robot2'):
        super().__init__('amr1_node')

        self.robot_ns = robot_ns
        self.state_lock = threading.Lock()

        # ── 순찰 관련 상태 ──────────────────────────────
        self.start_patrol = False
        self.goto_requested = False
        self.stop_requested = False
        self.target_x = None
        self.target_y = None
        self.battery_low = False

        # ── 현재 상태 문자열 ────────────────────────────
        self.current_status = STATUS_CHARGING

        # ── Navigator ──────────────────────────────────
        self.navigator = TurtleBot4Navigator()

        # ── 명령 토픽 구독 ─────────────────────────────
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

        self.create_subscription(
            BatteryState,
            f'/{robot_ns}/battery_state',
            self.battery_callback,
            10
        )

        # ── 상태 퍼블리셔 ──────────────────────────────
        self.status_pub = self.create_publisher(
            String,
            f'robot2_status',
            10
        )

        self.create_timer(
            STATUS_PUBLISH_PERIOD_SEC,
            self.publish_status
        )

        self.get_logger().info(f'[{robot_ns}] 초기화 완료')

    def set_status(self, status: str):
        """
        현재 AMR 상태 문자열을 갱신.
        """
        with self.state_lock:
            if self.current_status != status:
                self.current_status = status
                self.get_logger().info(f'status 변경 → {status}')

    def publish_status(self):
        """
        현재 상태를 주기적으로 publish.
        charging 상태도 계속 publish되도록 타이머 기반으로 전송.
        """
        with self.state_lock:
            status = self.current_status

        msg = String()
        msg.data = status
        self.status_pub.publish(msg)

    def patrol_command_callback(self, msg):
        """
        /{robot_ns}/patrol_command 토픽 콜백.
        """
        with self.state_lock:
            if msg.data == 'start':
                self.start_patrol = True
                self.get_logger().info('출발 신호 수신 → 순찰 시작!')
            elif msg.data == 'stop':
                self.stop_requested = True
                self.get_logger().info('정지 신호 수신 → waypoint 1까지 완료 후 도킹')

    def goto_target_callback(self, msg):
        """
        /{robot_ns}/goto_target 토픽 콜백.
        """
        try:
            x, y = map(float, msg.data.split(','))
            with self.state_lock:
                self.target_x = x
                self.target_y = y
                self.goto_requested = True
            self.get_logger().info(f'목표 좌표 수신: x={x}, y={y}')
        except Exception as e:
            self.get_logger().error(f'좌표 파싱 실패: {msg.data} → {e}')

    def battery_callback(self, msg):
        """
        /{robot_ns}/battery_state 토픽 콜백.
        """
        with self.state_lock:
            if not self.battery_low and msg.percentage < BATTERY_LOW_THRESHOLD:
                self.battery_low = True
                self.get_logger().warn(
                    f'배터리 부족 ({msg.percentage*100:.1f}%) → '
                    f'현재 작업 완료 후 도킹 복귀'
                )


# ================================
# 순찰 한 사이클
# ================================
def patrol_cycle(node: AMR2Node, is_first_cycle: bool = False):
    """
    start 신호 수신부터 도킹 완료까지 한 사이클을 처리.
    """
    navigator = node.navigator

    # ── 출발 신호 대기 ────────────────────────────────
    navigator.info('[robot2] 출발 신호 대기 중...')

    with node.state_lock:
        node.start_patrol = False
        node.stop_requested = False
        node.battery_low = False

    node.set_status(STATUS_CHARGING)

    while rclpy.ok():
        with node.state_lock:
            if node.start_patrol:
                break
        time.sleep(0.1)

    # ── 초기 설정 (첫 사이클만 실행) ─────────────────
    if is_first_cycle:
        if not navigator.getDockedStatus():
            navigator.info('도킹 상태 아님 → 도킹 후 초기 포즈 설정')
            navigator.dock()

        actual_yaw = get_current_yaw('robot2')
        initial_pose = navigator.getPoseStamped(INITIAL_POSITION, actual_yaw)
        navigator.setInitialPose(initial_pose)
        navigator.waitUntilNav2Active()

    navigator.undock()
    node.set_status(STATUS_PATROL)
    navigator.info('순찰 시작!')

    current_mode = MODE_PATROL
    patrol_pos = 1

    target_zone_id = None
    target_zone_first_index = None
    target_zone_first_patrol_pos = None
    target_zone_resume_patrol_pos = None
    enforcement_x = None
    enforcement_y = None

    while rclpy.ok():
        with node.state_lock:
            local_goto = node.goto_requested
            local_stop = node.stop_requested
            local_battery_low = node.battery_low
            local_target_x = node.target_x
            local_target_y = node.target_y

        # ── 배터리 부족 처리 ───────────────────────────
        if local_battery_low and not local_stop:
            with node.state_lock:
                node.stop_requested = True
            local_stop = True
            node.set_status(STATUS_RETURNING)
            navigator.warn('[배터리] 부족 → 현재 작업 완료 후 도킹 복귀')

        # ── 단속 모드 ────────────────────────────────
        if current_mode == MODE_ENFORCEMENT:
            do_enforcement(
                navigator,
                enforcement_x,
                enforcement_y,
                wait_sec=ENFORCEMENT_WAIT
            )

            with node.state_lock:
                node.stop_requested = True

            node.set_status(STATUS_RETURNING)
            current_mode = MODE_PATROL
            navigator.info(
                f'단속 완료 → '
                f'waypoint {patrol_pos_to_waypoint_index(patrol_pos)+1} 부터 '
                f'waypoint 1까지 이동 후 도킹'
            )
            continue

        # ── 목표 좌표 요청 처리 ──────────────────────
        if local_goto and current_mode != MODE_ROUTE_TO_ZONE:
            zone = find_zone_by_point(local_target_x, local_target_y)

            if zone is None:
                navigator.info(
                    f'좌표 ({local_target_x}, {local_target_y}) → 어떤 구역에도 속하지 않음'
                )
            else:
                target_zone_id = zone["zone_id"]
                target_zone_first_index = zone["waypoint_indices"][0]
                target_zone_first_patrol_pos = waypoint_index_to_patrol_pos(target_zone_first_index)
                target_zone_resume_patrol_pos = next_patrol_pos(target_zone_first_patrol_pos)
                enforcement_x = local_target_x
                enforcement_y = local_target_y
                current_mode = MODE_ROUTE_TO_ZONE
                navigator.info(
                    f'목표 ({local_target_x}, {local_target_y}) → '
                    f'구역 {target_zone_id}, '
                    f'첫 waypoint {target_zone_first_index+1} 으로 이동 시작'
                )

            with node.state_lock:
                node.goto_requested = False

        # ── stop 조건 확인 ───────────────────────────
        if local_stop and patrol_pos == 0:
            node.set_status(STATUS_RETURNING)
            navigator.info('마지막 waypoint 1 완료 → Pre-dock 이동 후 도킹')
            break

        # ── 현재 웨이포인트 이동 ─────────────────────
        current_waypoint_index = patrol_pos_to_waypoint_index(patrol_pos)

        if current_mode == MODE_ROUTE_TO_ZONE:
            node.set_status(STATUS_ENFORCE)
            log = f'구역 {target_zone_id}로 이동 중'
        elif local_stop:
            node.set_status(STATUS_RETURNING)
            log = '종료 전 이동'
        else:
            node.set_status(STATUS_PATROL)
            log = '순찰'

        move_to_waypoint(navigator, current_waypoint_index, log)

        arrived_waypoint_index = current_waypoint_index
        patrol_pos = next_patrol_pos(patrol_pos)

        # ── 목표 구역 첫 waypoint 도착 시 단속 모드 진입 ──
        if (current_mode == MODE_ROUTE_TO_ZONE and
            arrived_waypoint_index == target_zone_first_index):
            current_mode = MODE_ENFORCEMENT
            patrol_pos = target_zone_resume_patrol_pos
            node.set_status(STATUS_ENFORCE)
            navigator.info(f'구역 {target_zone_id} 첫 waypoint 도착 → 단속 실행')

    # ── Pre-dock 이동 후 도킹 ────────────────────────
    node.set_status(STATUS_RETURNING)
    move_to_pre_dock_and_dock(navigator)

    node.set_status(STATUS_CHARGING)
    navigator.info('도킹 완료 → 다음 출발 신호 대기')


def main():
    rclpy.init()

    node = AMR2Node(robot_ns='robot2')

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    try:
        is_first_cycle = True
        while rclpy.ok():
            patrol_cycle(node, is_first_cycle)
            is_first_cycle = False

    except KeyboardInterrupt:
        node.navigator.info('Ctrl+C → 순찰 중단')
        node.navigator.cancelTask()
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()