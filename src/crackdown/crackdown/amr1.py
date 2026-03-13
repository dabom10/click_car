#!/usr/bin/env python3
"""
amr1.py
-------
robot3 기준 테스트 - 출발 신호 수신 후 순찰 시작
구역 기반 이동 + 단속 + waypoint 9까지 복귀 + 도킹 + 재시작 대기

[전체 동작 흐름]
  1. start 신호 대기
  2. 현재의 yaw 값을 추출하고 initial_pose 선정 -> 언독 후 waypoint 순서대로 순찰 (시계 반대 방향)
  3. cctv_done 또는 amr_done 수신 시 해당 구역 첫 waypoint로 이동
  4. 구역 첫 waypoint 도착 → 목표 좌표로 이동
       - CCTV 경로: cctv_start(True) pub → capture_done 대기
       - AMR 경로:  amr_start(True)  pub → capture_done 대기
  5. 단속 완료 → 남은 waypoint 순서대로 waypoint 9까지 이동
  6. Pre-dock 위치로 이동 → 도킹
  7. 다시 1번으로 (무한 반복)

[단속 출처 구분]
  CCTV 경로  : cctv_done(x,y) 수신 → 이동 → cctv_start(True) pub → capture_done 대기
  AMR  경로  : amr_done(x,y)  수신 → 이동 → amr_start(True)  pub → capture_done 대기
  capture_done은 두 경로 공통 토픽 (동시 단속 없으므로 충돌 없음)
  단속 시작 직전 capture_done = False 초기화 필수 (잔류값 방지)

[모드 전환]
  MODE_PATROL
    └─ cctv_done / amr_done 수신 ──→ MODE_ROUTE_TO_ZONE
                                          └─ 구역 첫 waypoint 도착 ──→ MODE_ENFORCEMENT
                                                                          └─ 단속 완료 ──→ MODE_PATROL (stop ON)

[토픽 목록]
  구독:
    /{robot_ns}/patrol_command  std_msgs/String        'start' / 'stop'
    /{robot_ns}/cctv_done       std_msgs/String        'x,y'  ← CCTV 좌표
    /{robot_ns}/amr_done        std_msgs/String        'x,y'  ← AMR 카메라 좌표
    /{robot_ns}/capture_done    std_msgs/Bool          촬영 완료 신호 (공통)
    /{robot_ns}/battery_state   sensor_msgs/BatteryState
    /{robot_ns}/amcl_pose       geometry_msgs/PoseWithCovarianceStamped

  발행:
    /{robot_ns}/cctv_start      std_msgs/Bool          True = 촬영 시작 (CCTV 경로)
    /{robot_ns}/amr_start       std_msgs/Bool          True = 촬영 시작 (AMR 경로)

실행:
  ros2 run crackdown amr1 --ros-args -r __ns:=/robot3

토픽 송신 테스트:
  출발: ros2 topic pub /robot3/patrol_command std_msgs/String "{data: 'start'}" --once
  CCTV: ros2 topic pub /robot3/cctv_done std_msgs/String "{data: '0.5,-4.7'}" --once
  AMR:  ros2 topic pub /robot3/amr_done  std_msgs/String "{data: '0.5,-4.7'}" --once
  촬영완료: ros2 topic pub /robot3/capture_done std_msgs/Bool "{data: true}" --once
  정지: ros2 topic pub /robot3/patrol_command std_msgs/String "{data: 'stop'}" --once
"""

import math
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Directions, TurtleBot4Navigator
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Bool
from sensor_msgs.msg import BatteryState
from geometry_msgs.msg import PoseWithCovarianceStamped

# ── 웨이포인트 정의 ──────────────────────────────────────
WAYPOINTS = [
    ([-0.725, -0.2],  TurtleBot4Directions.WEST),
    ([-0.725,  1.9],  TurtleBot4Directions.SOUTH),
    ([-2.2,    2.2],  TurtleBot4Directions.NORTH_EAST),
    ([-2.15,  -0.3],  TurtleBot4Directions.EAST),
    ([-2.3,   -2.0],  TurtleBot4Directions.EAST),
    ([-2.4,   -4.0],  TurtleBot4Directions.NORTH),
    ([ 2.1,   -4.0],  TurtleBot4Directions.WEST),
    ([ 1.97,  -2.5],  TurtleBot4Directions.SOUTH),
    ([-1.5,   -2.2],  TurtleBot4Directions.WEST),
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

# ── robot3 (AMR 1번) 기준 ─────────────────────────────
INITIAL_POSITION   = [0.0, 0.0]             # dock 1 기준
# INITIAL_POSITION = [0.1424, 1.769]        # dock 2 기준

PRE_DOCK_POSITION  = [-0.63, -0.20]         # dock 1
# PRE_DOCK_POSITION = [-2.5, -1.7]          # dock 2
PRE_DOCK_DIRECTION = TurtleBot4Directions.NORTH

ENFORCEMENT_WAIT_CCTV = 10.0   # CCTV 경로 단속 대기 시간 (초)
ENFORCEMENT_WAIT_AMR  = 30.0   # AMR 카메라 경로 단속 대기 시간 (초)
ENFORCEMENT_STOP_DIST = 0.7    # 목표 차량으로부터 정지할 거리 (미터) - 조절 가능
TASK_POLL_PERIOD_SEC  = 0.1
BATTERY_LOW_THRESHOLD = 0.25   # 배터리 복귀 기준 (25%)

# ── 상태 publish 설정 ─────────────────────────────────
STATUS_PUBLISH_PERIOD_SEC = 0.5

STATUS_PATROL = "patrol"
STATUS_ENFORCE = "enforce"
STATUS_RETURNING = "returning"
STATUS_CHARGING = "charging"

# ================================
# 단속 출처 상수
# ================================
SOURCE_CCTV = "CCTV"
SOURCE_AMR  = "AMR"

# ================================
# 모드 상수
# ================================
MODE_PATROL        = "PATROL"
MODE_ROUTE_TO_ZONE = "ROUTE_TO_ZONE"
MODE_ENFORCEMENT   = "ENFORCEMENT"

# ================================
# 유틸 함수
# ================================
def quaternion_to_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

def get_current_yaw(robot_ns):
    """
    Odometry 토픽에서 현재 로봇의 yaw 값을 한 번 읽어서 반환.
    첫 사이클 시작 전(executor 충돌 없는 시점)에만 호출.
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

def do_enforcement(navigator, node, target_x, target_y, source):
    """
    단속 실행: 목표 좌표 앞 ENFORCEMENT_STOP_DIST 지점으로 이동 후 촬영 대기.

    [출처별 동작]
      SOURCE_CCTV : 이동 완료 → cctv_start(True) pub → capture_done 대기
      SOURCE_AMR  : 이동 완료 → amr_start(True)  pub → capture_done 대기

    [방향 계산]
      node.current_x/y (amcl_pose 콜백으로 지속 갱신) 에서 현재 위치를 읽고
      atan2로 목표 방향을 계산 → degree 변환 → getPoseStamped에 전달.

    [capture_done 초기화]
      촬영 시작 pub 직전에 capture_done = False 로 초기화.
      이전 단속의 잔류 True 값이 즉시 통과하는 버그를 방지.
    """
    navigator.info(f'[단속/{source}] 목표 좌표 ({target_x}, {target_y}) 로 이동 시작')

    # 현재 위치 읽기 (amcl_pose 콜백으로 갱신된 공유 변수)
    with node.state_lock:
        rx = node.current_x
        ry = node.current_y
    navigator.info(f'[단속/{source}] 현재 위치: ({rx:.2f}, {ry:.2f})')

    # 목표 방향 계산: atan2(라디안) → degree 변환
    yaw_rad = math.atan2(target_y - ry, target_x - rx)
    yaw_deg = math.degrees(yaw_rad)
    navigator.info(f'[단속/{source}] 목표 방향: {yaw_deg:.1f}°')

    # 목표에서 ENFORCEMENT_STOP_DIST만큼 앞 정지 좌표 계산
    stop_x = target_x - math.cos(yaw_rad) * ENFORCEMENT_STOP_DIST
    stop_y = target_y - math.sin(yaw_rad) * ENFORCEMENT_STOP_DIST
    navigator.info(
        f'[단속/{source}] 정지 좌표: ({stop_x:.2f}, {stop_y:.2f}) '
        f'← 목표에서 {ENFORCEMENT_STOP_DIST}m 앞'
    )

    goal_pose = navigator.getPoseStamped([stop_x, stop_y], yaw_deg)
    navigator.goToPose(goal_pose)
    while not navigator.isTaskComplete():
        time.sleep(TASK_POLL_PERIOD_SEC)
    navigator.info(f'[단속/{source}] 도착 완료 → 촬영 시작 신호 pub')

    # capture_done 잔류값 초기화 후 촬영 시작 신호 pub
    with node.state_lock:
        node.capture_done = False

    if source == SOURCE_CCTV:
        node.cctv_start_pub.publish(Bool(data=True))
        navigator.info('[단속/CCTV] cctv_start(True) pub → capture_done 대기')
    else:
        node.amr_start_pub.publish(Bool(data=True))
        navigator.info('[단속/AMR] amr_start(True) pub → capture_done 대기')

    # capture_done True 수신까지 대기
    while True:
        with node.state_lock:
            done = node.capture_done
        if done:
            break
        time.sleep(TASK_POLL_PERIOD_SEC)

    navigator.info(f'[단속/{source}] capture_done 수신 → 단속 완료!')


class AMRNode(Node):
    def __init__(self, robot_ns='robot3'):
        super().__init__('amr1_node')

        self.robot_ns   = robot_ns
        self.state_lock = threading.Lock()

        # ── 순찰 제어 플래그 ─────────────────────────
        self.start_patrol   = False
        self.stop_requested = False
        self.battery_low    = False

        # ── 단속 요청 플래그 ─────────────────────────
        # source: SOURCE_CCTV / SOURCE_AMR / None
        self.goto_requested = False
        self.goto_source    = None   # 어느 경로에서 수신했는지
        self.target_x       = None
        self.target_y       = None

        # ── 촬영 완료 플래그 ─────────────────────────
        self.capture_done   = False  # capture_done 토픽 수신 시 True

        # ── 현재 위치 (amcl_pose 콜백으로 갱신) ──────
        self.current_x      = 0.0
        self.current_y      = 0.0

        # ── 현재 상태 문자열 ────────────────────────────
        self.current_status = STATUS_CHARGING

        # ── Navigator ────────────────────────────────
        self.navigator = TurtleBot4Navigator()

        # ── 구독 ─────────────────────────────────────
        self.create_subscription(
            String,
            f'/{robot_ns}/patrol_command',
            self.patrol_command_callback,
            10
        )
        self.create_subscription(
            String,
            f'/{robot_ns}/cctv_done',
            self.cctv_done_callback,
            10
        )
        self.create_subscription(
            String,
            f'/{robot_ns}/amr_done',
            self.amr_done_callback,
            10
        )
        self.create_subscription(
            Bool,
            f'/{robot_ns}/capture_done',
            self.capture_done_callback,
            10
        )
        self.create_subscription(
            BatteryState,
            f'/{robot_ns}/battery_state',
            self.battery_callback,
            10
        )
        self.create_subscription(
            PoseWithCovarianceStamped,
            f'/{robot_ns}/amcl_pose',
            self.amcl_pose_callback,
            10
        )

        self.create_timer(
            STATUS_PUBLISH_PERIOD_SEC,
            self.publish_status
        )

        # ── 발행 ─────────────────────────────────────
        self.cctv_start_pub = self.create_publisher(Bool, f'/{robot_ns}/cctv_start', 10)
        self.amr_start_pub  = self.create_publisher(Bool, f'/{robot_ns}/amr_start',  10)
        # ── 상태 퍼블리셔 ──────────────────────────────
        self.status_pub = self.create_publisher(
            String,
            f'{robot_ns}_status',
            10
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

    # ── 콜백 ─────────────────────────────────────────
    def patrol_command_callback(self, msg):
        with self.state_lock:
            if msg.data == 'start':
                self.start_patrol = True
                self.get_logger().info('출발 신호 수신 → 순찰 시작!')
            elif msg.data == 'stop':
                self.stop_requested = True
                self.get_logger().info('정지 신호 수신 → waypoint 9까지 완료 후 도킹')

    def _parse_coord(self, msg, source_label):
        """'x,y' 문자열 파싱 공통 함수. 파싱 실패 시 None 반환."""
        try:
            x, y = map(float, msg.data.split(','))
            return x, y
        except Exception as e:
            self.get_logger().error(f'[{source_label}] 좌표 파싱 실패: {msg.data} → {e}')
            return None, None

    def cctv_done_callback(self, msg):
        """CCTV에서 목표 좌표 수신. 단속 중(stop_requested)이면 무시."""
        x, y = self._parse_coord(msg, 'cctv_done')
        if x is None:
            return
        with self.state_lock:
            if self.stop_requested:
                self.get_logger().warn('[cctv_done] 단속/복귀 중 → 무시')
                return
            self.target_x       = x
            self.target_y       = y
            self.goto_source    = SOURCE_CCTV
            self.goto_requested = True
        self.get_logger().info(f'[cctv_done] 목표 좌표 수신: ({x}, {y})')

    def amr_done_callback(self, msg):
        """AMR 카메라에서 목표 좌표 수신. 단속 중(stop_requested)이면 무시."""
        x, y = self._parse_coord(msg, 'amr_done')
        if x is None:
            return
        with self.state_lock:
            if self.stop_requested:
                self.get_logger().warn('[amr_done] 단속/복귀 중 → 무시')
                return
            self.target_x       = x
            self.target_y       = y
            self.goto_source    = SOURCE_AMR
            self.goto_requested = True
        self.get_logger().info(f'[amr_done] 목표 좌표 수신: ({x}, {y})')

    def capture_done_callback(self, msg):
        """촬영 완료 신호 수신 (CCTV/AMR 경로 공통)."""
        if msg.data:
            with self.state_lock:
                self.capture_done = True
            self.get_logger().info('[capture_done] 촬영 완료 수신')

    def battery_callback(self, msg):
        with self.state_lock:
            if not self.battery_low and msg.percentage < BATTERY_LOW_THRESHOLD:
                self.battery_low = True
                self.get_logger().warn(
                    f'배터리 부족 ({msg.percentage*100:.1f}%) → '
                    f'현재 작업 완료 후 도킹 복귀'
                )

    def amcl_pose_callback(self, msg):
        """
        amcl_pose 콜백 - 현재 위치를 공유 변수에 지속 갱신.
        임시 노드로 읽으면 MultiThreadedExecutor와 충돌해 무한 대기하므로
        노드 멤버로 관리.
        """
        with self.state_lock:
            self.current_x = msg.pose.pose.position.x
            self.current_y = msg.pose.pose.position.y


# ================================
# 순찰 한 사이클
# ================================
def patrol_cycle(node: AMRNode, is_first_cycle: bool = False):
    navigator = node.navigator

    navigator.info('[robot3] 출발 신호 대기 중...')
    with node.state_lock:
        node.start_patrol   = False
        node.stop_requested = False
        node.battery_low    = False
        node.capture_done   = False

    node.set_status(STATUS_CHARGING)

    while rclpy.ok():
        with node.state_lock:
            if node.start_patrol:
                break
        time.sleep(0.1)

    if is_first_cycle:
        if not navigator.getDockedStatus():
            navigator.info('도킹 상태 아님 → 도킹 후 초기 포즈 설정')
            navigator.dock()
        actual_yaw   = get_current_yaw('robot3')
        initial_pose = navigator.getPoseStamped(INITIAL_POSITION, actual_yaw)
        navigator.setInitialPose(initial_pose)
        navigator.waitUntilNav2Active()
    navigator.undock()
    node.set_status(STATUS_PATROL)
    navigator.info('순찰 시작!')

    current_mode = MODE_PATROL
    patrol_pos   = 0

    target_zone_id                = None
    target_zone_first_index       = None
    target_zone_resume_patrol_pos = None
    enforcement_x                 = None
    enforcement_y                 = None
    enforcement_source            = None   # SOURCE_CCTV or SOURCE_AMR

    while rclpy.ok():

        with node.state_lock:
            local_goto        = node.goto_requested
            local_stop        = node.stop_requested
            local_battery_low = node.battery_low
            local_target_x    = node.target_x
            local_target_y    = node.target_y
            local_source      = node.goto_source

        # 배터리 부족 → stop과 동일하게 처리
        if local_battery_low and not local_stop:
            with node.state_lock:
                node.stop_requested = True
            local_stop = True
            node.set_status(STATUS_RETURNING)
            navigator.warn('[배터리] 부족 → 현재 작업 완료 후 도킹 복귀')

        # ── 단속 실행 ────────────────────────────────
        if current_mode == MODE_ENFORCEMENT:
            node.set_status(STATUS_ENFORCE)
            do_enforcement(
                navigator, node,
                enforcement_x, enforcement_y,
                source=enforcement_source
            )
            with node.state_lock:
                node.stop_requested = True

            node.set_status(STATUS_RETURNING)
            current_mode = MODE_PATROL
            navigator.info(
                f'단속 완료 → '
                f'waypoint {patrol_pos_to_waypoint_index(patrol_pos)+1} 부터 '
                f'waypoint 9까지 이동 후 도킹'
            )
            continue

        # ── 단속 요청 처리 ───────────────────────────
        if local_goto and current_mode != MODE_ROUTE_TO_ZONE and not local_stop:
            zone = find_zone_by_point(local_target_x, local_target_y)
            if zone is None:
                navigator.info(
                    f'좌표 ({local_target_x}, {local_target_y}) → '
                    f'어떤 구역에도 속하지 않음'
                )
            else:
                target_zone_id                = zone["zone_id"]
                target_zone_first_index       = zone["waypoint_indices"][0]
                target_zone_resume_patrol_pos = next_patrol_pos(
                    waypoint_index_to_patrol_pos(target_zone_first_index)
                )
                enforcement_x      = local_target_x
                enforcement_y      = local_target_y
                enforcement_source = local_source
                current_mode = MODE_ROUTE_TO_ZONE

                node.set_status(STATUS_ENFORCE)
                navigator.info(
                    f'[{local_source}] 목표 ({local_target_x}, {local_target_y}) → '
                    f'구역 {target_zone_id}, '
                    f'첫 waypoint {target_zone_first_index+1} 으로 이동 시작'
                )
            with node.state_lock:
                node.goto_requested = False

        # ── stop 조건: waypoint 0 도착 후 도킹 ──────
        if local_stop and patrol_pos == 0:
            node.set_status(STATUS_RETURNING)
            navigator.info('마지막 waypoint 9 완료 → Pre-dock 이동 후 도킹')
            break

        # ── 현재 waypoint 이동 ───────────────────────
        current_waypoint_index = patrol_pos_to_waypoint_index(patrol_pos)

        if current_mode == MODE_ROUTE_TO_ZONE:
            node.set_status(STATUS_ENFORCE)
            log = f'구역 {target_zone_id}로 이동 중'
        elif local_stop:
            node.set_status(STATUS_RETURNING)
            log = '도킹 복귀 중'
        else:
            node.set_status(STATUS_PATROL)
            log = '순찰'

        move_to_waypoint(navigator, current_waypoint_index, log)

        arrived_waypoint_index = current_waypoint_index
        patrol_pos = next_patrol_pos(patrol_pos)

        if (
            current_mode == MODE_ROUTE_TO_ZONE
            and arrived_waypoint_index == target_zone_first_index
        ):
            current_mode = MODE_ENFORCEMENT
            patrol_pos   = target_zone_resume_patrol_pos
            node.set_status(STATUS_ENFORCE)
            navigator.info(f'구역 {target_zone_id} 첫 waypoint 도착 → 단속 실행')

    node.set_status(STATUS_RETURNING)
    move_to_pre_dock_and_dock(navigator)
    node.set_status(STATUS_CHARGING)
    navigator.info('도킹 완료 → 다음 출발 신호 대기')

def main():
    rclpy.init()

    node = AMRNode(robot_ns='robot3')

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