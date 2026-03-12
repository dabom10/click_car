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
    ([-0.725,  1.9],  TurtleBot4Directions.SOUTH),
    ([-2.1,   1.95],  TurtleBot4Directions.EAST),
    ([-2.15, -0.3],   TurtleBot4Directions.EAST),
    ([-2.3,  -2.0],   TurtleBot4Directions.EAST),
    ([-2.4,  -4.0],   TurtleBot4Directions.NORTH),
    ([ 2.1,  -4.0],   TurtleBot4Directions.WEST),
    ([ 1.97, -2.5],   TurtleBot4Directions.SOUTH),
    ([-1.5,  -2.2],   TurtleBot4Directions.WEST),
    ([-0.725, -0.2],  TurtleBot4Directions.WEST),
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

# ── robot2 (AMR 1번) 기준 ──────────────────────────
# INITIAL_POSITION  = [0.0, 0.0]      # dock 1 기준
INITIAL_POSITION  = [0.1424, 1.769]   # dock 2 기준

# PRE_DOCK_POSITION = [-0.63, -0.20]   # dock 1
PRE_DOCK_POSITION  = [-0.15, 1.83]      # dock 2
PRE_DOCK_DIRECTION = TurtleBot4Directions.NORTH

ENFORCEMENT_WAIT   = 10.0  # 단속 대기 시간 (초) - 파라미터로 조정 가능
TASK_POLL_PERIOD_SEC = 0.1
BATTERY_LOW_THRESHOLD = 0.25  # 25%

# ================================
# 모드 상수
# ================================
MODE_PATROL        = "PATROL"
MODE_ROUTE_TO_ZONE = "ROUTE_TO_ZONE"
MODE_WAIT_RESUME   = "WAIT_RESUME"
MODE_ENFORCEMENT   = "ENFORCEMENT"

# ================================
# 유틸 함수
# ================================
def quaternion_to_yaw(q):
    """
    쿼터니언(q.x, q.y, q.z, q.w)을 yaw 각도(라디안)로 변환.
    ROS2 Odometry의 orientation은 쿼터니언 형식이므로 변환 필요.
    수식: atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
    """
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)



def get_current_yaw(robot_ns):
    """
    Odometry 토픽에서 현재 로봇의 yaw 값을 한 번 읽어서 반환.

    임시 노드를 만들어 /{robot_ns}/odom 토픽을 구독하고,
    첫 메시지를 받으면 yaw를 추출 후 노드를 종료한다.

    초기 포즈(setInitialPose) 설정 시 실제 yaw 값을 넘겨주기 위해 사용.
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
    corner1, corner2는 대각선 꼭짓점이므로 min/max로 범위를 구한다.
    """
    x1, y1 = zone["corner1"]
    x2, y2 = zone["corner2"]
    xmin, xmax = min(x1, x2), max(x1, x2)
    ymin, ymax = min(y1, y2), max(y1, y2)
    return xmin <= x <= xmax and ymin <= y <= ymax

def find_zone_by_point(x, y):
    """
    좌표 (x, y)가 속하는 구역을 ZONES에서 찾아 반환.
    어떤 구역에도 속하지 않으면 None 반환.
    """
    for zone in ZONES:
        if point_in_zone(x, y, zone):
            return zone
    return None

def waypoint_index_to_patrol_pos(waypoint_index):
    """
    WAYPOINTS index → PATROL_ORDER 내 위치(patrol_pos)로 변환.
    예: waypoint index 5 → PATROL_ORDER에서 5번째 위치인 patrol_pos 4
    구역 첫 waypoint의 patrol_pos를 구할 때 사용.
    """
    return PATROL_ORDER.index(waypoint_index)

def patrol_pos_to_waypoint_index(patrol_pos):
    """
    patrol_pos(0~8) → WAYPOINTS index로 변환.
    순찰 루프에서 현재 이동할 waypoint를 결정할 때 사용.
    """
    return PATROL_ORDER[patrol_pos % PATROL_LENGTH]

def next_patrol_pos(patrol_pos):
    """
    현재 patrol_pos의 다음 순찰 위치를 반환 (0~8 순환).
    waypoint 이동 완료 후 다음 목적지를 결정할 때 사용.
    """
    return (patrol_pos + 1) % PATROL_LENGTH

def move_to_waypoint(navigator, waypoint_index, log_prefix):
    """
    WAYPOINTS[waypoint_index]로 이동하고 완료될 때까지 대기.

    goToPose()로 목표를 전송한 뒤 isTaskComplete()를 폴링해서
    도착 완료 여부를 확인한다.
    log_prefix: 로그에 표시할 현재 상태 문자열 (예: '순찰', '도킹 복귀 중')
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
    Pre-dock 위치(PRE_DOCK_POSITION)로 이동 후 도킹 실행.

    바로 dock()을 호출하면 정렬이 안 될 수 있어서
    도킹 직전 경유 위치로 먼저 이동한 뒤 dock()을 호출한다.
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

    목표 좌표는 goto_target 토픽으로 수신한 불법주차 차량 위치.
    wait_sec 동안 정지해 있는 것이 단속 행위를 나타낸다.
    (추후 카메라 촬영 등 단속 로직 추가 예정)
    """
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
    """
    순찰 명령 수신 및 상태 관리 노드.

    MultiThreadedExecutor 환경에서 토픽 콜백을 처리하며,
    메인 순찰 루프(patrol_cycle)와 상태를 공유한다.

    공유 상태 변수 (state_lock으로 보호):
      - start_patrol   : 'start' 신호 수신 여부
      - stop_requested : 'stop' 신호 수신 여부 (단속 완료 시 내부에서도 ON)
      - goto_requested : 목표 좌표 수신 여부
      - target_x/y     : 수신된 목표 좌표

    구독 토픽:
      /{robot_ns}/patrol_command  → 'start' / 'stop'
      /{robot_ns}/goto_target     → 'x,y' 형식 좌표 문자열
    """
    def __init__(self, robot_ns='robot2'):
        super().__init__('amr1_node')

        self.robot_ns = robot_ns
        self.state_lock = threading.Lock()

        # ── 탐지 플래그 ──────────────────────────────
        self.start_patrol   = False
        self.goto_requested = False
        self.stop_requested = False
        self.target_x       = None
        self.target_y       = None
        self.battery_low    = False 

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

        # 배터리 상태 구독
        self.create_subscription(
            BatteryState,
            f'/{robot_ns}/battery_state',
            self.battery_callback,
            10
        )

        self.get_logger().info(f'[{robot_ns}] 초기화 완료')

    def patrol_command_callback(self, msg):
        """
        /{robot_ns}/patrol_command 토픽 콜백.

        'start' 수신: start_patrol = True, stop_requested 초기화
        'stop'  수신: stop_requested = True
                      → 메인 루프에서 waypoint 1 도착 후 도킹 처리
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

        'x,y' 형식의 문자열을 파싱해서 목표 좌표 저장.
        goto_requested = True 로 메인 루프에 신호 전달.
        파싱 실패 시 에러 로그만 출력하고 무시.
        """
        try:
            x, y = map(float, msg.data.split(','))
            with self.state_lock:
                self.target_x       = x
                self.target_y       = y
                self.goto_requested = True
            self.get_logger().info(f'목표 좌표 수신: x={x}, y={y}')
        except Exception as e:
            self.get_logger().error(f'좌표 파싱 실패: {msg.data} → {e}')

    def battery_callback(self, msg):
        """
        /{robot_ns}/battery_state 토픽 콜백.

        msg.percentage: 0.0~1.0 범위의 배터리 잔량
        BATTERY_LOW_THRESHOLD(0.25) 미만이면 battery_low = True 설정.
        한 번 True가 되면 도킹 완료 후 다음 사이클 시작 시까지 유지.
        → 메인 루프에서 stop과 동일하게 처리되어 현재 작업 완료 후 복귀.
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

    [is_first_cycle]
      True  → 초기 포즈 설정 + waitUntilNav2Active 실행 (프로그램 첫 실행 시)
      False → 초기 설정 생략, 언독만 실행 (2번째 사이클부터)
              이미 Nav2가 활성화되어 있고 localization도 유지되므로
              다시 setInitialPose 하면 위치가 튈 수 있어서 생략.

    [내부 상태 변수]
      current_mode              : 현재 동작 모드 (PATROL / ROUTE_TO_ZONE / ENFORCEMENT)
      patrol_pos                : 현재 순찰 순서 위치 (0~8, PATROL_ORDER 인덱스)
      target_zone_first_index   : 목표 구역의 첫 waypoint (WAYPOINTS index)
      target_zone_resume_patrol_pos : 단속 후 순찰 재개 시작 patrol_pos
      enforcement_x/y           : 단속 목표 좌표

    [stop 조건]
      stop_requested == True 이고 patrol_pos == 0 일 때 break
      → waypoint 9(PATROL_ORDER 마지막)을 막 지나친 시점에서 탈출
      → 단속 완료 후에는 내부에서 stop_requested = True 로 자동 설정
    """
    navigator = node.navigator

    # ── 출발 신호 대기 ────────────────────────────────
    # 사이클 시작마다 이전 사이클의 플래그를 초기화해야
    # 잔류 stop 신호로 인해 바로 도킹되는 버그를 방지할 수 있다.
    navigator.info('[robot2] 출발 신호 대기 중...')
    with node.state_lock:
        node.start_patrol = False  # 이전 신호 초기화
        node.stop_requested = False
        node.battery_low    = False

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
        actual_yaw   = get_current_yaw('robot2')
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
            local_battery_low = node.battery_low
            local_target_x = node.target_x
            local_target_y = node.target_y

        # [추가] 배터리 부족 시 stop과 동일하게 처리
        # 현재 작업(단속 or waypoint 이동) 완료 후 자동으로 복귀 흐름 진입
        if local_battery_low and not local_stop:
            with node.state_lock:
                node.stop_requested = True
            local_stop = True
            navigator.warn('[배터리] 부족 → 현재 작업 완료 후 도킹 복귀')

        # ── 단속 모드 ────────────────────────────────
        # 구역 첫 waypoint 도착 후 진입.
        # 단속 완료 시 stop_requested = True 로 설정해
        # 이후 waypoint 순서대로 waypoint 1까지 이동 후 도킹한다.
        # 배터리 부족이어도 단속은 완료 후 복귀한다.
        if current_mode == MODE_ENFORCEMENT:
            do_enforcement(navigator, enforcement_x, enforcement_y, wait_sec=ENFORCEMENT_WAIT)

            # 단속 완료 → stop 플래그 ON → waypoint 1까지 이동 후 도킹
            with node.state_lock:
                node.stop_requested = True
            current_mode = MODE_PATROL
            navigator.info(
                f'단속 완료 → '
                f'waypoint {patrol_pos_to_waypoint_index(patrol_pos)+1} 부터 '
                f'waypoint 1까지 이동 후 도킹'
            )
            continue

        # ── 목표 좌표 요청 처리 ──────────────────────
        # goto_target 토픽 수신 시 목표 좌표가 속한 구역을 찾아
        # MODE_ROUTE_TO_ZONE으로 전환. 이미 이동 중이면 무시.
        # 배터리 부족(local_stop)이면 새로운 goto_target 무시
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

        # ── stop 조건 확인 ───────────────────────────
        # stop_requested == True 이고 patrol_pos == 0 이면
        # waypoint 1을 막 완료한 시점이므로 루프 탈출 후 도킹
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

        # ── 목표 구역 첫 waypoint 도착 시 단속 모드 진입
        # patrol_pos는 이미 next로 업데이트했으므로
        # target_zone_resume_patrol_pos로 덮어써서 단속 후 재개 위치를 설정
        if (
            current_mode == MODE_ROUTE_TO_ZONE
            and arrived_waypoint_index == target_zone_first_index
        ):
            current_mode = MODE_ENFORCEMENT
            patrol_pos   = target_zone_resume_patrol_pos
            navigator.info(f'구역 {target_zone_id} 첫 waypoint 도착 → 단속 실행')
    
    # ── Pre-dock 이동 후 도킹 ─────────────────────────
    move_to_pre_dock_and_dock(navigator)
    navigator.info('도킹 완료 → 다음 출발 신호 대기')

def main():
    """
    프로그램 진입점.

    [실행 구조]
      - AMR2Node: 토픽 콜백 처리 (MultiThreadedExecutor 별도 스레드)
      - patrol_cycle: 메인 스레드에서 순찰 루프 실행

    [is_first_cycle 플래그]
      첫 사이클에서만 초기 포즈 설정 + Nav2 활성화 대기를 수행.
      이후 사이클은 이미 localization이 유지되므로 생략.

    [종료 처리]
      Ctrl+C 시 현재 태스크를 cancelTask()로 중단하고 rclpy를 종료.
    """
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