#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import select
import sys
import termios
import threading
import time
import tty

import rclpy

from turtlebot4_navigation.turtlebot4_navigator import (
    TurtleBot4Directions,
    TurtleBot4Navigator,
)


# ================================
# 설정 상수
# ================================
# 초기 위치
INITIAL_POSITION = [0.142463, 1.76936]   # robot2
INITIAL_DIRECTION = -5.43

# 도킹 직전 위치
PRE_DOCK_POSITION = [-0.15, 1.83]
PRE_DOCK_DIRECTION = TurtleBot4Directions.NORTH

# g 입력 시 사용할 목표 좌표
TARGET_POSITION = (-0.5, -4.7)

# waypoint 정의 (0-based index)
WAYPOINTS = [
    ([-0.725, -0.2], TurtleBot4Directions.WEST),    # waypoint 1 -> index 0
    ([-0.725, 1.9], TurtleBot4Directions.SOUTH),    # waypoint 2 -> index 1
    ([-2.1, 1.95], TurtleBot4Directions.EAST),      # waypoint 3 -> index 2
    ([-2.15, -0.3], TurtleBot4Directions.EAST),     # waypoint 4 -> index 3
    ([-2.3, -2.0], TurtleBot4Directions.EAST),      # waypoint 5 -> index 4
    ([-2.4, -4.0], TurtleBot4Directions.NORTH),     # waypoint 6 -> index 5
    ([2.1, -4.0], TurtleBot4Directions.WEST),       # waypoint 7 -> index 6
    ([1.97, -2.5], TurtleBot4Directions.SOUTH),     # waypoint 8 -> index 7
    ([-1.5, -2.2], TurtleBot4Directions.WEST),      # waypoint 9 -> index 8
]

# 순찰 순서
# 사람 기준: [2,3,4,5,6,7,8,9,1]
# 코드 기준 index: [1,2,3,4,5,6,7,8,0]
PATROL_ORDER = [1, 2, 3, 4, 5, 6, 7, 8, 0]
PATROL_LENGTH = len(PATROL_ORDER)

# 마지막 웨이포인트는 1번
FINAL_WAYPOINT_INDEX = 0
FINAL_PATROL_POS = PATROL_LENGTH - 1

ZONES = [
    {
        "zone_id": 1,
        "waypoint_indices": [0, 1],  # waypoint 1, 2
        "corner1": (0.38, 2.55),
        "corner2": (-1.67, -1.37),
    },
    {
        "zone_id": 2,
        "waypoint_indices": [2, 3],  # waypoint 3, 4
        "corner1": (-1.67, 2.55),
        "corner2": (-2.91, -1.37),
    },
    {
        "zone_id": 3,
        "waypoint_indices": [4],     # waypoint 5
        "corner1": (-2.91, -1.37),
        "corner2": (-1.7, -3.23),
    },
    {
        "zone_id": 4,
        "waypoint_indices": [5, 6],  # waypoint 6, 7
        "corner1": (-3.0, -3.23),
        "corner2": (2.6, -5.15),
    },
    {
        "zone_id": 5,
        "waypoint_indices": [7, 8],  # waypoint 8, 9
        "corner1": (-1.67, -1.37),
        "corner2": (2.7, -3.23),
    },
]

TASK_POLL_PERIOD_SEC = 0.1
KEY_POLL_PERIOD_SEC = 0.05


# ================================
# 상태 상수
# ================================
MODE_PATROL = "PATROL"
MODE_ROUTE_TO_ZONE = "ROUTE_TO_ZONE"
MODE_WAIT_RESUME = "WAIT_RESUME"


# ================================
# 전역 상태
# ================================
state_lock = threading.Lock()

current_mode = MODE_PATROL

# q 입력 시 True
stop_requested = False

# g 입력 시 True
goto_zone_requested = False

# WAIT_RESUME 상태에서 g 입력 시 True
resume_requested = False

# g 입력으로 준비된 목표 구역 정보
target_zone_id = None
target_zone_first_index = None
target_zone_first_patrol_pos = None
target_zone_resume_patrol_pos = None


# ================================
# 유틸 함수
# ================================
def build_pose(navigator: TurtleBot4Navigator, waypoint_index: int):
    """
    waypoint index를 PoseStamped로 변환한다.
    """
    position, direction = WAYPOINTS[waypoint_index]
    return navigator.getPoseStamped(position, direction)


def point_in_zone(x: float, y: float, zone: dict) -> bool:
    """
    입력 좌표가 특정 구역 내부에 있는지 확인한다.
    """
    x1, y1 = zone["corner1"]
    x2, y2 = zone["corner2"]

    xmin = min(x1, x2)
    xmax = max(x1, x2)
    ymin = min(y1, y2)
    ymax = max(y1, y2)

    return xmin <= x <= xmax and ymin <= y <= ymax


def find_zone_by_point(x: float, y: float):
    """
    입력 좌표가 속한 구역을 반환한다.
    없으면 None을 반환한다.
    """
    for zone in ZONES:
        if point_in_zone(x, y, zone):
            return zone
    return None


def waypoint_index_to_patrol_pos(waypoint_index: int) -> int:
    """
    waypoint index를 PATROL_ORDER 상의 위치로 변환한다.
    예: waypoint 2(index 1) -> patrol pos 0
    """
    return PATROL_ORDER.index(waypoint_index)


def patrol_pos_to_waypoint_index(patrol_pos: int) -> int:
    """
    PATROL_ORDER 상의 위치를 실제 waypoint index로 변환한다.
    """
    return PATROL_ORDER[patrol_pos % PATROL_LENGTH]


def next_patrol_pos(patrol_pos: int) -> int:
    """
    다음 순찰 위치를 계산한다.
    """
    return (patrol_pos + 1) % PATROL_LENGTH


def prepare_zone_request():
    """
    TARGET_POSITION 기준으로 목표 구역 정보를 준비한다.

    반환값
    - True  : 구역 판별 성공
    - False : 어떤 구역에도 속하지 않음
    """
    global target_zone_id
    global target_zone_first_index
    global target_zone_first_patrol_pos
    global target_zone_resume_patrol_pos

    target_x, target_y = TARGET_POSITION
    zone = find_zone_by_point(target_x, target_y)

    if zone is None:
        return False

    first_index = zone["waypoint_indices"][0]
    first_patrol_pos = waypoint_index_to_patrol_pos(first_index)
    resume_patrol_pos = next_patrol_pos(first_patrol_pos)

    target_zone_id = zone["zone_id"]
    target_zone_first_index = first_index
    target_zone_first_patrol_pos = first_patrol_pos
    target_zone_resume_patrol_pos = resume_patrol_pos

    return True


def move_to_waypoint(
    navigator: TurtleBot4Navigator,
    waypoint_index: int,
    log_prefix: str,
):
    """
    특정 웨이포인트로 이동한다.
    """
    human_waypoint_number = waypoint_index + 1
    navigator.info(f"{log_prefix}: waypoint {human_waypoint_number} 이동 시작")

    pose = build_pose(navigator, waypoint_index)
    navigator.goToPose(pose)

    while not navigator.isTaskComplete():
        time.sleep(TASK_POLL_PERIOD_SEC)

    navigator.info(f"{log_prefix}: waypoint {human_waypoint_number} 도착 완료")


def move_to_pre_dock_and_dock(navigator: TurtleBot4Navigator):
    """
    pre-dock 위치로 이동한 뒤 dock를 수행한다.
    """
    navigator.info("Pre-dock 위치로 이동합니다.")

    pre_dock_pose = navigator.getPoseStamped(
        PRE_DOCK_POSITION,
        PRE_DOCK_DIRECTION,
    )
    navigator.goToPose(pre_dock_pose)

    while not navigator.isTaskComplete():
        time.sleep(TASK_POLL_PERIOD_SEC)

    navigator.info("Dock 실행")
    navigator.dock()


def should_shutdown_now(next_patrol_pos_value: int) -> bool:
    """
    q 입력 후 즉시 pre-dock -> dock 로 넘어가도 되는지 판단한다.

    기준:
    - 순찰 순서가 [2,3,4,5,6,7,8,9,1]
    - 마지막 웨이포인트는 1
    - next_patrol_pos_value == 0 이면 다음에 갈 웨이포인트가 2라는 뜻
      즉, 이미 마지막 웨이포인트 1을 지난 상태이므로 바로 종료 가능
    """
    return next_patrol_pos_value == 0


# ================================
# 키 입력 스레드
# ================================
def keyboard_listener():
    """
    비차단 방식으로 g / q 키를 감시한다.
    """
    global current_mode
    global stop_requested
    global goto_zone_requested
    global resume_requested

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        tty.setcbreak(fd)

        while True:
            readable, _, _ = select.select([sys.stdin], [], [], KEY_POLL_PERIOD_SEC)
            if not readable:
                continue

            key = sys.stdin.read(1)
            if not key:
                continue

            key = key.lower()

            if key == "g":
                with state_lock:
                    if current_mode == MODE_WAIT_RESUME:
                        resume_requested = True
                        print("\n[g] 입력됨 -> 다음 waypoint부터 순찰 재개")
                    else:
                        ok = prepare_zone_request()

                        if ok:
                            goto_zone_requested = True
                            print(
                                f"\n[g] 입력됨 -> TARGET_POSITION={TARGET_POSITION}, "
                                f"구역 {target_zone_id}, "
                                f"목표 첫 waypoint {target_zone_first_index + 1}"
                            )
                        else:
                            print(
                                f"\n[g] 입력됨 -> TARGET_POSITION={TARGET_POSITION} "
                                "가 어떤 구역에도 속하지 않습니다."
                            )

            elif key == "q":
                with state_lock:
                    stop_requested = True
                    print("\n[q] 입력됨 -> 마지막 waypoint 1까지 진행 후 pre-dock -> dock")

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    
def main():
    global current_mode
    global stop_requested
    global goto_zone_requested
    global resume_requested

    rclpy.init()
    navigator = TurtleBot4Navigator()

    # 키 입력 스레드 시작
    thread = threading.Thread(target=keyboard_listener, daemon=True)
    thread.start()

    # 시작 상태 보정
    if not navigator.getDockedStatus():
        navigator.info("Docking before initialising pose")
        navigator.dock()

    # 초기 위치 설정
    initial_pose = navigator.getPoseStamped(INITIAL_POSITION, INITIAL_DIRECTION)
    navigator.setInitialPose(initial_pose)

    # Nav2 활성화 대기
    navigator.waitUntilNav2Active()

    # 도킹 해제 후 바로 순찰 시작
    navigator.undock()
    navigator.info("순찰 시작: 순서 [2,3,4,5,6,7,8,9,1], g=구역 이동, q=종료")

    # patrol_pos는 "다음에 이동할 순찰 위치"를 의미한다.
    # 0이면 다음 이동은 waypoint 2
    patrol_pos = 0

    while rclpy.ok():
        with state_lock:
            local_mode = current_mode
            local_stop_requested = stop_requested
            local_goto_zone_requested = goto_zone_requested
            local_resume_requested = resume_requested
            local_target_zone_id = target_zone_id
            local_target_zone_first_index = target_zone_first_index
            local_target_zone_first_patrol_pos = target_zone_first_patrol_pos
            local_target_zone_resume_patrol_pos = target_zone_resume_patrol_pos

        # --------------------------------
        # WAIT_RESUME 상태 처리
        # --------------------------------
        if local_mode == MODE_WAIT_RESUME:
            # q가 들어왔을 때
            # 1번 웨이포인트에서 이미 멈춘 상태면 바로 종료
            # 아니면 다음 순번부터 이어서 1번까지 진행 후 종료
            if local_stop_requested:
                if local_target_zone_first_index == FINAL_WAYPOINT_INDEX:
                    navigator.info(
                        "현재 대기 위치가 마지막 waypoint 1 이므로 "
                        "바로 pre-dock -> dock 를 진행합니다."
                    )
                    break

                with state_lock:
                    current_mode = MODE_PATROL
                    if local_target_zone_resume_patrol_pos is not None:
                        patrol_pos = local_target_zone_resume_patrol_pos

                navigator.info(
                    f"q 요청 처리: waypoint "
                    f"{patrol_pos_to_waypoint_index(patrol_pos) + 1} 부터 이어서 "
                    "마지막 waypoint 1까지 진행합니다."
                )
                continue

            # g가 들어왔을 때
            # 목표 구역 첫 웨이포인트 다음 순번부터 재개
            if local_resume_requested:
                with state_lock:
                    resume_requested = False
                    current_mode = MODE_PATROL
                    if local_target_zone_resume_patrol_pos is not None:
                        patrol_pos = local_target_zone_resume_patrol_pos

                navigator.info(
                    f"순찰 재개: waypoint "
                    f"{patrol_pos_to_waypoint_index(patrol_pos) + 1} 부터 이동합니다."
                )
                continue

            time.sleep(TASK_POLL_PERIOD_SEC)
            continue

        # --------------------------------
        # g 요청 처리
        # --------------------------------
        if local_goto_zone_requested and local_mode != MODE_ROUTE_TO_ZONE:
            with state_lock:
                goto_zone_requested = False
                current_mode = MODE_ROUTE_TO_ZONE

            navigator.info(
                f"g 이벤트 처리: 현재 순번부터 시작해서 "
                f"구역 {local_target_zone_id}의 첫 waypoint "
                f"{local_target_zone_first_index + 1} 까지 순서대로 이동합니다."
            )
            continue

        # --------------------------------
        # q 종료 조건 확인
        # --------------------------------
        # q 이후 patrol_pos == 0 이면
        # 이미 마지막 waypoint 1을 지나 다음이 waypoint 2인 상태이므로
        # 바로 pre-dock -> dock 가능
        if local_stop_requested and should_shutdown_now(patrol_pos):
            navigator.info(
                "마지막 waypoint 1까지 완료된 상태이므로 "
                "pre-dock -> dock 를 시작합니다."
            )
            break

        # --------------------------------
        # 현재 순찰 순서의 waypoint 이동
        # --------------------------------
        current_waypoint_index = patrol_pos_to_waypoint_index(patrol_pos)

        move_log_prefix = "순찰"

        if local_mode == MODE_ROUTE_TO_ZONE:
            move_log_prefix = f"구역 {local_target_zone_id}로 이동 중"

        if local_stop_requested:
            move_log_prefix = "종료 전 이동"

        move_to_waypoint(
            navigator=navigator,
            waypoint_index=current_waypoint_index,
            log_prefix=move_log_prefix,
        )

        # 이번에 도착한 순찰 위치 / waypoint
        arrived_patrol_pos = patrol_pos
        arrived_waypoint_index = current_waypoint_index

        # 다음 순번으로 이동
        patrol_pos = next_patrol_pos(patrol_pos)

        # --------------------------------
        # 목표 구역 첫 웨이포인트 도착 처리
        # --------------------------------
        if (
            local_mode == MODE_ROUTE_TO_ZONE
            and arrived_waypoint_index == local_target_zone_first_index
        ):
            # q가 이미 들어온 상태라면 여기서 멈추지 않고
            # 계속 마지막 waypoint 1까지 진행한다.
            if local_stop_requested:
                with state_lock:
                    current_mode = MODE_PATROL

                navigator.info(
                    f"구역 {local_target_zone_id} 첫 waypoint 도착. "
                    "하지만 q 요청이 있으므로 계속 진행합니다."
                )
                continue

            # 일반 g 동작이면 여기서 멈추고 다시 g 입력을 기다린다.
            with state_lock:
                current_mode = MODE_WAIT_RESUME

            navigator.info(
                f"구역 {local_target_zone_id} 첫 waypoint 도착 -> 다음 g 입력 대기"
            )
            continue

    move_to_pre_dock_and_dock(navigator)
    rclpy.shutdown()


if __name__ == "__main__":
    main()