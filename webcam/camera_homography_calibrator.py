"""
Camera Homography Calibrator
==============================
4개의 기준점을 클릭하면 픽셀 → 실제 좌표(m) 변환 행렬을 생성하고 JSON으로 저장합니다.

조작 가이드:
  - 화면에서 기준점 4개를 순서대로 클릭 (P1 → P2 → P3 → P4)
  - [s] : 저장 (JSON)
  - [t] : 테스트 모드 토글 (클릭한 픽셀 → 실제 좌표 확인)
  - [r] : 초기화
  - [q] : 종료
"""

import cv2               # 카메라 입력 및 화면 표시
import numpy as np       # 행렬 연산 (Homography 계산)
import json              # 결과를 JSON 파일로 저장
from datetime import datetime   # 저장 시각 기록용


# ──────────────────────────────────────────────
# 기준점 실제 좌표 (단위: m) — 클릭 순서 P1~P4
# ──────────────────────────────────────────────
# 실제 공간에서 직접 측정한 좌표값이며, 클릭 순서(P1→P4)와 반드시 일치해야 한다.
WORLD_POINTS = [
    [-0.94,  -1.155],   # P1: 좌측 상단 기준점
    [-0.74,  -1.675],   # P2: 좌측 하단 기준점
    [ 2.3,   -1.68 ],   # P3: 우측 하단 기준점
    [ 2.52,  -1.175],   # P4: 우측 상단 기준점
]

CONFIG_PATH  = "webcam/homography.json"   # 계산된 행렬을 저장할 JSON 파일 경로
CAMERA_INDEX = 2                          # cv2.VideoCapture 에 전달할 카메라 인덱스


# ──────────────────────────────────────────────
# Homography 계산
# ──────────────────────────────────────────────

def compute_homography(image_pts, world_pts):
    """
    클릭으로 수집한 픽셀 좌표(image_pts)와 실제 좌표(world_pts)로
    Homography 행렬을 계산하고, RMS 재투영 오차를 함께 반환한다.

    cv2.findHomography(RANSAC): 아웃라이어에 강인한 최적 행렬을 추정한다.
    RMS 오차: 픽셀 좌표를 행렬로 변환한 결과가 실제 좌표와 얼마나 다른지를 나타내며,
              값이 낮을수록 캘리브레이션 정확도가 높다.
    """
    src = np.array(image_pts, dtype=np.float32)   # 픽셀 좌표 배열 (N×2)
    dst = np.array(world_pts,  dtype=np.float32)  # 실제 좌표 배열 (N×2)
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)   # RANSAC으로 아웃라이어 제거하며 행렬 추정

    # RMS 재투영 오차 계산
    src_h = np.concatenate([src, np.ones((len(src), 1))], axis=1)  # 동차 좌표로 변환 (N×3): [x, y, 1]
    proj  = (H @ src_h.T).T       # H 를 적용해 픽셀 → 실제 좌표 변환 (N×3)
    proj /= proj[:, 2:3]          # 동차 좌표 정규화 (w로 나눠 실제 (x,y) 복원)
    rms   = float(np.sqrt(np.mean((proj[:, :2] - dst) ** 2)))   # 변환 결과와 실제 좌표 간 평균 제곱근 오차
    return H, rms


def pixel_to_world(H, px, py):
    """
    단일 픽셀 좌표 (px, py) 를 Homography 행렬 H로 실제 좌표(m)로 변환한다.
    테스트 모드에서 클릭한 임의 픽셀의 실제 좌표를 확인하는 데 사용된다.
    """
    pt    = np.array([[[px, py]]], dtype=np.float32)   # perspectiveTransform 입력 형식 (1,1,2)
    world = cv2.perspectiveTransform(pt, H)            # Homography 적용
    return float(world[0, 0, 0]), float(world[0, 0, 1])   # (x, y) 실수값으로 반환


def load_homography(path=CONFIG_PATH):
    """다른 스크립트에서 행렬만 빠르게 로드할 때 사용"""
    with open(path) as f:
        d = json.load(f)                                   # JSON 파싱
    H = np.array(d["homography_matrix"], dtype=np.float64)  # 3×3 행렬로 복원
    print(f"[Load] {path}  RMS={d.get('rms_error', 'N/A')}")
    return H


# ──────────────────────────────────────────────
# 그리기
# ──────────────────────────────────────────────

# 기준점별 표시 색상 (P1~P4 순서)
POINT_COLORS = [
    (0,   255,   0),   # P1 초록
    (0,   255, 255),   # P2 노랑
    (255, 128,   0),   # P3 주황
    (255,   0, 255),   # P4 마젠타
]

def draw_overlay(frame, state):
    """
    현재 상태(state)를 바탕으로 프레임 위에 안내 정보와 시각 피드백을 그린다.

    등록된 기준점, 다음 클릭 안내, 행렬 계산 후 그리드 미리보기,
    테스트 모드 클릭 결과를 모두 이 함수에서 처리한다.
    """
    h, w = frame.shape[:2]   # 프레임 높이·너비 (텍스트 위치 계산용)

    # ── 등록된 기준점 표시 ──
    for i, (px, py) in enumerate(state["image_pts"]):   # 지금까지 클릭한 기준점 순회
        color = POINT_COLORS[i]                          # 해당 기준점 색상
        cv2.circle(frame, (px, py), 8, color, -1)        # 채운 원으로 기준점 위치 표시
        cv2.circle(frame, (px, py), 10, (255, 255, 255), 1)  # 흰 테두리로 가시성 향상
        wx, wy = WORLD_POINTS[i]
        # 기준점 옆에 실제 좌표 레이블 표시
        cv2.putText(frame, f"P{i+1} ({wx},{wy}m)",
                    (px + 12, py - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, color, 1, cv2.LINE_AA)

    # ── 다음 클릭할 기준점 안내 ──
    next_i = len(state["image_pts"])   # 아직 입력되지 않은 다음 기준점 인덱스
    if next_i < 4:                     # 4개 미만이면 다음 기준점 안내 출력
        wx, wy = WORLD_POINTS[next_i]
        color  = POINT_COLORS[next_i]
        cv2.putText(frame, f"Click  P{next_i+1}  ({wx}, {wy} m)",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2, cv2.LINE_AA)   # 굵게(두께 2) 표시해 눈에 잘 띄게

    # ── 행렬이 계산된 후: 그리드 오버레이로 변환 품질 시각화 ──
    if state["H"] is not None:
        H_inv  = np.linalg.inv(state["H"])   # 역 Homography: 실제 좌표 → 픽셀로 되돌리는 행렬
        wpts   = np.array(WORLD_POINTS)
        wx_min, wx_max = wpts[:, 0].min(), wpts[:, 0].max()   # 실제 좌표계 x 범위
        wy_min, wy_max = wpts[:, 1].min(), wpts[:, 1].max()   # 실제 좌표계 y 범위

        def wpt_to_px(wx, wy):
            """실제 좌표 한 점을 역 Homography로 픽셀 좌표로 변환 (그리드 그리기용)"""
            pt = np.array([[[wx, wy]]], dtype=np.float32)
            p  = cv2.perspectiveTransform(pt, H_inv)
            return int(p[0, 0, 0]), int(p[0, 0, 1])

        STEPS = 12   # 가로·세로 각 방향으로 그릴 그리드 선 수

        # 세로 방향 그리드선: x 를 STEPS 등분, 각 x에서 y 방향으로 선 그리기
        for xi in np.linspace(wx_min, wx_max, STEPS):
            pts = [wpt_to_px(xi, yi) for yi in np.linspace(wy_min, wy_max, 30)]
            pts = [(x, y) for x, y in pts if 0 <= x < w and 0 <= y < h]  # 프레임 범위 밖 점 제거
            for j in range(len(pts) - 1):
                cv2.line(frame, pts[j], pts[j+1], (255, 255, 0), 1)   # 노란 선으로 그리드 표시

        # 가로 방향 그리드선: y 를 STEPS 등분, 각 y에서 x 방향으로 선 그리기
        for yi in np.linspace(wy_min, wy_max, STEPS):
            pts = [wpt_to_px(xi, yi) for xi in np.linspace(wx_min, wx_max, 30)]
            pts = [(x, y) for x, y in pts if 0 <= x < w and 0 <= y < h]  # 프레임 범위 밖 점 제거
            for j in range(len(pts) - 1):
                cv2.line(frame, pts[j], pts[j+1], (255, 255, 0), 1)   # 노란 선으로 그리드 표시

        rms_str  = f"RMS={state['rms']:.5f} m"   # RMS 오차 문자열
        mode_str = "[TEST MODE]  클릭으로 좌표 확인" if state["test_mode"] else "[s]저장  [t]테스트모드"
        # 하단에 완료 상태 및 조작 안내 출력
        cv2.putText(frame, f"완료  {rms_str}  |  {mode_str}",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 0), 1, cv2.LINE_AA)
    else:
        # 행렬 미계산 상태: 현재까지 입력된 점 개수와 기본 조작 안내 표시
        cv2.putText(frame, f"{next_i}/4 점 입력됨  |  [r]초기화  [q]종료",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (200, 200, 200), 1, cv2.LINE_AA)

    # ── 테스트 모드: 마지막으로 클릭한 픽셀의 실제 좌표 표시 ──
    if state["test_mode"] and state["last_test"]:
        px, py, wx, wy = state["last_test"]            # 마지막 테스트 클릭 정보
        cv2.circle(frame, (px, py), 8, (0, 0, 255), -1)   # 빨간 원으로 클릭 위치 표시
        # 클릭 위치 옆에 변환된 실제 좌표 출력
        cv2.putText(frame, f"({wx:.3f}, {wy:.3f}) m",
                    (px + 12, py + 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 255), 1, cv2.LINE_AA)


# ──────────────────────────────────────────────
# 저장
# ──────────────────────────────────────────────

def save_config(state, frame_size):
    """
    계산된 Homography 행렬과 캘리브레이션 메타데이터를 JSON 파일로 저장한다.

    저장되는 항목: 행렬(homography_matrix), 픽셀 기준점(image_points),
    실제 기준점(world_points), 카메라 인덱스, 해상도, 저장 시각, RMS 오차.
    rc_car_tracker.py 는 이 파일을 읽어 좌표 변환에 사용한다.
    """
    data = {
        "homography_matrix": state["H"].tolist(),    # numpy 배열 → JSON 직렬화 가능한 리스트로 변환
        "image_points":      state["image_pts"],     # 클릭한 픽셀 좌표 4개
        "world_points":      WORLD_POINTS,           # 대응하는 실제 좌표 4개
        "camera_device":     CAMERA_INDEX,           # 사용한 카메라 인덱스 (재현성 확인용)
        "frame_size":        list(frame_size),       # 캡처 해상도 [width, height]
        "created_at":        datetime.now().isoformat(),  # 저장 시각 (ISO 8601 형식)
        "rms_error":         state["rms"],           # RMS 재투영 오차 (정확도 지표)
    }
    with open(CONFIG_PATH, "w") as f:
        json.dump(data, f, indent=2)   # 들여쓰기 2칸으로 사람이 읽기 쉬운 형태로 저장
    print(f"[Save] {CONFIG_PATH}  RMS={state['rms']:.5f} m")


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────

def main():
    """
    카메라를 열고 마우스 콜백을 등록한 뒤, 기준점 수집 → 행렬 계산 → 저장 루프를 실행한다.

    state 딕셔너리로 모든 UI 상태를 관리하며,
    mouse_callback 은 클릭 이벤트마다 state 를 갱신하고 행렬을 자동 계산한다.
    """
    cap = cv2.VideoCapture(CAMERA_INDEX)              # 지정 인덱스 카메라 열기
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)           # 캡처 너비 설정 (rc_car_tracker 와 동일해야 함)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)           # 캡처 높이 설정 (rc_car_tracker 와 동일해야 함)

    if not cap.isOpened():   # 카메라 열기 실패 시 즉시 종료
        print(f"[Error] 카메라 {CAMERA_INDEX}번을 열 수 없습니다.")
        return

    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))   # 실제 적용된 해상도 읽기 (설정값과 다를 수 있음)
    print(f"[Camera] index={CAMERA_INDEX}  {frame_size[0]}x{frame_size[1]}")
    print("\n기준점을 순서대로 클릭하세요:")
    for i, (wx, wy) in enumerate(WORLD_POINTS):
        print(f"  P{i+1} : ({wx}, {wy}) m")   # 클릭 순서별 실제 좌표 안내

    # UI 상태 딕셔너리: 모든 가변 상태를 한 곳에서 관리 (mouse_callback 과 공유)
    state = {
        "image_pts": [],     # 클릭으로 수집된 픽셀 좌표 리스트 (최대 4개)
        "H":         None,   # 계산된 Homography 행렬 (4개 전 None)
        "rms":       None,   # RMS 재투영 오차 (4개 전 None)
        "test_mode": False,  # 테스트 모드 활성 여부
        "last_test": None,   # 테스트 모드 마지막 클릭 결과 (px, py, wx, wy)
    }

    def mouse_callback(event, x, y, flags, param):
        """
        마우스 왼쪽 버튼 클릭 이벤트를 처리한다.

        테스트 모드이면 클릭 픽셀을 실제 좌표로 변환해 출력하고,
        기준점 수집 모드이면 state["image_pts"]에 추가하며
        4개가 채워지는 순간 Homography 행렬을 자동 계산한다.
        """
        if event != cv2.EVENT_LBUTTONDOWN:   # 왼쪽 버튼 클릭 이외의 이벤트는 무시
            return

        # ── 테스트 모드: 클릭 → 실제 좌표 변환 및 출력 ──
        if state["test_mode"] and state["H"] is not None:
            wx, wy = pixel_to_world(state["H"], x, y)   # 클릭한 픽셀을 실제 좌표로 변환
            state["last_test"] = (x, y, wx, wy)         # draw_overlay 에서 화면에 표시하기 위해 저장
            print(f"[Test] 픽셀({x:4d},{y:4d})  →  ({wx:.4f}, {wy:.4f}) m")
            return

        # ── 기준점 수집 모드 ──
        if len(state["image_pts"]) < 4:   # 4개가 채워지기 전까지만 수집
            i = len(state["image_pts"])       # 현재 입력 중인 기준점 인덱스
            state["image_pts"].append([x, y]) # 클릭 좌표를 기준점 리스트에 추가
            wx, wy = WORLD_POINTS[i]
            print(f"[P{i+1}] 픽셀({x},{y})  ↔  실제({wx},{wy}) m")

            if len(state["image_pts"]) == 4:   # 4개가 모두 입력되면 행렬 자동 계산
                H, rms = compute_homography(state["image_pts"], WORLD_POINTS)
                state["H"]   = H     # 계산된 행렬 저장
                state["rms"] = rms   # RMS 오차 저장
                print(f"\n[Homography] 완료  RMS={rms:.5f} m")
                print(H)
                print("[s]저장  [t]테스트  [r]초기화  [q]종료")

    cv2.namedWindow("Homography Calibrator")
    cv2.setMouseCallback("Homography Calibrator", mouse_callback)   # 창에 마우스 콜백 등록

    while True:
        ret, frame = cap.read()   # 카메라에서 프레임 읽기
        if not ret:               # 프레임 읽기 실패 시 루프 종료
            break

        display = frame.copy()    # 원본 프레임을 복사해 오버레이 그리기 (원본 보존)
        draw_overlay(display, state)             # 현재 상태 시각화
        cv2.imshow("Homography Calibrator", display)   # 화면에 표시

        key = cv2.waitKey(1) & 0xFF   # 1ms 대기 후 키 입력 확인 (& 0xFF: 8비트 마스킹)

        if key == ord('q'):   # 'q': 종료
            break
        elif key == ord('s'):   # 's': JSON 저장
            if state["H"] is not None:
                save_config(state, frame_size)
            else:
                print("[Save] 4개 점을 먼저 입력하세요.")
        elif key == ord('t'):   # 't': 테스트 모드 토글
            if state["H"] is not None:
                state["test_mode"] = not state["test_mode"]   # True ↔ False 전환
                print(f"[Test] {'ON' if state['test_mode'] else 'OFF'}")
            else:
                print("[Test] 먼저 4개 점을 입력하세요.")
        elif key == ord('r'):   # 'r': 전체 초기화
            state["image_pts"].clear()   # 수집된 기준점 삭제
            state["H"]         = None   # 행렬 초기화
            state["rms"]       = None   # RMS 초기화
            state["test_mode"] = False  # 테스트 모드 해제
            state["last_test"] = None   # 마지막 테스트 결과 초기화
            print("[Reset] 초기화.")

    cap.release()             # 카메라 자원 해제
    cv2.destroyAllWindows()   # 모든 OpenCV 창 닫기


if __name__ == "__main__":
    main()
