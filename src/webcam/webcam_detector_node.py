#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
[프로젝트: Click Car - CCTV 웹캠 기반 불법주정차 감시 노드]
- 최종 수정: 2026-03-12

[원본] webcam_detector.py → ROS2 노드화

[System Architecture & Role]
1. 카메라  : cv2.VideoCapture 로 웹캠 영상을 직접 읽는다
2. 탐지    : YOLOv8 + ByteTrack 으로 차량 ID를 프레임 간 추적한다
3. 좌표 변환: Homography(캘리브레이션 JSON) 으로 픽셀 → 실제 좌표(m) 변환
4. 감시    : ParkingWatcher 가 90초 이상 연속 감지된 차량을 불법주정차로 확정한다
5. 퍼블리시: confirmed 이벤트 발생 시 std_msgs/String 으로
             'cctv_done' 토픽에 '<x>,<y>' 형식의 실제 좌표를 퍼블리시한다
6. Firebase: ENABLE_FIREBASE=True 이면 실시간 위치 + 증거 로그를 Realtime DB에 업로드한다

[퍼블리시 토픽]
  /cctv_done  (std_msgs/String)
    - 메시지 내용: "<center_x>,<center_y>"  (단위: m, 소수점 4자리)
    - 발행 조건  : ILLEGAL_PARK_SEC(90초) 이상 연속 감지된 차량 1건당 1회만 발행
    - 수신 측    : AMR(ocr_node.py) 이 이 메시지를 받아 cctv_start 모드로 출동

[ROS2 노드 구조]
  - 1Hz 타이머 콜백에서 카메라 프레임을 읽고 YOLO 추론을 수행한다
    (카메라 루프를 spin() 과 분리하기 위해 create_timer 사용)
  - Firebase 업로드는 confirmed 이벤트 발생 시 직접 호출로 처리한다 (실시간 스트리밍 없음)
  - destroy_node() 에서 카메라 자원과 OpenCV 창을 정리한다
'''

# ── 표준 라이브러리 ──────────────────────────────
import base64          # Firebase 이미지 업로드용 base64 인코딩
import json            # homography.json 로드
import os              # 로컬 임시 이미지 파일 삭제
import threading       # Firebase 업로드 데몬 스레드
import time            # 타이머 계산용 (Unix 시각)
from datetime import datetime   # 로그 타임스탬프

# ── 외부 라이브러리 ──────────────────────────────
import cv2             # 카메라 입력, 이미지 처리, 시각화
import numpy as np     # 행렬 연산 (Homography)
from ultralytics import YOLO    # YOLOv8 + ByteTrack

# ── ROS2 ──────────────────────────────────────────
import rclpy
from rclpy.node import Node
from std_msgs.msg import String    # cctv_done 퍼블리시용


# ══════════════════════════════════════════════════
#  [CHAPTER 1: 하이퍼파라미터]
# ══════════════════════════════════════════════════    

HOMOGRAPHY_JSON  = "/home/rokey/click_car/src/webcam/homography.json"
MODEL_PATH       = "/home/rokey/click_car/models/webcam/v2/weights/best.pt"
CAMERA_INDEX     = 2       # cv2.VideoCapture 카메라 인덱스
CAM_WIDTH        = 640     # 캡처 해상도 너비 (캘리브레이션과 반드시 일치)
CAM_HEIGHT       = 480     # 캡처 해상도 높이 (캘리브레이션과 반드시 일치)

OFFSET_X_PLUS    =  0.0    # x 좌표 +보정
OFFSET_X_MINUS   =  0.0    # x 좌표 -보정
OFFSET_Y_PLUS    =  0.0    # y 좌표 +보정
OFFSET_Y_MINUS   =  0.45   # y 좌표 -보정

ENABLE_TERMINAL  = True    # 감지 결과를 터미널에 출력 (TERMINAL_INTERVAL 초마다 1회)
TERMINAL_INTERVAL = 10.0  # 터미널 출력 최소 간격 (초)
ENABLE_FIREBASE  = True   # True 이면 Realtime DB 업로드 (터미널 검증 후 전환)

FIREBASE_CRED   = "/home/rokey/Downloads/iligalstop-firebase-adminsdk-fbsvc-d989ef0f8c.json"              # 서비스 계정 키 파일 경로
FIREBASE_DB_URL = "https://iligalstop-default-rtdb.asia-southeast1.firebasedatabase.app"  # Realtime DB URL
DB_PATH          = "webcam"       # DB 내 최상위 경로
LOG_DB_PATH      = "webcam/logs"      # 기존 증거 로그 경로 (webcam/logs/car_<id>/...)
CCTV_DB_PATH     = "cctv_detections"  # AMR 과 공유하는 CCTV 연동 증거 경로

# 로컬 임시 이미지 저장 디렉토리 (confirmed 전까지 first_seen 프레임을 보관)
TEMP_IMAGE_DIR   = "/tmp/click_car_cctv"

ILLEGAL_PARK_SEC = 90.0    # 이 시간(초) 이상 연속 감지 시 불법주정차 확정 + cctv_done 퍼블리시

# ── cctv_done 재전송 횟수 ─────────────────────────────────────────────────────
CCTV_DONE_REPEAT   = 5     # cctv_done 을 몇 번 반복 발행할지
CCTV_DONE_INTERVAL = 0.2   # 반복 발행 간격 (초)

# ROS2 퍼블리시 토픽
TOPIC_CCTV_DONE  = "/cctv_done"   # 불법주정차 확정 시 '<x>,<y>' 문자열 퍼블리시

# ── 로봇 상태 연동 모드 ──────────────────────────────
# True  : /robot2/robot2_status, /robot3/robot3_status 를 구독하여
#         patrol 중인 로봇의 /<robot>/cctv_done 토픽으로 퍼블리시한다.
#         우선순위: 둘 다 patrol → robot2 우선 / 둘 다 아님 → 기존 /cctv_done 유지
# False : 기존대로 /cctv_done 으로 퍼블리시
ROBOT_STATUS_MODE = True


# ══════════════════════════════════════════════════
#  [CHAPTER 2: Homography 유틸리티]
# ══════════════════════════════════════════════════

def load_homography(path: str):
    """
    저장된 JSON 파일에서 Homography 행렬과 실제 기준점 좌표를 불러온다.

    camera_homography_calibrator.py 가 생성한 JSON을 읽으며,
    H (픽셀→실제 좌표 변환 행렬) 와 world_pts (ROI 폴리곤 생성용 기준점) 를 반환한다.
    """
    with open(path) as f:
        d = json.load(f)
    H         = np.array(d["homography_matrix"], dtype=np.float64)
    world_pts = np.array(d["world_points"],      dtype=np.float32)
    return H, world_pts


def pixel_to_world(H, px, py):
    """
    단일 픽셀 좌표 (px, py) 를 Homography 행렬 H 로 실제 좌표(m)로 변환한다.
    사용자 OFFSET 값을 적용해 최종 실제 좌표를 반환한다.
    """
    pt = np.array([[[float(px), float(py)]]], dtype=np.float32)
    w  = cv2.perspectiveTransform(pt, H)
    wx = float(w[0, 0, 0]) + OFFSET_X_PLUS - OFFSET_X_MINUS
    wy = float(w[0, 0, 1]) + OFFSET_Y_PLUS - OFFSET_Y_MINUS
    return wx, wy


def make_roi_polygon(H_inv, world_pts):
    """
    실제 좌표계 기준점 4개를 역 Homography(H_inv)로 픽셀 공간의 ROI 폴리곤을 생성한다.
    """
    pts = np.array(world_pts, dtype=np.float32).reshape(-1, 1, 2)
    px  = cv2.perspectiveTransform(pts, H_inv)
    return px.reshape(-1, 2).astype(np.int32)


def point_in_roi(roi_poly, px, py):
    """
    픽셀 좌표 (px, py) 가 ROI 폴리곤 내부인지 판별한다. 경계 포함 내부이면 True.
    """
    return cv2.pointPolygonTest(
        roi_poly.astype(np.float32),
        (float(px), float(py)),
        False
    ) >= 0


def bbox_corners_world(H, x1, y1, x2, y2):
    """
    YOLO bounding box의 픽셀 4꼭짓점을 실제 좌표(m)로 변환한다.
    반환 순서: 좌상 → 우상 → 우하 → 좌하
    """
    corners_px = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    return [pixel_to_world(H, px, py) for px, py in corners_px]


# ══════════════════════════════════════════════════
#  [CHAPTER 3: 터미널 출력]
# ══════════════════════════════════════════════════

def output_terminal(car_info: list):
    """
    현재 프레임에서 감지된 차량 정보를 터미널에 출력한다.
    ENABLE_TERMINAL=True 일 때만 호출된다.
    """
    if not car_info:
        return
    print(f"\n{'─'*62}")
    for c in car_info:
        print(f"  ID:{c['id']:>3}  {c['label']:<10}  conf:{c['conf']:.2f}"
              f"  center=({c['center_x']:+.3f}, {c['center_y']:+.3f}) m")
        label_map = {"top_left": "좌상", "top_right": "우상",
                     "bottom_right": "우하", "bottom_left": "좌하"}
        for key, corner in c["corners_m"].items():
            print(f"         {label_map[key]}: ({corner['x']:+.4f}, {corner['y']:+.4f}) m")


# ══════════════════════════════════════════════════
#  [CHAPTER 4: Firebase 업로더]
# ══════════════════════════════════════════════════

class FirebaseUploader:
    """
    Realtime Database에 불법주정차 confirmed 증거를 업로드하는 클래스.

    confirmed 확정 시에만 DB에 기록하며, 미확정 차량 정보는 업로드하지 않는다.
    (실시간 위치 스트리밍 기능 제거 — 맵 표시 불필요로 단순화)
    """

    def __init__(self, cred_path: str, db_url: str, db_path: str):
        import firebase_admin
        from firebase_admin import credentials, db as rtdb

        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred, {"databaseURL": db_url})
        self.db_path = db_path
        self.rtdb    = rtdb
        print(f"[Firebase] Realtime DB 연결 완료  경로: /{db_path}")


# ══════════════════════════════════════════════════
#  [CHAPTER 5: 불법주정차 감시]
# ══════════════════════════════════════════════════

class ParkingWatcher:
    """
    차량별 최초 감지 시각과 불법주정차 판정 여부를 추적하는 클래스.

    YOLO의 불완전한 탐지로 ID가 일시 소실될 수 있으므로,
    VANISH_GRACE_SEC(기본 3초) 이내 재등장한 차량은 타이머를 유지한다.

    매 프레임 update() 호출 시 반환 이벤트:
      "first_seen" : 신규 등장 차량
      "confirmed"  : ILLEGAL_PARK_SEC 이상 누적 감지 → 불법주정차 확정 (1회만)
      "cancelled"  : confirmed 없이 grace 초과 소멸 → 임시 데이터 정리용
    """

    def __init__(self, threshold_sec: float, grace_sec: float = 3.0):
        self.threshold = threshold_sec
        self.grace     = grace_sec
        self._tracks   = {}   # {tid: {"first_seen", "last_seen", "confirmed"}}

    def update(self, current_ids: set, car_info_by_id: dict) -> list:
        """
        현재 프레임의 감지 차량 ID 집합을 받아 상태를 갱신하고
        이번 프레임에서 발생한 이벤트 목록을 반환한다.

        반환 형식: [{"event": "first_seen" | "confirmed" | "cancelled", "car": <car_info dict> | None}, ...]
        """
        events = []
        now    = time.time()

        # ── 1단계: grace 초과 후 사라진 차량 기록 삭제 ──
        gone_ids = set(self._tracks.keys()) - current_ids
        for tid in list(gone_ids):
            if now - self._tracks[tid]["last_seen"] > self.grace:
                # confirmed 없이 소멸 → 임시 이미지 정리 이벤트 발행
                if not self._tracks[tid]["confirmed"]:
                    events.append({"event": "cancelled", "tid": tid})
                del self._tracks[tid]

        # 2단계: 현재 차량 갱신 및 경과 시간 계산
        for tid in current_ids:
            if tid not in self._tracks:
                self._tracks[tid] = {
                    "first_seen": now,
                    "last_seen":  now,
                    "confirmed":  False,
                }
                events.append({"event": "first_seen", "car": car_info_by_id[tid]})
            else:
                self._tracks[tid]["last_seen"] = now

            # car_info에 실시간 경과 시간 추가 (시각화용)
            elapsed = now - self._tracks[tid]["first_seen"]
            car_info_by_id[tid]["elapsed"] = elapsed
            car_info_by_id[tid]["confirmed"] = self._tracks[tid]["confirmed"]

            # 확정 이벤트 발생 (1회 한정)
            if elapsed >= self.threshold and not self._tracks[tid]["confirmed"]:
                self._tracks[tid]["confirmed"] = True 
                events.append({"event": "confirmed", "car": car_info_by_id[tid]})

        return events


# ══════════════════════════════════════════════════
#  [CHAPTER 6: 유틸리티]
# ══════════════════════════════════════════════════

def frame_to_b64(frame) -> str:
    """ OpenCV 프레임을 JPEG(품질 85)로 인코딩한 뒤 base64 문자열로 반환한다. """
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode("utf-8")


def save_frame_local(car_id: int, frame) -> str:
    """
    first_seen 프레임을 로컬 디스크에 임시 저장하고 파일 경로를 반환한다.
    confirmed 전까지 보관되며, confirmed 또는 cancelled 시 삭제된다.
    저장 경로: TEMP_IMAGE_DIR/car_<id>_<타임스탬프>.jpg
    """
    os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path   = os.path.join(TEMP_IMAGE_DIR, f"car_{car_id}_{ts_str}.jpg")
    cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return path


def delete_frame_local(path: str):
    """
    로컬에 임시 저장된 프레임 파일을 삭제한다.
    파일이 없거나 오류가 발생해도 조용히 무시한다.
    """
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception as e:
        print(f"[TempImage] 삭제 오류: {e}")


def upload_parking_log(rtdb, log_path: str, event: str, car: dict, frame):
    """
    불법주정차 증거 로그를 Firebase Realtime DB에 업로드한다.

    저장 경로: <log_path>/car_<id>/<event>_<타임스탬프>
    저장 내용: 이벤트 종류, 시각(ISO), 차량 ID, 중심 좌표(m), base64 이미지
    """
    now_dt    = datetime.now()
    now_str   = now_dt.isoformat()
    ts_key    = now_dt.strftime("%Y%m%d_%H%M%S")
    event_key = f"{event}_{ts_key}"   # ex) first_seen_20250311_142305

    log_data = {
        "event":     event,
        "timestamp": now_str,
        "car_id":    car["id"],
        "center_x":  car["center_x"],
        "center_y":  car["center_y"],
        "image_b64": frame_to_b64(frame),
    }

    path = f"{log_path}/car_{car['id']}/{event_key}"
    try:
        rtdb.reference(path).set(log_data)
        print(f"[Log] {event:12s}  ID:{car['id']}  {now_str}")
    except Exception as e:
        print(f"[Log] 업로드 오류: {e}")


def upload_cctv_detection(rtdb, cctv_db_path: str, event: str, car: dict,
                          frame=None, initial_image_path: str = None) -> str | None:
    """
    cctv_detections/<case_key>/ 경로에 CCTV 증거를 업로드한다.
    ocr_node(AMR)가 동일 케이스 키를 받아 AMR 증거를 merge 할 수 있도록
    confirmed 이벤트 시 생성한 case_key 를 반환한다.

    ┌─ confirmed 이벤트 (변경됨) ──────────────────────────────────────────────┐
    │  새 케이스 키(타임스탬프)로 cctv_detections/<key> 문서를 생성한다.        │
    │  로컬에 저장해 둔 first_seen 프레임(initial_image_path)과               │
    │  현재 confirmed 프레임(frame)을 한 번에 업로드한다.                       │
    │  업로드 완료 후 로컬 임시 파일을 삭제한다.                                │
    │  반환값: case_key                                                        │
    └──────────────────────────────────────────────────────────────────────────┘

    최종 DB 구조 (ocr_node 가 AMR 증거를 merge 한 후):
      cctv_detections/<YYYYMMDD_HHMMSSffffff>/
        ├── cctv_detected_at      CCTV 최초 감지 ISO 시각  ← car["first_seen_at"] 에서 주입
        ├── cctv_confirmed_at     CCTV 90초 확정 ISO 시각
        ├── cctv_initial_image    CCTV 최초 프레임 base64  ← 로컬 파일에서 읽어 인코딩
        ├── cctv_evidence_image   CCTV 90초 후 프레임 base64
        ├── center_x / center_y  차량 실제 좌표 (m)
        ├── amr_initial_image    AMR 도착 직후 전체 프레임  ← ocr_node 기록
        ├── amr_evidence_image   AMR 확정 시점 전체 프레임  ← ocr_node 기록
        ├── plate_image          번호판 크롭                ← ocr_node 기록
        └── plate_number         OCR 인식 결과              ← ocr_node 기록
    """
    now_dt  = datetime.now()
    now_iso = now_dt.isoformat()

    if event == "confirmed":
        case_key = now_dt.strftime("%Y%m%d_%H%M%S_%f")

        # 로컬에 저장된 first_seen 프레임을 읽어 base64 변환
        initial_b64 = None
        if initial_image_path and os.path.exists(initial_image_path):
            initial_frame = cv2.imread(initial_image_path)
            if initial_frame is not None:
                initial_b64 = frame_to_b64(initial_frame)

        try:
            rtdb.reference(f"{cctv_db_path}/{case_key}").set({
                "cctv_detected_at":    car.get("first_seen_at"),   # first_seen 시각 (호출자 주입)
                "cctv_confirmed_at":   now_iso,
                "cctv_initial_image":  initial_b64,                # 최초 프레임 (로컬 → b64)
                "cctv_evidence_image": frame_to_b64(frame),        # 90초 후 프레임
                "center_x":            car["center_x"],
                "center_y":            car["center_y"],
                # AMR 증거 필드는 ocr_node 가 채운다
                "amr_initial_image":   None,
                "amr_evidence_image":  None,
                "plate_image":         None,
                "plate_number":        None,
            })
            print(f"[cctv_detections/confirmed] key={case_key}  ID:{car['id']}")
        except Exception as e:
            print(f"[cctv_detections] confirmed 업로드 오류: {e}")
            return None

        # 업로드 완료 후 로컬 임시 파일 삭제
        delete_frame_local(initial_image_path)
        return case_key

    return None


def draw_overlay(frame, H_inv, car_info, roi_poly, threshold_sec):
    """
    카메라 프레임 위에 ROI 경계(노란 실선)와 차량 정보를 시각화한다.
    중심점(빨간 원), 실제 좌표 역변환 bbox(주황 사각형), ID+좌표 텍스트를 그린다.
    """
    cv2.polylines(frame, [roi_poly], True, (0, 255, 255), 2)

    for c in car_info:
        cx_px, cy_px = c["pixel_center"]
        elapsed = c.get("elapsed", 0.0)
        
        # --- 게이지(Progress Bar) 그리기 (ocr_node 참고) ---
        # 90초를 기준으로 비율 계산 (0.0 ~ 1.0)
        ratio = min(elapsed / threshold_sec, 1.0)
        
        # 게이지 위치 설정 (중심점 위쪽)
        bar_w = 100  # 게이지 총 너비
        bar_h = 8    # 게이지 높이
        bar_x = cx_px - (bar_w // 2)
        bar_y = cy_px - 40 
        
        # 배경 (회색)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (100, 100, 100), -1)
        # 진행 바 (색상 변경: 확정 전 주황, 확정 후 빨강)
        bar_color = (0, 0, 255) if c.get("confirmed") else (0, 165, 255)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * ratio), bar_y + bar_h), bar_color, -1)
        
        # 시간 텍스트
        cv2.putText(frame, f"{elapsed:.1f}s / {threshold_sec:.0f}s", 
                    (bar_x, bar_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        # -----------------------------------------------

        # 기존 앵커 및 좌표 정보
        cv2.circle(frame, (cx_px, cy_px), 5, (0, 0, 255), -1)
        
        # Homography 역변환 bbox 시각화 (생략 가능하나 유지)
        corners_m = list(c["corners_m"].values())
        corners_np = np.array([[p["x"], p["y"]] for p in corners_m], dtype=np.float32).reshape(-1, 1, 2)
        corners_px = cv2.perspectiveTransform(corners_np, H_inv).reshape(-1, 2).astype(np.int32)
        cv2.polylines(frame, [corners_px], True, (255, 128, 0), 2)


# ══════════════════════════════════════════════════
#  [CHAPTER 7: ROS2 노드]
# ══════════════════════════════════════════════════

class WebcamDetectorNode(Node):
    """
    웹캠 영상으로 불법주정차를 감시하는 ROS2 노드.

    설계 원칙:
      - 카메라 루프를 ROS2 spin() 과 공존시키기 위해 1ms 타이머 콜백에서 프레임을 처리한다
      - 불법주정차 확정 시 /cctv_done 토픽으로 '<x>,<y>' 문자열을 퍼블리시한다
      - Firebase 업로드는 FirebaseUploader 내부 데몬 스레드가 비동기로 처리한다
      - destroy_node() 에서 카메라 캡처 자원과 OpenCV 창을 정리한다
    """

    def __init__(self):
        super().__init__("webcam_detector_node")

        # ── Homography 로드 ──────────────────────
        self.H, world_pts = load_homography(HOMOGRAPHY_JSON)
        self.H_inv        = np.linalg.inv(self.H)
        self.roi_poly     = make_roi_polygon(self.H_inv, world_pts)
        self.get_logger().info(f"[Homography] 로드 완료")

        # ── YOLO 모델 로드 ───────────────────────
        self.model = YOLO(MODEL_PATH)
        self.get_logger().info(f"[YOLO] 모델 로드: {MODEL_PATH}")

        # ── 카메라 초기화 ────────────────────────
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        if not self.cap.isOpened():
            self.get_logger().error(f"카메라 {CAMERA_INDEX} 열기 실패!")
        else:
            self.get_logger().info(f"[Camera] index={CAMERA_INDEX}  {CAM_WIDTH}x{CAM_HEIGHT}")

        # ── Firebase 초기화 (선택적) ─────────────
        self.uploader = None
        if ENABLE_FIREBASE:
            self.uploader = FirebaseUploader(
                FIREBASE_CRED, FIREBASE_DB_URL, DB_PATH
            )
        else:
            self.get_logger().info("[Firebase] 비활성화 — 터미널 출력 전용 모드")

        # ── 터미널 출력 주기 제어 ────────────────
        self._last_terminal: float = 0.0   # 마지막 터미널 출력 시각 (Unix)

        # ── 불법주정차 감시 ──────────────────────
        self.watcher = ParkingWatcher(ILLEGAL_PARK_SEC)

        # first_seen 시 로컬에 저장한 임시 이미지 경로 보관
        # {tracker_id: {"path": str, "first_seen_at": str}}
        # confirmed 또는 cancelled 시 파일 삭제 후 딕셔너리에서 제거
        self._temp_images: dict = {}

        # ── ROS2 퍼블리셔 ────────────────────────
        # 불법주정차 확정 시 '<center_x>,<center_y>' 문자열을 퍼블리시한다.
        # AMR(ocr_node)이 이 메시지를 받아 cctv_start 모드로 해당 좌표로 출동한다.
        self.cctv_done_pub = self.create_publisher(String, TOPIC_CCTV_DONE, 10)

        # ── 로봇 상태 연동 모드 (ROBOT_STATUS_MODE) ─────────────
        # 각 로봇의 최신 상태 문자열을 보관한다 (구독 콜백에서 갱신).
        self._robot2_status: str = ""
        self._robot3_status: str = ""
        if ROBOT_STATUS_MODE:
            self.create_subscription(
                String, "/robot2/robot2_status", self._robot2_status_cb, 10
            )
            self.create_subscription(
                String, "/robot3/robot3_status", self._robot3_status_cb, 10
            )
            # patrol 로봇별 전용 퍼블리셔 (미리 생성해 둔다)
            self._pub_robot2_cctv = self.create_publisher(String, "/robot2/cctv_done", 10)
            self._pub_robot3_cctv = self.create_publisher(String, "/robot3/cctv_done", 10)
            self.get_logger().info(
                "[RobotStatusMode] ON — /robot2/robot2_status, /robot3/robot3_status 구독 시작"
            )
        else:
            self.get_logger().info("[RobotStatusMode] OFF — 기존 /cctv_done 모드")

        # ── 프레임 처리 타이머 ───────────────────
        # cv2.VideoCapture 루프를 ROS2 spin() 과 공존시키기 위해
        # 1ms 주기 타이머 콜백에서 프레임을 1장씩 처리한다.
        self.create_timer(0.001, self._frame_callback)

        cv2.namedWindow("CCTV", cv2.WINDOW_NORMAL)
        self.get_logger().info("[Start]  노드 준비 완료. [q] 로 종료.")

    # ── 프레임 처리 콜백 ────────────────────────

    def _frame_callback(self):
        """
        타이머 콜백: 카메라에서 프레임 1장을 읽어 전체 파이프라인을 수행한다.

        처리 순서:
          1. 카메라에서 프레임 읽기
          2. YOLO ByteTrack 추론 (persist=True 로 프레임 간 ID 유지)
          3. ROI 밖 차량 필터링 + 실제 좌표 변환
          4. 터미널 출력 / Firebase 업로드
          5. ParkingWatcher 상태 갱신 + 이벤트 처리
             - first_seen 이벤트: 프레임을 로컬에 임시 저장
             - confirmed 이벤트: 로컬 임시 이미지 + 현재 프레임을 DB에 한 번에 업로드 후 임시 파일 삭제
             - cancelled 이벤트: 로컬 임시 이미지 파일만 삭제 (DB 업로드 없음)
          6. 시각화 후 화면 표시
        """
        if not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("카메라 프레임 읽기 실패")
            return

        # ── YOLO ByteTrack 추론 ──────────────────
        results = self.model.track(
            frame,
            tracker="bytetrack.yaml",  # ByteTrack: 프레임 간 ID 연속성 보장
            persist=True,              # 이전 프레임 트랙 정보 유지
            conf=0.87,                 # 신뢰도 임계값
            verbose=False,
        )

        boxes    = results[0].boxes
        car_info = []

        if boxes is not None and boxes.id is not None:
            for tid, cls, conf, xyxy in zip(
                boxes.id.int().tolist(),
                boxes.cls.int().tolist(),
                boxes.conf.tolist(),
                boxes.xyxy.tolist(),
            ):
                x1, y1, x2, y2 = [int(v) for v in xyxy]
                cx_px = (x1 + x2) // 2
                cy_px = (y1 + y2) // 2

                if not point_in_roi(self.roi_poly, cx_px, cy_px):   # ROI 밖 차량 무시
                    continue

                corners_m  = bbox_corners_world(self.H, x1, y1, x2, y2)
                cx_m, cy_m = pixel_to_world(self.H, cx_px, cy_px)
                tags       = ["top_left", "top_right", "bottom_right", "bottom_left"]

                car_info.append({
                    "id":           tid,
                    "label":        self.model.names[cls],
                    "conf":         round(conf, 3),
                    "pixel_center": [cx_px, cy_px],      # 시각화 전용 (DB 미업로드)
                    "center_x":     round(cx_m, 4),      # 실제 x 좌표 (m)
                    "center_y":     round(cy_m, 4),      # 실제 y 좌표 (m)
                    "corners_m":    {
                        tag: {"x": round(p[0], 4), "y": round(p[1], 4)}
                        for tag, p in zip(tags, corners_m)
                    },
                    "timestamp":    time.time(),
                })

        # ── 터미널 출력 (10초 간격) ──────────────
        if ENABLE_TERMINAL:
            now = time.time()
            if now - self._last_terminal >= TERMINAL_INTERVAL:
                output_terminal(car_info)
                self._last_terminal = now

        # ── 불법주정차 감시 + 이벤트 처리 ────────
        car_info_by_id = {c["id"]: c for c in car_info}
        current_ids    = set(car_info_by_id.keys())
        events = self.watcher.update(current_ids, car_info_by_id)

        for ev in events:
            event_type = ev["event"]

            # ── cancelled: 확정 없이 소멸 → 로컬 임시 파일만 삭제, DB 업로드 없음 ──
            if event_type == "cancelled":
                tid = ev["tid"]
                temp = self._temp_images.pop(tid, None)
                if temp:
                    delete_frame_local(temp["path"])
                    self.get_logger().info(f"[cancelled] 임시 이미지 삭제  ID:{tid}")
                continue

            car   = ev["car"]
            label = "[최초감지]" if event_type == "first_seen" else "[불법주정차확정]"
            self.get_logger().info(
                f"{label}  ID:{car['id']}  "
                f"({car['center_x']:+.3f}, {car['center_y']:+.3f}) m"
            )

            if event_type == "first_seen":
                # ── 로컬에 first_seen 프레임 임시 저장 (DB 업로드 없음) ──
                temp_path = save_frame_local(car["id"], frame)
                self._temp_images[car["id"]] = {
                    "path":         temp_path,
                    "first_seen_at": datetime.now().isoformat(),
                }
                self.get_logger().info(
                    f"[first_seen] 임시 저장: {temp_path}  ID:{car['id']}"
                )

            if event_type == "confirmed":
                # ── cctv_detections: first_seen + confirmed 프레임을 한 번에 DB 업로드 ──
                # 업로드 완료 후 upload_cctv_detection() 내부에서 로컬 파일이 자동 삭제된다.
                if ENABLE_FIREBASE and self.uploader:
                    temp = self._temp_images.pop(car["id"], None)
                    initial_path   = temp["path"]         if temp else None
                    car["first_seen_at"] = temp["first_seen_at"] if temp else None
                    upload_cctv_detection(
                        self.uploader.rtdb,
                        CCTV_DB_PATH,
                        "confirmed",
                        car,
                        frame,
                        initial_image_path=initial_path,
                    )

                # ── /cctv_done 퍼블리시 (CCTV_DONE_REPEAT 회 반복) ─────────────
                # 메시지 형식: "<center_x>,<center_y>"  (단위: m)
                # 네트워크 불안정 대비: 별도 데몬 스레드에서 반복 발행한다.
                coord_str = f"{car['center_x']},{car['center_y']}"

                if ROBOT_STATUS_MODE:
                    r2_patrol = (self._robot2_status == "patrol")
                    r3_patrol = (self._robot3_status == "patrol")
                    if r2_patrol:
                        pub       = self._pub_robot2_cctv
                        topic_used = "/robot2/cctv_done"
                    elif r3_patrol:
                        pub       = self._pub_robot3_cctv
                        topic_used = "/robot3/cctv_done"
                    else:
                        pub       = self.cctv_done_pub
                        topic_used = TOPIC_CCTV_DONE
                else:
                    pub       = self.cctv_done_pub
                    topic_used = TOPIC_CCTV_DONE

                threading.Thread(
                    target=self._publish_cctv_done_repeated,
                    args=(pub, coord_str),
                    daemon=True,
                ).start()
                self.get_logger().info(
                    f"[cctv_done] 발송 시작 × {CCTV_DONE_REPEAT}: {coord_str}"
                    f"  (ID:{car['id']})  → {topic_used}"
                )

            if ENABLE_FIREBASE and self.uploader:
                # 기존 webcam/logs 증거 로그도 유지 (first_seen + confirmed 모두)
                upload_parking_log(
                    self.uploader.rtdb,
                    LOG_DB_PATH,
                    ev["event"],
                    car,
                    frame,   # annotation 없는 원본 프레임
                )

        # ── 시각화 ───────────────────────────────
        # 시각화 호출 시 threshold_sec 전달
        annotated = results[0].plot() # YOLO 기본 bbox+ID
        draw_overlay(annotated, self.H_inv, car_info, self.roi_poly, ILLEGAL_PARK_SEC) # ROI + 실제 좌표 오버레이
        cv2.imshow("CCTV", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            # 'q' 키 입력 시 ROS2 노드 종료
            self.get_logger().info("[q] 종료 요청")
            raise SystemExit

    # ── cctv_done 반복 발행 ─────────────────────

    def _publish_cctv_done_repeated(self, pub, coord_str: str):
        '''
        cctv_done 토픽을 CCTV_DONE_REPEAT 회 반복 발행한다.
        네트워크 패킷 손실 대비용. 별도 데몬 스레드에서 호출된다.
        pub: 사용할 퍼블리셔 객체 (robot2/robot3/기본 중 선택된 것)
        '''
        for i in range(CCTV_DONE_REPEAT):
            msg      = String()
            msg.data = coord_str
            pub.publish(msg)
            self.get_logger().info(f"[cctv_done] {i+1}/{CCTV_DONE_REPEAT}: {coord_str}")
            if i < CCTV_DONE_REPEAT - 1:
                time.sleep(CCTV_DONE_INTERVAL)

    # ── 로봇 상태 구독 콜백 ─────────────────────

    def _robot2_status_cb(self, msg: String):
        """ /robot2/robot2_status 수신 콜백: 최신 상태 문자열을 저장한다. """
        self._robot2_status = msg.data

    def _robot3_status_cb(self, msg: String):
        """ /robot3/robot3_status 수신 콜백: 최신 상태 문자열을 저장한다. """
        self._robot3_status = msg.data

    # ── 노드 종료 ────────────────────────────────

    def destroy_node(self):
        """
        노드 종료 시 카메라 캡처 자원과 OpenCV 창을 해제한다.
        미확정 상태로 남은 임시 이미지 파일도 모두 정리한다.
        Firebase 업로드 스레드는 daemon=True 이므로 자동 정리된다.
        """
        # 노드 종료 시 미확정 임시 이미지 파일 전체 정리
        for tid, temp in list(self._temp_images.items()):
            delete_frame_local(temp["path"])
        self._temp_images.clear()

        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


# ══════════════════════════════════════════════════
#  [CHAPTER 8: 진입점]
# ══════════════════════════════════════════════════

def main(args=None):
    """ ROS2 노드 실행 진입점. Ctrl+C 또는 'q' 키 입력 시 안전하게 종료한다. """
    rclpy.init(args=args)
    node = WebcamDetectorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
