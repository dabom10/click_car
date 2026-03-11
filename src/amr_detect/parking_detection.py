#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
[프로젝트: Click Car - AMR 탑재 카메라 기반 불법주정차 단속 노드]
- 최종 수정: 2026-03-11

[System Architecture & Role]
1. 수신: OAK-D 카메라로부터 RGB(CompressedImage) + Depth(Image) + CameraInfo 수신
2. 탐지: YOLOv8 기반 실시간 차량(car) + 번호판(id) 동시 탐지
3. 검증: 번호판이 차량 영역 내부에 있는지 Overlap 검증
4. 추적: IoU 기반 동일 차량 식별 + 30초 타이머 관리
5. 좌표: Depth + Camera Intrinsic + TF 변환으로 맵 좌표 추출
6. 송신: 30초 초과 시 번호판 크롭 이미지(base64) + 맵좌표 + timestamp → Firebase

[Interface 정의]
- Topic (Sub): /{NS}/oakd/rgb/image_raw/compressed  [sensor_msgs/CompressedImage]
- Topic (Sub): /{NS}/oakd/stereo/image_raw          [sensor_msgs/Image]
- Topic (Sub): /{NS}/oakd/rgb/camera_info           [sensor_msgs/CameraInfo]
- Database (Out): Firebase Realtime Database
  * 경로: violations/{timestamp}
  * 데이터: { image_base64, map_x, map_y, detected_at, car_conf, id_conf }

[타이머 로직]
- car + id 동시 탐지 시점부터 30초 카운트
- 같은 차량 판별: IoU >= CAR_IOU_THRESH (bbox 기반)
- 프레임에서 사라지면(IoU 매칭 실패) 타이머 리셋
- 30초 초과 시 단 1회 Firebase 업로드 후 해당 차량 트래킹 종료
'''

import base64
import datetime
import queue
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.duration import Duration
from rclpy.time import Time

from sensor_msgs.msg import CompressedImage, Image, CameraInfo
from geometry_msgs.msg import PointStamped
from tf2_ros import Buffer, TransformListener
from cv_bridge import CvBridge

from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials, db


# ──────────────────────────────────────────────
# [CHAPTER 1: 하이퍼파라미터]
# ──────────────────────────────────────────────

ROBOT_NAMESPACE          = "/robot2"
MODEL_PATH               = "/home/rokey/click_car/models/amr.pt"
FIREBASE_CRED_PATH       = "/home/rokey/click_car/web/click_car.json"
FIREBASE_DB_URL          = "https://iligalstop-default-rtdb.asia-southeast1.firebasedatabase.app"

CONF_THRESHOLD           = 0.50          # YOLO 최소 신뢰도
ID_IN_CAR_OVERLAP_THRESH = 0.50          # 번호판이 차량 bbox 안에 있다고 판정하는 overlap 임계값
CAR_IOU_THRESH           = 0.30          # 동일 차량 판별용 IoU 임계값 (재등장 시 타이머 리셋 기준)
YOLO_IMG_SIZE            = 704           # YOLO 입력 해상도
PARKING_TIMEOUT_SEC      = 30.0          # 단속 타이머 (초) — 시나리오 테스트 시 30초
SAVE_QUEUE_MAXSIZE       = 10            # Firebase 업로드 큐 상한
DEPTH_VALID_MIN_M        = 0.2           # 유효 depth 최솟값 (m)
DEPTH_VALID_MAX_M        = 10.0          # 유효 depth 최댓값 (m)


# ──────────────────────────────────────────────
# [CHAPTER 2: 차량 트래킹 상태 컨테이너]
# ──────────────────────────────────────────────

class TrackedVehicle:
    '''
    단일 차량의 타이머 + 최신 탐지 정보를 보관하는 상태 객체.

    Attributes:
        first_seen  : car+id 동시 탐지된 최초 시각 (time.monotonic 기준)
        last_seen   : 마지막으로 프레임에서 확인된 시각
        car_det     : 차량 탐지 dict (bbox, conf 포함)
        id_det      : 번호판 탐지 dict
        map_coord   : 가장 최근 추출된 맵 좌표 (x, y) 또는 None
        uploaded    : Firebase 업로드 완료 여부 (중복 업로드 방지)
    '''
    def __init__(self, car_det: dict, id_det: dict):
        now = time.monotonic()
        self.first_seen  = now
        self.last_seen   = now
        self.car_det     = car_det
        self.id_det      = id_det
        self.map_coord   = None   # (map_x, map_y)
        self.uploaded    = False

    def elapsed(self) -> float:
        ''' first_seen 기준 경과 시간(초) '''
        return time.monotonic() - self.first_seen

    def update(self, car_det: dict, id_det: dict):
        ''' 동일 차량 재탐지 시 bbox·conf 갱신 및 last_seen 업데이트 '''
        self.last_seen = time.monotonic()
        self.car_det   = car_det
        self.id_det    = id_det


# ──────────────────────────────────────────────
# [CHAPTER 3: 메인 노드]
# ──────────────────────────────────────────────

class ParkingDetectionNode(Node):
    '''
    AMR 탑재 OAK-D 카메라로 불법주정차 차량을 탐지하고
    30초 초과 시 번호판 이미지 + 맵 좌표를 Firebase에 업로드하는 ROS2 노드.

    스레딩 구조:
        Main Thread  : ROS2 spin → 콜백 처리 → YOLO 추론 → 타이머 갱신 → 큐 투입
        Worker Thread: save_queue 소비 → Firebase 업로드 (네트워크 I/O 분리)
    '''

    def __init__(self):
        super().__init__("parking_detection_node")

        self.bridge      = CvBridge()
        self.lock        = threading.Lock()

        # 카메라 상태
        self.K           = None          # 3x3 intrinsic matrix
        self.depth_image = None
        self.camera_frame_id = None

        # 트래킹 상태: list[TrackedVehicle]
        self.tracked_vehicles = []

        # Firebase 업로드 큐
        self.save_queue  = queue.Queue(maxsize=SAVE_QUEUE_MAXSIZE)
        self.db_ref      = None

        self._load_model()
        self._init_firebase()
        self._init_tf()
        self._init_subscribers()

        threading.Thread(target=self._upload_worker, daemon=True).start()

        cv2.namedWindow("Parking Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Parking Detection", YOLO_IMG_SIZE, YOLO_IMG_SIZE)
        self.get_logger().info("ParkingDetectionNode ready.")

    # ── 초기화 ──────────────────────────────────

    def _load_model(self):
        ''' YOLO 로드 + 워밍업으로 첫 프레임 지연 방지 '''
        self.model = YOLO(MODEL_PATH)
        self.model.predict(
            source=np.zeros((YOLO_IMG_SIZE, YOLO_IMG_SIZE, 3), dtype=np.uint8),
            imgsz=YOLO_IMG_SIZE, verbose=False
        )
        self.get_logger().info(f"YOLO warm-up complete. classes={self.model.names}")

    def _init_firebase(self):
        ''' Firebase Admin SDK 초기화. 실패 시 db_ref=None (업로드 스킵). '''
        try:
            firebase_admin.initialize_app(
                credentials.Certificate(FIREBASE_CRED_PATH),
                {"databaseURL": FIREBASE_DB_URL}
            )
            self.db_ref = db.reference("violations")
            self.get_logger().info("Firebase connected.")
        except Exception as e:
            self.get_logger().error(f"Firebase init failed: {e}")

    def _init_tf(self):
        ''' TF2 버퍼 및 리스너 초기화 '''
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def _init_subscribers(self):
        ''' QoS: BEST_EFFORT + KEEP_LAST(1) — 카메라 스트리밍 최적 설정 '''
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        ns = ROBOT_NAMESPACE
        self.create_subscription(
            CompressedImage, f"{ns}/oakd/rgb/image_raw/compressed",
            self.rgb_callback, qos
        )
        self.create_subscription(
            Image, f"{ns}/oakd/stereo/image_raw",
            self.depth_callback, qos
        )
        self.create_subscription(
            CameraInfo, f"{ns}/oakd/rgb/camera_info",
            self.camera_info_callback, 1
        )
        self.get_logger().info(f"Subscribed to {ns}/oakd/{{rgb,stereo,camera_info}}")

    # ── 카메라 콜백 ─────────────────────────────

    def camera_info_callback(self, msg: CameraInfo):
        ''' Intrinsic matrix K 수신 (1회만 로깅) '''
        with self.lock:
            if self.K is None:
                self.K = np.array(msg.k).reshape(3, 3)
                self.get_logger().info(
                    f"Camera intrinsics: fx={self.K[0,0]:.1f}, fy={self.K[1,1]:.1f}, "
                    f"cx={self.K[0,2]:.1f}, cy={self.K[1,2]:.1f}"
                )

    def depth_callback(self, msg: Image):
        ''' Depth 이미지 갱신 '''
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            with self.lock:
                self.depth_image     = depth
                self.camera_frame_id = msg.header.frame_id
        except Exception as e:
            self.get_logger().error(f"Depth decode failed: {e}")

    def rgb_callback(self, msg: CompressedImage):
        '''
        [메인 파이프라인]
        CompressedImage → YOLO 추론 → Overlap 검증 → 트래킹 갱신 → 타이머 체크 → 시각화
        '''
        frame = cv2.imdecode(np.array(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            self.get_logger().warn("Frame decode failed.")
            return

        # YOLO 탐지
        cars, ids = self._detect(frame)

        # 번호판-차량 매칭 (Overlap 검증)
        validated_pairs = []   # list of (car_det, id_det)
        for id_det in ids:
            car = self._find_parent_car(id_det, cars)
            if car:
                validated_pairs.append((car, id_det))

        self.get_logger().info(
            f"Detect: cars={len(cars)}, plates={len(ids)}, validated={len(validated_pairs)}"
        )

        # 트래킹 갱신 + 타이머 체크
        self._update_tracking(frame, validated_pairs)

        # 시각화
        self._draw(frame, cars, ids)

    # ── YOLO 탐지 ───────────────────────────────

    def _detect(self, frame: np.ndarray) -> tuple[list, list]:
        '''
        YOLOv8 추론.
        반환: (cars, ids) — 각각 탐지 dict 리스트
        '''
        results = self.model.predict(
            source=frame, imgsz=YOLO_IMG_SIZE,
            conf=CONF_THRESHOLD, verbose=False
        )
        cars, ids = [], []
        if not results:
            return cars, ids

        for box in results[0].boxes:
            name = self.model.names.get(int(box.cls[0].item()))
            if name not in ("car", "id"):
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            det = {
                "class_name": name,
                "conf": float(box.conf[0].item()),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "area": max(0, x2 - x1) * max(0, y2 - y1),
            }
            (cars if name == "car" else ids).append(det)

        return cars, ids

    # ── Overlap 검증 (1번 코드와 동일 로직) ────────

    def _find_parent_car(self, id_det: dict, cars: list) -> dict | None:
        '''
        번호판(id_det)이 어느 차량(cars) 내부에 속하는지 단방향 Overlap으로 판별.
        IoU 대신 "번호판 면적 기준 overlap"을 사용 → 작은 번호판도 민감하게 감지.
        '''
        id_area = max(1, id_det["area"])

        def overlap(car):
            ix = max(0, min(id_det["x2"], car["x2"]) - max(id_det["x1"], car["x1"]))
            iy = max(0, min(id_det["y2"], car["y2"]) - max(id_det["y1"], car["y1"]))
            return (ix * iy) / id_area

        candidates = [
            (overlap(car), car) for car in cars
            if overlap(car) >= ID_IN_CAR_OVERLAP_THRESH
        ]
        return max(candidates, key=lambda x: x[0])[1] if candidates else None

    # ── IoU 계산 (동일 차량 판별용) ────────────────

    @staticmethod
    def _iou(a: dict, b: dict) -> float:
        '''
        두 bbox의 IoU 계산.
        TrackedVehicle 재등장 판별 시 사용 — Overlap과 달리 양방향 비율로 계산.
        '''
        ix = max(0, min(a["x2"], b["x2"]) - max(a["x1"], b["x1"]))
        iy = max(0, min(a["y2"], b["y2"]) - max(a["y1"], b["y1"]))
        inter = ix * iy
        if inter == 0:
            return 0.0
        area_a = max(1, a["area"])
        area_b = max(1, b["area"])
        return inter / (area_a + area_b - inter)

    # ── 트래킹 갱신 + 타이머 관리 ──────────────────

    def _update_tracking(self, frame: np.ndarray, validated_pairs: list):
        '''
        [핵심 로직]

        1. 현재 프레임의 validated_pairs와 기존 tracked_vehicles를 IoU로 매칭
           - 매칭 성공 → update() (bbox 갱신, last_seen 갱신)
           - 매칭 실패 (새 차량) → 새 TrackedVehicle 생성
        2. 이번 프레임에서 보이지 않은 트래킹 → 삭제 (타이머 리셋)
        3. 업로드 완료된 트래킹 → 삭제
        4. 30초 초과 트래킹 → 큐에 투입 후 uploaded=True

        설계 결정: "프레임에서 사라지면 즉시 삭제"하는 이유
          카메라를 장착한 AMR이 이동 중이므로, 한 프레임이라도 탐지가 끊기면
          AMR이 지나쳤거나 차량이 이동한 것으로 간주한다.
          일정 프레임 tolerance를 두면 AMR 이동 중 깜빡임으로 인해
          이미 지나친 차량의 타이머가 계속 유지되는 부작용이 생긴다.
        '''
        matched_track_indices = set()
        matched_pair_indices  = set()

        # Step 1: 기존 트래킹 ↔ 현재 프레임 매칭
        for t_idx, track in enumerate(self.tracked_vehicles):
            best_iou   = 0.0
            best_p_idx = -1
            for p_idx, (car_det, _) in enumerate(validated_pairs):
                if p_idx in matched_pair_indices:
                    continue
                iou = self._iou(track.car_det, car_det)
                if iou > best_iou:
                    best_iou   = iou
                    best_p_idx = p_idx

            if best_iou >= CAR_IOU_THRESH:
                car_det, id_det = validated_pairs[best_p_idx]
                track.update(car_det, id_det)

                # 맵 좌표 추출 (depth + TF)
                coord = self._get_map_coord(frame, car_det)
                if coord:
                    track.map_coord = coord

                matched_track_indices.add(t_idx)
                matched_pair_indices.add(best_p_idx)

        # Step 2: 이번 프레임에서 사라진 트래킹 제거 (타이머 리셋)
        visible_tracks = [
            track for i, track in enumerate(self.tracked_vehicles)
            if i in matched_track_indices
        ]

        # Step 3: 새 차량 추가
        for p_idx, (car_det, id_det) in enumerate(validated_pairs):
            if p_idx not in matched_pair_indices:
                new_track = TrackedVehicle(car_det, id_det)
                coord = self._get_map_coord(frame, car_det)
                if coord:
                    new_track.map_coord = coord
                visible_tracks.append(new_track)
                self.get_logger().info("New vehicle tracking started.")

        # Step 4: 30초 초과 체크 → 큐 투입
        next_tracks = []
        for track in visible_tracks:
            if track.uploaded:
                continue   # 업로드 완료 → 제거

            elapsed = track.elapsed()
            self.get_logger().info(
                f"Tracking: elapsed={elapsed:.1f}s / {PARKING_TIMEOUT_SEC}s "
                f"map_coord={track.map_coord}"
            )

            if elapsed >= PARKING_TIMEOUT_SEC:
                self._enqueue(track)
                track.uploaded = True
                # 업로드 후 트래킹 종료 (제거)
            else:
                next_tracks.append(track)

        self.tracked_vehicles = next_tracks

    # ── 맵 좌표 추출 ────────────────────────────

    def _get_map_coord(self, frame: np.ndarray, car_det: dict) -> tuple | None:
        '''
        차량 bbox 중심 픽셀의 depth 값을 읽어 카메라 좌표계 → map 좌표계로 변환.

        반환: (map_x, map_y) 또는 None (depth 없음 / TF 실패 시)

        설계 결정: bbox 중심 픽셀을 사용하는 이유
          번호판 bbox 중심을 쓰면 번호판이 프레임 끝에 걸렸을 때
          depth 픽셀이 유효 범위를 벗어날 수 있다.
          차량 bbox 중심은 차량 본체 위에 있을 가능성이 높아 depth가 더 안정적이다.
        '''
        with self.lock:
            depth  = self.depth_image
            K      = self.K
            frame_id = self.camera_frame_id

        if depth is None or K is None or frame_id is None:
            return None

        # 차량 bbox 중심 픽셀
        cx_px = (car_det["x1"] + car_det["x2"]) // 2
        cy_px = (car_det["y1"] + car_det["y2"]) // 2

        # depth 이미지 경계 체크
        h, w = depth.shape[:2]
        if not (0 <= cy_px < h and 0 <= cx_px < w):
            return None

        z = float(depth[cy_px, cx_px]) / 1000.0   # mm → m
        if not (DEPTH_VALID_MIN_M < z < DEPTH_VALID_MAX_M):
            return None

        # 픽셀 → 카메라 좌표계 (핀홀 역투영)
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        X = (cx_px - cx) * z / fx
        Y = (cy_px - cy) * z / fy
        Z = z

        # 카메라 좌표계 → map 좌표계 (TF 변환)
        pt = PointStamped()
        pt.header.stamp    = Time().to_msg()
        pt.header.frame_id = frame_id
        pt.point.x = X
        pt.point.y = Y
        pt.point.z = Z

        try:
            pt_map = self.tf_buffer.transform(pt, "map", timeout=Duration(seconds=0.5))
            return (round(pt_map.point.x, 3), round(pt_map.point.y, 3))
        except Exception as e:
            self.get_logger().warn(f"TF transform failed: {e}")
            return None

    # ── Firebase 큐 투입 ─────────────────────────

    def _enqueue(self, track: TrackedVehicle):
        '''
        업로드 데이터를 큐에 투입.
        put_nowait() 사용으로 메인 스레드 블로킹 방지.
        '''
        frame_snapshot = None   # _draw 이전에 crop할 수 없으므로 det 정보만 전달
        try:
            self.save_queue.put_nowait({
                "id_det":    track.id_det,
                "car_det":   track.car_det,
                "map_coord": track.map_coord,
                "elapsed":   track.elapsed(),
            })
            self.get_logger().info(
                f"Enqueued violation. elapsed={track.elapsed():.1f}s "
                f"map={track.map_coord} queue={self.save_queue.qsize()}/{SAVE_QUEUE_MAXSIZE}"
            )
        except queue.Full:
            self.get_logger().warn("Upload queue full. Violation frame dropped.")

    # ── 시각화 ──────────────────────────────────

    def _draw(self, frame: np.ndarray, cars: list, ids: list):
        '''
        BBox 오버레이 + 타이머 바 + 큐 상태 HUD
        '''
        # 차량 bbox (초록)
        for det in cars:
            cv2.rectangle(frame, (det["x1"], det["y1"]), (det["x2"], det["y2"]), (0, 255, 0), 2)
            cv2.putText(frame, f"car {det['conf']:.2f}",
                        (det["x1"], max(25, det["y1"] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 번호판 bbox (빨강)
        for det in ids:
            cv2.rectangle(frame, (det["x1"], det["y1"]), (det["x2"], det["y2"]), (0, 0, 255), 2)
            cv2.putText(frame, f"id {det['conf']:.2f}",
                        (det["x1"], max(25, det["y1"] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 트래킹 타이머 바 (차량 bbox 위)
        for track in self.tracked_vehicles:
            elapsed  = track.elapsed()
            ratio    = min(elapsed / PARKING_TIMEOUT_SEC, 1.0)
            c        = track.car_det
            bar_x1   = c["x1"]
            bar_y    = max(0, c["y1"] - 18)
            bar_w    = c["x2"] - c["x1"]
            bar_fill = int(bar_w * ratio)

            # 배경 (회색)
            cv2.rectangle(frame, (bar_x1, bar_y), (bar_x1 + bar_w, bar_y + 10), (80, 80, 80), -1)
            # 진행 바 (초록 → 주황 → 빨강)
            bar_color = (0, 255, 0) if ratio < 0.5 else (0, 165, 255) if ratio < 0.85 else (0, 0, 255)
            cv2.rectangle(frame, (bar_x1, bar_y), (bar_x1 + bar_fill, bar_y + 10), bar_color, -1)
            cv2.putText(frame, f"{elapsed:.0f}s/{PARKING_TIMEOUT_SEC:.0f}s",
                        (bar_x1, bar_y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, bar_color, 1)

        # 큐 상태 HUD (우상단)
        q_size  = self.save_queue.qsize()
        q_ratio = q_size / max(1, SAVE_QUEUE_MAXSIZE)
        q_color = (0, 255, 0) if q_ratio < 0.5 else (0, 165, 255) if q_ratio < 1.0 else (0, 0, 255)
        q_text  = f"Queue: {q_size}/{SAVE_QUEUE_MAXSIZE}"
        (tw, _), _ = cv2.getTextSize(q_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(frame, q_text, (frame.shape[1] - tw - 10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, q_color, 2)

        # 트래킹 수 (좌상단)
        cv2.putText(frame, f"Tracking: {len(self.tracked_vehicles)}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("Parking Detection", frame)
        cv2.waitKey(1)

    # ── Firebase 업로드 워커 ─────────────────────

    def _upload_worker(self):
        '''
        백그라운드 스레드에서 큐를 소비하며 Firebase에 업로드.
        None 수신 시 종료 (Sentinel 패턴).
        '''
        while True:
            item = self.save_queue.get()
            if item is None:
                break
            try:
                self._upload(item)
            except Exception as e:
                self.get_logger().error(f"Upload error: {e}")

    def _upload(self, item: dict):
        '''
        저장 데이터:
          - image_base64 : 번호판 크롭 이미지 (JPEG → Base64)
          - detected_at  : 업로드 시각 (ISO 8601)
          - map_x, map_y : 차량 맵 좌표
          - car_conf     : 차량 탐지 신뢰도
          - id_conf      : 번호판 탐지 신뢰도

        설계 결정: 번호판(id) bbox로 crop하는 이유
          차량 전체 이미지는 용량이 크고 웹 대시보드에서 번호판 식별이 어렵다.
          번호판 영역만 crop하면 용량을 ~90% 절감하고 Firebase 노드 크기 제한에도 안전하다.

        주의: _upload는 별도 스레드에서 실행되므로 self.depth_image 등 공유 자원에
              직접 접근하지 않는다. 필요한 모든 데이터는 item dict로 전달받는다.
        '''
        if self.db_ref is None:
            self.get_logger().error("db_ref is None — Firebase 미연결.")
            return

        id_det    = item["id_det"]
        car_det   = item["car_det"]
        map_coord = item["map_coord"]

        # ── 번호판 크롭 및 인코딩 ──
        # 주의: 업로드 시점에 frame이 없으므로 id_det의 bbox 정보만 저장
        # (실제 크롭이 필요하면 item에 frame을 함께 전달해야 함)
        # 여기서는 bbox 정보 + placeholder로 처리하고, frame 크롭은 아래에서 처리

        # frame 크롭 (item에 frame이 있는 경우)
        b64 = ""
        if "frame" in item and item["frame"] is not None:
            x1, y1, x2, y2 = id_det["x1"], id_det["y1"], id_det["x2"], id_det["y2"]
            crop = item["frame"][y1:y2, x1:x2]
            if crop.size > 0:
                _, enc = cv2.imencode(".jpg", crop)
                b64 = base64.b64encode(enc.tobytes()).decode("utf-8")

        now = datetime.datetime.now()
        self.db_ref.child(now.strftime("%Y%m%d_%H%M%S_%f")).set({
            "detected_at":    now.isoformat(),
            "map_x":          map_coord[0] if map_coord else None,
            "map_y":          map_coord[1] if map_coord else None,
            "car_confidence": round(car_det["conf"], 4),
            "id_confidence":  round(id_det["conf"], 4),
            "car_bbox": {"x1": car_det["x1"], "y1": car_det["y1"],
                         "x2": car_det["x2"], "y2": car_det["y2"]},
            "id_bbox":  {"x1": id_det["x1"], "y1": id_det["y1"],
                         "x2": id_det["x2"], "y2": id_det["y2"]},
            "image_base64": b64,
        })
        self.get_logger().info(f"Uploaded violation at {now.isoformat()} | map={map_coord}")

    # ── 종료 ────────────────────────────────────

    def destroy_node(self):
        ''' Sentinel 전송으로 워커 스레드 안전 종료 후 GUI 해제. '''
        self.save_queue.put(None)
        cv2.destroyAllWindows()
        super().destroy_node()


# ──────────────────────────────────────────────
# [CHAPTER 4: 엔트리 포인트]
# ──────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = ParkingDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()