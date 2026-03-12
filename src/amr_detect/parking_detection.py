#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
[프로젝트: Click Car - AMR 카메라 기반 불법주정차 단속 노드]
- 최종 수정: 2026-03-11

[System Architecture & Role]
1. 수신: OAK-D RGB 카메라로부터 CompressedImage 수신 (ROS2)
2. 탐지: YOLOv8 기반 차량(car) + 번호판(id) 동시 탐지
3. 검증: 번호판이 차량 영역 내부에 있는지 Overlap 검증
4. 추적: IoU 기반 동일 차량 식별 + 30초 타이머 관리
5. 송신 1: car+id 동시 탐지 첫 프레임 → 차량 전체 이미지(base64) 즉시 DB 저장 (증빙용)
6. 송신 2: 30초 초과 시 번호판 크롭 이미지(base64) DB 저장 (단속용)

[Interface 정의]
- Topic (Sub): /{NS}/oakd/rgb/image_raw/compressed [sensor_msgs/CompressedImage]
- Topic (Sub): /{NS}/oakd/stereo/image_raw/compressedDepth [sensor_msgs/CompressedImage]  ← 추가
- Topic (Sub): /{NS}/oakd/stereo/camera_info [sensor_msgs/CameraInfo]                     ← 추가
- Database (Out): Firebase Realtime Database
  * 경로: detections/{timestamp}
  * 타입 car  : { type, detected_at, car_conf, id_conf, car_bbox, id_bbox, image_base64 }
  * 타입 plate: { type, detected_at, car_conf, id_conf, car_bbox, id_bbox, image_base64 }

[업로드 로직]
- car+id 동시 탐지 첫 프레임 → type="car"   로 프레임 전체 이미지 즉시 업로드
- 30초 초과 시              → type="plate" 로 번호판 크롭 이미지 업로드
- 동일 차량 판별: car bbox IoU >= CAR_IOU_THRESH
- 프레임에서 사라지면 즉시 타이머 리셋 (TrackedVehicle 삭제)
- car 이미지는 TrackedVehicle 생성 시 1회만 업로드 (car_uploaded 플래그로 중복 방지)
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
from sensor_msgs.msg import CompressedImage, CameraInfo  # ← CameraInfo 추가
from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials, db


# ──────────────────────────────────────────────
# [CHAPTER 1: 하이퍼파라미터]
# ──────────────────────────────────────────────
ROBOT_NAMESPACE          = "/robot3"
MODEL_PATH               = "/home/rokey/click_car/models/amr.pt"
FIREBASE_CRED_PATH       = "/home/rokey/click_car/web/click_car.json"
FIREBASE_DB_URL          = "https://iligalstop-default-rtdb.asia-southeast1.firebasedatabase.app"
FIREBASE_DB_PATH         = "detections"

CONF_THRESHOLD           = 0.50
ID_IN_CAR_OVERLAP_THRESH = 0.50
CAR_IOU_THRESH           = 0.30
YOLO_IMG_SIZE            = 704
PARKING_TIMEOUT_SEC      = 30.0
SAVE_QUEUE_MAXSIZE       = 10

# ── 토픽 경로 ──────────────────────────────────
TOPIC_RGB   = f"{ROBOT_NAMESPACE}/oakd/rgb/image_raw/compressed"
TOPIC_DEPTH = f"{ROBOT_NAMESPACE}/oakd/stereo/image_raw/compressedDepth"
TOPIC_INFO  = f"{ROBOT_NAMESPACE}/oakd/stereo/camera_info"


# ──────────────────────────────────────────────
# [CHAPTER 2: 차량 트래킹 상태 컨테이너]
# ──────────────────────────────────────────────

class TrackedVehicle:
    '''
    단일 차량의 타이머 + 최신 탐지 정보를 보관하는 상태 객체.

    Attributes:
        first_seen    : car+id 동시 탐지된 최초 시각 (time.monotonic 기준)
        last_seen     : 마지막으로 프레임에서 확인된 시각
        car_det       : 차량 탐지 dict (bbox, conf 포함)
        id_det        : 번호판 탐지 dict
        car_uploaded  : 차량 전체 이미지 업로드 완료 여부 (첫 프레임 1회만)
        plate_uploaded: 번호판 크롭 이미지 업로드 완료 여부 (30초 후 1회만)
    '''
    def __init__(self, car_det: dict, id_det: dict):
        now                  = time.monotonic()
        self.first_seen      = now
        self.last_seen       = now
        self.car_det         = car_det
        self.id_det          = id_det
        self.car_uploaded    = False   # 증빙용 차량 이미지 업로드 여부
        self.plate_uploaded  = False   # 단속용 번호판 이미지 업로드 여부

    def elapsed(self) -> float:
        return time.monotonic() - self.first_seen

    def update(self, car_det: dict, id_det: dict):
        self.last_seen = time.monotonic()
        self.car_det   = car_det
        self.id_det    = id_det


# ──────────────────────────────────────────────
# [CHAPTER 3: 메인 노드]
# ──────────────────────────────────────────────

class ParkingDetectionNode(Node):
    '''
    AMR 탑재 OAK-D RGB 카메라로 불법주정차 차량을 탐지하고
    - 첫 탐지 시  : 차량 전체 이미지를 Firebase에 즉시 업로드 (증빙)
    - 30초 초과 시: 번호판 크롭 이미지를 Firebase에 업로드 (단속)

    스레딩 구조:
        Main Thread  : ROS2 spin → image_callback → YOLO 추론 → 타이머 갱신 → 큐 투입
        Worker Thread: save_queue 소비 → Firebase 업로드 (네트워크 I/O 분리)
    '''

    def __init__(self):
        super().__init__("parking_detection_node")

        self.tracked_vehicles = []
        self.save_queue       = queue.Queue(maxsize=SAVE_QUEUE_MAXSIZE)
        self.db_ref           = None

        # ── [추가] Depth / CameraInfo 수신 버퍼 ──────
        self.latest_depth_frame = None  # depth_callback에서 매 프레임 갱신 (uint16, mm)
        self.camera_info        = None  # info_callback에서 1회 저장
        self._depth_lock        = threading.Lock()  # depth 버퍼 스레드 안전 접근용

        self._load_model()
        self._init_firebase()
        self._init_subscriber()

        threading.Thread(target=self._upload_worker, daemon=True).start()
        cv2.namedWindow("Parking Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Parking Detection", YOLO_IMG_SIZE, YOLO_IMG_SIZE)
        self.get_logger().info("Node ready.")

    # ── 초기화 ──────────────────────────────────

    def _load_model(self):
        ''' YOLO 로드 + Cold Start 방지 워밍업 '''
        self.model = YOLO(MODEL_PATH)
        self.model.predict(
            source=np.zeros((YOLO_IMG_SIZE, YOLO_IMG_SIZE, 3), dtype=np.uint8),
            imgsz=YOLO_IMG_SIZE, verbose=False
        )
        self.get_logger().info(f"[DEBUG-1] Model classes: {self.model.names}")
        self.get_logger().info("YOLO warm-up complete.")

    def _init_firebase(self):
        ''' Firebase Admin SDK 세션 생성. 실패 시 db_ref=None 유지 (업로드 스킵). '''
        try:
            firebase_admin.initialize_app(
                credentials.Certificate(FIREBASE_CRED_PATH),
                {"databaseURL": FIREBASE_DB_URL}
            )
            self.db_ref = db.reference(FIREBASE_DB_PATH)
            self.get_logger().info("Firebase connected.")
        except Exception as e:
            self.get_logger().error(f"[DEBUG-2] Firebase init failed: {e}")

    def _init_subscriber(self):
        '''
        카메라 토픽 3종 구독.
        - RGB   (BEST_EFFORT): image_callback → YOLO 추론 메인 파이프라인
        - Depth (BEST_EFFORT): depth_callback → latest_depth_frame 버퍼 갱신
        - Info  (RELIABLE)   : info_callback  → camera_info 1회 저장
        '''
        qos_be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # RGB 구독 (기존 동일)
        self.create_subscription(CompressedImage, TOPIC_RGB, self.image_callback, qos_be)
        self.get_logger().info(f"Subscribing RGB  : {TOPIC_RGB}")

        # ── [추가] Depth 구독 ──────────────────────────
        self.create_subscription(CompressedImage, TOPIC_DEPTH, self.depth_callback, qos_be)
        self.get_logger().info(f"Subscribing Depth: {TOPIC_DEPTH}")

        # ── [추가] CameraInfo 구독 (RELIABLE: latched topic 대응) ──
        qos_rel = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.create_subscription(CameraInfo, TOPIC_INFO, self.info_callback, qos_rel)
        self.get_logger().info(f"Subscribing Info : {TOPIC_INFO}")

    # ── [추가] Depth 콜백 ────────────────────────────

    def depth_callback(self, msg: CompressedImage):
        '''
        OAK-D compressedDepth 수신 → latest_depth_frame 버퍼 갱신.

        포맷: PNG 헤더(12 bytes 커스텀) + PNG 바디 → uint16 depth map (단위: mm)
        PNG 시그니처(\x89PNG)를 탐색해 오프셋을 자동 결정한 뒤 디코딩.
        image_callback에서 _depth_lock으로 안전하게 읽어 사용 가능.
        '''
        try:
            raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)

            # PNG 시그니처 위치 탐색 (최대 16바이트 내)
            png_start = 0
            for i in range(min(16, len(raw) - 4)):
                if (raw[i]   == 0x89 and raw[i+1] == 0x50 and
                        raw[i+2] == 0x4E and raw[i+3] == 0x47):
                    png_start = i
                    break

            depth_img = cv2.imdecode(raw[png_start:], cv2.IMREAD_UNCHANGED)

            if depth_img is None:
                self.get_logger().warn("[DEBUG-D1] Depth decode failed.")
                return

            with self._depth_lock:
                self.latest_depth_frame = depth_img  # uint16, mm 단위

        except Exception as e:
            self.get_logger().warn(f"[DEBUG-D2] depth_callback error: {e}")

    # ── [추가] CameraInfo 콜백 ──────────────────────

    def info_callback(self, msg: CameraInfo):
        '''
        카메라 내부 파라미터(fx, fy, cx, cy)를 1회만 저장.
        이후 메시지는 무시 (재저장 불필요).

        msg.k 레이아웃 (3×3 row-major):
            [fx,  0, cx,
              0, fy, cy,
              0,  0,  1]
        '''
        if self.camera_info is not None:
            return  # 이미 저장됨 → 무시

        self.camera_info = {
            "fx":     msg.k[0],
            "fy":     msg.k[4],
            "cx":     msg.k[2],
            "cy":     msg.k[5],
            "width":  msg.width,
            "height": msg.height,
        }
        self.get_logger().info(
            f"[INFO] CameraInfo saved — "
            f"fx={self.camera_info['fx']:.2f}, fy={self.camera_info['fy']:.2f}, "
            f"cx={self.camera_info['cx']:.2f}, cy={self.camera_info['cy']:.2f}, "
            f"res={self.camera_info['width']}×{self.camera_info['height']}"
        )

    # ── 메인 파이프라인 ──────────────────────────

    def image_callback(self, msg: CompressedImage):
        '''
        Bytes → NumPy 디코딩 → YOLO 추론 → Overlap 검증 → 트래킹/타이머 → 큐 투입
        '''
        frame = cv2.imdecode(np.array(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            self.get_logger().warn("[DEBUG-3] Frame decode failed. Check topic format.")
            return

        cars, ids = self._detect(frame)

        self.get_logger().info(
            f"[DEBUG-4] Detect result — cars: {len(cars)}, plates: {len(ids)}"
        )

        validated_pairs = []
        for id_det in ids:
            car = self._find_parent_car(id_det, cars)

            self.get_logger().info(
                f"[DEBUG-5] Plate conf={id_det['conf']:.2f} "
                f"bbox=({id_det['x1']},{id_det['y1']},{id_det['x2']},{id_det['y2']}) "
                f"→ parent_car={'FOUND' if car else 'NONE'}"
            )

            if car is None:
                if cars:
                    id_area = max(1, id_det["area"])
                    for i, c in enumerate(cars):
                        ix = max(0, min(id_det["x2"], c["x2"]) - max(id_det["x1"], c["x1"]))
                        iy = max(0, min(id_det["y2"], c["y2"]) - max(id_det["y1"], c["y1"]))
                        ov = (ix * iy) / id_area
                        self.get_logger().info(
                            f"[DEBUG-6]   car[{i}] overlap={ov:.3f} / threshold={ID_IN_CAR_OVERLAP_THRESH}"
                        )
                continue

            validated_pairs.append((car, id_det))

        self._update_tracking(frame, validated_pairs)
        self._draw(frame)

    # ── YOLO 탐지 ───────────────────────────────

    def _detect(self, frame: np.ndarray) -> tuple[list, list]:
        '''
        YOLOv8 추론.
        반환: (cars, ids) — 각각 탐지 dict 리스트
        '''
        results = self.model.predict(source=frame, imgsz=YOLO_IMG_SIZE,
                                     conf=CONF_THRESHOLD, verbose=False)
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
                "conf":       float(box.conf[0].item()),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "area":       max(0, x2 - x1) * max(0, y2 - y1),
            }
            (cars if name == "car" else ids).append(det)

        return cars, ids

    # ── Overlap 검증 ────────────────────────────

    def _find_parent_car(self, id_det: dict, cars: list) -> dict | None:
        '''
        번호판(id_det)이 어느 차량(cars) 내부에 속하는지 Overlap 비율로 판별.
        알고리즘: intersection(id, car) / area(id) >= ID_IN_CAR_OVERLAP_THRESH
        '''
        id_area = max(1, id_det["area"])

        def overlap(car):
            ix = max(0, min(id_det["x2"], car["x2"]) - max(id_det["x1"], car["x1"]))
            iy = max(0, min(id_det["y2"], car["y2"]) - max(id_det["y1"], car["y1"]))
            return (ix * iy) / id_area

        candidates = [(overlap(car), car) for car in cars if overlap(car) >= ID_IN_CAR_OVERLAP_THRESH]
        return max(candidates, key=lambda x: x[0])[1] if candidates else None

    # ── IoU (동일 차량 판별) ─────────────────────

    @staticmethod
    def _iou(a: dict, b: dict) -> float:
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
        1. 기존 TrackedVehicle ↔ 현재 validated_pairs IoU 매칭
           - 매칭 성공 → bbox·conf 갱신
           - 매칭 실패 → 새 TrackedVehicle 생성 + 차량 전체 이미지 즉시 큐 투입
        2. 사라진 트래킹 즉시 삭제 (타이머 리셋)
        3. 30초 초과 → 번호판 크롭 큐 투입 후 트래킹 종료
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
                matched_track_indices.add(t_idx)
                matched_pair_indices.add(best_p_idx)

        # Step 2: 사라진 트래킹 제거 (타이머 리셋)
        visible_tracks = [
            track for i, track in enumerate(self.tracked_vehicles)
            if i in matched_track_indices
        ]

        # Step 3: 신규 차량 → TrackedVehicle 생성 + 차량 전체 이미지 즉시 업로드
        for p_idx, (car_det, id_det) in enumerate(validated_pairs):
            if p_idx not in matched_pair_indices:
                new_track = TrackedVehicle(car_det, id_det)
                visible_tracks.append(new_track)
                self.get_logger().info("New vehicle tracking started.")

                # ── 증빙용 차량 전체 이미지 즉시 큐 투입 ──
                self._enqueue_car(frame, new_track)
                new_track.car_uploaded = True

        # Step 4: 30초 초과 체크 → 번호판 크롭 큐 투입
        next_tracks = []
        for track in visible_tracks:
            if track.plate_uploaded:
                continue

            elapsed = track.elapsed()
            self.get_logger().info(
                f"Tracking: elapsed={elapsed:.1f}s / {PARKING_TIMEOUT_SEC}s"
            )

            if elapsed >= PARKING_TIMEOUT_SEC:
                self._enqueue_plate(frame, track)
                track.plate_uploaded = True
                # 번호판 업로드 완료 → 트래킹 종료
            else:
                next_tracks.append(track)

        self.tracked_vehicles = next_tracks

    # ── Firebase 큐 투입 (차량 전체 이미지) ──────────

    def _enqueue_car(self, frame: np.ndarray, track: TrackedVehicle):
        '''
        증빙용: 차량이 처음 탐지된 프레임 전체를 큐에 투입.
        type="car" 로 저장되어 차주 증빙 자료로 활용.
        '''
        try:
            self.save_queue.put_nowait({
                "type":    "car",
                "image":   frame.copy(),   # 프레임 전체
                "car_det": track.car_det,
                "id_det":  track.id_det,
            })
            self.get_logger().info(
                f"[CAR] Enqueued full frame. queue={self.save_queue.qsize()}/{SAVE_QUEUE_MAXSIZE}"
            )
        except queue.Full:
            self.get_logger().warn("Queue full. Car image dropped.")

    # ── Firebase 큐 투입 (번호판 크롭) ──────────────

    def _enqueue_plate(self, frame: np.ndarray, track: TrackedVehicle):
        '''
        단속용: 30초 초과 시 번호판 bbox 크롭을 큐에 투입.
        type="plate" 로 저장.
        '''
        id_det = track.id_det
        x1, y1, x2, y2 = id_det["x1"], id_det["y1"], id_det["x2"], id_det["y2"]
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            self.get_logger().warn(
                f"Empty crop. id_bbox=({x1},{y1},{x2},{y2}), frame={frame.shape}"
            )
            return

        try:
            self.save_queue.put_nowait({
                "type":    "plate",
                "image":   crop.copy(),    # 번호판 크롭
                "car_det": track.car_det,
                "id_det":  track.id_det,
            })
            self.get_logger().info(
                f"[PLATE] Enqueued crop. elapsed={track.elapsed():.1f}s "
                f"queue={self.save_queue.qsize()}/{SAVE_QUEUE_MAXSIZE}"
            )
        except queue.Full:
            self.get_logger().warn("Queue full. Plate image dropped.")

    # ── 시각화 ──────────────────────────────────

    def _draw(self, frame: np.ndarray):
        for track in self.tracked_vehicles:
            c       = track.car_det
            elapsed = track.elapsed()
            ratio   = min(elapsed / PARKING_TIMEOUT_SEC, 1.0)

            # 차량 bbox (초록)
            cv2.rectangle(frame, (c["x1"], c["y1"]), (c["x2"], c["y2"]), (0, 255, 0), 2)
            cv2.putText(frame, f"car {c['conf']:.2f}",
                        (c["x1"], max(25, c["y1"] - 30)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 번호판 bbox (빨강)
            d = track.id_det
            cv2.rectangle(frame, (d["x1"], d["y1"]), (d["x2"], d["y2"]), (0, 0, 255), 2)
            cv2.putText(frame, f"id {d['conf']:.2f}",
                        (d["x1"], max(25, d["y1"] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # 타이머 진행 바
            bar_x1    = c["x1"]
            bar_y     = max(0, c["y1"] - 18)
            bar_w     = c["x2"] - c["x1"]
            bar_fill  = int(bar_w * ratio)
            bar_color = (0, 255, 0) if ratio < 0.5 else (0, 165, 255) if ratio < 0.85 else (0, 0, 255)
            cv2.rectangle(frame, (bar_x1, bar_y), (bar_x1 + bar_w, bar_y + 10), (80, 80, 80), -1)
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
        save_queue를 소비하며 Firebase에 직렬화 전송.
        None 수신 시 종료 (Sentinel 패턴).
        item["type"] 으로 car / plate 구분하여 저장.
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
        공통 업로드 함수.
        - type="car"  : 프레임 전체 이미지 → 증빙용
        - type="plate": 번호판 크롭 이미지 → 단속용
        경로: detections/{timestamp}
        '''
        if self.db_ref is None:
            self.get_logger().error(
                "[DEBUG-7] db_ref is None — Firebase 미연결. "
                "시작 로그의 [DEBUG-2] 확인 필요."
            )
            return

        upload_type = item["type"]
        image       = item["image"]
        car_det     = item["car_det"]
        id_det      = item["id_det"]

        _, enc = cv2.imencode(".jpg", image)
        b64    = base64.b64encode(enc.tobytes()).decode("utf-8")

        now = datetime.datetime.now()
        self.db_ref.child(now.strftime("%Y%m%d_%H%M%S_%f")).set({
            "type":           upload_type,           # "car" or "plate"
            "detected_at":    now.isoformat(),
            "car_confidence": round(car_det["conf"], 4),
            "id_confidence":  round(id_det["conf"], 4),
            "car_bbox": {"x1": car_det["x1"], "y1": car_det["y1"],
                         "x2": car_det["x2"], "y2": car_det["y2"]},
            "id_bbox":  {"x1": id_det["x1"], "y1": id_det["y1"],
                         "x2": id_det["x2"], "y2": id_det["y2"]},
            "image_base64": b64,
        })
        self.get_logger().info(
            f"Uploaded [{upload_type}]: {now.strftime('%Y%m%d_%H%M%S_%f')}"
        )

    # ── 종료 ────────────────────────────────────

    def destroy_node(self):
        ''' Sentinel(None) 전송으로 워커 스레드 안전 종료 후 GUI 해제. '''
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