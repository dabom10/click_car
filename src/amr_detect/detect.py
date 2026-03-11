#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
[프로젝트: Click Car - 객체 탐지 및 클라우드 연동 노드]
- 담당: 김다봄 개쩜
- 최종 수정: 2026-03-11

[System Architecture & Role]
1. 수신: OAK-D 카메라로부터 CompressedImage 수신 (ROS2)
2. 처리: YOLOv8 기반 실시간 차량 및 번호판 탐지 (Object Detection)
3. 검증: 번호판이 차량 영역 내부에 있는지 공간 필터링 (Overlap Logic)
4. 송신: 탐지된 크롭 이미지를 Base64 인코딩 후 Firebase 업로드 (Cloud Interface)

[Interface 정의]
- Topic (Sub): /robot2/oakd/rgb/image_raw/compressed [sensor_msgs/msg/CompressedImage]
  * 데이터 형식: JPEG 압축 이미지, 약 30 FPS
- Database (Out): Firebase Realtime Database
  * 데이터 형식: JSON (String: base64_image, Float: confidence, Dict: bbox)
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
from sensor_msgs.msg import CompressedImage
from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials, db

# ──────────────────────────────────────────────
# [CHAPTER 1: 하이퍼파라미터]
# ──────────────────────────────────────────────
ROBOT_NAMESPACE          = "/robot2"
MODEL_PATH               = "/home/rokey/click_car/models/AMR/v1/weights/best.pt"
FIREBASE_CRED_PATH       = "/home/rokey/click_car/web/database.json"
FIREBASE_DB_URL          = "https://click-car-2f586-default-rtdb.asia-southeast1.firebasedatabase.app"
CONF_THRESHOLD           = 0.70   # YOLO 탐지 및 DB 저장 공통 신뢰도 기준
ID_IN_CAR_OVERLAP_THRESH = 0.50   # 번호판이 차량 내부에 있다고 판단할 최소 중첩 비율
YOLO_IMG_SIZE            = 320    # 추론 해상도 (정밀도 ↔ 속도 트레이드오프)
SAVE_QUEUE_MAXSIZE       = 30     # 30fps × 최대 지연 1s 기준 메모리 상한
CAM_FPS_LOG_INTERVAL     = 5.0   # 카메라 FPS 로그 출력 주기 (초)


# ──────────────────────────────────────────────
# [CHAPTER 2: 노드 본체]
# ──────────────────────────────────────────────
class PlateDetectionNode(Node):
    '''
    OAK-D RGB 스트림을 수신하여 차량·번호판을 탐지하고
    검증된 번호판 이미지를 Firebase에 비동기 업로드하는 ROS2 노드.

    스레딩 구조:
        Main Thread  : ROS2 spin → image_callback → YOLO 추론 → 큐 투입
        Worker Thread: save_queue 소비 → Firebase 업로드 (네트워크 I/O 분리)
    '''

    def __init__(self):
        super().__init__("plate_detection_node")

        # 상태 변수
        self.last_detections   = []
        self.last_inference_ms = 0.0
        self.save_queue        = queue.Queue(maxsize=SAVE_QUEUE_MAXSIZE)
        self.db_ref            = None

        # 카메라 FPS 측정용 변수
        # - _cam_frame_count : 마지막 집계 이후 수신된 프레임 수
        # - _cam_fps_timer   : 마지막 집계 시각 (perf_counter 기준)
        # - _cam_fps         : 가장 최근에 계산된 카메라 FPS (오버레이 표시용)
        self._cam_frame_count = 0
        self._cam_fps_timer   = time.perf_counter()
        self._cam_fps         = 0.0

        self._load_model()
        self._init_firebase()
        self._init_subscriber()

        threading.Thread(target=self._upload_worker, daemon=True).start()
        cv2.namedWindow("Plate Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Plate Detection", 704, 704)
        self.get_logger().info("Node ready.")

    # ── 초기화 ──────────────────────────────────

    def _load_model(self):
        ''' YOLO 로드 + Cold Start 방지 워밍업 '''
        self.model = YOLO(MODEL_PATH)
        self.model.predict(
            source=np.zeros((YOLO_IMG_SIZE, YOLO_IMG_SIZE, 3), dtype=np.uint8),
            imgsz=YOLO_IMG_SIZE, verbose=False
        )
        self.get_logger().info("YOLO warm-up complete.")

    def _init_firebase(self):
        ''' Firebase Admin SDK 세션 생성. 실패 시 db_ref=None 유지 (업로드 스킵). '''
        try:
            firebase_admin.initialize_app(
                credentials.Certificate(FIREBASE_CRED_PATH),
                {"databaseURL": FIREBASE_DB_URL}
            )
            self.db_ref = db.reference("detections")
            self.get_logger().info("Firebase connected.")
        except Exception as e:
            self.get_logger().error(f"Firebase init failed: {e}")

    def _init_subscriber(self):
        ''' Best Effort QoS로 최신 프레임 우선 수신 '''
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        topic = f"{ROBOT_NAMESPACE}/oakd/rgb/image_raw/compressed"
        self.create_subscription(CompressedImage, topic, self.image_callback, qos)
        self.get_logger().info(f"Subscribing to: {topic}")

    # ── 메인 파이프라인 ──────────────────────────

    def image_callback(self, msg: CompressedImage):
        '''
        [CHAPTER 3: 실시간 데이터 파이프라인]
        Bytes → NumPy 디코딩 → 카메라 FPS 집계 → YOLO 추론 → Overlap 검증 → 큐 투입
        '''
        # 1. 디코딩
        frame = cv2.imdecode(np.array(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return

        # 2. 카메라 FPS 측정
        # callback 호출 주기 = 카메라가 실제로 프레임을 전달하는 주기이므로
        # 이 카운터가 가장 정확한 수신 FPS를 반영합니다.
        self._cam_frame_count += 1
        elapsed = time.perf_counter() - self._cam_fps_timer
        if elapsed >= CAM_FPS_LOG_INTERVAL:
            self._cam_fps = self._cam_frame_count / elapsed
            self.get_logger().info(f"Camera FPS: {self._cam_fps:.1f}")
            self._cam_frame_count = 0
            self._cam_fps_timer   = time.perf_counter()

        # 3. 추론
        cars, ids = self._detect(frame)
        self.last_detections = cars + ids

        # 4. 검증 후 큐 투입
        for id_det in ids:
            car = self._find_parent_car(id_det, cars)
            if car is None:
                continue

            self.get_logger().info(
                f"Validated: car={car['conf']:.2f}, plate={id_det['conf']:.2f}"
            )
            try:
                self.save_queue.put_nowait({"frame": frame, "car": car, "id": id_det})
            except queue.Full:
                self.get_logger().warn(
                    f"Queue full ({SAVE_QUEUE_MAXSIZE}). Frame dropped."
                )

        self._draw(frame)

    def _detect(self, frame: np.ndarray) -> tuple[list, list]:
        '''
        YOLOv8 추론. 추론 시간을 측정하여 로그 및 오버레이에 반영.
        반환: (cars, ids) — 각각 탐지 dict 리스트
        '''
        t0 = time.perf_counter()
        results = self.model.predict(source=frame, imgsz=YOLO_IMG_SIZE,
                                     conf=CONF_THRESHOLD, verbose=False)
        self.last_inference_ms = (time.perf_counter() - t0) * 1000
        self.get_logger().info(
            f"Inference: {self.last_inference_ms:.1f} ms "
            f"({1000 / self.last_inference_ms:.1f} FPS)"
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
                "conf":       float(box.conf[0].item()),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "area": max(0, x2 - x1) * max(0, y2 - y1),
            }
            (cars if name == "car" else ids).append(det)

        return cars, ids

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

    # ── 시각화 ──────────────────────────────────

    def _draw(self, frame: np.ndarray):
        '''
        Bounding Box 오버레이 + 상단 HUD
          좌상단: YOLO 추론 시간 / 추론 FPS
          중앙상단: 카메라 수신 FPS
          우상단: Firebase 업로드 큐 상태 (초록 → 주황 → 빨강)
        '''
        for det in self.last_detections:
            color = (0, 255, 0) if det["class_name"] == "car" else (0, 0, 255)
            cv2.rectangle(frame, (det["x1"], det["y1"]), (det["x2"], det["y2"]), color, 2)
            cv2.putText(frame, f"{det['class_name']} {det['conf']:.2f}",
                        (det["x1"], max(25, det["y1"] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 좌상단: YOLO 추론 성능
        infer_fps = 1000 / self.last_inference_ms if self.last_inference_ms > 0 else 0
        cv2.putText(frame, f"Infer: {self.last_inference_ms:.1f}ms ({infer_fps:.1f}FPS)",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # 중앙상단: 카메라 수신 FPS
        cam_text = f"Cam: {self._cam_fps:.1f}FPS"
        (cw, _), _ = cv2.getTextSize(cam_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(frame, cam_text,
                    ((frame.shape[1] - cw) // 2, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 우상단: 큐 상태
        q_size  = self.save_queue.qsize()
        q_ratio = q_size / SAVE_QUEUE_MAXSIZE
        q_color = (0, 255, 0) if q_ratio < 0.5 else (0, 165, 255) if q_ratio < 1.0 else (0, 0, 255)
        q_text  = f"Queue: {q_size}/{SAVE_QUEUE_MAXSIZE}"
        (tw, _), _ = cv2.getTextSize(q_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(frame, q_text, (frame.shape[1] - tw - 10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, q_color, 2)

        cv2.imshow("Plate Detection", frame)
        cv2.waitKey(1)

    # ── Firebase 업로드 워커 ─────────────────────

    def _upload_worker(self):
        '''
        [CHAPTER 4: 백그라운드 업로드 워커 (Consumer)]
        save_queue를 소비하며 Firebase에 직렬화 전송.
        None 수신 시 종료 (Sentinel 패턴).
        '''
        while True:
            item = self.save_queue.get()
            if item is None:
                break
            try:
                self._upload(item["frame"], item["car"], item["id"])
            except Exception as e:
                self.get_logger().error(f"Upload error: {e}")

    def _upload(self, frame: np.ndarray, car: dict, id_det: dict):
        '''
        번호판 영역 크롭 → JPG 재압축 → Base64 인코딩 → Firebase 전송.
        경로: detections/{timestamp}
        '''
        if self.db_ref is None:
            return

        x1, y1, x2, y2 = id_det["x1"], id_det["y1"], id_det["x2"], id_det["y2"]
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return

        now = datetime.datetime.now()
        _, enc = cv2.imencode(".jpg", crop)
        b64    = base64.b64encode(enc.tobytes()).decode("utf-8")

        self.db_ref.child(now.strftime("%Y%m%d_%H%M%S_%f")).set({
            "detected_at":    now.isoformat(),
            "car_confidence": round(car["conf"], 4),
            "id_confidence":  round(id_det["conf"], 4),
            "car_bbox": {"x1": car["x1"], "y1": car["y1"], "x2": car["x2"], "y2": car["y2"]},
            "id_bbox":  {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "image_base64": b64,
        })
        self.get_logger().info(f"Uploaded: {now.strftime('%Y%m%d_%H%M%S_%f')}")

    # ── 종료 ────────────────────────────────────

    def destroy_node(self):
        ''' Sentinel(None) 전송으로 워커 스레드 안전 종료 후 GUI 해제. '''
        self.save_queue.put(None)
        cv2.destroyAllWindows()
        super().destroy_node()


# ──────────────────────────────────────────────
# [CHAPTER 5: 엔트리 포인트]
# ──────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = PlateDetectionNode()
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