#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
detect.py
---------
단속봇(AMR) OAK-D 카메라 영상을 구독하여 YOLO 모델로
'car' → 'id'(번호판) 2단계 탐지를 수행하고,
car bbox 안에서 id가 감지되면 번호판 크롭 이미지를
Base64 인코딩하여 Firebase Realtime Database에 저장하는 ROS2 노드.

탐지 흐름:
  1. 매 프레임 YOLO 추론 → 'car' 클래스 감지
  2. 감지된 car bbox 안에 'id' bbox가 겹치는지 확인
  3. 조건 충족 시 번호판 크롭 이미지를 Base64로 Realtime DB에 저장

스레드 구조:
  - ROS spin      : 카메라 수신 + YOLO 탐지 (실시간성 보장)
  - Firebase 워커 : 저장 I/O만 분리 (탐지 블로킹 방지)

토픽:
  Subscribe : /{ns}/oakd/rgb/image_raw/compressed  (sensor_msgs/CompressedImage)

Firebase Realtime DB 구조:
  detections/{timestamp} : {
      detected_at, car_confidence, id_confidence,
      car_bbox, id_bbox, image_base64
  }
"""

import base64
import datetime
import queue
import threading

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from ultralytics import YOLO

import firebase_admin
from firebase_admin import credentials, db


# ================================
# 설정 상수
# ================================
MODEL_PATH         = "/home/rokey/click_car/yolov8n.pt"
FIREBASE_CRED_PATH = "/home/rokey/click_car/web/database.json"
FIREBASE_DB_URL    = "https://click-car-2f586-default-rtdb.firebaseio.com"  # Realtime DB URL

CONF_THRESHOLD        = 0.25   # YOLO 추론 기본 confidence
DETECT_CONF_THRESHOLD = 0.50   # DB 저장 트리거 confidence

# id bbox 가 car bbox 안에 포함되는 비율 기준
ID_IN_CAR_OVERLAP_THRESH = 0.5

YOLO_PERIOD_SEC    = 0.12
DISPLAY_PERIOD_SEC = 0.05

GUI_WINDOW_NAME = "AMR Plate Detection"
GUI_WIDTH       = 640
GUI_HEIGHT      = 480

ROBOT_NAMESPACE = "/robot2"   # 로봇 네임스페이스 변경: /robot2 또는 /robot3
# ================================


class PlateDetectionNode(Node):
    def __init__(self):
        super().__init__("plate_detection_node")

        # 상태 변수
        self.rgb_image        = None
        self.last_detections  = []
        self.logged_rgb_shape = False

        # DB 저장 큐 (탐지 → 저장 스레드)
        self.save_queue = queue.Queue()

        # 네임스페이스 기반 토픽
        self.rgb_topic = f"{ROBOT_NAMESPACE}/oakd/rgb/image_raw/compressed"

        self.get_logger().info(f"RGB topic  : {self.rgb_topic}")
        self.get_logger().info(f"YOLO model : {MODEL_PATH}")

        # YOLO 모델 로드
        self.get_logger().info("YOLO 모델 로드 중...")
        self.model = YOLO(MODEL_PATH)
        self.get_logger().info("YOLO 모델 로드 완료")

        # Firebase Realtime DB 초기화
        self._init_firebase()

        # DB 저장 전용 워커 스레드
        self.save_thread = threading.Thread(
            target=self._db_save_worker, daemon=True
        )
        self.save_thread.start()

        # GUI 윈도우 생성
        cv2.namedWindow(GUI_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(GUI_WINDOW_NAME, GUI_WIDTH, GUI_HEIGHT)

        # 구독
        self.create_subscription(
            CompressedImage, self.rgb_topic, self.rgb_callback, 10
        )

        # 타이머 - 노드 시작과 동시에 즉시 탐지 시작
        self.create_timer(YOLO_PERIOD_SEC,    self.run_detection_cycle)
        self.create_timer(DISPLAY_PERIOD_SEC, self.display_images)

        self.get_logger().info("번호판 탐지 노드 시작 - 즉시 탐지 중...")

    # ------------------------------------------------------------------
    # Firebase Realtime DB 초기화
    # ------------------------------------------------------------------
    def _init_firebase(self):
        try:
            cred = credentials.Certificate(FIREBASE_CRED_PATH)
            firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_DB_URL})
            self.db_ref = db.reference("detections")
            self.get_logger().info("Firebase Realtime DB 초기화 성공")
        except Exception as e:
            self.get_logger().error(f"Firebase Realtime DB 초기화 실패: {e}")
            self.db_ref = None

    # ------------------------------------------------------------------
    # 카메라 콜백
    # ------------------------------------------------------------------
    def rgb_callback(self, msg: CompressedImage):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            rgb    = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if rgb is None or rgb.size == 0:
                self.get_logger().error("Compressed RGB decode 실패: imdecode returned None")
                return

            if not self.logged_rgb_shape:
                self.get_logger().info(f"RGB 이미지 수신: {rgb.shape}")
                self.logged_rgb_shape = True

            self.rgb_image = rgb.copy()

        except Exception as e:
            self.get_logger().error(f"rgb_callback 오류: {e}")

    # ------------------------------------------------------------------
    # YOLO 탐지 사이클 (타이머 콜백 - 실시간)
    # ------------------------------------------------------------------
    def run_detection_cycle(self):
        if self.rgb_image is None:
            return

        frame = self.rgb_image.copy()
        cars, ids = self.run_yolo_on_frame(frame)

        # 표시용 저장
        self.last_detections = cars + ids

        if not cars or not ids:
            return

        # car bbox 안에 있는 id 찾기
        for id_det in ids:
            if id_det["conf"] < DETECT_CONF_THRESHOLD:
                continue

            matched_car = self._find_car_containing_id(id_det, cars)
            if matched_car is None:
                continue

            self.get_logger().info(
                f"차량 내 번호판 감지! "
                f"car_conf={matched_car['conf']:.2f}  "
                f"id_conf={id_det['conf']:.2f}  "
                f"id_bbox=({id_det['x1']},{id_det['y1']},{id_det['x2']},{id_det['y2']})"
            )

            # 저장 큐에 추가 (워커 스레드가 처리)
            self.save_queue.put({
                "frame":   frame,
                "car_det": matched_car,
                "id_det":  id_det,
            })

    def run_yolo_on_frame(self, frame):
        """
        Returns:
            cars : 'car' 클래스 bbox 리스트
            ids  : 'id'  클래스 bbox 리스트
        """
        cars = []
        ids  = []

        results = self.model.predict(source=frame, conf=CONF_THRESHOLD, verbose=False)
        if not results:
            return cars, ids

        for box in results[0].boxes:
            cls_id     = int(box.cls[0].item())
            conf       = float(box.conf[0].item())
            class_name = self.model.names.get(cls_id, str(cls_id))

            if class_name not in ("car", "id"):
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy().tolist())

            det = {
                "class_name": class_name,
                "conf": conf,
                "x1": x1, "y1": y1,
                "x2": x2, "y2": y2,
                "cx": (x1 + x2) // 2,
                "cy": (y1 + y2) // 2,
                "area": max(0, x2 - x1) * max(0, y2 - y1),
            }

            if class_name == "car":
                cars.append(det)
            else:
                ids.append(det)

        return cars, ids

    def _find_car_containing_id(self, id_det, cars):
        """
        id bbox가 car bbox 안에 충분히 포함되는 car를 반환.
        여러 car와 겹치면 overlap 비율이 가장 높은 car 반환.
        """
        best_car     = None
        best_overlap = 0.0
        id_area      = max(1, id_det["area"])

        for car in cars:
            ix1 = max(id_det["x1"], car["x1"])
            iy1 = max(id_det["y1"], car["y1"])
            ix2 = min(id_det["x2"], car["x2"])
            iy2 = min(id_det["y2"], car["y2"])

            inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            overlap    = inter_area / id_area

            if overlap >= ID_IN_CAR_OVERLAP_THRESH and overlap > best_overlap:
                best_overlap = overlap
                best_car     = car

        return best_car

    # ------------------------------------------------------------------
    # DB 저장 전용 워커 스레드
    # ------------------------------------------------------------------
    def _db_save_worker(self):
        while True:
            try:
                item = self.save_queue.get()
                if item is None:  # 종료 신호
                    break
                self._save_to_realtime_db(item["frame"], item["car_det"], item["id_det"])
                self.save_queue.task_done()
            except Exception as e:
                self.get_logger().error(f"저장 워커 오류: {e}")

    def _save_to_realtime_db(self, frame, car_det, id_det):
        if self.db_ref is None:
            self.get_logger().warn("Firebase Realtime DB 미초기화 - 저장 건너뜀")
            return
        try:
            now       = datetime.datetime.now()
            timestamp = now.strftime("%Y%m%d_%H%M%S_%f")

            # 번호판(id) 영역 크롭
            x1, y1, x2, y2 = id_det["x1"], id_det["y1"], id_det["x2"], id_det["y2"]
            plate_crop = frame[y1:y2, x1:x2]
            if plate_crop.size == 0:
                self.get_logger().warn("번호판 크롭 영역이 비어 있음 - 저장 건너뜀")
                return

            # JPEG 인코딩 → Base64 변환
            _, encoded   = cv2.imencode(".jpg", plate_crop)
            image_base64 = base64.b64encode(encoded.tobytes()).decode("utf-8")

            # Realtime DB 저장
            # 경로: detections/{timestamp}
            doc_data = {
                "detected_at":    now.isoformat(),
                "car_confidence": round(car_det["conf"], 4),
                "id_confidence":  round(id_det["conf"],  4),
                "car_bbox": {
                    "x1": car_det["x1"], "y1": car_det["y1"],
                    "x2": car_det["x2"], "y2": car_det["y2"],
                    "width":  car_det["x2"] - car_det["x1"],
                    "height": car_det["y2"] - car_det["y1"],
                },
                "id_bbox": {
                    "x1": x1, "y1": y1,
                    "x2": x2, "y2": y2,
                    "width":  x2 - x1,
                    "height": y2 - y1,
                },
                "image_base64": image_base64,
            }
            self.db_ref.child(timestamp).set(doc_data)
            self.get_logger().info(f"Realtime DB 저장 완료: detections/{timestamp}")

        except Exception as e:
            self.get_logger().error(f"Realtime DB 저장 실패: {e}")

    # ------------------------------------------------------------------
    # GUI 표시 (타이머 콜백)
    # ------------------------------------------------------------------
    def display_images(self):
        if self.rgb_image is None:
            return

        display = self.rgb_image.copy()

        for det in self.last_detections:
            # car → 초록, id(번호판) → 빨강
            color = (0, 255, 0) if det["class_name"] == "car" else (0, 0, 255)
            cv2.rectangle(display, (det["x1"], det["y1"]), (det["x2"], det["y2"]), color, 2)
            cv2.putText(
                display,
                f"{det['class_name']} {det['conf']:.2f}",
                (det["x1"], max(25, det["y1"] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
            )

        cv2.imshow(GUI_WINDOW_NAME, display)
        cv2.waitKey(1)

    # ------------------------------------------------------------------
    # 종료 처리
    # ------------------------------------------------------------------
    def destroy_node(self):
        self.save_queue.put(None)  # 워커 스레드 종료 신호
        cv2.destroyAllWindows()
        super().destroy_node()


# ======================================================================
# 엔트리포인트
# ======================================================================
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