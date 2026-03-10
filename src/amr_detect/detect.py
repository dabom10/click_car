#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
detect.py - 번호판 탐지 + Firebase Realtime DB 저장 ROS2 노드
"""

import base64
import datetime
import queue
import threading

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage
from ultralytics import YOLO

import firebase_admin
from firebase_admin import credentials, db


# ================================
# 설정
# ================================
ROBOT_NAMESPACE = "/robot2"

MODEL_PATH            = "/home/rokey/click_car/models/AMR/v1/weights/best.pt"
FIREBASE_CRED_PATH    = "/home/rokey/click_car/web/database.json"
FIREBASE_DB_URL       = "https://click-car-2f586-default-rtdb.asia-southeast1.firebasedatabase.app"

CONF_THRESHOLD        = 0.50   # YOLO 탐지 기준
DETECT_CONF_THRESHOLD = 0.70   # DB 저장 기준
ID_IN_CAR_OVERLAP_THRESH = 0.5

YOLO_IMG_SIZE = 320            # 작을수록 빠름
# ================================


class PlateDetectionNode(Node):
    def __init__(self):
        super().__init__("plate_detection_node")

        self.last_detections = []
        self.save_queue      = queue.Queue()

        rgb_topic = f"{ROBOT_NAMESPACE}/oakd/rgb/image_raw/compressed"
        self.get_logger().info(f"토픽: {rgb_topic} | 모델: {MODEL_PATH}")

        # YOLO 로드 + 워밍업 (첫 프레임 딜레이 제거)
        self.model = YOLO(MODEL_PATH)
        self.model.predict(
            source=np.zeros((YOLO_IMG_SIZE, YOLO_IMG_SIZE, 3), dtype=np.uint8),
            imgsz=YOLO_IMG_SIZE, verbose=False
        )
        self.get_logger().info("YOLO 워밍업 완료")

        # Firebase
        self._init_firebase()

        # Firebase 저장 전용 스레드 (I/O 블로킹 격리)
        threading.Thread(target=self._db_save_worker, daemon=True).start()

        # GUI
        cv2.namedWindow("Plate Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Plate Detection", 640, 480)

        # QoS: 최신 프레임만 유지 (오래된 프레임 쌓임 방지)
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.create_subscription(CompressedImage, rgb_topic, self.image_callback, qos)
        self.get_logger().info("노드 시작!")

    # ------------------------------------------------------------------
    def _init_firebase(self):
        try:
            cred = credentials.Certificate(FIREBASE_CRED_PATH)
            firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_DB_URL})
            self.db_ref = db.reference("detections")
            self.get_logger().info("Firebase 초기화 성공")
        except Exception as e:
            self.get_logger().error(f"Firebase 초기화 실패: {e}")
            self.db_ref = None

    # ------------------------------------------------------------------
    # 카메라 수신 → 즉시 탐지 (타이머 없음)
    # ------------------------------------------------------------------
    def image_callback(self, msg: CompressedImage):
        # ── 디코딩 (bytes() 변환 필수!) ──
        try:
            np_arr = np.frombuffer(bytes(msg.data), np.uint8)
            frame  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                self.get_logger().error("이미지 디코딩 실패")
                return
        except Exception as e:
            self.get_logger().error(f"디코딩 오류: {e}")
            return

        # ── YOLO 탐지 ──
        cars, ids = self._detect(frame)
        self.last_detections = cars + ids

        # ── 번호판이 차 안에 있으면 저장 큐에 추가 ──
        for id_det in ids:
            if id_det["conf"] < DETECT_CONF_THRESHOLD:
                continue
            car = self._find_car_for_id(id_det, cars)
            if car is None:
                continue
            self.get_logger().info(
                f"번호판 감지! car={car['conf']:.2f} id={id_det['conf']:.2f}"
            )
            self.save_queue.put({"frame": frame, "car": car, "id": id_det})

        # ── 화면 출력 ──
        self._draw(frame)

    # ------------------------------------------------------------------
    def _detect(self, frame):
        cars, ids = [], []
        results = self.model.predict(
            source=frame, imgsz=YOLO_IMG_SIZE,
            conf=CONF_THRESHOLD, verbose=False
        )
        if not results:
            return cars, ids

        for box in results[0].boxes:
            cls_id = int(box.cls[0].item())
            conf   = float(box.conf[0].item())
            name   = self.model.names.get(cls_id, str(cls_id))
            if name not in ("car", "id"):
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            det = {"class_name": name, "conf": conf,
                   "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                   "area": max(0, x2-x1) * max(0, y2-y1)}
            (cars if name == "car" else ids).append(det)

        return cars, ids

    def _find_car_for_id(self, id_det, cars):
        best, best_ov = None, 0.0
        id_area = max(1, id_det["area"])
        for car in cars:
            ix = max(0, min(id_det["x2"], car["x2"]) - max(id_det["x1"], car["x1"]))
            iy = max(0, min(id_det["y2"], car["y2"]) - max(id_det["y1"], car["y1"]))
            ov = (ix * iy) / id_area
            if ov >= ID_IN_CAR_OVERLAP_THRESH and ov > best_ov:
                best_ov, best = ov, car
        return best

    # ------------------------------------------------------------------
    def _draw(self, frame):
        for det in self.last_detections:
            color = (0, 255, 0) if det["class_name"] == "car" else (0, 0, 255)
            cv2.rectangle(frame, (det["x1"], det["y1"]), (det["x2"], det["y2"]), color, 2)
            cv2.putText(frame, f"{det['class_name']} {det['conf']:.2f}",
                        (det["x1"], max(25, det["y1"]-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imshow("Plate Detection", frame)
        cv2.waitKey(1)

    # ------------------------------------------------------------------
    # Firebase 저장 워커 (별도 스레드)
    # ------------------------------------------------------------------
    def _db_save_worker(self):
        while True:
            item = self.save_queue.get()
            if item is None:
                break
            try:
                self._save(item["frame"], item["car"], item["id"])
            except Exception as e:
                self.get_logger().error(f"저장 오류: {e}")

    def _save(self, frame, car, id_det):
        if self.db_ref is None:
            return
        x1, y1, x2, y2 = id_det["x1"], id_det["y1"], id_det["x2"], id_det["y2"]
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return

        now = datetime.datetime.now()
        ts  = now.strftime("%Y%m%d_%H%M%S_%f")

        _, enc   = cv2.imencode(".jpg", crop)
        b64      = base64.b64encode(enc.tobytes()).decode("utf-8")

        self.db_ref.child(ts).set({
            "detected_at":    now.isoformat(),
            "car_confidence": round(car["conf"], 4),
            "id_confidence":  round(id_det["conf"], 4),
            "car_bbox": {"x1": car["x1"], "y1": car["y1"],
                         "x2": car["x2"], "y2": car["y2"]},
            "id_bbox":  {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "image_base64": b64,
        })
        self.get_logger().info(f"DB 저장 완료: detections/{ts}")

    # ------------------------------------------------------------------
    def destroy_node(self):
        self.save_queue.put(None)
        cv2.destroyAllWindows()
        super().destroy_node()


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