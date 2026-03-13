#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage, CameraInfo
from std_msgs.msg import String
from ultralytics import YOLO


ROBOT_NAMESPACE    = "/robot3"
MODEL_PATH         = "/home/rokey/click_car/models/amr.pt"

CONF_THRESHOLD     = 0.50
YOLO_IMG_SIZE      = 704

TOPIC_RGB          = f"{ROBOT_NAMESPACE}/oakd/rgb/image_raw/compressed"
TOPIC_DEPTH        = f"{ROBOT_NAMESPACE}/oakd/stereo/image_raw/compressedDepth"
TOPIC_INFO         = f"{ROBOT_NAMESPACE}/oakd/rgb/camera_info"
TOPIC_AMR_TARGET   = f"{ROBOT_NAMESPACE}amr_done"   # std_msgs/String, payload: "x,y,z"

WINDOW_NAME        = "Parking Detection"
PUBLISH_INTERVAL   = 0.2


class ParkingDetectionNode(Node):
    def __init__(self):
        super().__init__("parking_detection_node")

        self.last_rgb_received = None
        self.last_publish_time = 0.0
        self.gui_enabled = True

        self.latest_depth_frame = None   # uint16 depth(mm)
        self.camera_info = None
        self._depth_lock = None

        import threading
        self._depth_lock = threading.Lock()
        self._depth_no_data_warned = False

        self._load_model()
        self._init_subscriber()
        self._init_publisher()

        try:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, YOLO_IMG_SIZE, YOLO_IMG_SIZE)
            self.get_logger().info("[GUI] OpenCV window created.")
        except Exception as e:
            self.gui_enabled = False
            self.get_logger().warn(f"[GUI] OpenCV window disabled: {e}")

        self.create_timer(0.5, self._watchdog_timer)

        self.get_logger().info("Node ready.")

    # ──────────────────────────────────────────
    # 초기화
    # ──────────────────────────────────────────
    def _load_model(self):
        self.model = YOLO(MODEL_PATH)
        self.model.predict(
            source=np.zeros((YOLO_IMG_SIZE, YOLO_IMG_SIZE, 3), dtype=np.uint8),
            imgsz=YOLO_IMG_SIZE,
            verbose=False
        )
        self.get_logger().info(f"[DEBUG-1] Model classes: {self.model.names}")
        self.get_logger().info("YOLO warm-up complete.")

    def _init_subscriber(self):
        qos_be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        qos_rel = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.create_subscription(
            CompressedImage,
            TOPIC_RGB,
            self.image_callback,
            qos_be
        )
        self.create_subscription(
            CompressedImage,
            TOPIC_DEPTH,
            self.depth_callback,
            qos_be
        )
        self.create_subscription(
            CameraInfo,
            TOPIC_INFO,
            self.info_callback,
            qos_rel
        )

        self.get_logger().info(f"Subscribing RGB   : {TOPIC_RGB}")
        self.get_logger().info(f"Subscribing Depth : {TOPIC_DEPTH}")
        self.get_logger().info(f"Subscribing Info  : {TOPIC_INFO}")

    def _init_publisher(self):
        qos_pub = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.amr_target_pub = self.create_publisher(String, TOPIC_AMR_TARGET, qos_pub)
        self.get_logger().info(f"Publishing target: {TOPIC_AMR_TARGET}")

    # ──────────────────────────────────────────
    # 대기 화면
    # ──────────────────────────────────────────
    def _watchdog_timer(self):
        if self.last_rgb_received is None:
            self._draw_waiting_screen()

    def _draw_waiting_screen(self):
        if not self.gui_enabled:
            return

        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(canvas, "Waiting for RGB topic...", (40, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(canvas, TOPIC_RGB, (40, 235),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(canvas, TOPIC_DEPTH, (40, 260),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        status_text = "RGB RECEIVED" if self.last_rgb_received is not None else "NO RGB FRAME"
        color = (0, 255, 0) if self.last_rgb_received is not None else (0, 0, 255)
        cv2.putText(canvas, status_text, (40, 310),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow(WINDOW_NAME, canvas)
        cv2.waitKey(1)

    # ──────────────────────────────────────────
    # CameraInfo / Depth
    # ──────────────────────────────────────────
    def info_callback(self, msg: CameraInfo):
        if self.camera_info is not None:
            return

        self.camera_info = {
            "fx": msg.k[0],
            "fy": msg.k[4],
            "cx": msg.k[2],
            "cy": msg.k[5],
            "width": msg.width,
            "height": msg.height,
        }

        self.get_logger().info(
            f"[INFO] CameraInfo saved — "
            f"fx={self.camera_info['fx']:.2f}, fy={self.camera_info['fy']:.2f}, "
            f"cx={self.camera_info['cx']:.2f}, cy={self.camera_info['cy']:.2f}, "
            f"res={self.camera_info['width']}x{self.camera_info['height']}"
        )

    def _decode_compressed_depth(self, msg: CompressedImage):
        try:
            data_bytes = bytes(msg.data)
            raw = np.frombuffer(data_bytes, dtype=np.uint8)

            # 1) 바로 디코딩
            depth_img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
            if depth_img is not None and depth_img.size > 0:
                return depth_img

            # 2) PNG 문자열 찾기
            idx = data_bytes.find(b'PNG')
            if idx != -1 and idx > 0:
                depth_img = cv2.imdecode(
                    np.frombuffer(data_bytes[idx - 1:], dtype=np.uint8),
                    cv2.IMREAD_UNCHANGED
                )
                if depth_img is not None and depth_img.size > 0:
                    return depth_img

            # 3) PNG 시그니처 찾기
            png_sig = b'\x89PNG\r\n\x1a\n'
            idx = data_bytes.find(png_sig)
            if idx != -1:
                depth_img = cv2.imdecode(
                    np.frombuffer(data_bytes[idx:], dtype=np.uint8),
                    cv2.IMREAD_UNCHANGED
                )
                if depth_img is not None and depth_img.size > 0:
                    return depth_img

            return None

        except Exception as e:
            self.get_logger().warn(f"[DEBUG-D0] compressedDepth decode error: {e}")
            return None

    def depth_callback(self, msg: CompressedImage):
        try:
            depth_img = self._decode_compressed_depth(msg)

            if depth_img is None or depth_img.size == 0:
                self.get_logger().warn("[DEBUG-D1] Depth decode failed.")
                return

            if depth_img.dtype != np.uint16:
                self.get_logger().warn(
                    f"[DEBUG-D1] Decoded depth dtype is {depth_img.dtype}, expected uint16."
                )

            with self._depth_lock:
                self.latest_depth_frame = depth_img

        except Exception as e:
            self.get_logger().warn(f"[DEBUG-D2] depth_callback error: {e}")

    # ──────────────────────────────────────────
    # RGB 콜백
    # ──────────────────────────────────────────
    def image_callback(self, msg: CompressedImage):
        self.last_rgb_received = time.monotonic()

        frame = cv2.imdecode(np.frombuffer(bytes(msg.data), dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            self.get_logger().warn("[DEBUG-2] Frame decode failed. Check RGB topic format.")
            return

        car_dets = self._detect_cars(frame)

        with self._depth_lock:
            depth_snap = None if self.latest_depth_frame is None else self.latest_depth_frame.copy()

        targets = []
        for det in car_dets:
            cx, cy = self._get_bbox_center(det)
            xyz = self._get_xyz_from_pixel(cx, cy, depth_snap)
            targets.append((cx, cy, xyz, det))

        self._publish_targets(targets)
        self._draw(frame, targets)

    # ──────────────────────────────────────────
    # YOLO 탐지
    # ──────────────────────────────────────────
    def _detect_cars(self, frame: np.ndarray) -> list:
        results = self.model.predict(
            source=frame,
            imgsz=YOLO_IMG_SIZE,
            conf=CONF_THRESHOLD,
            verbose=False
        )

        cars = []
        if not results:
            return cars

        for box in results[0].boxes:
            cls_id = int(box.cls[0].item())
            name = self.model.names.get(cls_id, str(cls_id))

            if name != "car":
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

            cars.append({
                "class_name": name,
                "conf": float(box.conf[0].item()),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            })

        cars.sort(key=lambda d: ((d["x1"] + d["x2"]) // 2))
        return cars

    # ──────────────────────────────────────────
    # 좌표 계산
    # ──────────────────────────────────────────
    @staticmethod
    def _get_bbox_center(det: dict):
        cx = (det["x1"] + det["x2"]) // 2
        cy = (det["y1"] + det["y2"]) // 2
        return cx, cy

    def _get_xyz_from_pixel(self, u: int, v: int, depth_frame: np.ndarray | None):
        if depth_frame is None:
            return None
        if self.camera_info is None:
            return None

        h, w = depth_frame.shape[:2]
        if not (0 <= v < h and 0 <= u < w):
            return None

        depth_mm = int(depth_frame[v, u])
        if depth_mm <= 0:
            return None

        Z = depth_mm / 1000.0
        X = (u - self.camera_info["cx"]) * Z / self.camera_info["fx"]
        Y = (v - self.camera_info["cy"]) * Z / self.camera_info["fy"]
        return (X, Y, Z)

    # ──────────────────────────────────────────
    # 발행
    # ──────────────────────────────────────────
    def _publish_targets(self, targets: list):
        now = time.monotonic()
        if (now - self.last_publish_time) < PUBLISH_INTERVAL:
            return

        summary_parts = []

        for idx, (_, _, xyz, _) in enumerate(targets, start=1):
            if xyz is None:
                summary_parts.append(f"car{idx}=NO_DEPTH")
                continue

            x, y, z = xyz
            msg = String()
            msg.data = f"{x:.3f},{y:.3f},{z:.3f}"
            self.amr_target_pub.publish(msg)

            summary_parts.append(
                f"car{idx}=X:{x:.3f},Y:{y:.3f},Z:{z:.3f}"
            )

        if summary_parts:
            self.get_logger().info(f"[AMR_TARGET] cars={len(targets)} | " + " | ".join(summary_parts))
        else:
            self.get_logger().info("[AMR_TARGET] cars=0")

        self.last_publish_time = now

    # ──────────────────────────────────────────
    # 시각화
    # ──────────────────────────────────────────
    def _draw(self, frame: np.ndarray, targets: list):
        if not self.gui_enabled:
            return

        for idx, (cx, cy, xyz, det) in enumerate(targets, start=1):
            x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
            conf = det["conf"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            cv2.putText(frame, f"car{idx} {conf:.2f}",
                        (x1, max(25, y1 - 40)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if xyz is None:
                coord_text = "X=?, Y=?, Z=?"
            else:
                x, y, z = xyz
                coord_text = f"X={x:.2f}, Y={y:.2f}, Z={z:.2f}"

            cv2.putText(frame, coord_text,
                        (x1, max(45, y1 - 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 255), 2)

        cv2.putText(frame, f"Cars: {len(targets)}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow(WINDOW_NAME, frame)
        cv2.waitKey(1)

    # ──────────────────────────────────────────
    # 종료
    # ──────────────────────────────────────────
    def destroy_node(self):
        if self.gui_enabled:
            cv2.destroyAllWindows()
        super().destroy_node()


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