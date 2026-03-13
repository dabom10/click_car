#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import threading
from collections import deque

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage, CameraInfo
from std_msgs.msg import String
from ultralytics import YOLO


ROBOT_NAMESPACE    = "/robot2"
MODEL_PATH         = "/home/rokey/click_car/models/amr.pt"

CONF_THRESHOLD     = 0.70
YOLO_IMG_SIZE      = 704

TOPIC_RGB          = f"{ROBOT_NAMESPACE}/oakd/rgb/image_raw/compressed"
TOPIC_DEPTH        = f"{ROBOT_NAMESPACE}/oakd/stereo/image_raw/compressedDepth"
TOPIC_INFO         = f"{ROBOT_NAMESPACE}/oakd/rgb/camera_info"
TOPIC_AMR_TARGET   = f"{ROBOT_NAMESPACE}/amr_done"   # std_msgs/String, payload: "x,y,z"

WINDOW_NAME        = "Parking Detection"
PUBLISH_INTERVAL   = 0.2

# 트래킹/스무딩 파라미터
IOU_THRESH         = 0.30
TRACK_TTL_SEC      = 1.0
SMOOTH_WINDOW      = 7           # 기존 5 -> 7
OUTLIER_THRESH_M   = 0.25        # 기존 0.35 -> 0.25

# 하단 샘플 파라미터
BOTTOM_Y_OFFSET    = 4           # y2 그대로 말고 약간 위에서 측정
DEPTH_ROI_KSIZE    = 5           # depth ROI median 크기
INFRAME_Z_OUTLIER_THRESH_M = 0.20  # 한 프레임 안 3점 샘플 간 Z 편차 허용


class Track:
    def __init__(self, det: dict, xyz_uv=None):
        now = time.monotonic()
        self.det = det
        self.created_at = now
        self.last_seen = now
        self.history = deque(maxlen=SMOOTH_WINDOW)   # [(x,y,z,u,v), ...]

        if xyz_uv is not None:
            self.history.append(xyz_uv)

    def update(self, det: dict, xyz_uv=None):
        self.det = det
        self.last_seen = time.monotonic()
        if xyz_uv is not None:
            self.history.append(xyz_uv)

    def is_alive(self):
        return (time.monotonic() - self.last_seen) <= TRACK_TTL_SEC

    def get_smoothed_xyz_uv(self):
        """
        최근 history에서 outlier 제거 후 median 좌표 반환
        반환: (x, y, z, u, v) 또는 None
        """
        if len(self.history) == 0:
            return None

        arr = np.array(self.history, dtype=np.float32)  # (N, 5)
        xyz = arr[:, :3]
        uv  = arr[:, 3:]

        # 1차 median
        xyz_med = np.median(xyz, axis=0)

        # median 기준 거리
        dists = np.linalg.norm(xyz - xyz_med, axis=1)

        # outlier 제거
        keep = dists <= OUTLIER_THRESH_M
        filtered_xyz = xyz[keep]
        filtered_uv  = uv[keep]

        if filtered_xyz.shape[0] == 0:
            final_xyz = xyz_med
            final_uv = np.median(uv, axis=0)
        else:
            final_xyz = np.median(filtered_xyz, axis=0)
            final_uv = np.median(filtered_uv, axis=0)

        x, y, z = map(float, final_xyz)
        u, v = map(int, np.round(final_uv))
        return (x, y, z, u, v)


class ParkingDetectionNode(Node):
    def __init__(self):
        super().__init__("parking_detection_node")

        self.last_rgb_received = None
        self.last_publish_time = 0.0
        self.gui_enabled = True

        self.latest_depth_frame = None
        self.camera_info = None
        self._depth_lock = threading.Lock()
        self._depth_no_data_warned = False

        self.tracks = []

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

        self.create_subscription(CompressedImage, TOPIC_RGB, self.image_callback, qos_be)
        self.create_subscription(CompressedImage, TOPIC_DEPTH, self.depth_callback, qos_be)
        self.create_subscription(CameraInfo, TOPIC_INFO, self.info_callback, qos_rel)

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

            depth_img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
            if depth_img is not None and depth_img.size > 0:
                return depth_img

            idx = data_bytes.find(b'PNG')
            if idx != -1 and idx > 0:
                depth_img = cv2.imdecode(
                    np.frombuffer(data_bytes[idx - 1:], dtype=np.uint8),
                    cv2.IMREAD_UNCHANGED
                )
                if depth_img is not None and depth_img.size > 0:
                    return depth_img

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
            self.get_logger().warn("[DEBUG-2] RGB decode failed.")
            return

        car_dets = self._detect_cars(frame)

        with self._depth_lock:
            depth_snap = None if self.latest_depth_frame is None else self.latest_depth_frame.copy()

        current_measurements = []
        for det in car_dets:
            xyz_uv = self._get_xyz_from_bottom_multi(det, depth_snap)
            current_measurements.append((det, xyz_uv))

        self._update_tracks(current_measurements)

        smoothed_targets = []
        for trk in self.tracks:
            smoothed = trk.get_smoothed_xyz_uv()
            smoothed_targets.append((trk.det, smoothed))

        self._publish_targets(smoothed_targets)
        self._draw(frame, smoothed_targets)

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
                "area": max(1, (x2 - x1) * (y2 - y1)),
            })

        cars.sort(key=lambda d: ((d["x1"] + d["x2"]) // 2))
        return cars

    # ──────────────────────────────────────────
    # 하단 다중 샘플 / depth
    # ──────────────────────────────────────────
    def _get_depth_mm_median(self, u: int, v: int, depth_frame: np.ndarray | None, ksize: int = DEPTH_ROI_KSIZE):
        if depth_frame is None:
            return None

        h, w = depth_frame.shape[:2]
        if not (0 <= v < h and 0 <= u < w):
            return None

        half = ksize // 2
        x1 = max(0, u - half)
        x2 = min(w, u + half + 1)
        y1 = max(0, v - half)
        y2 = min(h, v + half + 1)

        roi = depth_frame[y1:y2, x1:x2]
        valid = roi[roi > 0]

        if valid.size == 0:
            return None

        return int(np.median(valid))

    def _get_xyz_from_bottom_multi(self, det: dict, depth_frame: np.ndarray | None):
        """
        bbox 하단에서 3점 샘플링:
        - 35%, 50%, 65% 지점
        - y2보다 약간 위에서(depth hole 회피)
        - 프레임 내부에서도 outlier 제거 후 median
        반환: (X, Y, Z, u, v) 또는 None
        """
        if self.camera_info is None or depth_frame is None:
            return None

        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        w = x2 - x1
        if w <= 0:
            return None

        v = max(0, y2 - BOTTOM_Y_OFFSET)

        sample_us = [
            int(x1 + 0.35 * w),
            int(x1 + 0.50 * w),
            int(x1 + 0.65 * w),
        ]

        candidates = []
        for u in sample_us:
            depth_mm = self._get_depth_mm_median(u, v, depth_frame, ksize=DEPTH_ROI_KSIZE)
            if depth_mm is None:
                continue

            Z = depth_mm / 1000.0
            X = (u - self.camera_info["cx"]) * Z / self.camera_info["fx"]
            Y = (v - self.camera_info["cy"]) * Z / self.camera_info["fy"]
            candidates.append([X, Y, Z, u, v])

        if len(candidates) == 0:
            return None

        cand = np.array(candidates, dtype=np.float32)

        # 프레임 내 Z median 기준으로 outlier 제거
        z_med = np.median(cand[:, 2])
        z_keep = np.abs(cand[:, 2] - z_med) <= INFRAME_Z_OUTLIER_THRESH_M
        cand_f = cand[z_keep]

        if cand_f.shape[0] == 0:
            cand_f = cand

        final = np.median(cand_f, axis=0)
        X, Y, Z, u, v = final

        return (float(X), float(Y), float(Z), int(round(u)), int(round(v)))

    # ──────────────────────────────────────────
    # 간단 IoU 트래킹
    # ──────────────────────────────────────────
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

    def _update_tracks(self, current_measurements):
        self.tracks = [trk for trk in self.tracks if trk.is_alive()]

        matched_track_idx = set()
        matched_meas_idx = set()

        for t_idx, trk in enumerate(self.tracks):
            best_iou = 0.0
            best_m_idx = -1

            for m_idx, (det, xyz_uv) in enumerate(current_measurements):
                if m_idx in matched_meas_idx:
                    continue
                iou = self._iou(trk.det, det)
                if iou > best_iou:
                    best_iou = iou
                    best_m_idx = m_idx

            if best_iou >= IOU_THRESH:
                det, xyz_uv = current_measurements[best_m_idx]
                trk.update(det, xyz_uv)
                matched_track_idx.add(t_idx)
                matched_meas_idx.add(best_m_idx)

        for m_idx, (det, xyz_uv) in enumerate(current_measurements):
            if m_idx not in matched_meas_idx:
                self.tracks.append(Track(det, xyz_uv))

    # ──────────────────────────────────────────
    # 발행
    # ──────────────────────────────────────────
    def _publish_targets(self, smoothed_targets):
        now = time.monotonic()
        if (now - self.last_publish_time) < PUBLISH_INTERVAL:
            return

        summary_parts = []

        for idx, (_, smoothed) in enumerate(smoothed_targets, start=1):
            if smoothed is None:
                summary_parts.append(f"car{idx}=NO_DEPTH")
                continue

            x, y, z, _, _ = smoothed

            msg = String()
            msg.data = f"{x:.3f},{y:.3f},{z:.3f}"
            self.amr_target_pub.publish(msg)

            summary_parts.append(
                f"car{idx}=X:{x:.3f},Y:{y:.3f},Z:{z:.3f}"
            )

        if summary_parts:
            self.get_logger().info(f"[AMR_TARGET] cars={len(smoothed_targets)} | " + " | ".join(summary_parts))
        else:
            self.get_logger().info("[AMR_TARGET] cars=0")

        self.last_publish_time = now

    # ──────────────────────────────────────────
    # 시각화
    # ──────────────────────────────────────────
    def _draw(self, frame: np.ndarray, smoothed_targets):
        if not self.gui_enabled:
            return

        for idx, (det, smoothed) in enumerate(smoothed_targets, start=1):
            x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
            conf = det["conf"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if smoothed is None:
                coord_text = "X=?, Y=?, Z=?"
            else:
                x, y, z, u, v = smoothed
                cv2.circle(frame, (u, v), 5, (0, 0, 255), -1)
                coord_text = f"X={x:.2f}, Y={y:.2f}, Z={z:.2f}"

            cv2.putText(frame, f"car{idx} {conf:.2f}",
                        (x1, max(25, y1 - 40)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.putText(frame, coord_text,
                        (x1, max(45, y1 - 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 255), 2)

        cv2.putText(frame, f"Cars: {len(smoothed_targets)}",
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