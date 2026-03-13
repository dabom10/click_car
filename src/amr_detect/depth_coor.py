#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
depth_coor_improved.py
──────────────────────────────────────────────────────────────────────────────
[개선 핵심 요약]
  1. TF 기반 맵 좌표 변환  → 로봇 이동/회전에 의한 X/Y 튐 원천 제거
  2. ROI 기반 Depth 샘플링  → 단일 픽셀 → BB 하단 20% 영역 퍼센타일
  3. 맵 좌표 EMA 필터       → 카메라 좌표 아닌 맵 좌표에서 스무딩
  4. Depth wall clock 동기화 → 수신 시각 기준 150 ms 초과 stale depth 무시
  5. 최소 안정 프레임 발행  → 3 프레임 미만 track 발행 차단
──────────────────────────────────────────────────────────────────────────────
"""

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

import tf2_ros
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import PointStamped
import tf2_geometry_msgs

from ultralytics import YOLO


ROBOT_NAMESPACE    = "/robot2"
MODEL_PATH         = "/home/rokey/click_car/models/amr.pt"

CONF_THRESHOLD     = 0.70
YOLO_IMG_SIZE      = 704

TOPIC_RGB          = f"{ROBOT_NAMESPACE}/oakd/rgb/image_raw/compressed"
TOPIC_DEPTH        = f"{ROBOT_NAMESPACE}/oakd/stereo/image_raw/compressedDepth"
TOPIC_INFO         = f"{ROBOT_NAMESPACE}/oakd/rgb/camera_info"
TOPIC_AMR_TARGET   = f"{ROBOT_NAMESPACE}/amr_done"   # std_msgs/String, payload: "x,y,z"

WINDOW_NAME       = "Parking Detection"
PUBLISH_INTERVAL  = 0.2

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
    def __init__(self, det: dict, cam_xyz_uv=None, map_xyz=None):
        now = time.monotonic()
        self.det             = det
        self.created_at      = now
        self.last_seen       = now
        self.history         = deque(maxlen=SMOOTH_WINDOW)
        self.ema_map         = None
        self.last_cam_xyz_uv = cam_xyz_uv

        if map_xyz is not None:
            self._add_map(map_xyz)

    def _add_map(self, map_xyz):
        new = np.array(map_xyz, dtype=np.float32)
        if self.ema_map is not None:
            if np.linalg.norm(new - self.ema_map) > OUTLIER_THRESH_M:
                return  # outlier → 버림
        self.history.append(tuple(map_xyz))
        if self.ema_map is None:
            self.ema_map = new.copy()
        else:
            self.ema_map = EMA_ALPHA * new + (1.0 - EMA_ALPHA) * self.ema_map

    def update(self, det: dict, cam_xyz_uv=None, map_xyz=None):
        self.det       = det
        self.last_seen = time.monotonic()
        if cam_xyz_uv is not None:
            self.last_cam_xyz_uv = cam_xyz_uv
        if map_xyz is not None:
            self._add_map(map_xyz)

    def is_alive(self) -> bool:
        return (time.monotonic() - self.last_seen) <= TRACK_TTL_SEC

    def is_stable(self) -> bool:
        return len(self.history) >= MIN_STABLE_FRAMES

    def get_smoothed_map_xyz(self):
        if self.ema_map is not None:
            return tuple(float(v) for v in self.ema_map)
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

        self.last_rgb_received  = None
        self.last_publish_time  = 0.0
        self.gui_enabled        = True
        self.latest_depth_frame = None
        self.latest_depth_stamp = None   # ★ float (time.monotonic())
        self.camera_info        = None
        self._depth_lock        = threading.Lock()
        self.tracks             = []
        self._tf_warn_logged    = False

        self._tf_buffer   = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._load_model()
        self._init_subscriber()
        self._init_publisher()

        try:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, YOLO_IMG_SIZE, YOLO_IMG_SIZE)
        except Exception as e:
            self.gui_enabled = False
            self.get_logger().warn(f"[GUI] OpenCV disabled: {e}")

        self.create_timer(0.5, self._watchdog_timer)
        self.get_logger().info("Node ready.")

    # ── 초기화 ────────────────────────────────────────────────────────────────
    def _load_model(self):
        self.model = YOLO(MODEL_PATH)
        self.model.predict(
            source=np.zeros((YOLO_IMG_SIZE, YOLO_IMG_SIZE, 3), dtype=np.uint8),
            imgsz=YOLO_IMG_SIZE, verbose=False
        )
        self.get_logger().info(f"Model classes: {self.model.names}")
        self.get_logger().info("YOLO warm-up complete.")

    def _init_subscriber(self):
        qos_be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1
        )
        qos_rel = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST, depth=1
        )
        self.create_subscription(CompressedImage, TOPIC_RGB,   self.image_callback, qos_be)
        self.create_subscription(CompressedImage, TOPIC_DEPTH, self.depth_callback, qos_be)
        self.create_subscription(CameraInfo,      TOPIC_INFO,  self.info_callback,  qos_rel)

    def _init_publisher(self):
        qos_pub = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST, depth=10
        )
        self.amr_target_pub = self.create_publisher(String, TOPIC_AMR_TARGET, qos_pub)

    # ── 대기 화면 ─────────────────────────────────────────────────────────────
    def _watchdog_timer(self):
        if self.last_rgb_received is None:
            self._draw_waiting_screen()

    def _draw_waiting_screen(self):
        if not self.gui_enabled:
            return
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(canvas, "Waiting for RGB topic...", (40, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(canvas, TOPIC_RGB,   (40, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        cv2.putText(canvas, TOPIC_DEPTH, (40, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        cv2.imshow(WINDOW_NAME, canvas)
        cv2.waitKey(1)

    # ── CameraInfo ────────────────────────────────────────────────────────────
    def info_callback(self, msg: CameraInfo):
        if self.camera_info is not None:
            return
        self.camera_info = {
            "fx": msg.k[0], "fy": msg.k[4],
            "cx": msg.k[2], "cy": msg.k[5],
            "width": msg.width, "height": msg.height,
        }
        self.get_logger().info(
            f"CameraInfo — fx={self.camera_info['fx']:.1f}, fy={self.camera_info['fy']:.1f}, "
            f"cx={self.camera_info['cx']:.1f}, cy={self.camera_info['cy']:.1f}"
        )

    # ── Depth 디코드 ──────────────────────────────────────────────────────────
    def _decode_compressed_depth(self, msg: CompressedImage):
        try:
            data_bytes = bytes(msg.data)
            raw = np.frombuffer(data_bytes, dtype=np.uint8)

            depth_img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
            if depth_img is not None and depth_img.size > 0:
                return depth_img

            for sig in (b'PNG', b'\x89PNG\r\n\x1a\n'):
                idx = data_bytes.find(sig)
                if idx == -1:
                    continue
                start = max(0, idx - 1) if sig == b'PNG' else idx
                depth_img = cv2.imdecode(
                    np.frombuffer(data_bytes[start:], dtype=np.uint8),
                    cv2.IMREAD_UNCHANGED
                )
                if depth_img is not None and depth_img.size > 0:
                    return depth_img
            return None
        except Exception as e:
            self.get_logger().warn(f"compressedDepth decode error: {e}")
            return None

    def depth_callback(self, msg: CompressedImage):
        try:
            depth_img = self._decode_compressed_depth(msg)
            if depth_img is None or depth_img.size == 0:
                return
            with self._depth_lock:
                self.latest_depth_frame = depth_img
                self.latest_depth_stamp = time.monotonic()  # ★ wall clock 수신 시각
        except Exception as e:
            self.get_logger().warn(f"depth_callback error: {e}")

    # ── RGB 콜백 ──────────────────────────────────────────────────────────────
    def image_callback(self, msg: CompressedImage):
        self.last_rgb_received = time.monotonic()

        frame = cv2.imdecode(
            np.frombuffer(bytes(msg.data), dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if frame is None:
            return

        # ★ Depth 동기화 체크 (wall clock 기준)
        with self._depth_lock:
            depth_snap = None
            if self.latest_depth_frame is not None and self.latest_depth_stamp is not None:
                dt = time.monotonic() - self.latest_depth_stamp  # ★ wall clock 비교
                if dt <= DEPTH_SYNC_MAX_DT:
                    depth_snap = self.latest_depth_frame.copy()
                else:
                    self.get_logger().warn(
                        f"[SYNC] Depth stale ({dt*1000:.0f} ms > {DEPTH_SYNC_MAX_DT*1000:.0f} ms), skip",
                        throttle_duration_sec=1.0
                    )

        car_dets = self._detect_cars(frame)

        measurements = []
        for det in car_dets:
            xyz_uv = self._get_xyz_from_bottom_multi(det, depth_snap)
            current_measurements.append((det, xyz_uv))

        self._update_tracks(measurements)

        smoothed_results = [
            (trk, trk.get_smoothed_map_xyz()) for trk in self.tracks
        ]
        self._publish_targets(smoothed_results)
        self._draw(frame, smoothed_results)

    # ── YOLO 탐지 ─────────────────────────────────────────────────────────────
    def _detect_cars(self, frame: np.ndarray) -> list:
        results = self.model.predict(
            source=frame, imgsz=YOLO_IMG_SIZE,
            conf=CONF_THRESHOLD, verbose=False
        )
        cars = []
        if not results:
            return cars
        for box in results[0].boxes:
            cls_id = int(box.cls[0].item())
            name   = self.model.names.get(cls_id, str(cls_id))
            if name != "car":
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            cars.append({
                "class_name": name,
                "conf":  float(box.conf[0].item()),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "area": max(1, (x2 - x1) * (y2 - y1)),
            })
        cars.sort(key=lambda d: (d["x1"] + d["x2"]) // 2)
        return cars

    # ──────────────────────────────────────────
    # 하단 다중 샘플 / depth
    # ──────────────────────────────────────────
    def _get_depth_mm_median(self, u: int, v: int, depth_frame: np.ndarray | None, ksize: int = DEPTH_ROI_KSIZE):
        if depth_frame is None:
            return None

        roi   = depth_frame[roi_top:roi_bot, roi_left:roi_right]
        valid = roi[(roi >= DEPTH_MIN_MM) & (roi <= DEPTH_MAX_MM)]

        if valid.size < 5:
            return None

        return int(np.percentile(valid, 25))

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

    # ── TF: 카메라 → 맵 좌표 변환 ────────────────────────────────────────────
    def _cam_to_map(self, cx, cy, cz, stamp):
        """
        카메라 좌표 → 맵 좌표 변환.
        로봇이 이동/회전해도 맵 좌표는 고정 → X/Y 튐 원천 제거.
        """
        try:
            pt = PointStamped()
            pt.header.frame_id = CAMERA_FRAME
            pt.header.stamp    = stamp
            pt.point.x = cx
            pt.point.y = cy
            pt.point.z = cz

            pt_map = self._tf_buffer.transform(
                pt, MAP_FRAME,
                timeout=rclpy.duration.Duration(seconds=0.05)
            )
            return (pt_map.point.x, pt_map.point.y, pt_map.point.z)

        except tf2_ros.LookupException:
            if not self._tf_warn_logged:
                self.get_logger().warn(
                    f"[TF] 변환 실패: {CAMERA_FRAME} → {MAP_FRAME}\n"
                    f"      $ ros2 run tf2_tools view_frames 로 프레임명 확인 후\n"
                    f"      코드 상단 CAMERA_FRAME 상수를 수정하세요."
                )
                self._tf_warn_logged = True
            return None

        except tf2_ros.ExtrapolationException:
            # stamp extrapolation 실패 시 최신 transform으로 재시도
            try:
                pt2 = PointStamped()
                pt2.header.frame_id = CAMERA_FRAME
                pt2.header.stamp    = rclpy.time.Time().to_msg()
                pt2.point.x = cx; pt2.point.y = cy; pt2.point.z = cz
                pt_map2 = self._tf_buffer.transform(
                    pt2, MAP_FRAME,
                    timeout=rclpy.duration.Duration(seconds=0.05)
                )
                return (pt_map2.point.x, pt_map2.point.y, pt_map2.point.z)
            except Exception:
                return None

        except Exception as e:
            self.get_logger().debug(f"[TF] error: {e}")
            return None

    # ── IoU 트래킹 ────────────────────────────────────────────────────────────
    @staticmethod
    def _iou(a: dict, b: dict) -> float:
        ix    = max(0, min(a["x2"], b["x2"]) - max(a["x1"], b["x1"]))
        iy    = max(0, min(a["y2"], b["y2"]) - max(a["y1"], b["y1"]))
        inter = ix * iy
        if inter == 0:
            return 0.0
        return inter / (max(1, a["area"]) + max(1, b["area"]) - inter)

    def _update_tracks(self, current_measurements):
        self.tracks = [trk for trk in self.tracks if trk.is_alive()]

        matched_t = set()
        matched_m = set()

        for t_idx, trk in enumerate(self.tracks):
            best_iou, best_m = 0.0, -1
            for m_idx, (det, _, _) in enumerate(measurements):
                if m_idx in matched_m:
                    continue
                iou = self._iou(trk.det, det)
                if iou > best_iou:
                    best_iou, best_m = iou, m_idx
            if best_iou >= IOU_THRESH:
                det, cam_xyz_uv, map_xyz = measurements[best_m]
                trk.update(det, cam_xyz_uv, map_xyz)
                matched_t.add(t_idx)
                matched_m.add(best_m)

        for m_idx, (det, cam_xyz_uv, map_xyz) in enumerate(measurements):
            if m_idx not in matched_m:
                self.tracks.append(Track(det, cam_xyz_uv, map_xyz))

    # ── 발행 ──────────────────────────────────────────────────────────────────
    def _publish_targets(self, smoothed_results):
        now = time.monotonic()
        if (now - self.last_publish_time) < PUBLISH_INTERVAL:
            return

        parts     = []
        published = 0

        for idx, (trk, map_xyz) in enumerate(smoothed_results, start=1):
            if not trk.is_stable():
                parts.append(f"car{idx}=WARMING({len(trk.history)}/{MIN_STABLE_FRAMES})")
                continue
            if map_xyz is None:
                parts.append(f"car{idx}=NO_MAP_XYZ")
                continue

            mx, my, mz = map_xyz
            msg        = String()
            msg.data   = f"{mx:.3f},{my:.3f},{mz:.3f}"
            self.amr_target_pub.publish(msg)
            published += 1
            parts.append(f"car{idx}=MAP(X:{mx:.3f},Y:{my:.3f},Z:{mz:.3f})")

        self.get_logger().info(
            f"[PUBLISH] total={len(smoothed_results)} published={published} | " +
            " | ".join(parts)
        )
        self.last_publish_time = now

    # ── 시각화 ────────────────────────────────────────────────────────────────
    def _draw(self, frame: np.ndarray, smoothed_results):
        if not self.gui_enabled:
            return

        for idx, (trk, map_xyz) in enumerate(smoothed_results, start=1):
            det             = trk.det
            x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
            conf            = det["conf"]

            color = (0, 255, 0) if trk.is_stable() else (0, 200, 200)  # 녹=안정 / 노랑=워밍업
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            uv = trk.get_last_uv()
            if uv:
                cv2.circle(frame, uv, 5, (0, 0, 255), -1)

            coord_text = "Map: N/A" if map_xyz is None else \
                         f"Map X:{map_xyz[0]:.2f} Y:{map_xyz[1]:.2f} Z:{map_xyz[2]:.2f}"

            cv2.putText(frame, f"car{idx} {conf:.2f}",
                        (x1, max(25, y1 - 40)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, coord_text,
                        (x1, max(45, y1 - 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        stable_cnt = sum(1 for trk, _ in smoothed_results if trk.is_stable())
        cv2.putText(frame,
                    f"Cars: {len(smoothed_results)} (stable: {stable_cnt})",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow(WINDOW_NAME, frame)
        cv2.waitKey(1)

    # ── 종료 ──────────────────────────────────────────────────────────────────
    def destroy_node(self):
        if self.gui_enabled:
            cv2.destroyAllWindows()
        super().destroy_node()


# ──────────────────────────────────────────────────────────────────────────────
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