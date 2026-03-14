#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
my_test_v2.py
-------------
OAK-D RGB + CompressedDepth → YOLO car 탐지
→ amcl_pose (로봇 위치 + yaw) 기반 수동 좌표 변환 (TF2 미사용)
→ /robot3/amr_done (std_msgs/String, "x,y") 발행
→ amr1.py가 수신하여 단속 위치로 이동

[좌표 변환 수식]
  카메라 좌표계 (Z=전방, X=우측) → 맵 좌표계
  cam_x = (pixel_u - cx) * z / fx    # 좌우 오프셋 (카메라 기준 우측+)
  cam_z = z                           # 전방 거리 (depth)

  map_x = robot_x + cam_z * cos(yaw) + cam_x * sin(yaw)
  map_y = robot_y + cam_z * sin(yaw) - cam_x * cos(yaw)

실행:
  ros2 run amr_detect my_test_v2

토픽 수동 테스트:
  ros2 topic echo /robot3/amr_done
"""

import math
import threading
from collections import deque

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CameraInfo, CompressedImage
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import String
from ultralytics import YOLO

import time as _time

# ══════════════════════════════════════════════════════════
# 설정 상수
# ══════════════════════════════════════════════════════════
ROBOT_NS     = "/robot3"
MODEL_PATH   = "/home/rokey/click_car/models/amr.pt"
TARGET_CLASS = "car"

# ── YOLO ─────────────────────────────────────────────────
YOLO_IMG_SIZE          = 704
CONF_THRESHOLD         = 0.70   # bbox 표시 최소 confidence
MEASURE_CONF_THRESHOLD = 0.80   # depth 측정 시작 최소 confidence

# ── Depth 샘플링 ──────────────────────────────────────────
DEPTH_HEADER_BYTES = 12         # compressedDepth PNG 헤더 오프셋
DEPTH_KERNEL_SIZE  = 5          # median 커널 크기
DEPTH_SAMPLE_COUNT = 10         # 수집 목표 샘플 수
MIN_VALID_DEPTH_M  = 0.2
MAX_VALID_DEPTH_M  = 5.0
MAX_SYNC_DIFF_SEC  = 0.20       # RGB-Depth 타임스탬프 허용 차이
DEPTH_BUFFER_SIZE  = 30

# ── Outlier 제거 ──────────────────────────────────────────
OUTLIER_MIN_KEEP_COUNT    = 4
OUTLIER_ABS_THRESH_M      = 0.25
OUTLIER_MAD_SCALE         = 2.5
OUTLIER_FALLBACK_THRESH_M = 0.08
DISTANCE_EMA_ALPHA        = 0.6

# ── 허용 구역 (맵 좌표 기준) ─────────────────────────────
# 이 구역 안에 있는 차량 좌표만 amr_done 발행
PARKING_ZONES = [
    {"xmin": -1.91, "xmax": -0.80, "ymin":  0.46, "ymax":  1.60},  # 구역 A
    {"xmin": -1.24, "xmax":  0.504,"ymin": -5.0,  "ymax": -4.30},  # 구역 B
]

# ── 발행 ─────────────────────────────────────────────────
PUBLISH_COOLDOWN_SEC = 3.0      # amr_done 연속 발행 방지 (초)

# ── 타이밍 ───────────────────────────────────────────────
YOLO_PERIOD_SEC    = 0.15
DETECT_START_DELAY = 2.0
DISPLAY_PERIOD_SEC = 0.05

# ── GUI ──────────────────────────────────────────────────
GUI_WINDOW = "Car Detection v2 (RGB | Depth)"
GUI_W, GUI_H = 1280, 480
# ══════════════════════════════════════════════════════════


class CarDetectNodeV2(Node):
    """
    TF2 없이 amcl_pose (robot_x, robot_y, robot_yaw) 로
    카메라 좌표 → 맵 좌표를 수동 변환하는 버전.
    """

    def __init__(self):
        super().__init__("car_detect_node_v2")
        self.lock = threading.Lock()

        # ── 로봇 위치 (amcl_pose 콜백으로 갱신) ──────────
        self.robot_x   = 0.0
        self.robot_y   = 0.0
        self.robot_yaw = 0.0
        self._amcl_received = False

        # ── 카메라 데이터 ──────────────────────────────────
        self.K               = None
        self.rgb_image       = None
        self.rgb_stamp_sec   = None
        self.depth_image     = None
        self.depth_stamp_sec = None
        self.depth_buffer    = deque(maxlen=DEPTH_BUFFER_SIZE)

        # ── 탐지/측정 상태 ─────────────────────────────────
        self.last_detection    = None
        self.measurement_state = None
        self.ready_target      = None
        self.detection_enabled = False
        self.filtered_dist_ema = None

        # ── 발행 쿨다운 ────────────────────────────────────
        self.last_publish_time = 0.0

        # ── GUI ───────────────────────────────────────────
        self.display_image = None

        # ── 로그 중복 방지 ─────────────────────────────────
        self._logged_intrinsics = False
        self._logged_rgb_shape  = False
        self._logged_depth_info = False

        # ── 토픽 ──────────────────────────────────────────
        rgb_topic   = f"{ROBOT_NS}/oakd/rgb/image_raw/compressed"
        depth_topic = f"{ROBOT_NS}/oakd/stereo/image_raw/compressedDepth"
        info_topic  = f"{ROBOT_NS}/oakd/rgb/camera_info"
        amcl_topic  = f"{ROBOT_NS}/amcl_pose"

        self.get_logger().info(f"RGB  : {rgb_topic}")
        self.get_logger().info(f"Depth: {depth_topic}")
        self.get_logger().info(f"Info : {info_topic}")
        self.get_logger().info(f"AMCL : {amcl_topic}")

        # ── QoS ───────────────────────────────────────────
        qos_be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1
        )
        qos_rel = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST, depth=1
        )

        # ── YOLO 워밍업 ────────────────────────────────────
        self.model = YOLO(MODEL_PATH)
        self.model.predict(
            source=np.zeros((YOLO_IMG_SIZE, YOLO_IMG_SIZE, 3), dtype=np.uint8),
            imgsz=YOLO_IMG_SIZE, verbose=False
        )
        self.get_logger().info(f"YOLO warm-up 완료. classes: {self.model.names}")

        # ── 구독 ──────────────────────────────────────────
        self.create_subscription(CameraInfo,               info_topic,  self._info_cb,  qos_rel)
        self.create_subscription(CompressedImage,          depth_topic, self._depth_cb, qos_be)
        self.create_subscription(CompressedImage,          rgb_topic,   self._rgb_cb,   qos_be)
        self.create_subscription(PoseWithCovarianceStamped,amcl_topic,  self._amcl_cb,  qos_rel)

        # ── 발행 ──────────────────────────────────────────
        self.amr_done_pub = self.create_publisher(
            String, f"{ROBOT_NS}/amr_done", 10
        )
        self.get_logger().info(f"발행: {ROBOT_NS}/amr_done")

        # ── GUI 스레드 ────────────────────────────────────
        self.gui_stop = threading.Event()
        threading.Thread(target=self._gui_loop, daemon=True).start()

        # ── 타이머 시작 ───────────────────────────────────
        self.create_timer(YOLO_PERIOD_SEC,    self._detection_cycle)
        self.create_timer(0.1,                self._process_ready_target)
        self.create_timer(DISPLAY_PERIOD_SEC, self._update_display_data)

        self._det_enable_timer = self.create_timer(DETECT_START_DELAY, self._enable_detection)

        self.get_logger().info("초기화 완료. amcl_pose 수신 대기 중...")

    def _enable_detection(self):
        self._det_enable_timer.cancel()
        with self.lock:
            self.detection_enabled = True
        self.get_logger().info("YOLO 탐지 활성화.")

    # ──────────────────────────────────────────────────────
    # amcl_pose 콜백 — 로봇 위치/yaw 갱신
    # ──────────────────────────────────────────────────────
    def _amcl_cb(self, msg: PoseWithCovarianceStamped):
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw  = math.atan2(siny, cosy)

        with self.lock:
            self.robot_x   = msg.pose.pose.position.x
            self.robot_y   = msg.pose.pose.position.y
            self.robot_yaw = yaw
            if not self._amcl_received:
                self._amcl_received = True
                self.get_logger().info(
                    f"amcl_pose 첫 수신: x={self.robot_x:.3f}, "
                    f"y={self.robot_y:.3f}, yaw={math.degrees(yaw):.1f}°"
                )

    # ──────────────────────────────────────────────────────
    # 카메라 콜백
    # ──────────────────────────────────────────────────────
    def _info_cb(self, msg: CameraInfo):
        with self.lock:
            self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        if not self._logged_intrinsics:
            K = self.K
            self.get_logger().info(
                f"CameraInfo: fx={K[0,0]:.2f}, fy={K[1,1]:.2f}, "
                f"cx={K[0,2]:.2f}, cy={K[1,2]:.2f}"
            )
            self._logged_intrinsics = True

    def _depth_cb(self, msg: CompressedImage):
        try:
            depth = self._decode_depth(msg)
            if depth is None:
                return
            stamp_sec = self._to_sec(msg.header.stamp)

            if not self._logged_depth_info:
                self.get_logger().info(
                    f"Depth 수신: shape={depth.shape}, dtype={depth.dtype}"
                )
                self._logged_depth_info = True

            with self.lock:
                self.depth_image     = depth.copy()
                self.depth_stamp_sec = stamp_sec
                self.depth_buffer.append({
                    "stamp_sec": stamp_sec,
                    "depth":     depth.copy(),
                })

            self._add_depth_sample_from_callback(depth, stamp_sec)

        except Exception as e:
            self.get_logger().warn(f"depth_callback 오류: {e}")

    def _rgb_cb(self, msg: CompressedImage):
        try:
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            rgb = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if rgb is None or rgb.size == 0:
                return
            stamp_sec = self._to_sec(msg.header.stamp)

            if not self._logged_rgb_shape:
                self.get_logger().info(f"RGB 수신: shape={rgb.shape}")
                self._logged_rgb_shape = True

            with self.lock:
                self.rgb_image     = rgb.copy()
                self.rgb_stamp_sec = stamp_sec

        except Exception as e:
            self.get_logger().warn(f"rgb_callback 오류: {e}")

    # ──────────────────────────────────────────────────────
    # YOLO 탐지
    # ──────────────────────────────────────────────────────
    def _detection_cycle(self):
        with self.lock:
            if not self.detection_enabled:
                return
            frame        = self.rgb_image.copy() if self.rgb_image is not None else None
            rgb_stamp    = self.rgb_stamp_sec
            is_measuring = self.measurement_state is not None

        if frame is None:
            return

        det = self._run_yolo(frame)

        with self.lock:
            self.last_detection = det

        if det is None or det["conf"] < MEASURE_CONF_THRESHOLD:
            return
        if is_measuring:
            return

        self._start_measurement(det, frame.shape[:2], rgb_stamp)

    def _run_yolo(self, frame) -> dict | None:
        results = self.model.predict(
            source=frame, imgsz=YOLO_IMG_SIZE,
            conf=CONF_THRESHOLD, verbose=False
        )
        if not results:
            return None

        best = None
        for box in results[0].boxes:
            cls_id = int(box.cls[0].item())
            conf   = float(box.conf[0].item())
            name   = self.model.names.get(cls_id, str(cls_id))
            if name != TARGET_CLASS:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            area = max(0, x2 - x1) * max(0, y2 - y1)
            det = {
                "conf": conf,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "cx": (x1 + x2) // 2,
                "cy": (y1 + y2) // 2,
                "area": area,
            }
            if best is None or area > best["area"]:
                best = det
        return best

    # ──────────────────────────────────────────────────────
    # Depth 샘플 수집
    # ──────────────────────────────────────────────────────
    def _start_measurement(self, det, rgb_shape, rgb_stamp_sec):
        nearest = self._find_nearest_depth_frame(rgb_stamp_sec)

        with self.lock:
            self.measurement_state = {
                "x_rgb":         det["cx"],
                "y_rgb":         det["cy"],
                "x1": det["x1"], "y1": det["y1"],
                "x2": det["x2"], "y2": det["y2"],
                "conf":          det["conf"],
                "rgb_shape":     rgb_shape,
                "rgb_stamp_sec": rgb_stamp_sec,
                "samples_m":     [],
                "sync_diffs":    [],
                "used_stamps":   set(),
            }

        if nearest is not None:
            diff = abs(nearest["stamp_sec"] - rgb_stamp_sec)
            if diff <= MAX_SYNC_DIFF_SEC:
                self._do_add_sample(nearest["depth"], nearest["stamp_sec"], rgb_stamp_sec)

        self.get_logger().info(
            f"Depth 측정 시작: conf={det['conf']:.2f}, center=({det['cx']}, {det['cy']})"
        )

    def _add_depth_sample_from_callback(self, depth, depth_stamp_sec):
        with self.lock:
            if self.measurement_state is None:
                return
            rgb_stamp_sec = self.measurement_state["rgb_stamp_sec"]
        self._do_add_sample(depth, depth_stamp_sec, rgb_stamp_sec)

    def _do_add_sample(self, depth, depth_stamp_sec, rgb_stamp_sec):
        with self.lock:
            if self.measurement_state is None:
                return
            if len(self.measurement_state["samples_m"]) >= DEPTH_SAMPLE_COUNT:
                return
            if depth_stamp_sec in self.measurement_state["used_stamps"]:
                return
            x_rgb     = self.measurement_state["x_rgb"]
            y_rgb     = self.measurement_state["y_rgb"]
            rgb_shape = self.measurement_state["rgb_shape"]

        depth_px = self._scale_pixel(x_rgb, y_rgb, rgb_shape, depth.shape[:2])
        if depth_px is None:
            return
        xd, yd = depth_px

        raw       = self._get_depth_median(depth, xd, yd, DEPTH_KERNEL_SIZE)
        z         = raw / 1000.0
        sync_diff = abs(depth_stamp_sec - rgb_stamp_sec)

        with self.lock:
            if self.measurement_state is None:
                return
            self.measurement_state["used_stamps"].add(depth_stamp_sec)
            if MIN_VALID_DEPTH_M < z < MAX_VALID_DEPTH_M:
                self.measurement_state["samples_m"].append(z)
                self.measurement_state["sync_diffs"].append(sync_diff)
            count = len(self.measurement_state["samples_m"])

        if count >= DEPTH_SAMPLE_COUNT:
            self._finalize_measurement()

    def _finalize_measurement(self):
        with self.lock:
            if self.measurement_state is None:
                return
            state   = dict(self.measurement_state)
            samples = list(state["samples_m"])

        if len(samples) < OUTLIER_MIN_KEEP_COUNT:
            self.get_logger().warn(f"유효 샘플 부족 ({len(samples)}개). 재측정.")
            with self.lock:
                self.measurement_state = None
            return

        filtered = self._remove_outliers(samples)

        if len(filtered) < OUTLIER_MIN_KEEP_COUNT:
            self.get_logger().warn(f"Outlier 제거 후 부족 ({len(filtered)}개). 재측정.")
            with self.lock:
                self.measurement_state = None
            return

        avg_z = float(np.mean(filtered))

        with self.lock:
            prev_ema = self.filtered_dist_ema
        stable_z = avg_z if prev_ema is None else \
                   DISTANCE_EMA_ALPHA * avg_z + (1 - DISTANCE_EMA_ALPHA) * prev_ema

        with self.lock:
            self.filtered_dist_ema = stable_z
            self.ready_target = {
                "x_rgb":    state["x_rgb"],
                "y_rgb":    state["y_rgb"],
                "x1": state["x1"], "y1": state["y1"],
                "x2": state["x2"], "y2": state["y2"],
                "conf":     state["conf"],
                "z_stable": stable_z,
                "z_raw":    avg_z,
            }
            self.measurement_state = None

        self.get_logger().info(
            f"측정 완료: raw={avg_z:.3f}m, stable={stable_z:.3f}m, "
            f"samples={len(samples)}, filtered={len(filtered)}"
        )

    # ──────────────────────────────────────────────────────
    # amcl_pose 기반 좌표 변환 및 amr_done 발행
    # ──────────────────────────────────────────────────────
    def _process_ready_target(self):
        now = _time.monotonic()

        with self.lock:
            if self.ready_target is None:
                return
            if now - self.last_publish_time < PUBLISH_COOLDOWN_SEC:
                return
            if not self._amcl_received:
                self.get_logger().warn("amcl_pose 미수신. 발행 대기.")
                return
            K          = self.K.copy() if self.K is not None else None
            target     = dict(self.ready_target)
            robot_x    = self.robot_x
            robot_y    = self.robot_y
            robot_yaw  = self.robot_yaw

        if K is None:
            self.get_logger().warn("CameraInfo 미수신. 발행 스킵.")
            return

        fx    = K[0, 0]
        cx    = K[0, 2]
        z     = target["z_stable"]
        x_rgb = target["x_rgb"]

        # ── 카메라 좌표계 → 맵 좌표계 ────────────────────
        # cam_x: 카메라 기준 좌우 오프셋 (우측+)
        # cam_z: 전방 거리 (depth)
        # Y축(상하)은 2D 맵 변환에 불필요하므로 사용 안 함
        cam_x = (x_rgb - cx) * z / fx
        cam_z = z

        # 로봇 yaw 기준으로 맵 좌표 계산
        map_x = robot_x + cam_z * math.cos(robot_yaw) + cam_x * math.sin(robot_yaw)
        map_y = robot_y + cam_z * math.sin(robot_yaw) - cam_x * math.cos(robot_yaw)
        # ─────────────────────────────────────────────────

        self.get_logger().info(
            f"\n"
            f"[로봇]   x={robot_x:.3f}, y={robot_y:.3f}, yaw={math.degrees(robot_yaw):.1f}°\n"
            f"[카메라] cam_x={cam_x:.3f}, cam_z={cam_z:.3f}\n"
            f"[맵 좌표] map_x={map_x:.3f}, map_y={map_y:.3f}"
        )

        # ── 주차 구역 필터링 ───────────────────────────────
        zone = self._find_zone(map_x, map_y)
        if zone is None:
            self.get_logger().info(
                f"[구역 외] ({map_x:.3f}, {map_y:.3f}) → 발행 스킵"
            )
            with self.lock:
                self.ready_target = None
            return
        # ──────────────────────────────────────────────────

        msg = String()
        msg.data = f"{map_x:.3f},{map_y:.3f}"
        self.amr_done_pub.publish(msg)

        with self.lock:
            self.last_publish_time = now
            self.ready_target      = None

        self.get_logger().info(
            f"[amr_done] 발행 → ({map_x:.3f}, {map_y:.3f}) 구역{zone} | "
            f"depth={z:.2f}m, conf={target['conf']:.2f}"
        )

    # ──────────────────────────────────────────────────────
    # 유틸
    # ──────────────────────────────────────────────────────
    def _find_zone(self, x: float, y: float) -> int | None:
        for i, z in enumerate(PARKING_ZONES):
            if z["xmin"] <= x <= z["xmax"] and z["ymin"] <= y <= z["ymax"]:
                return i + 1
        return None

    def _decode_depth(self, msg: CompressedImage) -> np.ndarray | None:
        arr   = np.frombuffer(msg.data, dtype=np.uint8)
        depth = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if depth is not None and depth.size > 0:
            return depth
        if len(arr) > DEPTH_HEADER_BYTES:
            depth = cv2.imdecode(arr[DEPTH_HEADER_BYTES:], cv2.IMREAD_UNCHANGED)
            if depth is not None and depth.size > 0:
                return depth
        return None

    def _to_sec(self, stamp) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def _find_nearest_depth_frame(self, rgb_stamp_sec) -> dict | None:
        with self.lock:
            if not self.depth_buffer:
                return None
            buf = list(self.depth_buffer)
        best, best_diff = None, None
        for item in buf:
            diff = abs(item["stamp_sec"] - rgb_stamp_sec)
            if best_diff is None or diff < best_diff:
                best_diff, best = diff, item
        return best

    def _get_depth_median(self, depth: np.ndarray, x: int, y: int,
                          kernel_size: int = 5) -> float:
        h, w = depth.shape[:2]
        half = kernel_size // 2
        x1, x2 = max(0, x - half), min(w, x + half + 1)
        y1, y2 = max(0, y - half), min(h, y + half + 1)
        patch  = depth[y1:y2, x1:x2]
        valid  = patch[patch > 0]
        return float(np.median(valid)) if valid.size > 0 else 0.0

    def _scale_pixel(self, x_src, y_src, src_shape, dst_shape):
        sh, sw = src_shape
        dh, dw = dst_shape
        if sw <= 0 or sh <= 0:
            return None
        xd = int(round(x_src * dw / sw))
        yd = int(round(y_src * dh / sh))
        return max(0, min(dw - 1, xd)), max(0, min(dh - 1, yd))

    def _remove_outliers(self, samples_m: list) -> list:
        arr = np.array(samples_m, dtype=np.float64)
        if arr.size == 0:
            return []
        median   = float(np.median(arr))
        abs_diff = np.abs(arr - median)
        mad      = float(np.median(abs_diff))
        if mad < 1e-6:
            keep = abs_diff <= OUTLIER_ABS_THRESH_M
        else:
            sigma  = 1.4826 * mad
            thresh = max(OUTLIER_FALLBACK_THRESH_M, OUTLIER_MAD_SCALE * sigma)
            keep   = abs_diff <= thresh
        filtered = arr[keep]
        if filtered.size < OUTLIER_MIN_KEEP_COUNT:
            arr_sorted = np.sort(arr)
            trim = max(1, int(len(arr_sorted) * 0.1))
            filtered = arr_sorted[trim:-trim] if len(arr_sorted) > 2 * trim else arr_sorted
        return filtered.tolist()

    # ──────────────────────────────────────────────────────
    # GUI
    # ──────────────────────────────────────────────────────
    def _update_display_data(self):
        with self.lock:
            rgb    = self.rgb_image.copy()   if self.rgb_image   is not None else None
            depth  = self.depth_image.copy() if self.depth_image is not None else None
            det    = dict(self.last_detection)    if self.last_detection    is not None else None
            mstate = dict(self.measurement_state) if self.measurement_state is not None else None
            rtgt   = dict(self.ready_target)      if self.ready_target      is not None else None
            rx, ry, ryaw = self.robot_x, self.robot_y, self.robot_yaw
            amcl_ok = self._amcl_received

        if rgb is None:
            return

        disp = rgb.copy()

        # amcl 상태
        amcl_text = f"AMCL: ({rx:.2f},{ry:.2f}) yaw={math.degrees(ryaw):.0f}deg" \
                    if amcl_ok else "AMCL: 미수신"
        amcl_color = (0, 255, 0) if amcl_ok else (0, 0, 255)
        cv2.putText(disp, amcl_text, (10, disp.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, amcl_color, 1)

        # BBox
        if det is not None:
            x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(disp, f"car {det['conf']:.2f}",
                        (x1, max(25, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 측정 상태
        lock_target = mstate or rtgt
        if lock_target is not None:
            lx, ly = int(lock_target["x_rgb"]), int(lock_target["y_rgb"])
            cv2.circle(disp, (lx, ly), 8, (0, 255, 255), 2)
            cv2.line(disp, (lx - 12, ly), (lx + 12, ly), (0, 255, 255), 2)
            cv2.line(disp, (lx, ly - 12), (lx, ly + 12), (0, 255, 255), 2)

            if mstate is not None:
                text = f"MEASURING {len(mstate['samples_m'])}/{DEPTH_SAMPLE_COUNT}"
                cv2.putText(disp, text, (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            elif rtgt is not None:
                text = f"READY: z={rtgt['z_stable']:.2f}m"
                cv2.putText(disp, text, (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

        # Depth 시각화 병합
        if depth is not None:
            d_norm  = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
            d_color = cv2.applyColorMap(d_norm.astype(np.uint8), cv2.COLORMAP_JET)
            if disp.shape[:2] != d_color.shape[:2]:
                d_color = cv2.resize(d_color, (disp.shape[1], disp.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)
            combined = np.hstack((disp, d_color))
        else:
            combined = disp

        with self.lock:
            self.display_image = combined.copy()

    def _gui_loop(self):
        cv2.namedWindow(GUI_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(GUI_WINDOW, GUI_W, GUI_H)

        while not self.gui_stop.is_set():
            with self.lock:
                img = self.display_image.copy() if self.display_image is not None else None

            if img is not None:
                cv2.imshow(GUI_WINDOW, img)

            key = cv2.waitKey(10) & 0xFF
            if key == ord("q"):
                self.get_logger().info("q 키 입력 → 종료")
                self.gui_stop.set()
                break

    def destroy_node(self):
        self.gui_stop.set()
        cv2.destroyAllWindows()
        super().destroy_node()


# ══════════════════════════════════════════════════════════
# 엔트리포인트
# ══════════════════════════════════════════════════════════
def main(args=None):
    rclpy.init(args=args)
    node = CarDetectNodeV2()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
