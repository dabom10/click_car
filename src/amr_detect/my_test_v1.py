#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
my_test.py
----------
OAK-D RGB + CompressedDepth → YOLO car 탐지
→ TF2 카메라→맵 좌표 변환
→ /robot3/amr_done (std_msgs/String, "x,y") 발행
→ amr1.py가 수신하여 단속 위치로 이동

실행:
  ros2 run amr_detect my_test

토픽 수동 테스트:
  ros2 topic echo /robot3/amr_done
"""

import threading
from collections import deque

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CameraInfo, CompressedImage
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs  # noqa: F401 — PointStamped transform 등록 필수
from ultralytics import YOLO

import time as _time

# ══════════════════════════════════════════════════════════
# 설정 상수
# ══════════════════════════════════════════════════════════
ROBOT_NS   = "/robot3"
MODEL_PATH = "/home/rokey/click_car/models/amr.pt"
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

# ── Outlier 제거 (my_test_3 방식) ─────────────────────────
OUTLIER_MIN_KEEP_COUNT    = 4
OUTLIER_ABS_THRESH_M      = 0.25
OUTLIER_MAD_SCALE         = 2.5
OUTLIER_FALLBACK_THRESH_M = 0.08
DISTANCE_EMA_ALPHA        = 0.6

# ── 허용 구역 (맵 좌표 기준) ─────────────────────────────
# 이 구역 안에 있는 차량 좌표만 amr_done 발행
PARKING_ZONES = [
    {"xmin": -1.91, "xmax": -0.80, "ymin": 0.46,  "ymax": 1.60},   # 구역 A (좌상단)
    {"xmin": -1.24, "xmax":  0.504,"ymin": -5.0,  "ymax": -4.30},  # 구역 B (우하단)
]

# ── 발행 ─────────────────────────────────────────────────
PUBLISH_COOLDOWN_SEC = 3.0      # amr_done 연속 발행 방지 (초)

# ── 타이밍 ───────────────────────────────────────────────
YOLO_PERIOD_SEC     = 0.15
TF_START_DELAY_SEC  = 3.0       # TF 트리 안정화 대기
DETECT_START_DELAY  = 2.0       # 탐지 활성화 추가 대기
DISPLAY_PERIOD_SEC  = 0.05

# ── GUI ──────────────────────────────────────────────────
GUI_WINDOW = "Car Detection (RGB | Depth)"
GUI_W, GUI_H = 1280, 480
# ══════════════════════════════════════════════════════════


class CarDetectNode(Node):
    """
    OAK-D RGB + CompressedDepth → YOLO car 탐지
    → TF2로 카메라 좌표 → 맵 좌표 변환
    → /robot3/amr_done 발행 ("map_x,map_y")
    → amr1.py가 이동 처리 (네비게이션은 amr1.py에 위임)
    """

    def __init__(self):
        super().__init__("car_detect_node")
        self.lock = threading.Lock()

        # ── 카메라 데이터 ──────────────────────────────────
        self.K               = None
        self.rgb_image       = None
        self.rgb_stamp_sec   = None
        self.depth_image     = None
        self.depth_frame_id  = None
        self.depth_stamp_sec = None
        self.depth_buffer    = deque(maxlen=DEPTH_BUFFER_SIZE)

        # ── 탐지/측정 상태 ─────────────────────────────────
        self.last_detection    = None   # 최신 YOLO 탐지 결과
        self.measurement_state = None   # 샘플 수집 중인 상태
        self.ready_target      = None   # 측정 완료, 발행 대기
        self.detection_enabled = False
        self.filtered_dist_ema = None   # EMA 평활 거리

        # ── 발행 쿨다운 ────────────────────────────────────
        self.last_publish_time = 0.0

        # ── GUI 데이터 ────────────────────────────────────
        self.display_image = None

        # ── 로그 중복 방지 ─────────────────────────────────
        self._logged_intrinsics  = False
        self._logged_rgb_shape   = False
        self._logged_depth_info  = False

        # ── 토픽 정의 ─────────────────────────────────────
        rgb_topic   = f"{ROBOT_NS}/oakd/rgb/image_raw/compressed"
        depth_topic = f"{ROBOT_NS}/oakd/stereo/image_raw/compressedDepth"
        info_topic  = f"{ROBOT_NS}/oakd/rgb/camera_info"

        self.get_logger().info(f"RGB  topic: {rgb_topic}")
        self.get_logger().info(f"Depth topic: {depth_topic}")
        self.get_logger().info(f"Info  topic: {info_topic}")

        # ── QoS ───────────────────────────────────────────
        qos_be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1
        )
        qos_rel = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST, depth=1
        )

        # ── TF2 ───────────────────────────────────────────
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ── YOLO (워밍업) ──────────────────────────────────
        self.model = YOLO(MODEL_PATH)
        self.model.predict(
            source=np.zeros((YOLO_IMG_SIZE, YOLO_IMG_SIZE, 3), dtype=np.uint8),
            imgsz=YOLO_IMG_SIZE, verbose=False
        )
        self.get_logger().info(f"YOLO warm-up 완료. classes: {self.model.names}")

        # ── 구독 ──────────────────────────────────────────
        self.create_subscription(CameraInfo,      info_topic,  self._info_cb,  qos_rel)
        self.create_subscription(CompressedImage, depth_topic, self._depth_cb, qos_be)
        self.create_subscription(CompressedImage, rgb_topic,   self._rgb_cb,   qos_be)

        # ── 발행 ──────────────────────────────────────────
        self.amr_done_pub = self.create_publisher(
            String, f"{ROBOT_NS}/amr_done", 10
        )
        self.get_logger().info(f"발행 토픽: {ROBOT_NS}/amr_done")

        # ── GUI 스레드 ────────────────────────────────────
        self.gui_stop = threading.Event()
        threading.Thread(target=self._gui_loop, daemon=True).start()

        # ── TF 안정화 후 타이머 시작 ─────────────────────
        self.get_logger().info(f"TF 안정화 대기 중... ({TF_START_DELAY_SEC}초)")
        self._tf_timer = self.create_timer(TF_START_DELAY_SEC, self._on_tf_ready)

    # ──────────────────────────────────────────────────────
    # 초기화 타이머
    # ──────────────────────────────────────────────────────
    def _on_tf_ready(self):
        self._tf_timer.cancel()
        self.get_logger().info("TF 안정화 완료. 처리 타이머 시작.")

        self.create_timer(YOLO_PERIOD_SEC,    self._detection_cycle)
        self.create_timer(0.1,                self._process_ready_target)
        self.create_timer(DISPLAY_PERIOD_SEC, self._update_display_data)

        self._det_enable_timer = self.create_timer(DETECT_START_DELAY, self._enable_detection)

    def _enable_detection(self):
        self._det_enable_timer.cancel()
        with self.lock:
            self.detection_enabled = True
        self.get_logger().info("YOLO 탐지 활성화 완료.")

    # ──────────────────────────────────────────────────────
    # 카메라 콜백
    # ──────────────────────────────────────────────────────
    def _info_cb(self, msg: CameraInfo):
        with self.lock:
            self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        if not self._logged_intrinsics:
            K = self.K
            self.get_logger().info(
                f"CameraInfo 수신: fx={K[0,0]:.2f}, fy={K[1,1]:.2f}, "
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
                self.depth_frame_id  = msg.header.frame_id
                self.depth_stamp_sec = stamp_sec
                self.depth_buffer.append({
                    "stamp_sec": stamp_sec,
                    "frame_id":  msg.header.frame_id,
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
    # YOLO 탐지 사이클
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

        # confidence 부족하거나 이미 측정 중이면 새 측정 시작 안 함
        if det is None or det["conf"] < MEASURE_CONF_THRESHOLD:
            return
        if is_measuring:
            return

        self._start_measurement(det, frame.shape[:2], rgb_stamp)

    def _run_yolo(self, frame) -> dict | None:
        """YOLO 추론 → 가장 큰 car BBox 반환 (없으면 None)"""
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
        """샘플 수집 상태 초기화 후 가장 가까운 depth 프레임으로 첫 샘플 시도"""
        nearest = self._find_nearest_depth_frame(rgb_stamp_sec)

        with self.lock:
            self.measurement_state = {
                "x_rgb":        det["cx"],
                "y_rgb":        det["cy"],
                "x1": det["x1"], "y1": det["y1"],
                "x2": det["x2"], "y2": det["y2"],
                "conf":         det["conf"],
                "rgb_shape":    rgb_shape,
                "rgb_stamp_sec":rgb_stamp_sec,
                "samples_m":    [],
                "sync_diffs":   [],
                "used_stamps":  set(),
            }

        if nearest is not None:
            diff = abs(nearest["stamp_sec"] - rgb_stamp_sec)
            if diff <= MAX_SYNC_DIFF_SEC:
                self._do_add_sample(nearest["depth"], nearest["stamp_sec"], rgb_stamp_sec)

        self.get_logger().info(
            f"Depth 측정 시작: conf={det['conf']:.2f}, "
            f"center=({det['cx']}, {det['cy']})"
        )

    def _add_depth_sample_from_callback(self, depth, depth_stamp_sec):
        """depth 콜백에서 호출 — 측정 중인 경우에만 샘플 추가"""
        with self.lock:
            if self.measurement_state is None:
                return
            rgb_stamp_sec = self.measurement_state["rgb_stamp_sec"]
        self._do_add_sample(depth, depth_stamp_sec, rgb_stamp_sec)

    def _do_add_sample(self, depth, depth_stamp_sec, rgb_stamp_sec):
        """실제 샘플 추가 로직"""
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

        # RGB 해상도 → Depth 해상도로 픽셀 좌표 스케일링
        depth_px = self._scale_pixel(x_rgb, y_rgb, rgb_shape, depth.shape[:2])
        if depth_px is None:
            return
        xd, yd = depth_px

        raw = self._get_depth_median(depth, xd, yd, DEPTH_KERNEL_SIZE)
        z   = raw / 1000.0  # mm → m
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
        """샘플 수집 완료 → outlier 제거 → EMA 평활 → ready_target 설정"""
        with self.lock:
            if self.measurement_state is None:
                return
            state   = dict(self.measurement_state)
            samples = list(state["samples_m"])

        if len(samples) < OUTLIER_MIN_KEEP_COUNT:
            self.get_logger().warn(f"유효 샘플 부족 ({len(samples)}개). 재측정 대기.")
            with self.lock:
                self.measurement_state = None
            return

        filtered = self._remove_outliers(samples)

        if len(filtered) < OUTLIER_MIN_KEEP_COUNT:
            self.get_logger().warn(f"Outlier 제거 후 부족 ({len(filtered)}개). 재측정 대기.")
            with self.lock:
                self.measurement_state = None
            return

        avg_z = float(np.mean(filtered))

        # EMA 평활 (이전 측정 이력 반영)
        with self.lock:
            prev_ema = self.filtered_dist_ema
        if prev_ema is None:
            stable_z = avg_z
        else:
            stable_z = DISTANCE_EMA_ALPHA * avg_z + (1 - DISTANCE_EMA_ALPHA) * prev_ema

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
    # TF2 좌표 변환 및 amr_done 발행
    # ──────────────────────────────────────────────────────
    def _process_ready_target(self):
        """ready_target이 있으면 TF2로 맵 좌표 변환 후 amr_done 발행"""
        now = _time.monotonic()

        with self.lock:
            if self.ready_target is None:
                return
            if now - self.last_publish_time < PUBLISH_COOLDOWN_SEC:
                return
            K        = self.K.copy() if self.K is not None else None
            frame_id = self.depth_frame_id
            target   = dict(self.ready_target)

        if K is None or frame_id is None:
            self.get_logger().warn("CameraInfo 또는 depth frame_id 없음. 발행 스킵.")
            return

        try:
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            z      = target["z_stable"]
            x_rgb  = target["x_rgb"]
            y_rgb  = target["y_rgb"]

            # 픽셀 좌표 → 카메라 3D 좌표 (카메라 좌표계)
            X = (x_rgb - cx) * z / fx
            Y = (y_rgb - cy) * z / fy
            Z = z

            pt_cam = PointStamped()
            pt_cam.header.stamp    = Time().to_msg()
            pt_cam.header.frame_id = frame_id
            pt_cam.point.x = float(X)
            pt_cam.point.y = float(Y)
            pt_cam.point.z = float(Z)

            # TF2: 카메라 프레임 → map 프레임
            pt_map = self.tf_buffer.transform(
                pt_cam, "map",
                timeout=Duration(seconds=1.0)
            )

            map_x = pt_map.point.x
            map_y = pt_map.point.y

            # ── 주차 구역 필터링 ──────────────────────────
            zone = self._find_zone(map_x, map_y)
            if zone is None:
                self.get_logger().info(
                    f"[구역 외] ({map_x:.3f}, {map_y:.3f}) → 발행 스킵"
                )
                with self.lock:
                    self.ready_target = None
                return
            # ─────────────────────────────────────────────

            # amr1.py 인터페이스: "x,y" 형식
            msg = String()
            msg.data = f"{map_x:.3f},{map_y:.3f}"
            self.amr_done_pub.publish(msg)

            with self.lock:
                self.last_publish_time = now
                self.ready_target      = None

            self.get_logger().info(
                f"[amr_done] 발행 → ({map_x:.3f}, {map_y:.3f}) 구역{zone} | "
                f"depth={z:.2f}m, conf={target['conf']:.2f}, "
                f"cam=({X:.3f}, {Y:.3f}, {Z:.3f})"
            )

        except Exception as e:
            self.get_logger().warn(f"TF 변환 실패: {e}")
            # 실패 시 ready_target 초기화하여 재측정 유도
            with self.lock:
                self.ready_target = None

    # ──────────────────────────────────────────────────────
    # 유틸 함수
    # ──────────────────────────────────────────────────────
    def _find_zone(self, x: float, y: float) -> int | None:
        """맵 좌표 (x, y)가 PARKING_ZONES 중 하나에 속하면 구역 번호(1-based) 반환, 아니면 None"""
        for i, z in enumerate(PARKING_ZONES):
            if z["xmin"] <= x <= z["xmax"] and z["ymin"] <= y <= z["ymax"]:
                return i + 1
        return None

    def _decode_depth(self, msg: CompressedImage) -> np.ndarray | None:
        """compressedDepth 디코딩 (12바이트 헤더 스킵 fallback 포함)"""
        arr = np.frombuffer(msg.data, dtype=np.uint8)

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
        """(x, y) 중심 kernel_size × kernel_size ROI의 유효 depth 중앙값 (mm)"""
        h, w = depth.shape[:2]
        half = kernel_size // 2
        x1, x2 = max(0, x - half), min(w, x + half + 1)
        y1, y2 = max(0, y - half), min(h, y + half + 1)
        patch = depth[y1:y2, x1:x2]
        valid = patch[patch > 0]
        return float(np.median(valid)) if valid.size > 0 else 0.0

    def _scale_pixel(self, x_src, y_src, src_shape, dst_shape):
        """src 해상도 픽셀 → dst 해상도 픽셀 변환"""
        sh, sw = src_shape
        dh, dw = dst_shape
        if sw <= 0 or sh <= 0:
            return None
        xd = int(round(x_src * dw / sw))
        yd = int(round(y_src * dh / sh))
        return max(0, min(dw - 1, xd)), max(0, min(dh - 1, yd))

    def _remove_outliers(self, samples_m: list) -> list:
        """MAD 기반 outlier 제거 (my_test_3 방식)"""
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

        # 너무 많이 제거됐으면 trimmed mean fallback
        if filtered.size < OUTLIER_MIN_KEEP_COUNT:
            arr_sorted = np.sort(arr)
            trim = max(1, int(len(arr_sorted) * 0.1))
            filtered = arr_sorted[trim:-trim] if len(arr_sorted) > 2 * trim else arr_sorted

        return filtered.tolist()

    # ──────────────────────────────────────────────────────
    # GUI
    # ──────────────────────────────────────────────────────
    def _update_display_data(self):
        """타이머 기반 display_image 갱신 (GUI 스레드에서 표시)"""
        with self.lock:
            rgb    = self.rgb_image.copy()   if self.rgb_image   is not None else None
            depth  = self.depth_image.copy() if self.depth_image is not None else None
            det    = dict(self.last_detection)    if self.last_detection    is not None else None
            mstate = dict(self.measurement_state) if self.measurement_state is not None else None
            rtgt   = dict(self.ready_target)      if self.ready_target      is not None else None

        if rgb is None:
            return

        disp = rgb.copy()

        # BBox 표시
        if det is not None:
            x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
            color = (0, 255, 0)
            cv2.rectangle(disp, (x1, y1), (x2, y2), color, 2)
            cv2.putText(disp, f"car {det['conf']:.2f}",
                        (x1, max(25, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 측정 중심점 및 상태 텍스트
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
    node = CarDetectNode()
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
