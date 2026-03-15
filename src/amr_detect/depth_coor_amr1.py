#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# depth_coor.py
# AMR1 불법주정차 탐지 노드
# - YOLOv8 + EKF3D + odom→map 좌표 변환 + 불법주정차 구역 판별

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
from nav_msgs.msg import Odometry
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
from ultralytics import YOLO


# ──────────────────────────────────────────
# 설정값
# ──────────────────────────────────────────
ROBOT_NAMESPACE  = "/robot3"
MODEL_PATH       = "/home/kyb/click_car/models/amr.pt"

CONF_THRESHOLD   = 0.70
YOLO_IMG_SIZE    = 704

TOPIC_RGB        = f"{ROBOT_NAMESPACE}/oakd/rgb/image_raw/compressed"
TOPIC_DEPTH      = f"{ROBOT_NAMESPACE}/oakd/stereo/image_raw/compressedDepth"
TOPIC_INFO       = f"{ROBOT_NAMESPACE}/oakd/rgb/camera_info"
TOPIC_AMR_TARGET = f"{ROBOT_NAMESPACE}/amr_done"
TOPIC_ODOM       = f"{ROBOT_NAMESPACE}/odom"

# 카메라 → 로봇 베이스 오프셋 (단위: m, 실측값)
CAM_OFFSET_X = -0.10
CAM_OFFSET_Y =  0.00
CAM_OFFSET_Z =  0.25

WINDOW_NAME      = "Parking Detection - AMR1(robot3)"
PUBLISH_INTERVAL = 0.2

# 트래킹
IOU_THRESH       = 0.5
TRACK_TTL_SEC    = 1.0
SMOOTH_WINDOW    = 5
OUTLIER_THRESH_M = 0.35

# Depth 샘플링
DEPTH_ROI_HEIGHT_RATIO = 0.25
DEPTH_ROI_WIDTH_RATIO  = 0.50
DEPTH_PERCENTILE       = 10

# RGB-Depth 동기화
MAX_DEPTH_AGE_SEC = 0.15

# ROI 필터: 화면 상단 30%는 car가 없는 영역
ROI_TOP_RATIO = 0.30

# 회전 억제: 0.10 rad/s 이상이면 회전 중으로 판단
ROTATE_SUPPRESS_THRESH = 0.10
ANGULAR_AVG_WINDOW     = 5

# 불법주정차 구역 (map 좌표계, x_min, x_max, y_min, y_max)
ILLEGAL_ZONES = [
    (-2.04, -0.694,  0.0,  1.850),  # 구역1
    (-1.240,  0.504, -5.000, -4.300),  # 구역2
]

# EKF 파라미터
EKF_Q_STATIC     = 1e-4
EKF_SIGMA_XY_K   = 0.003
EKF_SIGMA_Z_K    = 0.005
EKF_GATE_CHI2    = 7.815

MIN_HISTORY_TO_PUBLISH = 5



# ================================================================
#  EKF3D — 정지 물체 위치 필터 (3-state: px, py, pz)
#  F=I (정지 모델), 관측 노이즈 R은 깊이 Z에 따라 동적 계산
# ================================================================
class EKF3D:
    def __init__(self, x0: float, y0: float, z0: float):
        self.x = np.array([x0, y0, z0], dtype=np.float64)
        self.F = np.eye(3, dtype=np.float64)
        self.H = np.eye(3, dtype=np.float64)
        self.Q = np.eye(3, dtype=np.float64) * EKF_Q_STATIC
        self.P = self._make_R(z0).copy()

    def _make_R(self, Z: float) -> np.ndarray:
        Z = max(Z, 0.1)
        sig_xy = EKF_SIGMA_XY_K * Z
        sig_z  = EKF_SIGMA_Z_K  * Z * Z
        return np.diag([sig_xy**2, sig_xy**2, sig_z**2])

    def predict(self) -> np.ndarray:
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, z_meas: np.ndarray) -> np.ndarray:
        z     = np.array(z_meas, dtype=np.float64)
        R     = self._make_R(max(self.x[2], 0.1))
        y     = z - self.H @ self.x
        S     = self.H @ self.P @ self.H.T + R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return self.x.copy()
        if float(y @ S_inv @ y) > EKF_GATE_CHI2:
            return self.x.copy()
        K        = self.P @ self.H.T @ S_inv
        self.x   = self.x + K @ y
        I_KH     = np.eye(3) - K @ self.H
        self.P   = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        return self.x.copy()

    def get_position(self) -> np.ndarray:
        return self.x.copy()

    def get_covariance_xyz(self) -> np.ndarray:
        return self.P.copy()


# ================================================================
#  Track — EKF3D 통합 버전
# ================================================================
class Track:
    def __init__(self, det: dict, xyz_uv=None):
        now = time.monotonic()
        self.det        = det
        self.created_at = now
        self.last_seen  = now
        self.last_ekf_time = now
        self.history    = deque(maxlen=SMOOTH_WINDOW)
        self.ekf: EKF3D | None = None

        if xyz_uv is not None:
            self.history.append(xyz_uv)
            self.ekf = EKF3D(xyz_uv[0], xyz_uv[1], xyz_uv[2])

    def update(self, det: dict, xyz_uv=None):
        now = time.monotonic()
        self.det       = det
        self.last_seen = now

        if xyz_uv is not None:
            self.history.append(xyz_uv)
            x_m, y_m, z_m = xyz_uv[0], xyz_uv[1], xyz_uv[2]

            if self.ekf is None:
                self.ekf = EKF3D(x_m, y_m, z_m)
            else:
                self.ekf.predict()                              # dt 불필요 (정지 모델)
                self.ekf.update(np.array([x_m, y_m, z_m]))
        else:
            # 측정값 없어도 predict는 수행 (P가 소폭 증가 → 다음 측정에 더 열린 자세)
            if self.ekf is not None:
                self.ekf.predict()

    def is_alive(self) -> bool:
        return (time.monotonic() - self.last_seen) <= TRACK_TTL_SEC

    def get_smoothed_xyz_uv(self):
        """
        EKF 추정값 반환 (history가 충분히 쌓인 경우)
        EKF 미초기화 시 median fallback
        """
        # ── EKF 추정값 (우선) ──────────────────────────
        if self.ekf is not None and len(self.history) >= MIN_HISTORY_TO_PUBLISH:
            pos = self.ekf.get_position()
            arr = np.array(self.history, dtype=np.float32)
            uv  = np.median(arr[:, 3:], axis=0)
            return (float(pos[0]), float(pos[1]), float(pos[2]),
                    int(np.round(uv[0])), int(np.round(uv[1])))

        # ── Median fallback (history 부족 시) ──────────
        if len(self.history) == 0:
            return None

        arr     = np.array(self.history, dtype=np.float32)
        xyz     = arr[:, :3]
        uv      = arr[:, 3:]
        med     = np.median(xyz, axis=0)
        dists   = np.linalg.norm(xyz - med, axis=1)
        keep    = dists <= OUTLIER_THRESH_M
        f_xyz   = xyz[keep] if keep.any() else xyz
        f_uv    = uv[keep]  if keep.any() else uv
        fin_xyz = np.median(f_xyz, axis=0)
        fin_uv  = np.median(f_uv,  axis=0)

        return (float(fin_xyz[0]), float(fin_xyz[1]), float(fin_xyz[2]),
                int(np.round(fin_uv[0])), int(np.round(fin_uv[1])))

    @property
    def std_xyz(self) -> str:
        """로그용: EKF 위치 표준편차 (cm 단위)"""
        if self.ekf is None:
            return "N/A"
        cov = self.ekf.get_covariance_xyz()
        std = np.sqrt(np.diag(cov)) * 100  # m → cm
        return f"σ=({std[0]:.1f},{std[1]:.1f},{std[2]:.1f})cm"


# ================================================================
#  ParkingDetectionNode
# ================================================================
class ParkingDetectionNode(Node):
    def __init__(self):
        super().__init__("parking_detection_node_amr2")

        self.last_publish_time  = 0.0
        self.gui_enabled        = True
        self.latest_depth_frame = None
        self.latest_depth_time  = None
        self.camera_info        = None
        self._depth_lock        = threading.Lock()
        self.tracks             = []

        self._odom_x         = None
        self._odom_y         = None
        self._odom_yaw       = None
        self._odom_angular_z = 0.0
        self._angular_z_buf  = deque(maxlen=ANGULAR_AVG_WINDOW)

        self._tf_buffer   = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

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

        self.get_logger().info("Node ready.")

    # ── 초기화 ───────────────────────────────────────────
    def _load_model(self):
        self.model = YOLO(MODEL_PATH)
        self.model.predict(
            source=np.zeros((YOLO_IMG_SIZE, YOLO_IMG_SIZE, 3), dtype=np.uint8),
            imgsz=YOLO_IMG_SIZE, verbose=False)
        self.get_logger().info(f"Model classes: {self.model.names}")
        self.get_logger().info("YOLO warm-up done.")

    def _init_subscriber(self):
        qos_be = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                            history=HistoryPolicy.KEEP_LAST, depth=1)
        qos_rel = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(CompressedImage, TOPIC_RGB,   self.image_callback, qos_be)
        self.create_subscription(CompressedImage, TOPIC_DEPTH, self.depth_callback, qos_be)
        self.create_subscription(CameraInfo,      TOPIC_INFO,  self.info_callback,  qos_rel)
        self.create_subscription(Odometry,        TOPIC_ODOM,  self.odom_callback,  qos_be)
        self.get_logger().info(f"Sub RGB  : {TOPIC_RGB}")
        self.get_logger().info(f"Sub Depth: {TOPIC_DEPTH}")
        self.get_logger().info(f"Sub Odom : {TOPIC_ODOM}")

    def _init_publisher(self):
        qos_pub = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             history=HistoryPolicy.KEEP_LAST, depth=10)
        self.amr_target_pub = self.create_publisher(String, TOPIC_AMR_TARGET, qos_pub)
        self.get_logger().info(f"Pub: {TOPIC_AMR_TARGET}")

    # ── CameraInfo ───────────────────────────────────────
    def info_callback(self, msg: CameraInfo):
        if self.camera_info is not None:
            return
        self.camera_info = {
            "fx": msg.k[0], "fy": msg.k[4],
            "cx": msg.k[2], "cy": msg.k[5],
            "width": msg.width, "height": msg.height,
        }
        self.get_logger().info(
            f"CameraInfo: fx={self.camera_info['fx']:.1f} "
            f"fy={self.camera_info['fy']:.1f} "
            f"cx={self.camera_info['cx']:.1f} "
            f"cy={self.camera_info['cy']:.1f} "
            f"res={msg.width}x{msg.height}")

    # ── Odometry ─────────────────────────────────────────
    def odom_callback(self, msg: Odometry):
        """
        AMR의 현재 world frame 위치/방향을 저장.
        quaternion → yaw 변환 (2D 이동이므로 yaw만 사용).
        """
        self._odom_x = msg.pose.pose.position.x
        self._odom_y = msg.pose.pose.position.y

        # ── angular.z 이동 평균 ──
        # TurtleBot4 wheel encoder odom은 순간값이 노이즈성으로 튀는 경우가 있음.
        # 순간값 그대로 쓰면 실제로 조금만 회전해도 임계값을 넘어 차단될 수 있음.
        # → 최근 N프레임 평균으로 안정화
        raw_angular_z = msg.twist.twist.angular.z
        self._angular_z_buf.append(raw_angular_z)
        self._odom_angular_z = float(np.mean(self._angular_z_buf))

        # quaternion → yaw
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._odom_yaw = float(np.arctan2(siny_cosp, cosy_cosp))

    def _camera_to_world(self, cam_x: float, cam_y: float, cam_z: float):
        """
        카메라 좌표계 → world(odom) 좌표계 변환.

        카메라 좌표계 (OAK-D, ROS convention):
          X = 오른쪽, Y = 아래, Z = 앞(depth)

        로봇 베이스 좌표계 (ROS convention):
          X = 앞, Y = 왼쪽, Z = 위

        카메라 → 로봇 베이스 변환:
          base_x =  cam_z + CAM_OFFSET_X   (depth → 앞방향)
          base_y = -cam_x + CAM_OFFSET_Y   (오른쪽 → 왼쪽 반전)
          base_z = -cam_y + CAM_OFFSET_Z   (아래 → 위 반전, 높이 오프셋)

        로봇 베이스 → world(odom) 변환:
          world_x = odom_x + base_x*cos(yaw) - base_y*sin(yaw)
          world_y = odom_y + base_x*sin(yaw) + base_y*cos(yaw)

        odom이 없으면 None 반환 → 상대 좌표 fallback.
        """
        if self._odom_x is None:
            return None   # odom 미수신 → fallback

        # ① camera → robot base
        base_x =  cam_z + CAM_OFFSET_X
        base_y = -cam_x + CAM_OFFSET_Y
        base_z = -cam_y + CAM_OFFSET_Z

        # ② robot base → world (2D rotation, yaw만 사용)
        cos_y = np.cos(self._odom_yaw)
        sin_y = np.sin(self._odom_yaw)
        world_x = self._odom_x + base_x * cos_y - base_y * sin_y
        world_y = self._odom_y + base_x * sin_y + base_y * cos_y
        world_z = base_z   # 높이는 회전 불필요

        return (world_x, world_y, world_z)

    def _odom_to_map(self, ox: float, oy: float, oz: float):
        """
        odom 좌표 → map 좌표 변환 (TF2 사용).

        SLAM localization 실행 중이면 map→odom TF가 발행되므로
        이를 역으로 적용해 odom 좌표를 map 좌표로 변환.

        TF 없을 때(localization 미실행):
          → None 반환 → 호출부에서 odom 좌표 그대로 fallback
        """
        try:
            # map→odom TF를 가져와서 odom→map 방향으로 적용
            t = self._tf_buffer.lookup_transform(
                "map", "odom",
                rclpy.time.Time(),          # 최신 TF
                timeout=rclpy.duration.Duration(seconds=0.05))

            # TF에서 translation + quaternion 추출
            tx = t.transform.translation.x
            ty = t.transform.translation.y
            tz = t.transform.translation.z
            qx = t.transform.rotation.x
            qy = t.transform.rotation.y
            qz = t.transform.rotation.z
            qw = t.transform.rotation.w

            # quaternion → yaw (2D 변환, roll/pitch 무시)
            siny = 2.0 * (qw * qz + qx * qy)
            cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
            yaw  = float(np.arctan2(siny, cosy))

            # odom → map: 회전 후 translation 더하기
            cos_y = np.cos(yaw)
            sin_y = np.sin(yaw)
            map_x = tx + ox * cos_y - oy * sin_y
            map_y = ty + ox * sin_y + oy * cos_y
            map_z = tz + oz

            return (map_x, map_y, map_z)

        except (LookupException, ConnectivityException, ExtrapolationException):
            # TF 없음 → localization 미실행 상태
            return None


    def _decode_compressed_depth(self, msg: CompressedImage):
        try:
            data = bytes(msg.data)
            raw  = np.frombuffer(data, dtype=np.uint8)

            img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
            if img is not None and img.size > 0:
                return img

            for sig in (b'PNG', b'\x89PNG\r\n\x1a\n'):
                idx = data.find(sig)
                if idx > 0:
                    img = cv2.imdecode(
                        np.frombuffer(data[idx:], dtype=np.uint8),
                        cv2.IMREAD_UNCHANGED)
                    if img is not None and img.size > 0:
                        return img
            return None
        except Exception as e:
            self.get_logger().warn(f"[DEPTH-DECODE] {e}")
            return None

    def depth_callback(self, msg: CompressedImage):
        try:
            img = self._decode_compressed_depth(msg)
            if img is None or img.size == 0:
                return
            with self._depth_lock:
                self.latest_depth_frame = img
                self.latest_depth_time  = time.monotonic()
        except Exception as e:
            self.get_logger().warn(f"[DEPTH-CB] {e}")

    def image_callback(self, msg: CompressedImage):
        frame = cv2.imdecode(
            np.frombuffer(bytes(msg.data), dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return

        # 회전 억제 — 탐지 스킵, TTL 카운트는 유지
        if abs(self._odom_angular_z) > ROTATE_SUPPRESS_THRESH:
            self.get_logger().info(
                f"[ROT] 회전 중 탐지 억제 (angular_z={self._odom_angular_z:.3f})",
                throttle_duration_sec=1.0)
            self._update_tracks([])
            self._draw_rotating(frame)
            return

        car_dets = self._detect_cars(frame)

        # Depth 스냅샷 — 나이 체크 포함
        with self._depth_lock:
            depth_snap = None
            if (self.latest_depth_frame is not None
                    and self.latest_depth_time is not None):
                age = time.monotonic() - self.latest_depth_time
                if age <= MAX_DEPTH_AGE_SEC:
                    depth_snap = self.latest_depth_frame.copy()
                else:
                    self.get_logger().warn(
                        f"[SYNC] Depth too old ({age*1000:.0f} ms), skipped.")

        measurements = [(det, self._get_xyz_from_bbox_roi(det, depth_snap))
                        for det in car_dets]

        self._update_tracks(measurements)

        smoothed = [(trk.det, trk.get_smoothed_xyz_uv()) for trk in self.tracks]

        self._publish_targets(smoothed)
        self._draw(frame, smoothed)

    # ── YOLO 탐지 ────────────────────────────────────────
    def _detect_cars(self, frame: np.ndarray) -> list:
        results = self.model.predict(
            source=frame, imgsz=YOLO_IMG_SIZE,
            conf=CONF_THRESHOLD, iou=0.45,
            agnostic_nms=True, verbose=False)
        cars = []
        if not results:
            return cars

        h_frame   = frame.shape[0]
        roi_top   = int(h_frame * ROI_TOP_RATIO)

        for box in results[0].boxes:
            cls_id = int(box.cls[0].item())
            if self.model.names.get(cls_id) != "car":
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            if y2 <= roi_top:
                continue
            cars.append({
                "conf": float(box.conf[0].item()),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "area": max(1, (x2-x1)*(y2-y1)),
            })

        cars = self._nms(cars, iou_thresh=0.45)
        cars.sort(key=lambda d: (d["x1"] + d["x2"]) // 2)
        return cars

    @staticmethod
    def _nms(dets: list, iou_thresh: float = 0.45) -> list:
        """
        conf 내림차순 정렬 후 IoU 기반 greedy NMS.
        YOLO 내장 NMS가 처리하지 못한 중복 박스 제거용.
        """
        if len(dets) <= 1:
            return dets

        dets_sorted = sorted(dets, key=lambda d: d["conf"], reverse=True)
        keep = []
        while dets_sorted:
            best = dets_sorted.pop(0)
            keep.append(best)
            remaining = []
            for d in dets_sorted:
                ix = max(0, min(best["x2"], d["x2"]) - max(best["x1"], d["x1"]))
                iy = max(0, min(best["y2"], d["y2"]) - max(best["y1"], d["y1"]))
                inter = ix * iy
                union = best["area"] + d["area"] - inter
                iou   = inter / max(1, union)
                if iou < iou_thresh:
                    remaining.append(d)
            dets_sorted = remaining
        return keep

    # ── Depth ROI 샘플링 ─────────────────────────────────
    def _get_depth_mm_from_bbox_roi(
            self, det: dict, depth_frame: np.ndarray) -> int | None:
        h_img, w_img = depth_frame.shape[:2]
        bh = det["y2"] - det["y1"]
        bw = det["x2"] - det["x1"]

        ry1 = max(0, int(det["y2"] - bh * DEPTH_ROI_HEIGHT_RATIO))
        ry2 = min(h_img, det["y2"])
        rx1 = max(0, int(det["x1"] + bw * (0.5 - DEPTH_ROI_WIDTH_RATIO / 2)))
        rx2 = min(w_img, int(det["x1"] + bw * (0.5 + DEPTH_ROI_WIDTH_RATIO / 2)))

        if ry2 <= ry1 or rx2 <= rx1:
            return None

        roi   = depth_frame[ry1:ry2, rx1:rx2]
        valid = roi[roi > 0].astype(np.float32)
        if valid.size < 5:
            return None

        return int(np.percentile(valid, DEPTH_PERCENTILE))

    def _get_xyz_from_bbox_roi(self, det: dict,
                               depth_frame: np.ndarray | None):
        if self.camera_info is None or depth_frame is None:
            return None

        depth_mm = self._get_depth_mm_from_bbox_roi(det, depth_frame)
        if depth_mm is None:
            return None

        u = (det["x1"] + det["x2"]) // 2
        v = det["y2"]

        # ── 카메라 좌표계 (상대 좌표) ──
        cam_z = depth_mm / 1000.0
        cam_x = (u - self.camera_info["cx"]) * cam_z / self.camera_info["fx"]
        cam_y = (v - self.camera_info["cy"]) * cam_z / self.camera_info["fy"]

        # ── camera → odom 좌표 변환 ──
        odom_xyz = self._camera_to_world(cam_x, cam_y, cam_z)
        if odom_xyz is None:
            self.get_logger().warn("[ODOM] Not received yet, using camera-relative coords.",
                                   throttle_duration_sec=5.0)
            return (cam_x, cam_y, cam_z, u, v)

        # ── odom → map 좌표 변환 (TF2) ──
        # SLAM localization 실행 중이면 map 좌표로 변환
        # localization 미실행 시 odom 좌표 그대로 fallback
        map_xyz = self._odom_to_map(odom_xyz[0], odom_xyz[1], odom_xyz[2])
        if map_xyz is not None:
            return (map_xyz[0], map_xyz[1], map_xyz[2], u, v)
        else:
            # TF 없음 → odom 좌표 그대로 사용 (fallback)
            self.get_logger().warn("[TF] map→odom TF 없음, odom 좌표 사용 (localization 실행 필요)",
                                   throttle_duration_sec=5.0)
            return (odom_xyz[0], odom_xyz[1], odom_xyz[2], u, v)

    # ── IoU 트래킹 ───────────────────────────────────────
    @staticmethod
    def _iou(a: dict, b: dict) -> float:
        ix = max(0, min(a["x2"], b["x2"]) - max(a["x1"], b["x1"]))
        iy = max(0, min(a["y2"], b["y2"]) - max(a["y1"], b["y1"]))
        inter = ix * iy
        if inter == 0:
            return 0.0
        return inter / (max(1, a["area"]) + max(1, b["area"]) - inter)

    def _update_tracks(self, measurements: list):
        self.tracks = [t for t in self.tracks if t.is_alive()]
        matched_t, matched_m = set(), set()

        for ti, trk in enumerate(self.tracks):
            best_iou, best_mi = 0.0, -1
            for mi, (det, _) in enumerate(measurements):
                if mi in matched_m:
                    continue
                iou = self._iou(trk.det, det)
                if iou > best_iou:
                    best_iou, best_mi = iou, mi
            if best_iou >= IOU_THRESH:
                det, xyz_uv = measurements[best_mi]
                trk.update(det, xyz_uv)
                matched_t.add(ti)
                matched_m.add(best_mi)

        for mi, (det, xyz_uv) in enumerate(measurements):
            if mi not in matched_m:
                self.tracks.append(Track(det, xyz_uv))

    # ── 불법주정차 구역 판별 ─────────────────────────────
    @staticmethod
    def _in_illegal_zone(x: float, y: float) -> bool:
        """
        map 좌표 (x, y)가 ILLEGAL_ZONES 중 하나라도 포함되면 True.
        구역은 직사각형 bbox (x_min, x_max, y_min, y_max) 로 정의.
        """
        for x_min, x_max, y_min, y_max in ILLEGAL_ZONES:
            if x_min <= x <= x_max and y_min <= y <= y_max:
                return True
        return False

    def _publish_targets(self, smoothed_targets: list):
        now = time.monotonic()
        if now - self.last_publish_time < PUBLISH_INTERVAL:
            return

        published = 0

        for idx, (trk_det, smoothed) in enumerate(smoothed_targets, 1):
            if smoothed is None:
                continue

            # 대응 트랙 찾기
            trk = next((t for t in self.tracks if t.det is trk_det), None)

            # Warm-up 중인 트랙은 발행 억제
            if trk is not None and len(trk.history) < MIN_HISTORY_TO_PUBLISH:
                continue

            x, y, z, _, _ = smoothed

            # ── 탐지 좌표 로그 (항상 출력) ──
            self.get_logger().info(
                f"[DET] car{idx} → x={x:.3f}, y={y:.3f}")

            # ── 불법주정차 구역 체크 ──
            if not self._in_illegal_zone(x, y):
                continue

            # ── 구역 내 차량: 노란색(WARN) 로그 + publish ──
            self.get_logger().warn(
                f"\033[33m[PUB] car{idx} IN_ZONE → x={x:.3f}, y={y:.3f}\033[0m")

            msg = String()
            msg.data = f"{x:.3f},{y:.3f}"
            self.amr_target_pub.publish(msg)
            published += 1

        self.last_publish_time = now

    # ── 회전 중 화면 표시 ────────────────────────────────
    def _draw_rotating(self, frame: np.ndarray):
        """회전 억제 중임을 화면에 표시 (탐지 결과 없음)"""
        if not self.gui_enabled:
            return
        overlay = frame.copy()
        cv2.putText(overlay, f"ROTATING — detection suppressed",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        cv2.putText(overlay, f"angular_z={self._odom_angular_z:.3f} rad/s",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        cv2.imshow(WINDOW_NAME, overlay)
        cv2.waitKey(1)

    # ── 시각화 ───────────────────────────────────────────
    def _draw(self, frame: np.ndarray, smoothed_targets: list):
        if not self.gui_enabled:
            return

        # ── ROI 경계선 표시 (상단 제외 영역) ──
        h_frame = frame.shape[0]
        roi_top_px = int(h_frame * ROI_TOP_RATIO)
        cv2.line(frame, (0, roi_top_px), (frame.shape[1], roi_top_px),
                 (0, 165, 255), 1)   # 주황색 점선 느낌
        cv2.putText(frame, f"ROI top ({int(ROI_TOP_RATIO*100)}%)",
                    (5, roi_top_px - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)

        for idx, (det, smoothed) in enumerate(smoothed_targets, 1):
            x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if smoothed is None:
                label = "X=? Y=? Z=?"
            else:
                x, y, z, u, v = smoothed
                cv2.circle(frame, (u, v), 5, (0, 0, 255), -1)
                label = f"X={x:.2f} Y={y:.2f} Z={z:.2f}"

                # EKF σ 표시 (수렴 확인용)
                trk = next(
                    (t for t in self.tracks if t.det is det), None)
                if trk is not None:
                    cv2.putText(frame, trk.std_xyz,
                                (x1, max(65, y1 + 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                                (200, 200, 0), 1)

            cv2.putText(frame, f"car{idx} {det['conf']:.2f}",
                        (x1, max(25, y1 - 40)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, label,
                        (x1, max(45, y1 - 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 255), 2)

        cv2.putText(frame, f"Cars: {len(smoothed_targets)}  [EKF]",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow(WINDOW_NAME, frame)
        cv2.waitKey(1)

    # ── 종료 ─────────────────────────────────────────────
    def destroy_node(self):
        if self.gui_enabled:
            cv2.destroyAllWindows()
        super().destroy_node()


# ================================================================
#  main
# ================================================================
def main(args=None):
    # TransformListener는 기본적으로 /tf, /tf_static을 구독.
    # TurtleBot4 네임스페이스 환경에서는 /robot3/tf, /robot3/tf_static으로
    # TF가 발행되므로 cli_args remapping으로 연결.
    remap_args = [
        "--ros-args",
        "-r", f"/tf:={ROBOT_NAMESPACE}/tf",
        "-r", f"/tf_static:={ROBOT_NAMESPACE}/tf_static",
    ]
    rclpy.init(args=remap_args)
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