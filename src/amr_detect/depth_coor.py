#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# ================================================================
#  depth_coor_ekf.py
#
#  이전 버전(depth_coor_improved.py)에서 KalmanFilter3D를
#  EKF3D로 완전 교체한 버전.
#
#  변경 사항 요약:
#    - KalmanFilter3D → EKF3D (관측 노이즈 R을 Z에 따라 동적 계산)
#    - 공분산 P Joseph form 업데이트 (수치 안정성)
#    - Mahalanobis 거리 기반 이상치 게이팅 (Chi-squared 기준)
#    - 나머지 파이프라인(ROI depth 샘플링, RGB-Depth 동기화,
#      Warm-up publish 조건)은 이전 버전과 동일하게 유지
#
#  논문 근거:
#    - "Robust Object Tracking for Mobile Robots Using Stereo Vision"
#      σ_z ∝ Z² (깊이 노이즈는 거리의 제곱에 비례)
#    - σ_xy ∝ Z  (횡방향 노이즈는 거리에 선형 비례)
#    - Chi-squared gating: 3-DOF, 95% 신뢰구간 = 7.815
# ================================================================

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


# ──────────────────────────────────────────
# 설정값 (기존과 동일)
# ──────────────────────────────────────────
ROBOT_NAMESPACE  = "/robot3"
MODEL_PATH       = "/home/rokey/click_car/models/amr.pt"

CONF_THRESHOLD   = 0.70
YOLO_IMG_SIZE    = 704

TOPIC_RGB        = f"{ROBOT_NAMESPACE}/oakd/rgb/image_raw/compressed"
TOPIC_DEPTH      = f"{ROBOT_NAMESPACE}/oakd/stereo/image_raw/compressedDepth"
TOPIC_INFO       = f"{ROBOT_NAMESPACE}/oakd/rgb/camera_info"
TOPIC_AMR_TARGET = f"{ROBOT_NAMESPACE}/amr_done"

WINDOW_NAME      = "Parking Detection"
PUBLISH_INTERVAL = 0.2

# 트래킹
IOU_THRESH       = 0.30
TRACK_TTL_SEC    = 1.0
SMOOTH_WINDOW    = 5
OUTLIER_THRESH_M = 0.35

# Depth 샘플링 (ROI 퍼센타일)
DEPTH_ROI_HEIGHT_RATIO = 0.25
DEPTH_ROI_WIDTH_RATIO  = 0.50
DEPTH_PERCENTILE       = 10

# RGB–Depth 동기화
MAX_DEPTH_AGE_SEC = 0.15

# ── EKF 파라미터 ──────────────────────────────
# 프로세스 노이즈 Q: 로봇 가속도 불확실성 (m/s²)
# 값이 클수록 필터가 측정값을 더 신뢰 (기동성 높은 물체면 올림)
EKF_ACCEL_STD    = 0.05   # 0.05 m/s² → 거의 정지한 주차 차량에 적합

# 관측 노이즈 계수 (논문 기반, OAK-D Pro 실측 근사)
# R은 매 측정마다 Z에 따라 동적으로 계산됨
EKF_SIGMA_XY_K   = 0.003  # σ_xy = K_xy × Z  (횡방향)
EKF_SIGMA_Z_K    = 0.005  # σ_z  = K_z  × Z² (깊이, 비선형)

# Mahalanobis 이상치 게이팅 임계값
# Chi-squared 분포 3-DOF, 95% 신뢰구간 = 7.815
# (95% 이상의 확률로 같은 물체라면 통과, 아니면 이상치로 차단)
EKF_GATE_CHI2    = 7.815

# Publish 신뢰도: history 몇 프레임 이상 쌓인 뒤 발행할지
MIN_HISTORY_TO_PUBLISH = 3


# ================================================================
#  EKF3D — Extended Kalman Filter (3D, Constant Velocity Model)
#
#  선형 KF와의 핵심 차이:
#    선형 KF : R = 고정 상수 행렬
#    EKF     : R = f(Z) — 측정할 때마다 Z(깊이)에 맞게 재계산
#
#  상태벡터 x = [px, py, pz, vx, vy, vz]  (6×1)
#  관측벡터 z = [px, py, pz]              (3×1)
#
#  비선형 요소:
#    관측 함수 h(x) = x[:3] 자체는 선형이지만
#    관측 노이즈 R(Z)이 상태(Z=pz)에 의존 → EKF 범주
# ================================================================
class EKF3D:
    def __init__(self, x0: float, y0: float, z0: float):
        # ── 상태 전이 행렬 F (등속도, dt는 predict() 시점에 갱신) ──
        self.F = np.eye(6, dtype=np.float64)
        # F의 상단 오른쪽 3×3 블록 = dt * I — predict()에서 채움

        # ── 관측 행렬 H (위치만 관측, 선형) ──
        self.H = np.zeros((3, 6), dtype=np.float64)
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = 1.0

        # ── 프로세스 노이즈 Q (이산화된 등가속도 모델, Singer 근사) ──
        # Q = diag([σ_a² · dt³/3, ..., σ_a² · dt, ...])
        # 여기서는 초기값으로 dt=0.1 기준으로 미리 세팅,
        # predict()에서 dt가 달라지면 재계산
        self._accel_var = EKF_ACCEL_STD ** 2
        self.Q = self._make_Q(dt=0.1)

        # ── 초기 상태 ──
        self.x = np.array([x0, y0, z0, 0.0, 0.0, 0.0], dtype=np.float64)

        # ── 초기 공분산 P ──
        # 위치: 측정 노이즈 수준, 속도: 크게 잡아 빠른 수렴 유도
        P_pos = self._make_R(z0)[0, 0]   # 첫 측정의 σ_xy² 정도
        P_vel = 1.0                        # 속도 초기 불확실성 (크게)
        self.P = np.diag([P_pos, P_pos, P_pos * 4,
                          P_vel, P_vel, P_vel]).astype(np.float64)

    # ── 내부 헬퍼 ──────────────────────────────────────
    def _make_Q(self, dt: float) -> np.ndarray:
        """
        Singer 이산화 프로세스 노이즈 행렬 (간략화)
        Q = σ_a² * [[dt³/3·I, dt²/2·I],
                     [dt²/2·I, dt·I   ]]
        """
        q  = self._accel_var
        q3 = q * dt**3 / 3.0
        q2 = q * dt**2 / 2.0
        q1 = q * dt

        Q = np.zeros((6, 6), dtype=np.float64)
        for i in range(3):
            Q[i,   i  ] = q3
            Q[i,   i+3] = q2
            Q[i+3, i  ] = q2
            Q[i+3, i+3] = q1
        return Q

    def _make_R(self, Z: float) -> np.ndarray:
        """
        논문 기반 동적 관측 노이즈 행렬
          σ_xy = K_xy × Z      (횡방향: 거리에 선형)
          σ_z  = K_z  × Z²     (깊이: 거리의 제곱, 비선형)

        Z가 클수록 R이 커짐 → 필터가 측정값을 덜 신뢰하고
        예측(모델)을 더 믿음 → 원거리에서 자동으로 스무딩 강화
        """
        Z = max(Z, 0.1)   # 0 나누기 방지
        sig_xy = EKF_SIGMA_XY_K * Z
        sig_z  = EKF_SIGMA_Z_K  * Z * Z
        return np.diag([sig_xy**2, sig_xy**2, sig_z**2])

    # ── predict / update ───────────────────────────────
    def predict(self, dt: float) -> np.ndarray:
        """
        예측 단계
          x̂ₖ₋ = F · x̂ₖ₋₁
          Pₖ₋  = F · Pₖ₋₁ · Fᵀ + Q
        """
        dt = max(dt, 1e-4)

        # F 갱신 (dt 반영)
        self.F[0, 3] = self.F[1, 4] = self.F[2, 5] = dt

        # Q 갱신 (dt 반영)
        self.Q = self._make_Q(dt)

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:3].copy()

    def update(self, z_meas: np.ndarray) -> np.ndarray:
        """
        보정 단계 (EKF update with Mahalanobis gating)

          혁신(innovation):  y = z - H·x̂
          혁신 공분산:       S = H·P·Hᵀ + R(Z)
          Mahalanobis 거리:  d² = yᵀ · S⁻¹ · y
            → d² > χ²₀.₉₅(3) = 7.815 이면 이상치로 폐기
          Kalman Gain:       K = P·Hᵀ·S⁻¹
          상태 갱신:         x = x̂ + K·y
          공분산 갱신(Joseph form):
            P = (I - KH)·P·(I - KH)ᵀ + K·R·Kᵀ
            ← 수치 오차 누적 시에도 P의 양반정치성(PSD) 보장
        """
        z = np.array(z_meas, dtype=np.float64)

        # 현재 깊이 추정값(Z)으로 R 동적 계산
        Z_est = max(self.x[2], 0.1)
        R = self._make_R(Z_est)

        # Innovation
        y = z - self.H @ self.x

        # 혁신 공분산
        S = self.H @ self.P @ self.H.T + R

        # ── Mahalanobis 이상치 게이팅 ──────────────────
        # d² = yᵀ S⁻¹ y : 측정값이 예측 분포 안에 있는지 확인
        # 물리적 의미: 예측 위치에서 측정값까지의 "통계적 거리"
        # 단순 유클리드 거리와 달리 방향별 불확실성을 고려함
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return self.x[:3].copy()   # S가 singular면 업데이트 건너뜀

        d2 = float(y @ S_inv @ y)

        if d2 > EKF_GATE_CHI2:
            # 95% 신뢰구간 밖 → 이상치로 판단, 예측 상태 유지
            return self.x[:3].copy()

        # ── Kalman Gain ─────────────────────────────────
        K = self.P @ self.H.T @ S_inv

        # ── 상태 갱신 ────────────────────────────────────
        self.x = self.x + K @ y

        # ── 공분산 갱신 (Joseph form) ────────────────────
        # 일반 형태: P = (I - KH)P
        # Joseph form: P = (I-KH)P(I-KH)ᵀ + KRKᵀ
        # → 수치 오차로 P가 비대칭/비PSD가 되는 현상 방지
        I_KH = np.eye(6) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

        return self.x[:3].copy()

    def get_position(self) -> np.ndarray:
        return self.x[:3].copy()

    def get_covariance_xyz(self) -> np.ndarray:
        """위치 공분산 (3×3) 반환 — 수렴 상태 모니터링용"""
        return self.P[:3, :3].copy()


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
        dt  = now - self.last_ekf_time
        self.det       = det
        self.last_seen = now

        if xyz_uv is not None:
            self.history.append(xyz_uv)
            x_m, y_m, z_m = xyz_uv[0], xyz_uv[1], xyz_uv[2]

            if self.ekf is None:
                self.ekf = EKF3D(x_m, y_m, z_m)
            else:
                self.ekf.predict(dt)
                self.ekf.update(np.array([x_m, y_m, z_m]))
        else:
            # 측정값 없어도 예측은 수행 (트랙 연속성 유지)
            if self.ekf is not None:
                self.ekf.predict(dt)

        self.last_ekf_time = now

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
        super().__init__("parking_detection_node")

        self.last_rgb_received  = None
        self.last_publish_time  = 0.0
        self.gui_enabled        = True
        self.latest_depth_frame = None
        self.latest_depth_time  = None
        self.camera_info        = None
        self._depth_lock        = threading.Lock()
        self.tracks             = []

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
        self.get_logger().info("Node ready (EKF mode).")

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
        self.get_logger().info(f"Sub RGB  : {TOPIC_RGB}")
        self.get_logger().info(f"Sub Depth: {TOPIC_DEPTH}")

    def _init_publisher(self):
        qos_pub = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             history=HistoryPolicy.KEEP_LAST, depth=10)
        self.amr_target_pub = self.create_publisher(String, TOPIC_AMR_TARGET, qos_pub)
        self.get_logger().info(f"Pub: {TOPIC_AMR_TARGET}")

    # ── 대기 화면 ─────────────────────────────────────────
    def _watchdog_timer(self):
        if self.last_rgb_received is None:
            self._draw_waiting_screen()

    def _draw_waiting_screen(self):
        if not self.gui_enabled:
            return
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(canvas, "Waiting for RGB topic...",
                    (40, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(canvas, TOPIC_RGB,
                    (40, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(canvas, TOPIC_DEPTH,
                    (40, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        status = "RGB RECEIVED" if self.last_rgb_received else "NO RGB FRAME"
        color  = (0, 255, 0) if self.last_rgb_received else (0, 0, 255)
        cv2.putText(canvas, status,
                    (40, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow(WINDOW_NAME, canvas)
        cv2.waitKey(1)

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

    # ── Depth 디코드 ─────────────────────────────────────
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

    # ── RGB 콜백 (메인 파이프라인) ───────────────────────
    def image_callback(self, msg: CompressedImage):
        self.last_rgb_received = time.monotonic()

        frame = cv2.imdecode(
            np.frombuffer(bytes(msg.data), dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
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
            conf=CONF_THRESHOLD, verbose=False)
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
        Z = depth_mm / 1000.0
        X = (u - self.camera_info["cx"]) * Z / self.camera_info["fx"]
        Y = (v - self.camera_info["cy"]) * Z / self.camera_info["fy"]
        return (X, Y, Z, u, v)

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

    # ── 발행 ─────────────────────────────────────────────
    def _publish_targets(self, smoothed_targets: list):
        now = time.monotonic()
        if now - self.last_publish_time < PUBLISH_INTERVAL:
            return

        parts, published = [], 0

        for idx, (trk_det, smoothed) in enumerate(smoothed_targets, 1):
            if smoothed is None:
                parts.append(f"car{idx}=NO_DEPTH")
                continue

            # 대응 트랙 찾기
            trk = next((t for t in self.tracks if t.det is trk_det), None)

            # Warm-up 중인 트랙은 발행 억제
            if trk is not None and len(trk.history) < MIN_HISTORY_TO_PUBLISH:
                parts.append(
                    f"car{idx}=WARMUP"
                    f"({len(trk.history)}/{MIN_HISTORY_TO_PUBLISH})")
                continue

            x, y, z, _, _ = smoothed
            std_str = trk.std_xyz if trk else "N/A"

            msg = String()
            msg.data = f"{x:.3f},{y:.3f},{z:.3f}"
            self.amr_target_pub.publish(msg)
            published += 1
            parts.append(f"car{idx}=({x:.3f},{y:.3f},{z:.3f}) {std_str}")

        self.get_logger().info(
            f"[PUB] n={len(smoothed_targets)} pub={published} | "
            + " | ".join(parts))
        self.last_publish_time = now

    # ── 시각화 ───────────────────────────────────────────
    def _draw(self, frame: np.ndarray, smoothed_targets: list):
        if not self.gui_enabled:
            return

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