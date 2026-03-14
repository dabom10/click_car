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
from nav_msgs.msg import Odometry
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
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
TOPIC_AMR_TARGET = f"{ROBOT_NAMESPACE}amr_done"
TOPIC_ODOM       = f"{ROBOT_NAMESPACE}/odom"   # world frame 변환용

# ── 카메라 → 로봇 베이스 오프셋 (단위: m) ──────────────
# OAK-D Lite가 로봇 중심에서 앞쪽으로 얼마나 떨어져 있는지.
# TurtleBot4 기준 대략적인 값 — 실측 후 교체 권장.
# 실측 방법: 줄자로 로봇 중심(회전축)에서 카메라 렌즈까지 측정
CAM_OFFSET_X = -0.10  # 로봇 중심 기준 앞뒤 (m) — 뒤쪽이면 음수, 실측 -10cm
CAM_OFFSET_Y =  0.00  # 로봇 중심 기준 좌우 (m) — 정중앙
CAM_OFFSET_Z =  0.25  # 지면에서 카메라 렌즈까지 높이 (m) — 실측 25cm

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

# ── ROI 필터 ──────────────────────────────────
# 카메라 뷰 상단 30%는 car가 절대 나올 수 없는 영역 (천장/배경)
# → 해당 영역에 bbox 중심이 있으면 오탐으로 간주하고 무시
ROI_TOP_RATIO = 0.30   # 0.0~1.0, 상단 몇 % 를 제외할지

# ── 회전 억제 ──────────────────────────────────
# angular.z 실측 분석 결과:
#   직진/정지 노이즈: 최대 0.041 rad/s
#   실제 회전 구간:   0.10 rad/s 이상
# TurtleBot4 odom은 wheel encoder 기반으로 순간값이 노이즈성으로 튀는 경우 있음
# → 이동 평균(ANGULAR_AVG_WINDOW 프레임)으로 안정화 후 임계값 비교
ROTATE_SUPPRESS_THRESH = 0.10   # rad/s (평균값 기준)
ANGULAR_AVG_WINDOW     = 5      # 평균 낼 프레임 수 (odom ~50Hz → 약 0.1초)

# ── Hard Negative 자동 저장 ───────────────────
# ROI 필터에 걸린 오탐 프레임을 원본 이미지로 저장
# → 나중에 라벨링 후 재학습 데이터로 활용
HARD_NEG_SAVE       = True    # False로 바꾸면 저장 안 함
HARD_NEG_SAVE_DIR   = "/home/rokey/click_car/hard_negatives"
HARD_NEG_INTERVAL   = 3.0     # 같은 오탐이 연속으로 저장되는 것 방지 (초)

# ── EKF 파라미터 ──────────────────────────────
# 프로세스 노이즈 Q: 정지 모델용 (매우 작게)
# 값이 작을수록 "물체가 절대 안 움직인다"고 강하게 가정
# 너무 작으면 초기 수렴이 느려지므로 1e-4 권장
EKF_Q_STATIC     = 1e-4

# 관측 노이즈 계수 (논문 기반, OAK-D Pro 실측 근사)
EKF_SIGMA_XY_K   = 0.003  # σ_xy = K_xy × Z  (횡방향)
EKF_SIGMA_Z_K    = 0.005  # σ_z  = K_z  × Z² (깊이, 비선형)

# Mahalanobis 이상치 게이팅 임계값
# Chi-squared 분포 3-DOF, 95% 신뢰구간 = 7.815
EKF_GATE_CHI2    = 7.815

# Publish 신뢰도: history 몇 프레임 이상 쌓인 뒤 발행할지
MIN_HISTORY_TO_PUBLISH = 3


# ================================================================
#  EKF3D — Static Object Position Filter
#
#  [이전 버전과의 핵심 차이]
#  이전: CV 모델 (Constant Velocity), 상태벡터 [px,py,pz,vx,vy,vz] 6-state
#  현재: 정지 모델 (Static),          상태벡터 [px,py,pz]           3-state
#
#  CV 모델이 정지 물체에 드리프트를 유발하는 이유:
#    depth 노이즈(±3cm)가 매 프레임 조금씩 다른 방향으로 발생하면
#    CV 모델은 이를 "실제 이동"으로 해석해 vz에 속도를 학습함.
#    로봇·차 모두 정지 상태여도 vz ≠ 0이 되면
#    predict()마다 pz += vz × dt 로 계속 밀려 드리프트 발생.
#
#  정지 모델(F = I)은 "물체가 이전 위치 그대로" 라고만 예측.
#    → 속도 성분 자체가 없으므로 드리프트 원천 차단.
#    → 불법 주차 차량처럼 정지한 물체에 최적.
#
#  EKF 범주를 유지하는 이유:
#    관측 노이즈 R(Z)이 깊이 Z에 비선형적으로 의존하기 때문.
#    (선형 KF는 R을 고정 상수로만 쓸 수 있음)
# ================================================================
class EKF3D:
    def __init__(self, x0: float, y0: float, z0: float):
        # ── 상태벡터: 위치만 (3-state) ──
        self.x = np.array([x0, y0, z0], dtype=np.float64)

        # ── 상태 전이 행렬 F = I (정지 모델: "다음도 지금과 같은 위치") ──
        # CV 모델의 F에는 dt 블록이 있어 속도를 위치에 더했음.
        # F = I 이면 predict()에서 x = F @ x = x → 위치 불변.
        self.F = np.eye(3, dtype=np.float64)

        # ── 관측 행렬 H = I (위치를 직접 관측) ──
        self.H = np.eye(3, dtype=np.float64)

        # ── 프로세스 노이즈 Q ──
        # 실제로 차가 안 움직이므로 Q는 매우 작게.
        # Q가 작을수록 필터가 모델(정지)을 강하게 신뢰 → 안정적.
        # Q가 너무 0이면 P가 수렴 후 새 측정값을 완전히 무시하므로
        # 아주 작은 값(1e-4)을 유지해 적응성 보존.
        self.Q = np.eye(3, dtype=np.float64) * EKF_Q_STATIC

        # ── 초기 공분산 P: 첫 측정의 불확실성 수준으로 시작 ──
        R0 = self._make_R(z0)
        self.P = R0.copy()

    # ── 내부 헬퍼 ──────────────────────────────────────
    def _make_R(self, Z: float) -> np.ndarray:
        """
        논문 기반 동적 관측 노이즈 행렬 — 이전과 동일하게 유지
          σ_xy = K_xy × Z      (횡방향: 거리에 선형)
          σ_z  = K_z  × Z²     (깊이: 거리의 제곱, 비선형)
        """
        Z = max(Z, 0.1)
        sig_xy = EKF_SIGMA_XY_K * Z
        sig_z  = EKF_SIGMA_Z_K  * Z * Z
        return np.diag([sig_xy**2, sig_xy**2, sig_z**2])

    # ── predict ────────────────────────────────────────
    def predict(self) -> np.ndarray:
        """
        예측 단계: x = F @ x = x (위치 그대로)
        P만 Q만큼 조금 커짐 → 시간이 지날수록 불확실성 소폭 증가.
        dt 인자 불필요 (정지 모델에서 dt는 의미 없음).
        """
        # x는 변하지 않음 (F = I)
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, z_meas: np.ndarray) -> np.ndarray:
        """
        보정 단계 — Mahalanobis 게이팅 + Joseph form 유지
        """
        z = np.array(z_meas, dtype=np.float64)

        # 현재 깊이 Z로 R 동적 계산
        Z_est = max(self.x[2], 0.1)
        R = self._make_R(Z_est)

        # Innovation
        y = z - self.H @ self.x

        # 혁신 공분산
        S = self.H @ self.P @ self.H.T + R

        # ── Mahalanobis 이상치 게이팅 ──────────────────
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return self.x.copy()

        d2 = float(y @ S_inv @ y)
        if d2 > EKF_GATE_CHI2:
            return self.x.copy()

        # ── Kalman Gain ─────────────────────────────────
        K = self.P @ self.H.T @ S_inv

        # ── 상태 갱신 ────────────────────────────────────
        self.x = self.x + K @ y

        # ── 공분산 갱신 (Joseph form) ────────────────────
        # Joseph form: P = (I-KH)P(I-KH)ᵀ + KRKᵀ
        # → 수치 오차로 P가 비대칭/비PSD가 되는 현상 방지
        I_KH = np.eye(3) - K @ self.H   # 3-state이므로 eye(3)
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

        return self.x.copy()

    def get_position(self) -> np.ndarray:
        return self.x.copy()

    def get_covariance_xyz(self) -> np.ndarray:
        """위치 공분산 (3×3) 반환 — 수렴 상태 모니터링용"""
        return self.P.copy()   # 이미 3×3


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
        super().__init__("parking_detection_node")

        self.last_rgb_received  = None
        self.last_publish_time  = 0.0
        self.gui_enabled        = True
        self.latest_depth_frame = None
        self.latest_depth_time  = None
        self.camera_info        = None
        self._depth_lock        = threading.Lock()
        self.tracks             = []

        # ── odom: AMR 현재 위치/방향 (world frame 변환용) ──
        # odom이 없으면 카메라 상대 좌표 그대로 publish (fallback)
        self._odom_x         = None   # AMR x (m)
        self._odom_y         = None   # AMR y (m)
        self._odom_yaw       = None   # AMR yaw (rad)
        self._odom_angular_z = 0.0    # AMR 회전 각속도 이동 평균 (rad/s)
        self._angular_z_buf  = deque(maxlen=ANGULAR_AVG_WINDOW)  # 이동 평균 버퍼

        # ── TF2: odom → map 좌표 변환용 ──
        # SLAM localization 실행 중이면 map→odom TF가 발행됨
        # TF 없을 때(localization 미실행)는 odom 좌표 그대로 fallback
        self._tf_buffer   = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # ── Hard Negative 저장 초기화 ──
        self._last_hard_neg_time = 0.0
        self._latest_raw_frame   = None   # S키 저장용 원본 프레임
        if HARD_NEG_SAVE:
            import os
            os.makedirs(HARD_NEG_SAVE_DIR, exist_ok=True)
            self.get_logger().info(f"[HardNeg] 저장 경로: {HARD_NEG_SAVE_DIR}")

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
        self.create_subscription(Odometry,        TOPIC_ODOM,  self.odom_callback,  qos_be)
        self.get_logger().info(f"Sub RGB  : {TOPIC_RGB}")
        self.get_logger().info(f"Sub Depth: {TOPIC_DEPTH}")
        self.get_logger().info(f"Sub Odom : {TOPIC_ODOM}")

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

    # ── RGB 콜백 (메인 파이프라인) ───────────────────────
    def image_callback(self, msg: CompressedImage):
        self.last_rgb_received = time.monotonic()

        frame = cv2.imdecode(
            np.frombuffer(bytes(msg.data), dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return

        # 원본 프레임 보존 — S키 저장 시 bbox 없는 이미지를 내보내기 위함
        self._latest_raw_frame = frame.copy()

        # ── 회전 억제 ──────────────────────────────────
        # 회전 중에는 모션 블러로 FP 폭발 → 탐지 자체를 스킵
        if abs(self._odom_angular_z) > ROTATE_SUPPRESS_THRESH:
            self.get_logger().info(
                f"[ROT] 회전 중 탐지 억제 (angular_z={self._odom_angular_z:.3f})",
                throttle_duration_sec=1.0)
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

    # ── Hard Negative 저장 ───────────────────────────
    def _save_hard_negative_manual(self):
        """
        S키 입력 시 호출 — 현재 원본 프레임(bbox 없음)을 저장.
        화면에는 bbox가 그려지지만 저장되는 이미지는 원본 그대로.
        """
        if not HARD_NEG_SAVE:
            self.get_logger().warn("[HardNeg] HARD_NEG_SAVE=False, 저장 비활성화 상태")
            return
        if self._latest_raw_frame is None:
            return
        import os
        ts    = int(time.time() * 1000)
        fname = f"hn_{ts}_manual.jpg"
        fpath = os.path.join(HARD_NEG_SAVE_DIR, fname)
        cv2.imwrite(fpath, self._latest_raw_frame)
        self.get_logger().info(f"[HardNeg] S키 저장: {fname}")

    def _save_hard_negative(self, frame: np.ndarray,
                            x1: int, y1: int, x2: int, y2: int,
                            reason: str = ""):
        """
        오탐으로 판단된 프레임을 bbox 없는 원본으로 저장.

        저장 파일명: hn_<타임스탬프>_<reason>.jpg
        - 연속 저장 방지: HARD_NEG_INTERVAL 초 이내 재저장 안 함
        - 저장되는 이미지: bbox/텍스트 없는 원본 프레임
          → Roboflow 등에서 바로 라벨링 가능
        """
        if not HARD_NEG_SAVE:
            return
        now = time.monotonic()
        if now - self._last_hard_neg_time < HARD_NEG_INTERVAL:
            return   # 너무 자주 저장 방지

        import os
        ts = int(time.time() * 1000)   # ms 타임스탬프
        fname = f"hn_{ts}_{reason}.jpg"
        fpath = os.path.join(HARD_NEG_SAVE_DIR, fname)

        # 원본 프레임 저장 (bbox 없음)
        cv2.imwrite(fpath, frame)
        self._last_hard_neg_time = now
        self.get_logger().info(
            f"[HardNeg] 저장: {fname}  bbox=({x1},{y1},{x2},{y2})")

    # ── YOLO 탐지 ────────────────────────────────────────
    def _detect_cars(self, frame: np.ndarray) -> list:
        results = self.model.predict(
            source=frame, imgsz=YOLO_IMG_SIZE,
            conf=CONF_THRESHOLD, verbose=False)
        cars = []
        if not results:
            return cars

        h_frame = frame.shape[0]
        roi_top_px = int(h_frame * ROI_TOP_RATIO)  # 상단 제외 픽셀 경계선

        for box in results[0].boxes:
            cls_id = int(box.cls[0].item())
            name   = self.model.names.get(cls_id, str(cls_id))
            if name != "car":
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

            # ── 상단 ROI 필터 ──
            # bbox 하단(y2)이 roi_top_px보다 위에 있으면 제거
            # bbox 중심이 아닌 하단 기준: 차량 하단이 ROI 경계 아래 있어야 유효
            if y2 <= roi_top_px:
                self._save_hard_negative(frame, x1, y1, x2, y2,
                                         reason="roi_top")
                continue

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
        cv2.putText(frame, "S: save hard-neg",
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') or key == ord('S'):
            self._save_hard_negative_manual()

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