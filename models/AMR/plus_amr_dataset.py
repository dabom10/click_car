#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# ================================================================
#  collect_dataset.py
#
#  Hard Negative Mining용 이미지 데이터셋 수집 노드.
#  YOLO 추론 없이 원본 프레임만 저장 — 라벨링은 별도 수행.
#
#  저장 모드:
#    S키      — 수동 저장 (오탐 발견 즉시)
#    정지 중  — AMR 정지 상태에서 N초마다 자동 저장
#    직진 중  — 직진 상태에서 N초마다 자동 저장
#    회전 중  — 회전 상태에서 N초마다 자동 저장 (FP 주요 원인)
#
#  저장 경로: SAVE_BASE_DIR/<모드>/hn_<타임스탬프>.jpg
#    → Roboflow 업로드 후 모드별로 분리 라벨링 가능
#
#  사용법:
#    ros2 run <pkg> collect_dataset
#    → cv2 창에서 S키로 수동 저장
#    → AMR 이동 상태에 따라 자동 저장
#    → Ctrl+C로 종료, 저장 통계 출력
# ================================================================

import os
import time
import threading

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage
from nav_msgs.msg import Odometry


# ──────────────────────────────────────────
#  설정값
# ──────────────────────────────────────────
ROBOT_NAMESPACE = "/robot3"
TOPIC_RGB       = f"{ROBOT_NAMESPACE}/oakd/rgb/image_raw/compressed"
TOPIC_ODOM      = f"{ROBOT_NAMESPACE}/odom"

# ── 저장 경로 ──
SAVE_BASE_DIR   = "/home/rokey/click_car/models/AMR/dataset/plus_dataset"
# 모드별 하위 폴더 자동 생성:
#   dataset_collection/manual/   ← S키 수동 저장
#   dataset_collection/still/    ← 정지 중 자동 저장
#   dataset_collection/straight/ ← 직진 중 자동 저장
#   dataset_collection/rotating/ ← 회전 중 자동 저장

# ── 자동 저장 간격 (초) ──
# 너무 짧으면 거의 동일한 프레임이 쌓임 → 다양성 확보를 위해 여유있게 설정
INTERVAL_STILL    = 2.0   # 정지 중: 2초마다
INTERVAL_STRAIGHT = 1.0   # 직진 중: 1초마다
INTERVAL_ROTATING = 0.5   # 회전 중: 0.5초마다 (FP 원인 장면 집중 수집)

# ── AMR 상태 판단 임계값 ──
# angular.z 실측 분석 결과:
#   직진/정지 노이즈: 최대 0.041 rad/s
#   실제 회전 구간:   0.10 rad/s 이상
ANGULAR_THRESH  = 0.10   # rad/s 이상이면 회전 중
LINEAR_THRESH   = 0.03   # m/s 이상이면 이동 중 (정지 판단용)

# ── 화면 ──
WINDOW_NAME     = "Dataset Collector"


# ================================================================
#  DatasetCollectorNode
# ================================================================
class DatasetCollectorNode(Node):
    def __init__(self):
        super().__init__("dataset_collector_node")

        # ── 상태 변수 ──
        self._latest_frame    = None
        self._frame_lock      = threading.Lock()
        self._angular_z       = 0.0
        self._linear_x        = 0.0

        # ── 마지막 저장 시각 (모드별) ──
        self._last_save = {
            "manual":   0.0,
            "still":    0.0,
            "straight": 0.0,
            "rotating": 0.0,
        }

        # ── 저장 카운터 ──
        self._save_count = {k: 0 for k in self._last_save}

        # ── 저장 디렉토리 생성 ──
        for mode in self._last_save:
            path = os.path.join(SAVE_BASE_DIR, mode)
            os.makedirs(path, exist_ok=True)
        self.get_logger().info(f"[Collector] 저장 경로: {SAVE_BASE_DIR}")

        # ── 구독 ──
        qos_be = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                            history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(CompressedImage, TOPIC_RGB,
                                 self._rgb_callback, qos_be)
        self.create_subscription(Odometry, TOPIC_ODOM,
                                 self._odom_callback, qos_be)

        # ── GUI ──
        try:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, 800, 500)
            self._gui = True
        except Exception:
            self._gui = False

        # ── 통계 출력 타이머 (10초마다) ──
        self.create_timer(10.0, self._log_stats)
        self.get_logger().info("[Collector] 준비 완료. S키로 수동 저장.")

    # ── odom 콜백 ────────────────────────────────────────
    def _odom_callback(self, msg: Odometry):
        self._angular_z = msg.twist.twist.angular.z
        self._linear_x  = msg.twist.twist.linear.x

    # ── AMR 상태 판단 ─────────────────────────────────────
    def _get_motion_mode(self) -> str:
        """
        angular_z, linear_x 기준으로 현재 AMR 상태 반환.
          rotating  : |angular_z| ≥ ANGULAR_THRESH
          straight  : |linear_x|  ≥ LINEAR_THRESH  (회전 아닌 이동)
          still     : 나머지
        """
        if abs(self._angular_z) >= ANGULAR_THRESH:
            return "rotating"
        if abs(self._linear_x) >= LINEAR_THRESH:
            return "straight"
        return "still"

    # ── RGB 콜백 ─────────────────────────────────────────
    def _rgb_callback(self, msg: CompressedImage):
        frame = cv2.imdecode(
            np.frombuffer(bytes(msg.data), dtype=np.uint8),
            cv2.IMREAD_COLOR)
        if frame is None:
            return

        with self._frame_lock:
            self._latest_frame = frame.copy()

        mode = self._get_motion_mode()
        now  = time.monotonic()

        # ── 자동 저장 ──
        intervals = {
            "still":    INTERVAL_STILL,
            "straight": INTERVAL_STRAIGHT,
            "rotating": INTERVAL_ROTATING,
        }
        interval = intervals[mode]
        if now - self._last_save[mode] >= interval:
            self._save(frame, mode)

        # ── 화면 표시 ──
        self._draw(frame, mode)

    # ── 저장 ─────────────────────────────────────────────
    def _save(self, frame: np.ndarray, mode: str):
        """원본 프레임(bbox 없음)을 모드별 폴더에 저장."""
        ts    = int(time.time() * 1000)
        fname = f"img_{ts}.jpg"
        fpath = os.path.join(SAVE_BASE_DIR, mode, fname)
        cv2.imwrite(fpath, frame)
        self._last_save[mode]  = time.monotonic()
        self._save_count[mode] += 1

    # ── 화면 표시 ─────────────────────────────────────────
    def _draw(self, frame: np.ndarray, mode: str):
        if not self._gui:
            return

        disp = frame.copy()

        # 상태별 색상
        colors = {
            "still":    (0, 255, 0),     # 초록
            "straight": (255, 255, 0),   # 노랑
            "rotating": (0, 165, 255),   # 주황
        }
        color = colors[mode]

        # 상태 표시
        mode_kor = {"still": "정지", "straight": "직진", "rotating": "회전"}
        cv2.putText(disp, f"[{mode_kor[mode]}]  angular_z={self._angular_z:.3f}",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # 저장 카운트
        total = sum(self._save_count.values())
        cv2.putText(disp,
                    f"저장: 정지={self._save_count['still']}  "
                    f"직진={self._save_count['straight']}  "
                    f"회전={self._save_count['rotating']}  "
                    f"수동={self._save_count['manual']}  "
                    f"합계={total}",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # 조작 안내
        h = disp.shape[0]
        cv2.putText(disp, "S: 수동 저장  |  Q / Ctrl+C: 종료",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (180, 180, 180), 1)

        cv2.imshow(WINDOW_NAME, disp)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('s'), ord('S')):
            with self._frame_lock:
                raw = self._latest_frame
            if raw is not None:
                self._save(raw, "manual")
                self.get_logger().info(
                    f"[Collector] S키 수동 저장 (total manual={self._save_count['manual']})")
        elif key in (ord('q'), ord('Q')):
            self.get_logger().info("[Collector] Q키 종료")
            self._print_summary()
            rclpy.shutdown()

    # ── 통계 ─────────────────────────────────────────────
    def _log_stats(self):
        total = sum(self._save_count.values())
        self.get_logger().info(
            f"[Collector] 저장 현황 — "
            f"정지={self._save_count['still']}  "
            f"직진={self._save_count['straight']}  "
            f"회전={self._save_count['rotating']}  "
            f"수동={self._save_count['manual']}  "
            f"합계={total}")

    def _print_summary(self):
        total = sum(self._save_count.values())
        print()
        print("=" * 45)
        print("  수집 완료 요약")
        print("=" * 45)
        for mode, cnt in self._save_count.items():
            print(f"  {mode:<10}: {cnt:>4}장")
        print(f"  {'합계':<10}: {total:>4}장")
        print(f"  저장 경로: {SAVE_BASE_DIR}")
        print("=" * 45)

    def destroy_node(self):
        self._print_summary()
        if self._gui:
            cv2.destroyAllWindows()
        super().destroy_node()


# ================================================================
#  main
# ================================================================
def main(args=None):
    rclpy.init(args=args)
    node = DatasetCollectorNode()
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