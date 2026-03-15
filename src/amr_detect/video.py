#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# ================================================================
#  record_video.py
#
#  ROS2 토픽에서 받아오는 실제 RGB 영상을 그대로 저장.
#  bbox/오버레이 없는 원본 프레임만 저장.
#  Q키 또는 Ctrl+C 종료 시 mp4 파일로 저장.
#
#  사용법:
#    python3 record_video.py
#    python3 record_video.py --output /home/rokey/Documents/test.mp4
#    python3 record_video.py --namespace /robot2
#
#  조작키:
#    Q / ESC   : 저장 후 종료
#    Ctrl+C    : 저장 후 종료
# ================================================================

import argparse
import os
import time
import threading

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage


# ──────────────────────────────────────────
#  설정값
# ──────────────────────────────────────────
ROBOT_NAMESPACE = "/robot3"
TOPIC_RGB       = f"{ROBOT_NAMESPACE}/oakd/rgb/image_raw/compressed"
OUTPUT_PATH     = f"/home/rokey/Documents/test_video_{int(time.time())}.mp4"  # ← 경로 변경
FPS             = 10.0
WINDOW_NAME     = "Recording... (Q/ESC: 저장 후 종료)"


# ================================================================
#  RecorderNode
# ================================================================
class RecorderNode(Node):
    def __init__(self, output_path: str):
        super().__init__("video_recorder_node")

        self._output_path  = output_path
        self._frames       = []
        self._frame_lock   = threading.Lock()
        self._writer       = None
        self._frame_count  = 0
        self._start_time   = None
        self._running      = True

        # ── 구독 ──
        qos_be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(
            CompressedImage, TOPIC_RGB,
            self._rgb_callback, qos_be)

        self.get_logger().info(f"[Recorder] 구독: {TOPIC_RGB}")
        self.get_logger().info(f"[Recorder] 저장 경로: {output_path}")
        self.get_logger().info(f"[Recorder] Q 또는 ESC 키로 저장 후 종료")

        # ── GUI ──
        try:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, 800, 500)
            self._gui = True
        except Exception:
            self._gui = False

        # ── 통계 타이머 (10초마다) ──
        self.create_timer(10.0, self._log_stats)

    # ── RGB 콜백 ─────────────────────────────────────────
    def _rgb_callback(self, msg: CompressedImage):
        frame = cv2.imdecode(
            np.frombuffer(bytes(msg.data), dtype=np.uint8),
            cv2.IMREAD_COLOR)
        if frame is None:
            return

        if self._start_time is None:
            self._start_time = time.monotonic()

        # ── VideoWriter 초기화 (첫 프레임에서 해상도 확정) ──
        if self._writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"avc1")  # ← H.264 코덱으로 변경 (호환성 최고)
            os.makedirs(os.path.dirname(self._output_path), exist_ok=True)
            self._writer = cv2.VideoWriter(
                self._output_path, fourcc, FPS, (w, h))

            # avc1 실패 시 mp4v 폴백
            if not self._writer.isOpened():
                self.get_logger().warn("[Recorder] avc1 실패 → mp4v 폴백")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self._writer = cv2.VideoWriter(
                    self._output_path, fourcc, FPS, (w, h))

            self.get_logger().info(
                f"[Recorder] VideoWriter 초기화: {w}x{h} @ {FPS}fps")

        # ── 프레임 저장 ──
        self._writer.write(frame)
        self._frame_count += 1

        # ── 화면 표시 ──
        if self._gui:
            disp = frame.copy()
            elapsed = time.monotonic() - self._start_time if self._start_time else 0
            cv2.circle(disp, (20, 20), 8, (0, 0, 255), -1)
            cv2.putText(disp, f"REC  {elapsed:.1f}s  |  {self._frame_count} frames",
                        (35, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(disp, f"저장: {self._output_path}",
                        (10, disp.shape[0] - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.putText(disp, "Q / ESC : 저장 후 종료",
                        (10, disp.shape[0] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
            cv2.imshow(WINDOW_NAME, disp)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                self.get_logger().info("[Recorder] Q/ESC 키 입력 → 종료")
                self._stop()

    # ── 종료 처리 ─────────────────────────────────────────
    def _stop(self):
        if not self._running:
            return
        self._running = False

        if self._writer is not None:
            self._writer.release()
            self.get_logger().info(f"[Recorder] 저장 완료: {self._output_path}")
            self.get_logger().info(
                f"[Recorder] 총 {self._frame_count}프레임 "
                f"({self._frame_count / FPS:.1f}초)")
        else:
            self.get_logger().warn("[Recorder] 저장된 프레임 없음 (영상 수신 전 종료)")

        if self._gui:
            cv2.destroyAllWindows()

        rclpy.shutdown()

    # ── 통계 ─────────────────────────────────────────────
    def _log_stats(self):
        if self._start_time is None:
            self.get_logger().info("[Recorder] 영상 수신 대기 중...")
            return
        elapsed = time.monotonic() - self._start_time
        actual_fps = self._frame_count / max(elapsed, 0.1)
        self.get_logger().info(
            f"[Recorder] {elapsed:.0f}s 경과  |  "
            f"{self._frame_count}프레임  |  실제 FPS={actual_fps:.1f}")

    def destroy_node(self):
        self._stop()
        super().destroy_node()


# ================================================================
#  main
# ================================================================
def main():
    global TOPIC_RGB, FPS

    parser = argparse.ArgumentParser(description="ROS2 RGB 토픽 영상 녹화")
    parser.add_argument(
        "--output", default=OUTPUT_PATH,
        help=f"저장할 mp4 경로 (기본값: {OUTPUT_PATH})")
    parser.add_argument(
        "--namespace", default=ROBOT_NAMESPACE,
        help=f"로봇 네임스페이스 (기본값: {ROBOT_NAMESPACE})")
    parser.add_argument(
        "--fps", type=float, default=FPS,
        help=f"저장 FPS (기본값: {FPS})")
    args = parser.parse_args()

    TOPIC_RGB = f"{args.namespace}/oakd/rgb/image_raw/compressed"
    FPS = args.fps

    rclpy.init()
    node = RecorderNode(args.output)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("[Recorder] Ctrl+C → 종료")
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()