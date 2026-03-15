#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[진단 3] start 신호 → 카메라 구독 전체 파이프라인 모의 테스트
  - /robotN/start 토픽을 직접 발행하고
  - 이후 이미지 수신이 되는지 + start 로그가 찍히는지 확인

  ocr_node 를 실행하지 않고, 진단 노드 단독으로 다음을 검증:
    ① start 토픽 발행 → 수신 확인
    ② 카메라 토픽 구독 → 수신 여부 확인
    ③ 수신된 이미지가 정상 디코딩되는지 확인

실행:
  python3 diag_3_pipeline.py robot2   # /robot2/start 발행 후 카메라 수신 확인
  python3 diag_3_pipeline.py robot3
  python3 diag_3_pipeline.py          # 기본값 robot2

주의: ocr_node 를 실행한 상태에서 이 스크립트를 돌리면
      start 신호가 ocr_node 에도 전달됩니다.
      단독 진단 시에는 ocr_node 를 종료하세요.
"""
import sys
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String, Bool
import numpy as np
import cv2

WAIT_AFTER_START = 2.0   # start 발행 후 카메라 수신 대기 시간 (초)
RECV_TIMEOUT     = 8.0   # 이미지 수신 최대 대기 시간 (초)

class PipelineTester(Node):
    def __init__(self, robot_ns: str):
        super().__init__("diag_pipeline_tester")
        self.robot_ns  = robot_ns
        self.recv_count = 0
        self.recv_shape = None
        self.cam_sub    = None

        # start 토픽 발행자
        self.start_pub = self.create_publisher(
            String, f"{robot_ns}/start", 10
        )
        # capture_done 수신 확인용
        self.create_subscription(
            Bool, f"{robot_ns}/capture_done",
            self._done_cb, 10
        )
        self.capture_done_received = False

    def _done_cb(self, msg: Bool):
        if msg.data:
            self.capture_done_received = True
            print(f"  ★ capture_done=True 수신됨!")

    def send_start(self, mode: str):
        msg      = String()
        msg.data = mode
        self.start_pub.publish(msg)
        print(f"  ▶ '{mode}' 발행 → {self.robot_ns}/start")

    def attach_cam_sub(self):
        """start 발행 후 카메라 구독 생성"""
        topic = f"{self.robot_ns}/oakd/rgb/image_raw/compressed"
        qos   = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.cam_sub = self.create_subscription(
            CompressedImage, topic, self._img_cb, qos
        )
        print(f"  카메라 구독 생성: {topic}")

    def _img_cb(self, msg: CompressedImage):
        self.recv_count += 1
        if self.recv_shape is None:
            frame = cv2.imdecode(
                np.frombuffer(bytes(msg.data), dtype=np.uint8),
                cv2.IMREAD_COLOR
            )
            self.recv_shape = frame.shape if frame is not None else "디코딩 실패"
            print(f"  ✓ 첫 프레임 수신 — 해상도: {self.recv_shape}")


def main():
    ns = "/" + (sys.argv[1].lstrip("/") if len(sys.argv) > 1 else "robot2")

    print(f"\n{'='*60}")
    print(f" 파이프라인 진단  (로봇 네임스페이스: {ns})")
    print(f"{'='*60}\n")

    rclpy.init()
    node     = PipelineTester(ns)
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)

    # ── STEP 1: DDS 안정화 ──────────────────────
    print("[STEP 1] DDS 디스커버리 대기 (2초)...")
    t = time.monotonic()
    while time.monotonic() - t < 2.0:
        executor.spin_once(timeout_sec=0.1)

    # ── STEP 2: start 신호 발행 ──────────────────
    print("\n[STEP 2] start 신호 발행")
    node.send_start("amr_start")

    # ── STEP 3: 카메라 구독 연결 ────────────────
    print("\n[STEP 3] 카메라 구독 생성 후 이미지 수신 대기")
    time.sleep(0.3)   # start 처리 대기
    node.attach_cam_sub()

    t = time.monotonic()
    while time.monotonic() - t < RECV_TIMEOUT:
        executor.spin_once(timeout_sec=0.1)
        if node.recv_count >= 5:
            break

    # ── STEP 4: 결과 출력 ───────────────────────
    print(f"\n{'='*60}")
    print(f" [결과]")
    print(f"  수신 프레임 수 : {node.recv_count}")
    if node.recv_shape:
        print(f"  첫 프레임 해상도 : {node.recv_shape}")
    print(f"  capture_done 수신 : {'예' if node.capture_done_received else '아니오 (정상 — ocr_node 미실행)'}")

    print(f"\n  [진단]")
    if node.recv_count >= 5:
        print("  ✓ 정상: 이미지 토픽 수신 가능")
        print("    ocr_node 가 이미지를 못 받는다면 코드 문제 (구독 생성 타이밍 등)")
    elif node.recv_count > 0:
        print(f"  △ 부분 수신 ({node.recv_count}프레임) — 네트워크 불안정 가능성")
        print("    QoS depth=1 + BEST_EFFORT 에서 패킷 드랍이 발생 중일 수 있음")
        print("    → 허브 교통량이 많을 때 재시도 권장")
    else:
        print("  ✗ 수신 없음")
        print()
        print("  [체크리스트]")
        checks = [
            ("ROS_DOMAIN_ID 동일 여부",
             "echo $ROS_DOMAIN_ID  (로봇과 PC 모두 같아야 함)"),
            ("토픽 존재 여부",
             f"ros2 topic list | grep {ns}"),
            ("토픽 발행 속도",
             f"ros2 topic hz {ns}/oakd/rgb/image_raw/compressed"),
            ("QoS 정보",
             f"ros2 topic info -v {ns}/oakd/rgb/image_raw/compressed"),
            ("네트워크 연결",
             f"ping <로봇 IP>"),
        ]
        for i, (desc, cmd) in enumerate(checks, 1):
            print(f"    {i}. {desc}")
            print(f"       $ {cmd}")
    print(f"{'='*60}\n")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
