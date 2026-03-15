#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[진단 2] 이미지 수신 테스터
  - 지정한 토픽을 직접 구독해서 수신 여부 + 수신 빈도 확인
  - QoS를 BEST_EFFORT / RELIABLE 두 가지로 각각 시도해
    어느 쪽에서 수신되는지 보여줌 (QoS 불일치 진단)
  - 수신 성공 시 프레임 해상도 출력

실행:
  python3 diag_2_image_recv.py /robot2/oakd/rgb/image_raw/compressed
  python3 diag_2_image_recv.py /robot3/oakd/rgb/image_raw/compressed
  python3 diag_2_image_recv.py  # 인자 없으면 robot2 기본값

결과 해석:
  BEST_EFFORT 수신됨  → ocr_node 현재 설정(BEST_EFFORT)이 맞음
  RELIABLE 만 수신됨  → 발행자가 RELIABLE → ocr_node QoS를 RELIABLE로 바꿔야 함
  둘 다 수신 안됨     → 토픽 자체가 없거나 네트워크 문제
"""
import sys
import time
import threading
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import CompressedImage
import numpy as np
import cv2

TOPIC_DEFAULT = "/robot2/oakd/rgb/image_raw/compressed"
TEST_DURATION = 6.0   # 각 QoS로 테스트할 시간 (초)

class ImageReceiver(Node):
    def __init__(self, topic: str, reliability: str):
        super().__init__(f"diag_image_recv_{reliability.lower()}")
        self.topic       = topic
        self.reliability = reliability
        self.count       = 0
        self.last_shape  = None
        self.first_recv  = None

        rel = (ReliabilityPolicy.BEST_EFFORT
               if reliability == "BEST_EFFORT"
               else ReliabilityPolicy.RELIABLE)

        qos = QoSProfile(
            reliability=rel,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.sub = self.create_subscription(
            CompressedImage, topic, self._cb, qos
        )

    def _cb(self, msg: CompressedImage):
        self.count += 1
        if self.first_recv is None:
            self.first_recv = time.monotonic()

        # 첫 수신 시 디코딩 테스트
        if self.count == 1:
            frame = cv2.imdecode(
                np.frombuffer(bytes(msg.data), dtype=np.uint8),
                cv2.IMREAD_COLOR
            )
            if frame is not None:
                self.last_shape = frame.shape
            else:
                self.last_shape = "디코딩 실패"


def test_qos(topic: str, reliability: str) -> dict:
    """단일 QoS 설정으로 TEST_DURATION 초 동안 수신 시도"""
    rclpy.init()
    node     = ImageReceiver(topic, reliability)
    start    = time.monotonic()
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)

    print(f"  [{reliability}] 수신 대기 중... ({TEST_DURATION:.0f}초)")
    while time.monotonic() - start < TEST_DURATION:
        executor.spin_once(timeout_sec=0.1)

    result = {
        "reliability": reliability,
        "count":       node.count,
        "rate_hz":     node.count / TEST_DURATION,
        "shape":       node.last_shape,
        "first_recv":  node.first_recv,
    }
    node.destroy_node()
    rclpy.shutdown()
    return result


def main():
    topic = sys.argv[1] if len(sys.argv) > 1 else TOPIC_DEFAULT

    print(f"\n{'='*60}")
    print(f" 이미지 수신 진단")
    print(f" 토픽: {topic}")
    print(f"{'='*60}\n")

    results = []
    for rel in ["BEST_EFFORT", "RELIABLE"]:
        r = test_qos(topic, rel)
        results.append(r)
        status = f"{r['count']}프레임 ({r['rate_hz']:.1f} Hz)" if r['count'] > 0 else "수신 없음"
        print(f"  [{rel}] {status}")
        if r["shape"]:
            print(f"         해상도: {r['shape']}")
        print()

    print(f"{'='*60}")
    print(" [종합 진단]")
    be = results[0]
    re = results[1]

    if be["count"] > 0 and re["count"] > 0:
        print("  BEST_EFFORT / RELIABLE 둘 다 수신됨 — 정상")
        print("  ocr_node BEST_EFFORT 설정 유지 권장")
    elif be["count"] > 0:
        print("  BEST_EFFORT 수신됨 — ocr_node 현재 설정 정상")
    elif re["count"] > 0:
        print("  ★ RELIABLE 만 수신됨!")
        print("    → 발행자 QoS가 RELIABLE")
        print("    → ocr_node 의 cam_qos 를 RELIABLE 로 변경 필요:")
        print("       reliability=ReliabilityPolicy.RELIABLE")
    else:
        print("  ★ 수신 없음!")
        print("    가능한 원인:")
        print("    1) 토픽이 발행되지 않음 → diag_1_topic_scan.py 로 확인")
        print("    2) DDS 도메인 불일치 (ROS_DOMAIN_ID 환경변수 확인)")
        print("    3) 네트워크 방화벽/스위치 멀티캐스트 차단")
        print("    4) 로봇과 PC가 같은 서브넷에 없음")
        print()
        print("    체크 명령:")
        print("    $ echo $ROS_DOMAIN_ID")
        print("    $ ros2 topic list | grep oakd")
        print("    $ ros2 topic hz " + topic)
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
