#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[진단 1] 토픽 스캐너
  - 현재 ROS2 네트워크에 떠 있는 모든 이미지 관련 토픽 목록 출력
  - 각 토픽의 publisher 수 / 타입 / QoS 정보 출력
  - ocr_node 가 구독해야 할 토픽이 실제로 존재하는지 확인하는 용도

실행:
  python3 diag_1_topic_scan.py
  python3 diag_1_topic_scan.py robot2   # 특정 네임스페이스만 필터
"""
import sys
import rclpy
from rclpy.node import Node

KEYWORDS = ["image", "camera", "oakd", "compressed", "depth"]

class TopicScanner(Node):
    def __init__(self, ns_filter=None):
        super().__init__("diag_topic_scanner")
        self.ns_filter = ns_filter

    def scan(self):
        # 모든 토픽 목록 수집
        topic_list = self.get_topic_names_and_types()

        matched = []
        for topic_name, type_list in topic_list:
            name_lower = topic_name.lower()
            if not any(kw in name_lower for kw in KEYWORDS):
                continue
            if self.ns_filter and self.ns_filter not in topic_name:
                continue
            matched.append((topic_name, type_list))

        if not matched:
            print("\n[결과] 이미지 관련 토픽이 전혀 없음 — 로봇이 연결되지 않았거나 네임스페이스가 다를 수 있음\n")
            return

        print(f"\n{'='*70}")
        print(f" 이미지 관련 토픽 {len(matched)}개 발견")
        if self.ns_filter:
            print(f" 필터: '{self.ns_filter}'")
        print(f"{'='*70}")

        # 토픽별 QoS 정보 조회
        for topic_name, type_list in sorted(matched):
            publishers = self.get_publishers_info_by_topic(topic_name)
            print(f"\n  토픽 : {topic_name}")
            print(f"  타입 : {', '.join(type_list)}")
            print(f"  발행자 수 : {len(publishers)}")
            for i, pub in enumerate(publishers):
                qos = pub.qos_profile
                print(f"    발행자[{i}]")
                print(f"      node      : {pub.node_name} ({pub.node_namespace})")
                print(f"      reliability: {qos.reliability.name}")
                print(f"      durability : {qos.durability.name}")
                print(f"      history    : {qos.history.name}  depth={qos.depth}")

        # ocr_node 구독 대상 직접 체크
        print(f"\n{'='*70}")
        print(" [핵심 체크] ocr_node 가 구독할 토픽 존재 여부")
        print(f"{'='*70}")
        for ns in ["/robot2", "/robot3"]:
            target = f"{ns}/oakd/rgb/image_raw/compressed"
            found  = any(t == target for t, _ in matched)
            mark   = "✓ 있음" if found else "✗ 없음  ← 문제 가능성"
            print(f"  {target}  →  {mark}")

        print()

def main():
    rclpy.init()
    ns_filter = sys.argv[1] if len(sys.argv) > 1 else None
    node = TopicScanner(ns_filter)

    # 토픽 목록이 DDS 디스커버리를 거쳐 반영되도록 잠시 대기
    import time
    print("DDS 디스커버리 대기 중... (2초)")
    time.sleep(2.0)

    node.scan()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
