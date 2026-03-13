#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
[테스트 노드: ocr_node 커맨더]

robot2 / robot3 를 선택한 뒤 amr_start / cctv_start 를 발행하고,
해당 로봇의 /robotN/capture_done 을 수신한다.
'''

import threading
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool

NAMESPACES = ["/robot2", "/robot3"]


class OcrCommanderNode(Node):

    def __init__(self):
        super().__init__('test_ocr_commander')

        # 두 로봇의 start 퍼블리셔를 미리 생성
        self._start_pubs = {
            ns: self.create_publisher(String, f"{ns}/start", 10)
            for ns in NAMESPACES
        }
        # 두 로봇의 capture_done 구독
        for ns in NAMESPACES:
            self.create_subscription(
                Bool, f"{ns}/capture_done",
                lambda msg, n=ns: self._capture_done_cb(msg, n), 10
            )

        self.get_logger().info("퍼블리시: /robot2/start, /robot3/start")
        self.get_logger().info("구독:     /robot2/capture_done, /robot3/capture_done")

        threading.Thread(target=self._menu_loop, daemon=True).start()

    def _capture_done_cb(self, msg: Bool, ns: str):
        mark = '✅' if msg.data else '❌'
        self.get_logger().info(f'[capture_done ({ns})]  {mark}  data={msg.data}')

    def _menu_loop(self):
        ns = NAMESPACES[0]   # 기본값 robot2
        while rclpy.ok():
            print(
                f'\n┌──────────────────────────────────┐'
                f'\n│  ocr_node 커맨더  (현재: {ns})  │'
                f'\n├──────────────────────────────────┤'
                f'\n│  r  로봇 전환 (robot2 ↔ robot3)  │'
                f'\n│  1  amr_start  발행               │'
                f'\n│  2  cctv_start 발행               │'
                f'\n│  q  종료                          │'
                f'\n└──────────────────────────────────┘'
            )
            try:
                choice = input('선택 > ').strip()
            except (EOFError, KeyboardInterrupt):
                break

            if choice == 'r':
                ns = "/robot3" if ns == "/robot2" else "/robot2"
                self.get_logger().info(f'로봇 전환 → {ns}')
            elif choice == '1':
                self._publish(ns, 'amr_start')
            elif choice == '2':
                self._publish(ns, 'cctv_start')
            elif choice == 'q':
                break
            else:
                print(f'  알 수 없는 입력: {choice!r}')

    def _publish(self, ns: str, data: str):
        msg = String()
        msg.data = data
        self._start_pubs[ns].publish(msg)
        self.get_logger().info(f'[발행] {ns}/start  →  "{data}"')


def main(args=None):
    rclpy.init(args=args)
    node = OcrCommanderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
