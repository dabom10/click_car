#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
[테스트 노드: webcam_detector_node의 /cctv_done 수신 확인]

webcam_detector_node 가 불법주정차 확정 시 발행하는 /cctv_done 토픽을 구독하여
메시지 형식이 올바른지 확인한다.

기대 메시지 형식: "cctv_start:<case_key>:<center_x>,<center_y>"
  ex) "cctv_start:20250311_142305_123456:1.2340,-0.5670"
'''

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

TOPIC_CCTV_DONE = '/cctv_done'


class CctvDoneListenerNode(Node):

    def __init__(self):
        super().__init__('test_cctv_done_listener')
        self.create_subscription(String, TOPIC_CCTV_DONE, self._cb, 10)
        self.get_logger().info(f'[대기 중]  {TOPIC_CCTV_DONE}  수신 준비 완료')

    def _cb(self, msg: String):
        raw = msg.data.strip()
        self.get_logger().info(f'[수신]  raw="{raw}"')

        # 형식 검증
        if not raw.startswith('cctv_start:'):
            self.get_logger().error('형식 오류 — "cctv_start:" 로 시작해야 합니다')
            return

        parts = raw.split(':', 2)   # ["cctv_start", "<key>", "<x>,<y>"]
        if len(parts) != 3:
            self.get_logger().error(f'형식 오류 — 콜론 구분자가 부족합니다 (parts={parts})')
            return

        case_key = parts[1]
        coord    = parts[2]

        try:
            cx_str, cy_str = coord.split(',')
            cx, cy = float(cx_str), float(cy_str)
        except ValueError:
            self.get_logger().error(f'좌표 파싱 실패 — "{coord}"')
            return

        self.get_logger().info(f'  ✅ 케이스 키 : {case_key}')
        self.get_logger().info(f'  ✅ 목표 좌표 : x={cx:+.4f} m,  y={cy:+.4f} m')


def main(args=None):
    rclpy.init(args=args)
    node = CctvDoneListenerNode()
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
