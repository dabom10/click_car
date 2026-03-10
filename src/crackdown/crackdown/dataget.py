#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from datetime import datetime

import cv2
import numpy as np
import rclpy

from rclpy.node import Node
from sensor_msgs.msg import CompressedImage


# ================================
# 설정 상수
# ================================
IMAGE_TOPIC = "/robot3/oakd/rgb/image_raw/compressed"
WINDOW_NAME = "robot3 RGB Viewer"

# 원하는 저장 경로로 수정
SAVE_DIR = "/home/kyb/click_car/src/data"

# 화면 갱신 주기
TIMER_PERIOD_SEC = 0.03


class CompressedImageViewer(Node):
    def __init__(self):
        super().__init__("compressed_image_viewer")

        # 최신 프레임 저장 변수
        self.latest_frame = None

        # 저장 폴더가 없으면 생성
        os.makedirs(SAVE_DIR, exist_ok=True)

        # CompressedImage 구독
        self.subscription = self.create_subscription(
            CompressedImage,
            IMAGE_TOPIC,
            self.image_callback,
            10,
        )

        # OpenCV 창 갱신용 타이머
        self.timer = self.create_timer(TIMER_PERIOD_SEC, self.display_loop)

        self.get_logger().info(f"구독 토픽: {IMAGE_TOPIC}")
        self.get_logger().info(f"저장 경로: {SAVE_DIR}")
        self.get_logger().info("창에서 c 키를 누르면 이미지 저장, q 키를 누르면 종료")

    def image_callback(self, msg: CompressedImage):
        """
        CompressedImage 데이터를 OpenCV 이미지로 디코딩한다.
        """
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            self.get_logger().warn("이미지 디코딩 실패")
            return

        self.latest_frame = frame

    def display_loop(self):
        """
        최신 프레임을 화면에 표시하고 키 입력을 처리한다.
        """
        if self.latest_frame is None:
            return

        cv2.imshow(WINDOW_NAME, self.latest_frame)
        key = cv2.waitKey(1) & 0xFF

        # c 키를 누르면 현재 프레임 저장
        if key == ord("c"):
            self.save_current_frame()

        # q 키를 누르면 종료
        elif key == ord("q"):
            self.get_logger().info("q 키 입력으로 종료합니다.")
            rclpy.shutdown()

    def save_current_frame(self):
        """
        현재 프레임을 파일로 저장한다.
        """
        if self.latest_frame is None:
            self.get_logger().warn("저장할 프레임이 없습니다.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        file_path = os.path.join(SAVE_DIR, f"capture_{timestamp}.jpg")

        success = cv2.imwrite(file_path, self.latest_frame)
        if success:
            self.get_logger().info(f"이미지 저장 완료: {file_path}")
        else:
            self.get_logger().error(f"이미지 저장 실패: {file_path}")


def main(args=None):
    rclpy.init(args=args)

    node = CompressedImageViewer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("사용자 종료")
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()