import rclpy
from rclpy.node import Node
from sensor_msgs.msg import BatteryState
import json
import urllib.request
from std_msgs.msg import String
import firebase_admin
from firebase_admin import credentials, db
from rclpy.executors import MultiThreadedExecutor
from nav_msgs.msg import Odometry
import math
import time

#========================================배터리 체크 노드========================================
class BatteryNode(Node):
    def __init__(self, robot_id, base_url):
        super().__init__(f'{robot_id}_battery_node')
        self.robot_id = robot_id
        self.base_url = base_url
        
        self.subscription = self.create_subscription(
            BatteryState,
            f'/{self.robot_id}/battery_state', # battery_state topic 전송
            self.battery_callback,
            10
        )

    def battery_callback(self, msg):
        battery_percent = int(msg.percentage * 100)
        # REST API 대신 Firebase SDK 사용 시:
        try:
            ref = db.reference(f'robot_status/{self.robot_id}')
            ref.update({"battery": battery_percent}) # HTML UI가 찾는 'battery' 키와 일치
            self.get_logger().info(f'Battery Update: {battery_percent}%')
        except Exception as e:
            self.get_logger().error(f'Battery Update Failed: {e}')

#========================================상태 체크 노드========================================
class StatusControlNode(Node):
    def __init__(self, robot_id, base_url):
        super().__init__(f'{robot_id}_status_control_node')
        self.robot_id = robot_id
        self.base_url = base_url
        
        # 1. UI 명령을 로봇에게 전달할 발행자
        self.publisher = self.create_publisher(String, f'/{self.robot_id}/patrol_command', 10)
        
        # 2. [추가] 로봇 하위 시스템의 실제 상태를 받는 구독자 (Feedback)
        self.status_subscription = self.create_subscription(
            String,
            f'/{self.robot_id}/{self.robot_id}_status',
            self.status_feedback_callback,
            10 
        )

        # Firebase 명령 리스너
        self.command_ref = db.reference(f'robot_command/{self.robot_id}/patrol_command')
        self.command_ref.listen(self.command_callback)

    def command_callback(self, event):
        command = event.data

        if not isinstance(command, str):
            return

        command = command.strip().lower()

        if command not in ("start", "stop"):
            return

        self.get_logger().info(f"RECEIVED FROM FIREBASE: {command}")

        msg = String()
        msg.data = command

        # 7번 publish
        for i in range(7):
            self.publisher.publish(msg)
            self.get_logger().info(
                f"PUBLISHED {i+1}/7 : /{self.robot_id}/patrol_command -> {command}"
            )
            time.sleep(0.05)

        # ★ 처리 후 DB 초기화
        try:
            self.command_ref.delete()
            self.get_logger().info("Firebase patrol_command deleted after processing")
        except Exception as e:
            self.get_logger().error(f"Failed to delete Firebase command: {e}")

    def status_feedback_callback(self, msg):
        # [핵심] 로봇이 실제로 상태 토픽을 보냈을 때만 Firebase를 업데이트
        actual_status = msg.data # 'patrol', 'enforce' 등
        ref = db.reference(f'robot_status/{self.robot_id}')
        ref.update({"status": actual_status})
        self.get_logger().info(f'Firebase status updated by feedback: {actual_status}')
#========================================상대 상태 체크 노드========================================
class WatcherNode(Node):
    def __init__(self, my_id, other_id):
        super().__init__(f'{my_id}_watcher')
        self.my_id = my_id
        self.other_id = other_id

        # 상대 로봇 상태 변화 감시
        self.other_status_ref = db.reference(f'robot_status/{self.other_id}/status')
        self.other_status_ref.listen(self.check_dispatch_logic)

        # 상대 로봇 배터리 변화 감시
        self.other_battery_ref = db.reference(f'robot_status/{self.other_id}/battery')
        self.other_battery_ref.listen(self.check_dispatch_logic)

        # 자동 출동 시 수동 버튼과 동일한 경로를 사용하기 위한 Firebase 명령 경로
        self.my_command_ref = db.reference(f'robot_command/{self.my_id}/patrol_command')

    def check_dispatch_logic(self, event):
        try:
            other_status = db.reference(f'robot_status/{self.other_id}/status').get()
            other_battery = db.reference(f'robot_status/{self.other_id}/battery').get()
            my_status = db.reference(f'robot_status/{self.my_id}/status').get()

            if my_status == 'idle':
                my_status = 'charging'

            should_start = (
                (other_status == 'enforce' and my_status == 'charging')
                or
                (
                    other_status == 'returning'
                    and isinstance(other_battery, (int, float))
                    and other_battery < 25
                    and my_status == 'charging'
                )
            )

            if should_start:
                current_cmd = self.my_command_ref.get()
                if current_cmd == 'start':
                    return

                self.get_logger().info(
                    f"{self.other_id} status={other_status}, battery={other_battery} -> "
                    f"write start to robot_command/{self.my_id}/patrol_command"
                )
                self.my_command_ref.set("start")

        except Exception as e:
            self.get_logger().error(f"Watcher dispatch check failed: {e}")

#========================================위치 체크 노드========================================
class OdomNode(Node):
    def __init__(self, robot_id, base_url):
        super().__init__(f'{robot_id}_odom_node')
        self.robot_id = robot_id
        
        # /robot_id/odom 토픽 구독
        self.subscription = self.create_subscription(
            Odometry,
            f'/{self.robot_id}/odom',
            self.odom_callback,
            10
        )

    def odom_callback(self, msg):
        # 위치 데이터 추출
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        
        # 쿼터니언을 Yaw(도)로 변환 (간이 수식)
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        yaw_deg = math.degrees(yaw)

        # Firebase 업데이트 (HTML UI가 요구하는 필드명 사용)
        try:
            # REST API 방식(urllib) 또는 SDK 방식(db.reference) 선택 가능
            # 여기서는 일관성을 위해 SDK 방식 사용 예시
            ref = db.reference(f'robot_status/{self.robot_id}')
            ref.update({
                "odom_x": x,
                "odom_y": y,
                "odom_yaw": yaw_deg
            })
            # 로거는 너무 자주 찍히면 성능에 영향을 줄 수 있으므로 필요시 주석 해제
            # self.get_logger().info(f'Odom Update: x={x:.2f}, y={y:.2f}, yaw={yaw_deg:.1f}')
        except Exception as e:
            self.get_logger().error(f'Odom Update Failed: {e}')

#========================================멀티 쓰레드========================================
def main(args=None):
    rclpy.init(args=args)
    
    robot_id = "robot2" 
    other_id = "robot3"
    base_url = "1"

    # 1. Firebase 초기화
    # (이미 되어 있는 코드 유지)
    if not firebase_admin._apps:
        cred = credentials.Certificate("serviceAccountKey.json")
        firebase_admin.initialize_app(cred, {'databaseURL': base_url})

    # 2. [중요] 노드 생성 전 DB 초기값 설정
    # OdomNode 안에 있던 코드를 이리로 가져오세요.
    ref = db.reference(f'robot_status/{robot_id}')
    ref.update({
        "status": "charging",
        "battery": 0,
        "odom_x": 0.0,
        "odom_y": 0.0,
        "odom_yaw": 0.0
    })

    # 3. 노드 생성 (OdomNode 내부의 초기화 코드는 삭제됨)
    bat_node = BatteryNode(robot_id, base_url)
    ctrl_node = StatusControlNode(robot_id, base_url)
    watch_node = WatcherNode(robot_id, other_id)
    odom_node = OdomNode(robot_id, base_url) 

    # 4. 멀티 스레드 실행
    executor = MultiThreadedExecutor()
    executor.add_node(bat_node)
    executor.add_node(ctrl_node)
    executor.add_node(watch_node)
    executor.add_node(odom_node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        # 종료 로직
        rclpy.shutdown()

if __name__ == '__main__':
    main()