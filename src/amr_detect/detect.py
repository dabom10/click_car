#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
[프로젝트 개요]
본 노드는 ROS2 환경에서 동작하며, 카메라로부터 수신된 차량 이미지에서 
YOLO 모델을 통해 '차량'과 '번호판(id)'을 실시간으로 탐지합니다.
탐지된 데이터는 Firebase Realtime Database에 업로드되어 웹 시스템과 연동됩니다.
'데이터 전송 효율(JPG)'과 'AI 추론 정밀도(NumPy)'의 균형을 맞춘 파이프라인이 핵심입니다.
'''

import base64
import datetime
import queue
import threading
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage
from ultralytics import YOLO

import firebase_admin
from firebase_admin import credentials, db

'''
[CHAPTER 1: 시스템 설정 및 경로 정의]
로봇 네임스페이스, 모델 가중치 경로, Firebase 인증 정보 및 
탐지 임계값(Threshold) 등 노드 운영에 필요한 핵심 파라미터를 설정합니다.
'''
ROBOT_NAMESPACE = "/robot2"
MODEL_PATH            = "/home/rokey/click_car/models/AMR/v1/weights/best.pt"
FIREBASE_CRED_PATH    = "/home/rokey/click_car/web/database.json"
FIREBASE_DB_URL       = "https://click-car-2f586-default-rtdb.asia-southeast1.firebasedatabase.app"

CONF_THRESHOLD        = 0.50   
DETECT_CONF_THRESHOLD = 0.70   
ID_IN_CAR_OVERLAP_THRESH = 0.5
YOLO_IMG_SIZE = 320            

class PlateDetectionNode(Node):
    def __init__(self):
        '''
        [CHAPTER 2: 노드 초기화 및 환경 구성]
        1. YOLO 모델 로드 및 '워밍업' 수행: 첫 프레임의 추론 지연을 방지합니다.
        2. Firebase 연결 초기화: 클라우드 DB 연동을 준비합니다.
        3. 병렬 처리 스레드 생성: DB I/O 작업이 메인 루프를 방해하지 않도록 격리합니다.
        4. ROS2 구독 설정: 최신성 유지를 위해 BEST_EFFORT 통신 정책을 적용합니다.
        '''
        super().__init__("plate_detection_node")

        self.last_detections = []
        self.save_queue      = queue.Queue()

        rgb_topic = f"{ROBOT_NAMESPACE}/oakd/rgb/image_raw/compressed"
        self.get_logger().info(f"토픽: {rgb_topic} | 모델: {MODEL_PATH}")

        self.model = YOLO(MODEL_PATH)
        self.model.predict(
            source=np.zeros((YOLO_IMG_SIZE, YOLO_IMG_SIZE, 3), dtype=np.uint8),
            imgsz=YOLO_IMG_SIZE, verbose=False
        )
        self.get_logger().info("YOLO 워밍업 완료")

        self._init_firebase()

        threading.Thread(target=self._db_save_worker, daemon=True).start()

        cv2.namedWindow("Plate Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Plate Detection", 640, 480)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.create_subscription(CompressedImage, rgb_topic, self.image_callback, qos)
        self.get_logger().info("노드 시작!")

    def _init_firebase(self):
        '''
        [로직: Firebase 관리자 인증]
        제공된 JSON 키 파일을 사용하여 데이터베이스 접근 권한을 획득합니다.
        '''
        try:
            cred = credentials.Certificate(FIREBASE_CRED_PATH)
            firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_DB_URL})
            self.db_ref = db.reference("detections")
            self.get_logger().info("Firebase 초기화 성공")
        except Exception as e:
            self.get_logger().error(f"Firebase 초기화 실패: {e}")
            self.db_ref = None

    def image_callback(self, msg: CompressedImage):
        '''
        [CHAPTER 3: 실시간 데이터 파이프라인]
        
        1. 수신 및 디코딩: 
           전송 효율을 위해 JPG로 압축된 데이터를 수신하고, 
           AI 분석을 위해 정밀한 픽셀 정보를 가진 NumPy 배열로 복원합니다.
        
        2. 탐지 및 필터링:
           YOLO를 통해 사물을 찾고, 번호판이 실제 차량 영역 내부에 있는지 
           공간적 검증을 거쳐 신뢰할 수 있는 데이터만 선별합니다.
        '''
        try:
            np_arr = np.frombuffer(bytes(msg.data), np.uint8)
            frame  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                self.get_logger().error("이미지 디코딩 실패")
                return
        except Exception as e:
            self.get_logger().error(f"디코딩 오류: {e}")
            return

        cars, ids = self._detect(frame)
        self.last_detections = cars + ids

        for id_det in ids:
            if id_det["conf"] < DETECT_CONF_THRESHOLD:
                continue
            car = self._find_car_for_id(id_det, cars)
            if car is None:
                continue
            self.get_logger().info(
                f"번호판 감지! car={car['conf']:.2f} id={id_det['conf']:.2f}"
            )
            self.save_queue.put({"frame": frame, "car": car, "id": id_det})

        self._draw(frame)

    def _detect(self, frame):
        '''
        [로직: YOLO 객체 탐지]
        입력된 NumPy 프레임에서 'car'와 'id' 클래스를 추출합니다.
        연산 속도 최적화를 위해 이미지 사이즈를 320으로 조정하여 추론합니다.
        '''
        cars, ids = [], []
        results = self.model.predict(
            source=frame, imgsz=YOLO_IMG_SIZE,
            conf=CONF_THRESHOLD, verbose=False
        )
        if not results:
            return cars, ids

        for box in results[0].boxes:
            cls_id = int(box.cls[0].item())
            conf   = float(box.conf[0].item())
            name   = self.model.names.get(cls_id, str(cls_id))
            if name not in ("car", "id"):
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            det = {"class_name": name, "conf": conf,
                   "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                   "area": max(0, x2-x1) * max(0, y2-y1)}
            (cars if name == "car" else ids).append(det)

        return cars, ids

    def _find_car_for_id(self, id_det, cars):
        '''
        [로직: 공간 포함 관계 검증]
        탐지된 번호판(id)의 좌표가 어떤 차량(car) 영역 내부에 위치하는지 계산합니다.
        이는 주변의 텍스트 오탐지를 걸러내어 탐지 정확성을 보장하는 안전장치입니다.
        '''
        best, best_ov = None, 0.0
        id_area = max(1, id_det["area"])
        for car in cars:
            ix = max(0, min(id_det["x2"], car["x2"]) - max(id_det["x1"], car["x1"]))
            iy = max(0, min(id_det["y2"], car["y2"]) - max(id_det["y1"], car["y1"]))
            ov = (ix * iy) / id_area
            if ov >= ID_IN_CAR_OVERLAP_THRESH and ov > best_ov:
                best_ov, best = ov, car
        return best

    def _draw(self, frame):
        '''
        [로직: 결과 시각화]
        모니터링을 위해 OpenCV 창에 탐지된 객체의 바운딩 박스와 신뢰도를 표시합니다.
        '''
        for det in self.last_detections:
            color = (0, 255, 0) if det["class_name"] == "car" else (0, 0, 255)
            cv2.rectangle(frame, (det["x1"], det["y1"]), (det["x2"], det["y2"]), color, 2)
            cv2.putText(frame, f"{det['class_name']} {det['conf']:.2f}",
                        (det["x1"], max(25, det["y1"]-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imshow("Plate Detection", frame)
        cv2.waitKey(1)

    def _db_save_worker(self):
        '''
        [CHAPTER 4: 백그라운드 데이터 처리]
        메인 스레드와 분리된 작업자가 큐에서 데이터를 하나씩 꺼내 DB 업로드를 수행합니다.
        네트워크 지연 발생 시에도 로봇의 실시간 탐지 주기가 유지되도록 설계되었습니다.
        '''
        while True:
            item = self.save_queue.get()
            if item is None:
                break
            try:
                self._save(item["frame"], item["car"], item["id"])
            except Exception as e:
                self.get_logger().error(f"저장 오류: {e}")

    def _save(self, frame, car, id_det):
        '''
        [로직: 데이터 영속화 및 최적화]
        1. 이미지 크롭: 번호판 영역만 NumPy 슬라이싱으로 추출합니다.
        2. 재압축 및 인코딩: 
           무거운 NumPy 이미지를 다시 효율적인 JPG로 압축한 뒤 Base64로 인코딩합니다.
           이는 DB 업로드 속도를 높이고 저장 공간을 절약하기 위함입니다.
        '''
        if self.db_ref is None:
            return
        x1, y1, x2, y2 = id_det["x1"], id_det["y1"], id_det["x2"], id_det["y2"]
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return

        now = datetime.datetime.now()
        ts  = now.strftime("%Y%m%d_%H%M%S_%f")

        _, enc   = cv2.imencode(".jpg", crop)
        b64      = base64.b64encode(enc.tobytes()).decode("utf-8")

        self.db_ref.child(ts).set({
            "detected_at":    now.isoformat(),
            "car_confidence": round(car["conf"], 4),
            "id_confidence":  round(id_det["conf"], 4),
            "car_bbox": {"x1": car["x1"], "y1": car["y1"],
                         "x2": car["x2"], "y2": car["y2"]},
            "id_bbox":  {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "image_base64": b64,
        })
        self.get_logger().info(f"DB 저장 완료: detections/{ts}")

    def destroy_node(self):
        '''
        [로직: 노드 종료 및 자원 정리]
        작업 스레드에 정지 신호를 보내고 생성된 모든 GUI 창을 닫습니다.
        '''
        self.save_queue.put(None)
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    '''
    [CHAPTER 5: 프로그램 실행부]
    ROS2 통신 환경을 초기화하고 노드를 실행(Spin)합니다.
    사용자 종료 시 자원을 안전하게 해제한 후 시스템을 종료합니다.
    '''
    rclpy.init(args=args)
    node = PlateDetectionNode()
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