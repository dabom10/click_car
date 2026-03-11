#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
[프로젝트: Click Car - 객체 탐지 및 클라우드 연동 노드]
- 담당: 김다봄 개쩜
- 최종 수정: 2026-03-11

[System Architecture & Role]
1. 수신: OAK-D 카메라로부터 CompressedImage 수신 (ROS2)
2. 처리: YOLOv8 기반 실시간 차량 및 번호판 탐지 (Object Detection)
3. 검증: 번호판이 차량 영역 내부에 있는지 공간 필터링 (Overlap Logic)
4. 송신: 탐지된 크롭 이미지를 Base64 인코딩 후 Firebase 업로드 (Cloud Interface)

[Interface 정의]
- Topic (Sub): /robot2/oakd/rgb/image_raw/compressed [sensor_msgs/msg/CompressedImage]
  * 데이터 형식: JPEG 압축 이미지, 약 30 FPS
- Database (Out): Firebase Realtime Database
  * 데이터 형식: JSON (String: base64_image, Float: confidence, Dict: bbox)
'''

import base64
import datetime
import queue
import threading
import time
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
════════════════════════════════════════════════════════════════
[CHAPTER 1 코드 리뷰: 하이퍼파라미터 설계]
════════════════════════════════════════════════════════════════

[설계 결정 1] 전역 상수로 분리한 이유
  클래스 내부나 함수 안에 하드코딩하는 대신 모듈 최상단에 상수로 모은 것은
  운영 환경 변경 시 코드 수정 범위를 최소화하기 위함이다.
  예를 들어 로봇 네임스페이스가 /robot1으로 바뀌거나 모델 경로가 변경될 때,
  해당 상수 한 줄만 수정하면 노드 전체에 즉시 반영된다.

[설계 결정 2] CONF_THRESHOLD 단일화
  초기 설계에는 YOLO 탐지용(0.50)과 DB 저장용(0.70) 두 개의 임계값이 있었다.
  이를 0.50 하나로 통일한 이유는 필터링 단계를 줄여 파이프라인을 단순하게
  유지하면서, Overlap 검증이 2차 필터 역할을 대신하기 때문이다.
  두 임계값을 나누면 "탐지는 됐지만 저장 안 된" 케이스가 발생해
  디버깅 시 혼란을 줄 수 있다.

[설계 결정 3] SAVE_QUEUE_MAXSIZE = 30
  RGB 카메라가 30fps이고 Firebase 평균 지연이 최대 1초라고 가정할 때,
  그 사이에 쌓일 수 있는 최대 프레임은 30장이다.
  큐 상한 없이 방치하면 네트워크 장애 시 메모리가 무한 증가하므로
  maxsize로 상한을 설정하고, 초과 시 최신 프레임을 드롭하는 방식을 선택했다.
  오래된 데이터보다 최신 데이터를 잃는 쪽이 운영상 허용 가능하다고 판단했기
  때문이다. (번호판 탐지 특성상 같은 차량이 여러 프레임에 걸쳐 탐지됨)

[설계 결정 4] CAM_FPS_LOG_INTERVAL = 5.0
  매 프레임마다 FPS를 계산해 로그를 찍으면 로그가 초당 30회 출력되어
  다른 중요한 로그를 묻히게 된다. 5초 간격 집계로 로그 노이즈를 줄이면서
  실시간성도 충분히 유지한다.
════════════════════════════════════════════════════════════════
'''

# ──────────────────────────────────────────────
# [CHAPTER 1: 하이퍼파라미터]
# ──────────────────────────────────────────────
ROBOT_NAMESPACE          = "/robot2"
MODEL_PATH               = "/home/rokey/click_car/models/amr.pt"
FIREBASE_CRED_PATH       = "/home/rokey/click_car/web/click_car.json"
FIREBASE_DB_URL          = "https://clickcar-38016-default-rtdb.asia-southeast1.firebasedatabase.app"
CONF_THRESHOLD           = 0.50   # YOLO 탐지 및 DB 저장 공통 신뢰도 기준
ID_IN_CAR_OVERLAP_THRESH = 0.50   # 번호판이 차량 내부에 있다고 판단할 최소 중첩 비율
YOLO_IMG_SIZE            = 320    # 추론 해상도 (정밀도 ↔ 속도 트레이드오프)
SAVE_QUEUE_MAXSIZE       = 30     # 30fps × 최대 지연 1s 기준 메모리 상한
CAM_FPS_LOG_INTERVAL     = 5.0    # 카메라 FPS 로그 출력 주기 (초)


'''
════════════════════════════════════════════════════════════════
[CHAPTER 2 코드 리뷰: 노드 초기화 및 스레딩 구조]
════════════════════════════════════════════════════════════════

[설계 결정 1] __init__을 3개의 초기화 함수로 분리한 이유
  _load_model / _init_firebase / _init_subscriber 로 역할을 분리한 것은
  단일 책임 원칙(SRP)을 따른 것이다.
  하나의 __init__에 모두 몰아넣으면 어느 단계에서 예외가 발생했는지
  추적하기 어렵고, 테스트 시 특정 초기화 단계만 모킹하기도 힘들다.
  분리 후에는 Firebase만 실패해도 로그로 명확히 표시되고,
  노드가 Firebase 없이도 탐지 기능만으로 동작할 수 있다.

[설계 결정 2] 생산자-소비자 패턴 (Producer-Consumer with Queue)
  Firebase 업로드를 메인 루프와 같은 스레드에서 처리하면 안 되는 이유:
  Firebase REST API 호출은 평균 100~500ms 걸리는 네트워크 I/O다.
  30fps 루프에서 한 프레임당 허용 시간은 약 33ms이므로, 업로드를
  메인 스레드에서 직접 호출하면 탐지 FPS가 2~3fps로 급감한다.

  asyncio 대신 threading + queue를 선택한 이유:
  ROS2 rclpy.spin()이 GIL을 점유하는 동기 루프이기 때문에 asyncio와
  혼용하면 이벤트 루프 충돌이 발생한다. threading.Thread(daemon=True)는
  노드 종료 시 메인 프로세스와 함께 자동으로 정리되므로
  별도의 종료 처리가 단순해진다.

[설계 결정 3] daemon=True 설정
  daemon=True로 설정하지 않으면, 프로세스가 Ctrl+C를 받아도
  워커 스레드가 큐를 비울 때까지 종료되지 않는다.
  Sentinel(None)을 큐에 넣어 정상 종료를 유도하되,
  daemon=True로 강제 종료도 보장해 이중 안전망을 구성했다.

[설계 결정 4] QoS: BEST_EFFORT + KEEP_LAST(depth=1)
  RELIABLE 정책은 패킷 손실 시 재전송을 요청하므로,
  이미 수십ms 지난 프레임이 뒤늦게 도착해 처리 지연을 유발한다.
  카메라 스트리밍처럼 "최신 프레임이 가장 중요한" 상황에서는
  BEST_EFFORT + depth=1 조합이 항상 최신 1장만 유지하므로 최적이다.
════════════════════════════════════════════════════════════════
'''

# ──────────────────────────────────────────────
# [CHAPTER 2: 노드 본체]
# ──────────────────────────────────────────────
class PlateDetectionNode(Node):
    '''
    OAK-D RGB 스트림을 수신하여 차량·번호판을 탐지하고
    검증된 번호판 이미지를 Firebase에 비동기 업로드하는 ROS2 노드.

    스레딩 구조:
        Main Thread  : ROS2 spin → image_callback → YOLO 추론 → 큐 투입
        Worker Thread: save_queue 소비 → Firebase 업로드 (네트워크 I/O 분리)
    '''

    def __init__(self):
        super().__init__("plate_detection_node")

        # 상태 변수
        self.last_detections   = []
        self.last_inference_ms = 0.0
        self.save_queue        = queue.Queue(maxsize=SAVE_QUEUE_MAXSIZE)
        self.db_ref            = None

        # 카메라 FPS 측정용 변수
        # - _cam_frame_count : 마지막 집계 이후 수신된 프레임 수
        # - _cam_fps_timer   : 마지막 집계 시각 (perf_counter 기준)
        # - _cam_fps         : 가장 최근에 계산된 카메라 FPS (오버레이 표시용)
        self._cam_frame_count = 0
        self._cam_fps_timer   = time.perf_counter()
        self._cam_fps         = 0.0

        self._load_model()
        self._init_firebase()
        self._init_subscriber()

        threading.Thread(target=self._upload_worker, daemon=True).start()
        cv2.namedWindow("Plate Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Plate Detection", 704, 704)
        self.get_logger().info("Node ready.")

    # ── 초기화 ──────────────────────────────────

    def _load_model(self):
        ''' YOLO 로드 + Cold Start 방지 워밍업 '''
        self.model = YOLO(MODEL_PATH)
        self.model.predict(
            source=np.zeros((YOLO_IMG_SIZE, YOLO_IMG_SIZE, 3), dtype=np.uint8),
            imgsz=YOLO_IMG_SIZE, verbose=False
        )
        # [DEBUG-1] 모델이 인식하는 클래스명 전체 출력
        # → "car"/"id"가 아닌 다른 이름이 찍히면 아래 필터 조건 수정 필요
        self.get_logger().info(f"[DEBUG-1] Model classes: {self.model.names}")
        self.get_logger().info("YOLO warm-up complete.")

    def _init_firebase(self):
        ''' Firebase Admin SDK 세션 생성. 실패 시 db_ref=None 유지 (업로드 스킵). '''
        try:
            firebase_admin.initialize_app(
                credentials.Certificate(FIREBASE_CRED_PATH),
                {"databaseURL": FIREBASE_DB_URL}
            )
            self.db_ref = db.reference("detections")
            self.get_logger().info("Firebase connected.")
        except Exception as e:
            # [DEBUG-2] 인증 실패 상세 원인 출력 → 경로/URL/권한 문제 확인
            self.get_logger().error(f"[DEBUG-2] Firebase init failed: {e}")

    def _init_subscriber(self):
        ''' Best Effort QoS로 최신 프레임 우선 수신 '''
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        topic = f"{ROBOT_NAMESPACE}/oakd/rgb/image_raw/compressed"
        self.create_subscription(CompressedImage, topic, self.image_callback, qos)
        self.get_logger().info(f"Subscribing to: {topic}")


    '''
    ════════════════════════════════════════════════════════════════
    [CHAPTER 3 코드 리뷰: 실시간 데이터 파이프라인]
    ════════════════════════════════════════════════════════════════

    [설계 결정 1] np.array(msg.data) 사용 이유
      기존 bytes(msg.data) + np.frombuffer() 조합은 데이터를 두 번 복사한다.
      (ROS2 버퍼 → bytes 객체 → NumPy 배열)
      np.array(msg.data, dtype=np.uint8)은 한 번의 복사로 동일한 결과를 얻는다.
      30fps 환경에서 매 프레임 불필요한 복사를 줄이면 GC 압박도 감소한다.

    [설계 결정 2] 카메라 FPS를 callback 호출 횟수로 측정하는 이유
      msg.header.stamp 차이로도 FPS를 계산할 수 있지만,
      ROS2 시간 동기화 문제나 카메라 드라이버의 타임스탬프 부정확성에 영향받는다.
      callback 호출 주기는 실제로 이 노드가 프레임을 수신하는 속도이므로
      "이 노드 관점의 실효 FPS"를 가장 정확하게 반영한다.

    [설계 결정 3] YOLO 추론 시간 측정에 time.perf_counter() 사용
      datetime.datetime.now() 차이 연산은 OS 시스템 클럭을 사용하므로
      수십 ms 단위 측정에서 오차가 발생할 수 있다.
      time.perf_counter()는 단조증가(monotonic) 고해상도 타이머로,
      짧은 구간 측정에서 더 정확하다.

    [설계 결정 4] Overlap 기반 검증 로직을 별도 함수로 분리한 이유
      image_callback 내부에 인라인으로 넣으면 콜백 함수가 지나치게 길어지고,
      임계값 조정이나 알고리즘 교체 시 콜백 전체를 수정해야 한다.
      _find_parent_car()로 분리하면 단위 테스트도 가능하고,
      향후 IoU 계산 방식을 변경해도 콜백 로직에 영향이 없다.

    [설계 결정 5] save_queue.put_nowait() 선택 이유
      blocking put()을 사용하면 큐가 가득 찼을 때 메인 스레드가 대기 상태에
      빠져 ROS2 spin이 지연된다. put_nowait()는 즉시 Full 예외를 발생시키므로
      콜백이 블로킹 없이 반환된다. 프레임 드롭은 허용하되 실시간성을 유지하는
      트레이드오프다.
    ════════════════════════════════════════════════════════════════
    '''

    # ── 메인 파이프라인 ──────────────────────────

    def image_callback(self, msg: CompressedImage):
        '''
        [CHAPTER 3: 실시간 데이터 파이프라인]
        Bytes → NumPy 디코딩 → 카메라 FPS 집계 → YOLO 추론 → Overlap 검증 → 큐 투입
        '''
        # 1. 디코딩
        frame = cv2.imdecode(np.array(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            # [DEBUG-3] 디코딩 실패 → 토픽 데이터 포맷 문제
            self.get_logger().warn("[DEBUG-3] Frame decode failed. Check topic format.")
            return

        # 2. 카메라 FPS 측정
        # callback 호출 주기 = 카메라가 실제로 프레임을 전달하는 주기이므로
        # 이 카운터가 가장 정확한 수신 FPS를 반영합니다.
        self._cam_frame_count += 1
        elapsed = time.perf_counter() - self._cam_fps_timer
        if elapsed >= CAM_FPS_LOG_INTERVAL:
            self._cam_fps = self._cam_frame_count / elapsed
            self.get_logger().info(f"Camera FPS: {self._cam_fps:.1f}")
            self._cam_frame_count = 0
            self._cam_fps_timer   = time.perf_counter()

        # 3. 추론
        cars, ids = self._detect(frame)
        self.last_detections = cars + ids

        # [DEBUG-4] 매 프레임 탐지 결과 요약
        # cars=0, ids=0 → YOLO 미탐 (신뢰도/클래스명 문제)
        # cars>0, ids=0 → 차량만 탐지, 번호판 미탐
        # cars=0, ids>0 → 번호판만 탐지 → Overlap 검증 항상 실패
        self.get_logger().info(
            f"[DEBUG-4] Detect result — cars: {len(cars)}, plates: {len(ids)}"
        )

        # 4. 검증 후 큐 투입
        for id_det in ids:
            car = self._find_parent_car(id_det, cars)

            # [DEBUG-5] 번호판별 Overlap 검증 결과
            self.get_logger().info(
                f"[DEBUG-5] Plate conf={id_det['conf']:.2f} "
                f"bbox=({id_det['x1']},{id_det['y1']},{id_det['x2']},{id_det['y2']}) "
                f"→ parent_car={'FOUND' if car else 'NONE'}"
            )

            if car is None:
                # [DEBUG-6] car가 있는데도 NONE → 실제 overlap 수치 출력
                # overlap 값이 threshold보다 낮으면 ID_IN_CAR_OVERLAP_THRESH 하향 조정 필요
                if cars:
                    id_area = max(1, id_det["area"])
                    for i, c in enumerate(cars):
                        ix = max(0, min(id_det["x2"], c["x2"]) - max(id_det["x1"], c["x1"]))
                        iy = max(0, min(id_det["y2"], c["y2"]) - max(id_det["y1"], c["y1"]))
                        ov = (ix * iy) / id_area
                        self.get_logger().info(
                            f"[DEBUG-6]   car[{i}] "
                            f"bbox=({c['x1']},{c['y1']},{c['x2']},{c['y2']}) "
                            f"overlap={ov:.3f} / threshold={ID_IN_CAR_OVERLAP_THRESH}"
                        )
                continue

            self.get_logger().info(
                f"Validated: car={car['conf']:.2f}, plate={id_det['conf']:.2f}"
            )
            try:
                self.save_queue.put_nowait({"frame": frame, "car": car, "id": id_det})
                # [DEBUG-7] 큐 투입 성공 확인
                self.get_logger().info(
                    f"[DEBUG-7] Queued OK. size={self.save_queue.qsize()}/{SAVE_QUEUE_MAXSIZE}"
                )
            except queue.Full:
                self.get_logger().warn(
                    f"Queue full ({SAVE_QUEUE_MAXSIZE}). Frame dropped."
                )

        self._draw(frame)


    '''
    ════════════════════════════════════════════════════════════════
    [_detect 코드 리뷰: YOLO 추론 엔진]
    ════════════════════════════════════════════════════════════════

    [설계 결정 1] imgsz=320 선택 이유
      YOLOv8은 입력 해상도가 클수록 정밀도는 높아지지만 추론 시간이 증가한다.
      640의 경우 CPU에서 약 200ms(5fps), 320은 약 50ms(20fps) 수준이다.
      번호판은 비교적 작은 객체이므로 320에서도 충분한 탐지율을 보이며,
      실시간 30fps 처리를 위해 속도를 우선했다.

    [설계 결정 2] verbose=False 설정
      YOLO의 기본 verbose=True는 매 추론마다 터미널에 결과를 출력한다.
      30fps 환경에서 초당 30줄이 출력되면 ROS2 로그가 묻히고
      터미널 I/O 자체가 성능 병목이 될 수 있어 비활성화했다.

    [설계 결정 3] results[0].boxes 순회 방식
      results가 리스트로 반환되는 이유는 배치 추론(여러 이미지 동시 처리)을
      지원하기 위해서다. 이 노드는 단일 프레임 처리이므로 results[0]만 사용한다.
      results가 빈 리스트일 경우의 예외 처리를 앞에 두어 불필요한 순회를 막는다.

    [설계 결정 4] box.xyxy[0].cpu().numpy() 변환 이유
      YOLO 출력은 기본적으로 GPU 텐서(torch.Tensor)다.
      OpenCV와 NumPy는 CPU 메모리를 사용하므로 .cpu()로 디바이스를 이동 후
      .numpy()로 변환해야 한다. GPU가 없는 환경에서도 .cpu()는 무해하다.
    ════════════════════════════════════════════════════════════════
    '''

    def _detect(self, frame: np.ndarray) -> tuple[list, list]:
        '''
        YOLOv8 추론. 추론 시간을 측정하여 로그 및 오버레이에 반영.
        반환: (cars, ids) — 각각 탐지 dict 리스트
        '''
        t0 = time.perf_counter()
        results = self.model.predict(source=frame, imgsz=YOLO_IMG_SIZE,
                                     conf=CONF_THRESHOLD, verbose=False)
        self.last_inference_ms = (time.perf_counter() - t0) * 1000
        self.get_logger().info(
            f"Inference: {self.last_inference_ms:.1f} ms "
            f"({1000 / self.last_inference_ms:.1f} FPS)"
        )

        cars, ids = [], []
        if not results:
            return cars, ids

        for box in results[0].boxes:
            name = self.model.names.get(int(box.cls[0].item()))
            if name not in ("car", "id"):
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            det = {
                "class_name": name,
                "conf":       float(box.conf[0].item()),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "area": max(0, x2 - x1) * max(0, y2 - y1),
            }
            (cars if name == "car" else ids).append(det)

        return cars, ids


    '''
    ════════════════════════════════════════════════════════════════
    [_find_parent_car 코드 리뷰: Overlap 기반 포함 관계 검증]
    ════════════════════════════════════════════════════════════════

    [설계 결정 1] IoU 대신 단방향 Overlap을 선택한 이유
      일반적인 IoU(Intersection over Union)는 두 박스의 합집합 대비 교집합 비율이다.
      하지만 번호판은 차량보다 훨씬 작기 때문에, IoU를 쓰면 번호판이 차량 안에
      완전히 들어있어도 분모(합집합)가 차량 면적에 지배되어 IoU 값이 낮게 나온다.
      따라서 "번호판 면적 대비 교집합 비율"인 단방향 overlap을 사용해
      번호판이 차량 내부에 있는지를 더 민감하게 판별한다.

    [설계 결정 2] candidates + max() 패턴 사용 이유
      여러 차량이 동시에 탐지될 수 있는 상황에서 단순 for문으로 best를 갱신하는
      방식 대신, list comprehension으로 후보를 먼저 걸러낸 뒤 max()로 최고값을
      선택하는 방식은 더 읽기 쉽고, 후보가 없는 경우(return None)도 한 줄로 처리된다.

    [설계 결정 3] id_area = max(1, id_det["area"]) 처리
      bbox 좌표 오류나 단일 픽셀 탐지로 area가 0이 될 경우 ZeroDivisionError가
      발생한다. max(1, area)로 최솟값을 보장해 예외 없이 안전하게 처리한다.
    ════════════════════════════════════════════════════════════════
    '''

    def _find_parent_car(self, id_det: dict, cars: list) -> dict | None:
        '''
        번호판(id_det)이 어느 차량(cars) 내부에 속하는지 Overlap 비율로 판별.
        알고리즘: intersection(id, car) / area(id) >= ID_IN_CAR_OVERLAP_THRESH
        '''
        id_area = max(1, id_det["area"])

        def overlap(car):
            ix = max(0, min(id_det["x2"], car["x2"]) - max(id_det["x1"], car["x1"]))
            iy = max(0, min(id_det["y2"], car["y2"]) - max(id_det["y1"], car["y1"]))
            return (ix * iy) / id_area

        candidates = [(overlap(car), car) for car in cars if overlap(car) >= ID_IN_CAR_OVERLAP_THRESH]
        return max(candidates, key=lambda x: x[0])[1] if candidates else None


    '''
    ════════════════════════════════════════════════════════════════
    [_draw 코드 리뷰: 모니터링 시각화]
    ════════════════════════════════════════════════════════════════

    [설계 결정 1] 시각화 정보를 3구역으로 나눈 이유
      좌상단(추론 성능) / 중앙상단(카메라 FPS) / 우상단(큐 상태)으로 분리해
      운영자가 한눈에 시스템 상태를 파악할 수 있도록 했다.
      세 지표는 각각 AI 처리 속도 / 카메라 입력 속도 / 백엔드 업로드 속도를
      나타내므로, 어느 구간에서 병목이 발생했는지 즉시 진단 가능하다.

    [설계 결정 2] 큐 색상 3단계 (초록→주황→빨강) 이유
      큐가 50% 미만이면 정상(초록), 50~99%면 경고(주황), 가득 차면 위험(빨강)으로
      구분했다. 50% 시점부터 경고를 주는 것은 Firebase 지연이 갑자기 증가할 때
      운영자가 사전에 인지할 수 있도록 하기 위함이다.

    [설계 결정 3] cv2.getTextSize()로 우상단 텍스트 위치를 동적 계산하는 이유
      큐 사이즈 텍스트는 "Queue: 1/30"처럼 숫자에 따라 길이가 달라진다.
      고정 좌표를 사용하면 텍스트가 화면 밖으로 나가거나 중앙 텍스트와 겹칠 수 있다.
      getTextSize()로 픽셀 폭을 측정해 frame.shape[1]에서 역산하면
      항상 우측 정렬이 유지된다.
    ════════════════════════════════════════════════════════════════
    '''

    # ── 시각화 ──────────────────────────────────

    def _draw(self, frame: np.ndarray):
        '''
        Bounding Box 오버레이 + 상단 HUD
          좌상단: YOLO 추론 시간 / 추론 FPS
          중앙상단: 카메라 수신 FPS
          우상단: Firebase 업로드 큐 상태 (초록 → 주황 → 빨강)
        '''
        for det in self.last_detections:
            color = (0, 255, 0) if det["class_name"] == "car" else (0, 0, 255)
            cv2.rectangle(frame, (det["x1"], det["y1"]), (det["x2"], det["y2"]), color, 2)
            cv2.putText(frame, f"{det['class_name']} {det['conf']:.2f}",
                        (det["x1"], max(25, det["y1"] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 좌상단: YOLO 추론 성능
        infer_fps = 1000 / self.last_inference_ms if self.last_inference_ms > 0 else 0
        cv2.putText(frame, f"Infer: {self.last_inference_ms:.1f}ms ({infer_fps:.1f}FPS)",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # 중앙상단: 카메라 수신 FPS
        cam_text = f"Cam: {self._cam_fps:.1f}FPS"
        (cw, _), _ = cv2.getTextSize(cam_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(frame, cam_text,
                    ((frame.shape[1] - cw) // 2, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 우상단: 큐 상태
        q_size  = self.save_queue.qsize()
        q_ratio = q_size / SAVE_QUEUE_MAXSIZE
        q_color = (0, 255, 0) if q_ratio < 0.5 else (0, 165, 255) if q_ratio < 1.0 else (0, 0, 255)
        q_text  = f"Queue: {q_size}/{SAVE_QUEUE_MAXSIZE}"
        (tw, _), _ = cv2.getTextSize(q_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(frame, q_text, (frame.shape[1] - tw - 10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, q_color, 2)

        cv2.imshow("Plate Detection", frame)
        cv2.waitKey(1)


    '''
    ════════════════════════════════════════════════════════════════
    [CHAPTER 4 코드 리뷰: 백그라운드 업로드 워커]
    ════════════════════════════════════════════════════════════════

    [설계 결정 1] _upload_worker와 _upload를 분리한 이유
      _upload_worker는 큐 소비 루프만 담당하고,
      실제 데이터 처리는 _upload에 위임한다.
      이렇게 분리하면 _upload만 단독으로 테스트하거나 재사용할 수 있고,
      예외가 발생해도 _upload_worker의 while True 루프는 계속 유지된다.
      만약 하나의 함수에 몰아넣으면 _upload 내부 예외가 루프 전체를 종료시킨다.

    [설계 결정 2] Sentinel 패턴(None 투입)으로 종료하는 이유
      threading.Event나 flag 변수로도 종료를 신호할 수 있지만,
      Sentinel 방식은 큐에 남아있는 모든 작업을 처리한 후 종료된다는 것이 보장된다.
      즉, Ctrl+C 직전에 투입된 탐지 결과까지 Firebase에 업로드를 시도한 뒤 종료된다.
      flag 방식은 큐에 남은 항목을 버리고 즉시 종료할 수 있다.

    [설계 결정 3] 번호판(id) 기준으로 crop하는 이유
      차량(car) 전체 이미지를 저장하면 데이터 용량이 크고,
      웹 대시보드에서 번호판을 식별하기 위해 불필요한 부분이 많다.
      번호판 영역만 crop하면 용량을 약 90% 절감하면서
      Firebase Realtime Database의 노드 크기 제한(~10MB)에도 안전하게 유지된다.

    [설계 결정 4] Base64 인코딩을 선택한 이유
      Firebase Realtime Database는 바이너리 저장을 직접 지원하지 않는다.
      Base64는 바이너리를 JSON 문자열로 직렬화하는 표준 방식이며,
      웹 클라이언트에서 <img src="data:image/jpeg;base64,..."> 형태로
      별도 다운로드 없이 이미지를 바로 렌더링할 수 있다.
    ════════════════════════════════════════════════════════════════
    '''

    # ── Firebase 업로드 워커 ─────────────────────

    def _upload_worker(self):
        '''
        [CHAPTER 4: 백그라운드 업로드 워커 (Consumer)]
        save_queue를 소비하며 Firebase에 직렬화 전송.
        None 수신 시 종료 (Sentinel 패턴).
        '''
        while True:
            item = self.save_queue.get()
            if item is None:
                break
            try:
                self._upload(item["frame"], item["car"], item["id"])
            except Exception as e:
                self.get_logger().error(f"Upload error: {e}")

    def _upload(self, frame: np.ndarray, car: dict, id_det: dict):
        '''
        번호판 영역 크롭 → JPG 재압축 → Base64 인코딩 → Firebase 전송.
        경로: detections/{timestamp}
        '''
        # [DEBUG-8] Firebase 연결 상태 확인
        if self.db_ref is None:
            self.get_logger().error(
                "[DEBUG-8] db_ref is None — Firebase 미연결. "
                "시작 로그의 [DEBUG-2] 확인 필요."
            )
            return

        x1, y1, x2, y2 = id_det["x1"], id_det["y1"], id_det["x2"], id_det["y2"]
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            # [DEBUG-9] bbox 좌표 오류로 crop 영역이 비어있는 경우
            self.get_logger().warn(
                f"[DEBUG-9] Empty crop. "
                f"id_bbox=({x1},{y1},{x2},{y2}), frame_shape={frame.shape}"
            )
            return

        now = datetime.datetime.now()
        _, enc = cv2.imencode(".jpg", crop)
        b64    = base64.b64encode(enc.tobytes()).decode("utf-8")

        self.db_ref.child(now.strftime("%Y%m%d_%H%M%S_%f")).set({
            "detected_at":    now.isoformat(),
            "car_confidence": round(car["conf"], 4),
            "id_confidence":  round(id_det["conf"], 4),
            "car_bbox": {"x1": car["x1"], "y1": car["y1"], "x2": car["x2"], "y2": car["y2"]},
            "id_bbox":  {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "image_base64": b64,
        })
        self.get_logger().info(f"Uploaded: {now.strftime('%Y%m%d_%H%M%S_%f')}")


    '''
    ════════════════════════════════════════════════════════════════
    [destroy_node 코드 리뷰: 자원 회수 루틴]
    ════════════════════════════════════════════════════════════════

    [설계 결정 1] super().destroy_node()를 마지막에 호출하는 이유
      ROS2 노드 자원(구독, 퍼블리셔, 타이머 등)은 super().destroy_node()가
      해제한다. 이 호출 이후에는 get_logger() 등 ROS2 API를 사용할 수 없다.
      따라서 큐 Sentinel 투입과 GUI 종료를 먼저 처리한 뒤 마지막에 호출해야
      종료 시퀀스 중 로깅이 가능하고 순서 의존성이 없다.

    [설계 결정 2] cv2.destroyAllWindows()를 여기서 호출하는 이유
      OpenCV GUI는 메인 스레드에서만 안전하게 종료할 수 있다.
      _upload_worker 스레드에서 호출하면 세그폴트가 발생할 수 있어
      항상 메인 스레드인 destroy_node()에서 처리한다.
    ════════════════════════════════════════════════════════════════
    '''

    # ── 종료 ────────────────────────────────────

    def destroy_node(self):
        ''' Sentinel(None) 전송으로 워커 스레드 안전 종료 후 GUI 해제. '''
        self.save_queue.put(None)
        cv2.destroyAllWindows()
        super().destroy_node()


'''
════════════════════════════════════════════════════════════════
[CHAPTER 5 코드 리뷰: 엔트리 포인트]
════════════════════════════════════════════════════════════════

[설계 결정 1] try/finally 구조로 종료를 보장하는 이유
  rclpy.spin()은 Ctrl+C 시 KeyboardInterrupt를 발생시킨다.
  except로 이를 조용히 처리하고 finally에서 항상 destroy_node()와
  rclpy.shutdown()을 호출함으로써 어떤 종료 시나리오에서도
  (정상 종료, Ctrl+C, 예외) ROS2 컨텍스트가 정상적으로 해제된다.

[설계 결정 2] rclpy.ok() 체크 후 shutdown() 호출하는 이유
  rclpy.shutdown()을 이미 종료된 컨텍스트에 중복 호출하면 예외가 발생한다.
  런치 파일이나 외부 신호로 이미 shutdown이 호출된 경우를 방어하기 위해
  rclpy.ok()로 상태를 먼저 확인한다.
════════════════════════════════════════════════════════════════
'''

# ──────────────────────────────────────────────
# [CHAPTER 5: 엔트리 포인트]
# ──────────────────────────────────────────────
def main(args=None):
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