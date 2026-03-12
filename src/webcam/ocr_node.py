#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
[프로젝트: Click Car - AMR 카메라 기반 불법주정차 단속 노드]
- 최종 수정: 2026-03-12

[System Architecture & Role]
1. 수신     : OAK-D RGB 카메라로부터 CompressedImage 수신 (ROS2 토픽)
2. 모드 수신 : std_msgs/String 토픽으로 동작 모드 수신
               - "amr_start"  → AMR 직접 출동 (단속 타이머 30초 + Santa Claus 알림음)
               - "cctv_start" → CCTV 연동 모드 (단속 타이머 5초, 알림음 없음)
3. 탐지     : YOLOv8 기반 차량(car) + 번호판(id) 동시 탐지
4. 필터링   : 화면 내 여러 차량 중 Bounding Box 면적이 가장 큰 차량 1대만 단속 대상
5. 검증     : 번호판이 차량 영역 내부에 있는지 Overlap 비율로 검증
6. 추적     : IoU 기반 동일 차량 식별 + 모드별 타이머 관리
7. 업로드   : Firebase Realtime DB에 케이스별 누적 저장
               - initial  : 최초 감지 시 전체 프레임 저장, status="watching"
               - confirmed: 타이머 초과 시 증거 프레임 + 번호판 크롭 + OCR 결과 기록
8. 알림음   : amr_start 모드에서 타이머 동작 중
               "Santa Claus is Coming to Town" 멜로디를 AudioNoteVector 토픽으로 퍼블리시

[Firebase 데이터 구조]
  detections/<YYYYMMDD_HHMMSSffffff>/
    ├── status          "watching" | "confirmed"
    ├── detected_at     최초 감지 ISO 시각
    ├── confirmed_at    확정 ISO 시각 (confirmed 시에만 채워짐)
    ├── initial_image   최초 감지 전체 프레임 JPEG → base64
    ├── evidence_image  확정 시 전체 프레임 JPEG → base64
    ├── plate_image     확정 시 번호판 크롭 JPEG → base64  ← confirmed 시에만 기록
    └── plate_number    OCR 인식 결과                      ← confirmed 시에만 기록
'''

import base64        # 이미지를 Firebase에 저장하기 위한 base64 인코딩
import datetime      # 감지 시각을 ISO 문자열로 기록
import queue         # 메인 루프와 업로드 워커 간 스레드 안전 데이터 전달
import threading     # OCR·업로드를 메인 루프와 분리하기 위한 스레드
import time          # 단속 타이머 계산용 (monotonic clock)

import cv2           # 이미지 디코딩, 크롭, 엣지 검출, 시점 변환
import numpy as np   # 행렬 연산 (시점 변환 좌표 계산 등)
import rclpy                                   # ROS2 Python 클라이언트 라이브러리
from rclpy.node import Node                    # ROS2 노드 기반 클래스
from rclpy.qos import (                        # 카메라 토픽 구독 품질 설정
    QoSProfile, ReliabilityPolicy, HistoryPolicy
)
from sensor_msgs.msg import CompressedImage            # 카메라 이미지 수신
from std_msgs.msg import String                        # 모드 명령 수신 ("amr_start" / "cctv_start")
from irobot_create_msgs.msg import AudioNoteVector, AudioNote  # AMR 알림음 송신
from builtin_interfaces.msg import Duration            # AudioNote 지속 시간 표현
from ultralytics import YOLO                           # YOLOv8 추론 엔진
import firebase_admin                                  # Firebase Admin SDK
from firebase_admin import credentials, db             # 인증 및 Realtime DB 접근
from paddleocr import PaddleOCR                        # 번호판 OCR 엔진


# ──────────────────────────────────────────────
# [CHAPTER 1: 하이퍼파라미터]
# ──────────────────────────────────────────────

ROBOT_NAMESPACE          = "/robot2"                                                               # ROS2 네임스페이스
MODEL_PATH               = "/home/rokey/click_car/models/amr.pt"                                  # YOLOv8 가중치 경로
FIREBASE_CRED_PATH       = "/home/rokey/click_car/web/click_car.json"                             # Firebase 서비스 계정 키 경로
FIREBASE_DB_URL          = "https://iligalstop-default-rtdb.asia-southeast1.firebasedatabase.app" # Realtime DB URL
FIREBASE_DB_PATH         = "detections"          # 단속 로그를 저장할 DB 상위 경로 (하위에 케이스별 노드가 누적됨)

CONF_THRESHOLD           = 0.50    # YOLO 탐지 신뢰도 임계값: 이 값 미만의 bbox는 무시
ID_IN_CAR_OVERLAP_THRESH = 0.50    # 번호판이 차량 bbox 내부에 이 비율 이상 겹쳐야 유효한 쌍으로 인정
CAR_IOU_THRESH           = 0.30    # 이전·현재 프레임 차량을 동일 객체로 판단하는 IoU 최솟값
YOLO_IMG_SIZE            = 704     # YOLO 추론 이미지 크기 (픽셀, 정사각형)
PARKING_TIMEOUT_AMR      = 30.0    # amr_start  모드: 30초 (AMR 직접 출동, 사이렌 동반)
PARKING_TIMEOUT_CCTV     =  5.0    # cctv_start 모드:  5초 (CCTV 연동 빠른 확인)
SAVE_QUEUE_MAXSIZE       = 50      # 업로드 큐 최대 항목 수: 초과 시 신규 항목 드랍

# ROS2 토픽 경로
TOPIC_RGB   = f"{ROBOT_NAMESPACE}/oakd/rgb/image_raw/compressed"  # 카메라 이미지 구독
TOPIC_CMD   = f"{ROBOT_NAMESPACE}/parking_cmd"                    # 모드 명령 수신
TOPIC_AUDIO = f"{ROBOT_NAMESPACE}/cmd_audio"                      # AMR 알림음 퍼블리시


# ──────────────────────────────────────────────
# [CHAPTER 2: Santa Claus is Coming to Town 멜로디]
# ──────────────────────────────────────────────
# 각 튜플: (주파수 Hz, 지속 시간 ms)
# BPM ≈ 120 → 4분음표 = 500ms, 8분음표 = 250ms
# 음계: C4=262, D4=294, E4=330, F4=349, F#4=370, G4=392, A4=440, B4=494, C5=523
# 총 재생 시간 ≈ 35초 → amr_start 모드의 30초 타이머 전체를 커버한다.

SANTA_NOTES = [
    # ── "You better watch out" ──
    (392, 250), (392, 250), (392, 500), (330, 500), (392, 1000),
    # ── "You better not cry" ──
    (440, 250), (440, 250), (440, 500), (370, 500), (440, 1000),
    # ── "Better not pout" ──
    (392, 250), (392, 250), (392, 500), (330, 500), (392,  500),
    # ── "I'm telling you why" ──
    (523, 500), (523, 500), (494, 500), (440, 500), (392, 1000),
    # ── "Santa Claus is coming to town" ──
    (330, 250), (330, 250), (349, 250), (392, 250),
    (392, 250), (349, 250), (330, 250), (294, 250), (262, 1000),
    # ── "He sees you when you're sleeping" ──
    (392, 250), (392, 250), (392, 250), (392, 250),
    (440, 250), (392, 250), (330, 250), (262,  750),
    # ── "He knows when you're awake" ──
    (330, 250), (330, 250), (349, 250), (392, 250), (440, 1000),
    # ── "He knows if you've been bad or good" ──
    (392, 250), (392, 250), (440, 250), (392, 250), (330, 250), (262, 750),
    # ── "So be good for goodness sake!" ──
    (294, 250), (294, 250), (330, 250), (294, 500), (392, 1000),
    # ── "Oh! You better watch out" ──
    (392, 250), (392, 250), (392, 500), (330, 500), (392, 1000),
    # ── "You better not cry" ──
    (440, 250), (440, 250), (440, 500), (370, 500), (440, 1000),
    # ── "Better not pout, I'm telling you why" ──
    (392, 250), (392, 250), (392, 500), (330, 500), (392,  500),
    (523, 500), (523, 500), (494, 500), (440, 500), (392, 1000),
    # ── "Santa Claus is coming to town" (마지막, 롱 엔딩) ──
    (330, 250), (330, 250), (349, 250), (392, 250),
    (392, 250), (349, 250), (330, 250), (294, 250), (262, 1500),
]


# ──────────────────────────────────────────────
# [CHAPTER 3: 차량 추적 상태 컨테이너]
# ──────────────────────────────────────────────

class TrackedVehicle:
    '''
    단일 차량의 타이머 및 최신 탐지 상태를 보관하는 데이터 컨테이너.

    최초 발견 시각(first_seen)을 기준으로 30초 단속 타이머를 계산하며,
    DB 업로드 중복을 막기 위한 플래그(initial_uploaded, confirmed_uploaded)를 관리한다.
    '''

    def __init__(self, car_det: dict, id_det: dict):
        '''
        최초 객체 생성 시 monotonic clock으로 타이머를 시작한다.
        monotonic clock은 시스템 시각 변경에 영향받지 않아 경과 시간 측정에 적합하다.
        '''
        now                      = time.monotonic()
        self.first_seen          = now          # 타이머 기준 시각 (이후 절대 변경하지 않음)
        self.last_seen           = now          # 마지막 감지 시각 (grace period 판단용)
        self.car_det             = car_det      # 최신 차량 bbox 딕셔너리
        self.id_det              = id_det       # 최신 번호판 bbox 딕셔너리
        self.initial_uploaded    = False        # 최초 감지 업로드 완료 여부 (중복 방지)
        self.confirmed_uploaded  = False        # 30초 확정 업로드 완료 여부 (중복 방지)

    def elapsed(self) -> float:
        ''' 최초 감지 시점으로부터 현재까지 경과한 시간(초)을 반환한다. '''
        return time.monotonic() - self.first_seen

    def update(self, car_det: dict, id_det: dict):
        '''
        다음 프레임에서 동일 차량이 재감지되었을 때 최신 bbox 좌표와 감지 시각을 갱신한다.
        first_seen은 갱신하지 않아 타이머가 계속 누적된다.
        '''
        self.last_seen = time.monotonic()   # 마지막 감지 시각 갱신
        self.car_det   = car_det            # 최신 차량 bbox 로 교체
        self.id_det    = id_det             # 최신 번호판 bbox 로 교체


# ──────────────────────────────────────────────
# [CHAPTER 3: 메인 ROS2 노드]
# ──────────────────────────────────────────────

class ParkingDetectionNode(Node):
    '''
    카메라 영상을 받아 불법주정차를 단속하는 메인 ROS2 노드.

    설계 원칙:
      - 모드 분리  : "amr_start"(30초+알림음) / "cctv_start"(5초) 를 토픽으로 수신해 동적 전환
      - 단일 타겟  : 화면에 여러 차량이 있어도 bbox 면적 최대 1대만 추적
      - DB 누적    : 케이스마다 타임스탬프 키로 새 노드 생성 → 이전 로그 영구 보존
      - 스레드 분리: 메인 스레드는 영상 수신·YOLO 탐지만 수행하고,
                     무거운 OCR·네트워크 업로드는 워커 스레드가 비동기 처리
      - 알림음     : amr_start 모드에서 타이머 동작 중
                     Santa Claus 멜로디를 별도 스레드에서 음표 단위로 퍼블리시
    '''

    def __init__(self):
        super().__init__("parking_detection_node")

        # ── 상태 변수 ──
        self.tracked_vehicles  = []                                      # 현재 추적 중인 차량 목록
        self.save_queue        = queue.Queue(maxsize=SAVE_QUEUE_MAXSIZE) # 업로드 작업 큐
        self.db_ref            = None   # Firebase Admin DB 모듈 참조 (초기화 후 설정)
        self.current_case_ref  = None   # 현재 단속 케이스 DB 경로 참조
                                        # initial 이벤트 시 타임스탬프 키로 새로 생성되며,
                                        # confirmed 이벤트는 같은 참조에 update() 하여 로그를 이어쓴다.
        self.ocr_reader        = None   # PaddleOCR 인스턴스 (초기화 후 설정)
        self.mode              = None   # 현재 동작 모드: "amr_start" | "cctv_start" | None(대기)
        self.parking_timeout   = None   # 현재 모드의 단속 타이머 값(초)
        self._audio_stop_event = threading.Event()  # 알림음 스레드 종료 신호용 이벤트

        # ── 초기화 ──
        self._load_model()        # YOLO 모델 로드 및 워밍업
        self._init_firebase()     # Firebase 연결 및 DB 참조 생성
        self._init_ocr()          # PaddleOCR 초기화
        self._init_publisher()    # AudioNoteVector 퍼블리셔 생성
        self._init_subscribers()  # ROS2 토픽 구독 등록 (카메라 + 모드 명령)

        # 업로드·OCR 전담 데몬 스레드 시작 (메인 루프 블로킹 방지)
        threading.Thread(target=self._upload_worker, daemon=True).start()

        cv2.namedWindow("Parking Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Parking Detection", YOLO_IMG_SIZE, YOLO_IMG_SIZE)
        self.get_logger().info("Node ready. Waiting for 'amr_start' or 'cctv_start'...")

    # ── 초기화 메서드들 ──────────────────────────

    def _load_model(self):
        '''
        YOLOv8 모델을 로드하고 빈 이미지로 한 번 추론(워밍업)하여
        첫 프레임에서 발생하는 Cold Start 지연을 방지한다.
        '''
        self.model = YOLO(MODEL_PATH)
        self.model.predict(
            source=np.zeros((YOLO_IMG_SIZE, YOLO_IMG_SIZE, 3), dtype=np.uint8),   # 더미 입력 이미지
            imgsz=YOLO_IMG_SIZE,
            verbose=False   # 워밍업 추론 로그 억제
        )
        self.get_logger().info("YOLO warm-up complete.")

    def _init_firebase(self):
        '''
        Firebase Admin SDK를 초기화하고 DB 모듈 참조를 저장한다.

        실제 쓰기 경로(detections/<타임스탬프>)는 initial 이벤트 발생 시
        _upload() 안에서 동적으로 생성되므로, 여기서는 SDK 초기화만 수행한다.
        실패해도 노드는 계속 실행되며, 업로드 시 db_ref None 체크로 스킵된다.
        '''
        try:
            firebase_admin.initialize_app(
                credentials.Certificate(FIREBASE_CRED_PATH),
                {"databaseURL": FIREBASE_DB_URL}
            )
            self.db_ref = db   # DB 모듈 자체를 보관 (경로는 업로드 시 동적 생성)
            self.get_logger().info("Firebase connected.")
        except Exception as e:
            self.get_logger().error(f"Firebase init failed: {e}")

    def _init_ocr(self):
        '''
        PaddleOCR 인스턴스를 초기화한다.

        전처리(이진화·Unwarp) 없이 원본 컬러 이미지를 직접 입력할 때
        가장 높은 인식률을 보였으므로, 이미지 전처리 파이프라인은 제거되었다.

        주요 파라미터:
          lang="korean"              : 한국어 + 영어·숫자 혼합 인식 (번호판 형식에 적합)
          use_textline_orientation=True : 기울어진 텍스트 자동 각도 보정
          enable_mkldnn=False        : C++ 백엔드(oneDNN) 에러 우회
        '''
        try:
            self.ocr_reader = PaddleOCR(
                lang="korean",                 # 한국어+숫자 혼합 인식 모드
                use_textline_orientation=True, # 기울어진 텍스트 자동 각도 보정
                enable_mkldnn=False            # ★ 핵심: C++ 백엔드(oneDNN) 에러 우회
            )
            self.get_logger().info("PaddleOCR initialized.")
        except Exception as e:
            self.get_logger().error(f"PaddleOCR init failed: {e}")

    def _init_publisher(self):
        ''' AudioNoteVector 퍼블리셔를 생성한다. amr_start 모드에서 알림음 송신에 사용된다. '''
        self.audio_pub = self.create_publisher(AudioNoteVector, TOPIC_AUDIO, 10)

    def _init_subscribers(self):
        '''
        두 종류의 토픽을 구독한다.
          - TOPIC_RGB : 카메라 이미지 (BEST_EFFORT QoS, 최신 1프레임만 유지)
          - TOPIC_CMD : 모드 명령 문자열 ("amr_start" | "cctv_start")
        카메라는 모드가 설정되지 않아도 항상 수신 준비 상태를 유지하며,
        image_callback 내부에서 mode가 None이면 탐지를 스킵한다.
        '''
        cam_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,   # 패킷 손실 허용, 실시간성 우선
            history=HistoryPolicy.KEEP_LAST,             # 가장 최근 메시지만 유지
            depth=1                                       # 큐 크기 1: 오래된 프레임 즉시 폐기
        )
        self.create_subscription(CompressedImage, TOPIC_RGB, self.image_callback, cam_qos)
        self.create_subscription(String, TOPIC_CMD, self.cmd_callback, 10)

    # ── 모드 제어 ────────────────────────────────

    def cmd_callback(self, msg: String):
        '''
        std_msgs/String 토픽으로 모드 명령을 수신한다.

        "amr_start"  → 단속 타이머 30초, Santa Claus 알림음 스레드 시작
        "cctv_start" → 단속 타이머 5초,  알림음 없음

        동일 모드가 중복 수신되어도 재설정 없이 무시된다.
        다른 모드가 수신되면 기존 추적 목록을 초기화하고 새 모드로 전환한다.
        '''
        cmd = msg.data.strip()

        if cmd not in ("amr_start", "cctv_start"):
            self.get_logger().warn(f"알 수 없는 명령: '{cmd}'. 'amr_start' 또는 'cctv_start'만 허용.")
            return

        if self.mode == cmd:   # 동일 모드 중복 수신 → 무시
            return

        # 모드 전환: 기존 추적 초기화 및 진행 중인 알림음 스레드 정리
        self.mode             = cmd
        self.tracked_vehicles = []
        self._audio_stop_event.set()   # 기존 알림음 스레드에 종료 신호

        if cmd == "amr_start":
            self.parking_timeout = PARKING_TIMEOUT_AMR
            self.get_logger().info(f"[MODE] amr_start — 타이머 {PARKING_TIMEOUT_AMR:.0f}초, 알림음 활성")
            # 알림음 스레드 시작: 30초 동안 Santa Claus 멜로디 퍼블리시
            self._audio_stop_event.clear()   # 이전 종료 신호 리셋 후 새 스레드 시작
            threading.Thread(target=self._play_santa, daemon=True).start()
        else:
            self.parking_timeout = PARKING_TIMEOUT_CCTV
            self.get_logger().info(f"[MODE] cctv_start — 타이머 {PARKING_TIMEOUT_CCTV:.0f}초, 알림음 없음")

    # ── 알림음 ───────────────────────────────────

    def _play_santa(self):
        '''
        "Santa Claus is Coming to Town" 첫 절(≈35초)을 AudioNoteVector 토픽으로 퍼블리시한다.

        SANTA_NOTES 배열의 각 음표를 순서대로 재생하며,
        음표 하나씩 개별 메시지로 전송하고 해당 음표의 지속 시간만큼 대기한다.

        threading.Event.wait(timeout)을 sleep 대신 사용하면
        _audio_stop_event 수신 시 즉각 루프를 탈출할 수 있다.

        AudioNote 파라미터:
          frequency   : 재생할 음 높이 (Hz)
          max_runtime : 음표 지속 시간 (Duration, nanosec 단위)
        AudioNoteVector 파라미터:
          append=False: 이전 재생 큐를 비우고 이 음표만 즉시 재생
        '''
        self.get_logger().info("[Audio] Santa Claus 멜로디 시작")
        for freq, dur_ms in SANTA_NOTES:
            if self._audio_stop_event.is_set():   # 모드 전환 또는 노드 종료 시 즉시 중단
                break

            msg        = AudioNoteVector()
            msg.append = False   # 이전 큐 비우고 이 음표만 즉시 재생
            msg.notes  = [
                AudioNote(
                    frequency   = freq,
                    max_runtime = Duration(
                        sec     = 0,
                        nanosec = dur_ms * 1_000_000   # ms → ns 변환
                    )
                )
            ]
            self.audio_pub.publish(msg)

            # 다음 음표 전송 전 현재 음표 지속 시간만큼 대기
            # Event.wait()를 사용해 중단 신호를 즉각 감지한다
            self._audio_stop_event.wait(timeout=dur_ms / 1000.0)

        self.get_logger().info("[Audio] Santa Claus 멜로디 종료")

    # ── 메인 파이프라인 ──────────────────────────

    def image_callback(self, msg: CompressedImage):
        '''
        새 프레임 수신 시 호출되는 핵심 파이프라인.

        처리 순서:
          1. CompressedImage → OpenCV 프레임 디코딩
          2. YOLO로 차량·번호판 탐지
          3. 번호판이 차량 내부에 있는지 Overlap 검증 → 유효한 쌍 생성
          4. 여러 쌍 중 bbox 면적 최대 차량 1대만 선택
          5. IoU 기반 추적 갱신 및 단속 타이머 판정
          6. 모니터링 화면 업데이트
        '''
        if self.mode is None:   # 모드 미설정 시 탐지 스킵 (amr_start/cctv_start 수신 전)
            return

        frame = cv2.imdecode(np.array(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:   # 이미지 디코딩 실패 시 (손상된 패킷 등) 즉시 반환
            return

        cars, ids       = self._detect(frame)   # 차량·번호판 탐지
        validated_pairs = []                     # 유효한 (차량, 번호판) 쌍 목록

        # 각 번호판에 대해 귀속 차량을 찾고 유효한 쌍 구성
        for id_det in ids:
            car = self._find_parent_car(id_det, cars)   # 번호판이 속한 차량 탐색
            if car is not None:
                validated_pairs.append((car, id_det))   # 유효한 차량-번호판 쌍 추가

        if validated_pairs:
            # bbox 면적 기준 내림차순 정렬 후 가장 큰 차량 1대만 유지
            validated_pairs.sort(key=lambda pair: pair[0]["area"], reverse=True)
            validated_pairs = [validated_pairs[0]]   # 면적 최대 1대만 단속 대상으로 선정

        self._update_tracking(frame, validated_pairs)   # 추적 상태 갱신 및 업로드 큐 투입
        self._draw(frame)                               # 모니터링 화면 갱신

    def _detect(self, frame: np.ndarray) -> tuple[list, list]:
        '''
        YOLOv8으로 프레임을 추론하고 탐지된 객체를 차량(car)과 번호판(id)으로 분리한다.

        반환되는 각 항목은 딕셔너리:
          {"class_name", "conf", "x1", "y1", "x2", "y2", "area"}
        '''
        results = self.model.predict(
            source=frame,
            imgsz=YOLO_IMG_SIZE,
            conf=CONF_THRESHOLD,
            verbose=False   # 매 프레임 YOLO 로그 억제
        )
        cars, ids = [], []
        if not results:
            return cars, ids

        for box in results[0].boxes:
            name = self.model.names.get(int(box.cls[0].item()))
            if name not in ("car", "id"):   # 차량·번호판 외 클래스는 무시
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            det = {
                "class_name": name,
                "conf":       float(box.conf[0].item()),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "area":       max(0, x2-x1) * max(0, y2-y1),   # 면적: 음수 방지
            }
            (cars if name == "car" else ids).append(det)   # 클래스에 따라 분류
        return cars, ids

    def _find_parent_car(self, id_det: dict, cars: list) -> dict | None:
        '''
        번호판 bbox가 어느 차량 bbox 안에 속하는지 Overlap 비율로 판별한다.

        교집합 면적 / 번호판 면적 ≥ ID_IN_CAR_OVERLAP_THRESH 인 차량 중
        가장 많이 겹치는 차량을 반환한다. 해당 차량이 없으면 None 반환.
        '''
        id_area = max(1, id_det["area"])   # 0 나눗셈 방지를 위해 최솟값 1

        def overlap(car):
            ''' 번호판과 차량 bbox의 교집합 면적 / 번호판 면적 비율 계산 '''
            ix = max(0, min(id_det["x2"], car["x2"]) - max(id_det["x1"], car["x1"]))  # 교집합 x 길이
            iy = max(0, min(id_det["y2"], car["y2"]) - max(id_det["y1"], car["y1"]))  # 교집합 y 길이
            return (ix * iy) / id_area   # 교집합 넓이 / 번호판 넓이

        # 임계값 이상 겹치는 차량만 후보로 추림 후 가장 많이 겹치는 것 반환
        candidates = [(overlap(car), car) for car in cars if overlap(car) >= ID_IN_CAR_OVERLAP_THRESH]
        return max(candidates, key=lambda x: x[0])[1] if candidates else None

    @staticmethod
    def _iou(a: dict, b: dict) -> float:
        '''
        두 bbox 간의 IoU(Intersection over Union)를 계산한다.
        이전 프레임과 현재 프레임의 차량이 동일 객체인지 판단하는 추적 기준으로 사용된다.
        IoU가 CAR_IOU_THRESH 이상이면 같은 차량으로 간주한다.
        '''
        ix    = max(0, min(a["x2"], b["x2"]) - max(a["x1"], b["x1"]))   # 교집합 x 길이
        iy    = max(0, min(a["y2"], b["y2"]) - max(a["y1"], b["y1"]))   # 교집합 y 길이
        inter = ix * iy                                                    # 교집합 면적
        if inter == 0:
            return 0.0
        area_a = max(1, a["area"])
        area_b = max(1, b["area"])
        return inter / (area_a + area_b - inter)   # IoU = 교집합 / 합집합

    def _update_tracking(self, frame: np.ndarray, validated_pairs: list):
        '''
        IoU 기반으로 이전 프레임 추적 목록과 현재 탐지 결과를 매칭하고
        단속 타이머에 따른 업로드 이벤트를 큐에 투입한다.

        처리 순서:
          1. 기존 추적 차량과 현재 탐지 쌍을 IoU로 매칭 → 위치·시각 갱신
          2. 매칭되지 않은 신규 탐지 → TrackedVehicle 생성 + 최초 감지 큐 투입
          3. 30초 초과 차량 → 확정 증거 큐 투입 후 추적 종료
          4. 화면에서 사라진 추적 차량은 자동 제거
        '''
        matched_track_indices = set()   # 이번 프레임에서 매칭된 기존 추적 인덱스
        matched_pair_indices  = set()   # 이번 프레임에서 매칭된 탐지 쌍 인덱스

        # ── 1단계: 기존 추적 차량과 현재 탐지 쌍 IoU 매칭 ──
        for t_idx, track in enumerate(self.tracked_vehicles):
            best_iou   = 0.0
            best_p_idx = -1
            for p_idx, (car_det, _) in enumerate(validated_pairs):
                if p_idx in matched_pair_indices:   # 이미 매칭된 쌍은 건너뜀
                    continue
                iou = self._iou(track.car_det, car_det)
                if iou > best_iou:   # 가장 높은 IoU 쌍을 최적 매칭으로 선택
                    best_iou   = iou
                    best_p_idx = p_idx

            if best_iou >= CAR_IOU_THRESH:   # IoU 임계값 이상이면 동일 차량으로 판단
                car_det, id_det = validated_pairs[best_p_idx]
                track.update(car_det, id_det)   # 최신 bbox 및 last_seen 갱신
                matched_track_indices.add(t_idx)
                matched_pair_indices.add(best_p_idx)

        # 이번 프레임에서 매칭된 추적 차량만 유지 (사라진 차량 자동 제거)
        visible_tracks = [
            track for i, track in enumerate(self.tracked_vehicles)
            if i in matched_track_indices
        ]

        # ── 2단계: 매칭 실패한 신규 탐지 → 새 추적 차량 등록 + 최초 감지 업로드 ──
        for p_idx, (car_det, id_det) in enumerate(validated_pairs):
            if p_idx not in matched_pair_indices:               # 기존 추적과 매칭 실패 = 신규
                new_track = TrackedVehicle(car_det, id_det)    # 타이머 시작
                visible_tracks.append(new_track)
                self._enqueue(frame, new_track, event="initial")   # 최초 감지 이벤트 큐 투입
                new_track.initial_uploaded = True               # 중복 큐 투입 방지 플래그

        # ── 3단계: 30초 초과 차량 → 확정 증거 업로드, 이후에도 추적 유지 ──
        #
        # [핵심 설계] confirmed 이후에도 visible_tracks에서 제거하지 않는다.
        # 제거하면 다음 프레임에서 IoU 매칭 대상이 없어 동일 차량을
        # 신규 차량으로 오인식하고 initial 이벤트를 재발생시키는 문제가 생긴다.
        # 대신 confirmed_uploaded 플래그로 큐 투입만 막고,
        # 실제로 화면에서 사라질 때 (matched_track_indices 미포함) 자연스럽게 제거된다.
        next_tracks = []
        for track in visible_tracks:
            if not track.confirmed_uploaded and track.elapsed() >= self.parking_timeout:
                self._enqueue(frame, track, event="confirmed")   # 30초 초과 시 확정 업로드
                track.confirmed_uploaded = True                  # 이후 재업로드 방지
            next_tracks.append(track)   # confirmed 여부 무관하게 추적 목록 유지

        self.tracked_vehicles = next_tracks   # 추적 목록을 다음 프레임용으로 교체

    # ── 큐 투입 헬퍼 ────────────────────────────

    def _enqueue(self, frame: np.ndarray, track: 'TrackedVehicle', event: str):
        '''
        업로드 워커 스레드로 전달할 작업 딕셔너리를 큐에 넣는다.

        event:
          "initial"   최초 감지 → 전체 프레임 저장, status="watching"
          "confirmed" 30초 초과 → 전체 프레임 + 번호판 크롭, status="confirmed"
        큐가 가득 찼을 경우 put_nowait 예외를 잡아 드랍 경고를 출력하고 건너뛴다.
        '''
        id_det = track.id_det
        px1, py1, px2, py2 = id_det["x1"], id_det["y1"], id_det["x2"], id_det["y2"]
        plate_crop = frame[py1:py2, px1:px2]   # 번호판 영역 크롭 (OCR 및 plate_image 공용)

        try:
            self.save_queue.put_nowait({
                "event":      event,                                         # "initial" 또는 "confirmed"
                "frame":      frame.copy(),                                  # 전체 프레임 (원본 보존을 위해 복사)
                "plate_crop": plate_crop.copy() if plate_crop.size > 0 else None,  # 번호판 크롭 (없으면 None)
                "car_det":    track.car_det,                                 # 차량 bbox 딕셔너리
                "id_det":     track.id_det,                                  # 번호판 bbox 딕셔너리
            })
        except queue.Full:
            self.get_logger().warn(f"Queue full. '{event}' image dropped.")  # 큐 포화 시 드랍 경고

    # ── 시각화 ──────────────────────────────────

    def _draw(self, frame: np.ndarray):
        '''
        모니터링 창에 차량(초록)·번호판(빨강) bbox와 단속 타이머 진행률 바를 그린다.

        진행률 바: 차량 bbox 상단에 가로 막대로 표시되며,
                   경과 시간 / 30초 비율만큼 채워진다.
        '''
        for track in self.tracked_vehicles:
            c       = track.car_det
            elapsed = track.elapsed()
            ratio   = min(elapsed / self.parking_timeout, 1.0)   # 0.0 ~ 1.0 클램핑

            cv2.rectangle(frame, (c["x1"], c["y1"]), (c["x2"], c["y2"]), (0, 255, 0), 2)   # 차량 bbox 초록

            d = track.id_det
            cv2.rectangle(frame, (d["x1"], d["y1"]), (d["x2"], d["y2"]), (0, 0, 255), 2)   # 번호판 bbox 빨강

            bar_x1   = c["x1"]
            bar_y    = max(0, c["y1"] - 18)    # 차량 bbox 위쪽 18픽셀 위치 (화면 상단 클램핑)
            bar_w    = c["x2"] - c["x1"]       # 바 전체 너비 = 차량 bbox 너비
            bar_fill = int(bar_w * ratio)       # 경과 비율만큼 채워지는 너비
            cv2.rectangle(frame,
                          (bar_x1, bar_y), (bar_x1 + bar_fill, bar_y + 10),
                          (0, 255, 0), -1)      # 초록 채운 사각형으로 진행률 표시

            cv2.putText(frame,
                        f"Target: {elapsed:.0f}s / {self.parking_timeout:.0f}s",
                        (bar_x1, bar_y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)   # 경과/목표 시간 텍스트

        cv2.imshow("Parking Detection", frame)
        cv2.waitKey(1)   # 1ms 대기 (GUI 이벤트 처리용, 프레임 블로킹 없음)

    # ── Firebase 업로드 및 OCR 워커 ─────────────

    def _upload_worker(self):
        '''
        메인 영상 처리 속도를 늦추지 않기 위해 백그라운드에서 동작하는 데몬 스레드.
        큐(save_queue)에서 데이터를 꺼내 OCR과 Firebase 네트워크 업로드를 수행한다.
        None을 수신하면 안전하게 종료된다 (Sentinel 패턴).
        '''
        while True:
            item = self.save_queue.get()   # 블로킹 대기: 큐에 항목이 생길 때까지 슬립
            if item is None:               # Sentinel 수신 → 워커 스레드 종료
                break
            try:
                self._upload(item)
            except Exception as e:
                self.get_logger().error(f"Upload error: {e}")

    def _upload(self, item: dict):
        '''
        OCR 수행과 Firebase 누적 업로드를 처리하는 함수.

        event="initial" (최초 감지):
          - 현재 시각을 키로 detections/<YYYYMMDD_HHMMSSffffff> 경로를 새로 생성하고
            self.current_case_ref 에 저장한다.
          - 문서를 status="watching" 으로 초기화하고 initial_image 와 plate_number 를 기록한다.
          - 이전 케이스 경로는 그대로 남아 누적 로그로 보존된다.

        event="confirmed" (30초 확정):
          - self.current_case_ref (initial 때 저장한 동일 경로)에 update() 로
            증거 필드(evidence_image, plate_image, plate_number, confirmed_at)만 추가한다.
          - initial_image 와 detected_at 은 건드리지 않는다.

        결과적으로 DB 구조:
          detections/
            ├── 20250311_142305_123456/   ← 첫 번째 차량 (영구 보존)
            ├── 20250311_153012_456789/   ← 두 번째 차량 (영구 보존)
            └── ...
        '''
        if self.db_ref is None:   # Firebase 미연결 시 업로드 스킵
            return

        event      = item["event"]
        frame      = item["frame"]
        plate_crop = item["plate_crop"]
        now_dt     = datetime.datetime.now()
        now_iso    = now_dt.isoformat()   # "2025-03-11T14:23:05.123456"

        if event == "initial":
            # ── 최초 감지: 새 케이스 경로 생성 + 전체 프레임 저장 ──
            # plate_number / plate_image 는 confirmed 시에만 기록하므로 여기서는 포함하지 않는다.
            case_key = now_dt.strftime("%Y%m%d_%H%M%S_%f")   # ex) 20250311_142305_123456
            self.current_case_ref = self.db_ref.reference(
                f"{FIREBASE_DB_PATH}/{case_key}"              # ex) detections/20250311_142305_123456
            )
            self.current_case_ref.set({
                "status":         "watching",           # 현재 감시 중
                "detected_at":    now_iso,              # 최초 감지 시각
                "confirmed_at":   None,                 # 확정 시 채워짐
                "initial_image":  self._to_b64(frame),  # 최초 전체 프레임
                "evidence_image": None,                 # 확정 시 채워짐
                "plate_image":    None,                 # 확정 시 채워짐
                "plate_number":   None,                 # 확정 시 채워짐 (OCR은 confirmed 에서만 수행)
            })
            self.get_logger().info(f"[initial]   key: {case_key}")

        elif event == "confirmed":
            if self.current_case_ref is None:   # initial 없이 confirmed 가 먼저 오는 예외 상황 방어
                self.get_logger().warn("[confirmed] current_case_ref 없음. 업로드 스킵.")
                return

            # ── 확정: OCR 수행 후 증거 필드 추가 ──
            # PaddleOCR은 원본 컬러 크롭 이미지를 직접 입력할 때 인식률이 가장 높다.
            plate_number = self._ocr_paddle(plate_crop) if plate_crop is not None else "UNKNOWN"

            # initial 때 저장한 동일 경로에 증거 필드 추가 (initial_image 등 기존 필드 보존)
            self.current_case_ref.update({
                "status":         "confirmed",                  # 불법주정차 확정
                "confirmed_at":   now_iso,                      # 확정 시각
                "evidence_image": self._to_b64(frame),          # 확정 시점 전체 프레임
                "plate_image":    self._to_b64(plate_crop) if plate_crop is not None else None,
                "plate_number":   plate_number,                 # OCR 결과 (확정 시 최초 기록)
            })
            self.get_logger().info(
                f"[confirmed] Plate: {plate_number}  at {now_dt.strftime('%H:%M:%S')}"
            )

    def _ocr_paddle(self, img: np.ndarray) -> str:
        '''
        PaddleOCR로 번호판 이미지를 인식하고 텍스트를 반환한다.

        원본 컬러 크롭 이미지를 전처리 없이 직접 입력한다.
        여러 차례 테스트 결과, 전처리가 없을 때 인식률이 가장 높았다.

        반환값 구조 처리:
          PaddleOCR 최신 버전(PaddleX 기반)은 predict() 메서드를 사용하며,
          결과 객체가 get_res()를 가진 경우와 리스트 형태 두 가지를 모두 처리한다.
          신뢰도(score) 0.6 미만 결과는 오인식으로 간주하고 제외한다.
        '''
        if self.ocr_reader is None or img is None:
            return "UNKNOWN"
        try:
            results = self.ocr_reader.predict(img)
            if not isinstance(results, list):
                results = list(results)   # 제너레이터일 경우 리스트로 변환

            texts = []
            for res in results:
                if hasattr(res, "get_res"):                          # PaddleX 객체 형태
                    d = res.get_res()
                    for t, s in zip(d.get("rec_text", []), d.get("rec_score", [])):
                        if s >= 0.6:
                            texts.append(t)
                elif isinstance(res, list):                          # 구버전 리스트 형태
                    for line in res:
                        if len(line) >= 2 and isinstance(line[1], (list, tuple)):
                            t, s = line[1][0], line[1][1]
                            if s >= 0.6:
                                texts.append(t)

            result = "".join(texts).replace(" ", "")
            return result if result else "UNKNOWN"
        except Exception as e:
            self.get_logger().warn(f"PaddleOCR 호출 실패: {e}")
            return "UNKNOWN"

    @staticmethod
    def _to_b64(img: np.ndarray) -> str:
        '''
        OpenCV 이미지(numpy 배열)를 JPEG 품질 85로 압축한 뒤 base64 문자열로 변환한다.
        Firebase Realtime DB는 바이너리를 직접 저장할 수 없으므로 문자열로 변환해 업로드한다.
        '''
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])   # JPEG 품질 85로 인코딩
        return base64.b64encode(buf).decode("utf-8")                          # bytes → base64 문자열

    # ── 노드 종료 ───────────────────────────────

    def destroy_node(self):
        '''
        노드 종료 시 알림음 스레드와 업로드 워커 스레드를 안전하게 종료하고
        OpenCV GUI 자원을 해제한다.
        '''
        self._audio_stop_event.set()   # 알림음 스레드 종료 신호 (Santa 루프 즉시 탈출)
        self.save_queue.put(None)      # 업로드 워커 스레드 종료 신호 (Sentinel)
        cv2.destroyAllWindows()        # 모든 OpenCV 창 닫기
        super().destroy_node()


# ──────────────────────────────────────────────
# [CHAPTER 4: 진입점]
# ──────────────────────────────────────────────

def main(args=None):
    ''' ROS2 노드 실행 진입점. Ctrl+C 입력 시 노드를 안전하게 종료한다. '''
    rclpy.init(args=args)
    node = ParkingDetectionNode()
    try:
        rclpy.spin(node)       # 토픽 수신 이벤트 루프 실행
    except KeyboardInterrupt:
        pass                   # Ctrl+C 는 정상 종료로 처리
    finally:
        node.destroy_node()    # 워커 스레드 종료 + GUI 자원 해제
        if rclpy.ok():
            rclpy.shutdown()   # ROS2 런타임 정리


if __name__ == "__main__":
    main()
