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
               - amr_start 모드: confirmed 시에만 업로드 (로컬 임시 저장 → confirmed 후 일괄 업로드)
                 미확정 시 로컬 파일 자동 삭제
               - cctv_start 모드: confirmed 시 cctv_detections 경로에만 업로드
8. 알림음   : amr_start 모드에서 타이머 동작 중
               "Santa Claus is Coming to Town" 멜로디를 AudioNoteVector 토픽으로 퍼블리시

[Firebase 데이터 구조]
  detections/<YYYYMMDD_HHMMSSffffff>/
    ├── status          "confirmed" (confirmed 시에만 생성됨)
    ├── detected_at     최초 감지 ISO 시각 (case_key에서 복원)
    ├── confirmed_at    확정 ISO 시각
    ├── initial_image   최초 감지 전체 프레임 JPEG → base64 (로컬 임시파일에서 복원)
    ├── evidence_image  확정 시 전체 프레임 JPEG → base64
    ├── plate_image     확정 시 번호판 크롭 JPEG → base64
    └── plate_number    OCR 인식 결과
'''

import base64        # 이미지를 Firebase에 저장하기 위한 base64 인코딩
import datetime      # 감지 시각을 ISO 문자열로 기록
import os            # 로컬 임시 파일 저장/삭제
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
from std_msgs.msg import String, Bool                  # 모드 명령 수신 ("amr_start" / "cctv_start")
from irobot_create_msgs.msg import AudioNoteVector, AudioNote  # AMR 알림음 송신
from builtin_interfaces.msg import Duration            # AudioNote 지속 시간 표현
from ultralytics import YOLO                           # YOLOv8 추론 엔진
import firebase_admin                                  # Firebase Admin SDK
from firebase_admin import credentials, db             # 인증 및 Realtime DB 접근
from paddleocr import PaddleOCR                        # 번호판 OCR 엔진 (fallback)
from google.cloud import vision                        # Google Cloud Vision OCR (1순위)


# ──────────────────────────────────────────────
# [CHAPTER 1: 하이퍼파라미터]
# ──────────────────────────────────────────────

ROBOT_NAMESPACE          = "/robot2"                                                               # ROS2 기본 네임스페이스 (robot2 / robot3 동적 전환)
MODEL_PATH               = "/home/rokey/click_car/models/amr.pt"                                  # YOLOv8 가중치 경로
FIREBASE_CRED_PATH       = "/home/rokey/Downloads/iligalstop-firebase-adminsdk-fbsvc-d989ef0f8c.json"                             # Firebase 서비스 계정 키 경로
FIREBASE_DB_URL          = "https://iligalstop-default-rtdb.asia-southeast1.firebasedatabase.app" # Realtime DB URL
FIREBASE_DB_PATH         = "detections"          # 단속 로그를 저장할 DB 상위 경로 (하위에 케이스별 노드가 누적됨)

CONF_THRESHOLD           = 0.50    # YOLO 탐지 신뢰도 임계값: 이 값 미만의 bbox는 무시
ID_IN_CAR_OVERLAP_THRESH = 0.50    # 번호판이 차량 bbox 내부에 이 비율 이상 겹쳐야 유효한 쌍으로 인정
CAR_IOU_THRESH           = 0.30    # 이전·현재 프레임 차량을 동일 객체로 판단하는 IoU 최솟값
YOLO_IMG_SIZE            = 704     # YOLO 추론 이미지 크기 (픽셀, 정사각형)
PARKING_TIMEOUT_AMR      = 30.0    # amr_start  모드: 30초 (AMR 직접 출동, 사이렌 동반)
PARKING_TIMEOUT_CCTV     =  5.0    # cctv_start 모드:  5초 (CCTV 연동 빠른 확인)
SAVE_QUEUE_MAXSIZE       = 50      # 업로드 큐 최대 항목 수: 초과 시 신규 항목 드랍

# ── 멀티 로봇 지원 ────────────────────────────────────────────────────────────
# robot2 / robot3 두 대를 하나의 노드 인스턴스로 처리한다.
# /robot2/start 또는 /robot3/start 를 수신하면 해당 네임스페이스를 self.ns 에 저장하고,
# 이후 카메라 구독·오디오 퍼블리시·capture_done 퍼블리시를 모두 그 네임스페이스로 전환한다.
SUPPORTED_NAMESPACES = ["/robot2", "/robot3"]

# CMD 토픽은 두 로봇 모두 구독 (카메라/오디오/done 은 명령 수신 후 동적 생성)
TOPIC_CMD_ROBOT2  = "/robot2/start"
TOPIC_CMD_ROBOT3  = "/robot3/start"

# CCTV 연동 전용 Firebase 경로
FIREBASE_CCTV_DB_PATH = "cctv_detections"   # webcam_detector 가 생성한 케이스에 AMR 증거를 추가하는 경로

GOOGLE_VISION_CRED_PATH = "/home/rokey/Downloads/google_vision.json"  # GCV 서비스 계정 키 경로

# ── 로컬 임시 저장 (amr_start 전용) ──────────────────────────────────────────
# confirmed 전까지 이미지를 로컬에만 보관하다가, confirmed 시에 한 번에 Firebase 업로드.
# confirmed 없이 노드가 종료되거나 새 start 신호가 오면 자동 삭제.
LOCAL_TEMP_DIR        = "/tmp/click_car_amr"   # 임시 저장 디렉토리

# ── capture_done 재전송 횟수 ──────────────────────────────────────────────────
CAPTURE_DONE_REPEAT   = 5     # capture_done 을 몇 번 반복 발행할지
CAPTURE_DONE_INTERVAL = 0.2   # 반복 발행 간격 (초)

# ──────────────────────────────────────────────
# [CHAPTER 2-B: 캐논 변주곡 D장조 하이라이트 멜로디]
# ──────────────────────────────────────────────
# ★ 핵심 원칙:
#   AudioNoteVector 의 notes 필드는 배열이다.
#   음표 전체를 한 메시지에 담아 단 한 번 publish 하면
#   로봇 내부 오디오 매니저가 직접 시퀀싱한다.
#   음표를 1개씩 개별 메시지로 보내면 append=False 가
#   이전 음표를 강제 중단시켜 반드시 음표 누락이 발생한다.
#
# 각 튜플: (주파수 Hz, 지속 시간 nanosec)
# D장조: D4=294 E4=330 F#4=370 G4=392 A4=440 B4=494
#        C#5=554 D5=587 E5=659 F#5=740 G5=784 A5=880
# BPM=100: Q(4분)=600ms, E(8분)=300ms, H(2분)=1200ms, W(온)=2400ms
# ── Create3 오디오 스펙 및 설계 근거 ────────────────────────────────────────
# AudioNote.msg: frequency = uint16 (이론 범위 0~65535 Hz)
# 실용 범위: Create3 스피커는 피에조 계열로 공진 주파수 근처(2~4kHz)에서 최대 음량.
#   - 300Hz 이하 : 거의 들리지 않음
#   - 440~880Hz  : 음정 인식 가능, 멜로디 표현에 최적
#   - 880Hz 이상 : 들리지만 쏘는 음색
# 이전 버전 문제: D4(294Hz)·E4(330Hz) 등 낮은 음이 피에조에서 거의 묵음 처리됨
#   → 음역을 440~880Hz(A4~A5)로 올려 전 음표가 또렷하게 들리도록 재설계
#
# 구성: 음표를 33개로 줄여 버퍼 오버플로 방지 (이전 38개 → 33개)
# 재생시간: 29.6초 (30초 타이머와 자연스럽게 맞물림)
#
# 캐논 D장조 핵심 테마 (5옥타브, 알아들을 수 있는 최소 구성):
#   하강 테마: F#5-E5-D5-C#5-B4-A4-B4-C#5  (가장 유명한 8음 하강 패턴)
#   상승 응답: D5-E5-F#5-G5-F#5-E5-D5-C#5  (대위 성부 느낌)
#   클라이맥스: 상승 후 A5 정점 → D5 종지
_N  =   800_000_000   # 보통 음표 (800ms)
_L  = 1_600_000_000   # 긴 음표  (1600ms)
_XL = 3_200_000_000   # 종지 음표 (3200ms)
CANON_NOTES = [
    # ── 하강 테마 (8음 × 800ms = 6.4초) — 캐논의 핵심 멜로디 ─────────────────
    (740, _N), (659, _N), (587, _N), (554, _N),   # F#5 E5  D5  C#5
    (494, _N), (440, _N), (494, _N), (554, _N),   # B4  A4  B4  C#5

    # ── 하강 테마 재현 (8음 × 800ms = 6.4초) ─────────────────────────────────
    (740, _N), (659, _N), (587, _N), (554, _N),   # F#5 E5  D5  C#5
    (494, _N), (440, _N), (494, _N), (554, _N),   # B4  A4  B4  C#5

    # ── 상승 응답 (8음 × 800ms = 6.4초) ─────────────────────────────────────
    (587, _N), (659, _N), (740, _N), (784, _N),   # D5  E5  F#5 G5
    (740, _N), (659, _N), (587, _N), (554, _N),   # F#5 E5  D5  C#5

    # ── 클라이맥스 + 종지 (10.4초) ───────────────────────────────────────────
    (440, _N), (494, _N), (587, _N), (740, _N),   # A4  B4  D5  F#5 (상승)
    (880, _L),                                     # A5 (정점 1600ms)
    (784, _N), (740, _N), (659, _N),               # G5  F#5 E5  (하강)
    (587, _XL),                                    # D5  (종지 3200ms)
]
# 총 재생시간: 6.4 + 6.4 + 6.4 + 10.4 = 29.6초  음표 수: 33개
del _N, _L, _XL


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
        self.ns                = None   # 현재 통신 중인 로봇 네임스페이스 ("/robot2" | "/robot3")
        self.tracked_vehicles  = []                                      # 현재 추적 중인 차량 목록
        self.save_queue        = queue.Queue(maxsize=SAVE_QUEUE_MAXSIZE) # 업로드 작업 큐
        self.db_ref            = None   # Firebase Admin DB 모듈 참조 (초기화 후 설정)
        self.current_case_ref  = None   # 현재 단속 케이스 DB 경로 참조
        self.cctv_case_key     = None   # cctv_start 모드: webcam_detector 가 생성한 케이스 키
        self.ocr_reader        = None   # PaddleOCR 인스턴스 (초기화 후 설정)
        self.gcv_client        = None   # Google Cloud Vision 클라이언트 (1순위 OCR)
        self.mode              = None   # 현재 동작 모드: "amr_start" | "cctv_start" | None(대기)
        self.parking_timeout   = None   # 현재 모드의 단속 타이머 값(초)
        self._audio_stop_event = threading.Event()  # 알림음 스레드 종료 신호용 이벤트
        self._local_case_key   = None   # amr_start 로컬 임시 저장 케이스 키 (confirmed 전까지 보관)
        self._pending_ns       = None   # 타이머에서 처리할 로봇 네임스페이스 대기열

        # ── 동적 퍼블리셔·구독자 (네임스페이스 확정 후 생성) ──
        self.audio_pub        = None   # /robotN/cmd_audio        — cmd_callback 에서 생성
        self.capture_done_pub = None   # /robotN/capture_done     — cmd_callback 에서 생성
        self._cam_sub         = None   # /robotN/oakd/.../compressed — cmd_callback 에서 생성

        # ── 초기화 ──
        self._load_model()        # YOLO 모델 로드 및 워밍업
        self._init_firebase()     # Firebase 연결 및 DB 참조 생성
        self._init_ocr()          # PaddleOCR 초기화 (fallback)
        self._init_gcv()          # Google Cloud Vision 초기화 (1순위)
        self._init_cmd_subscribers()  # /robot2/start + /robot3/start 구독

        # ── 로봇 활성화 타이머 (10Hz) ──
        # create_subscription / destroy_subscription 을 콜백 안에서 직접 호출하면
        # executor 의 wait set 재구성과 충돌할 수 있다.
        # cmd_callback 은 _pending_ns 플래그만 세팅하고,
        # 실제 구독 생성·해제는 이 타이머 콜백에서 spin 루프와 동일 컨텍스트로 처리한다.
        self.create_timer(0.1, self._activation_timer)

        # 업로드·OCR 전담 데몬 스레드 시작 (메인 루프 블로킹 방지)
        threading.Thread(target=self._upload_worker, daemon=True).start()

        # 로컬 임시 저장 디렉토리 생성
        os.makedirs(LOCAL_TEMP_DIR, exist_ok=True)

        cv2.namedWindow("Plate Checking", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Plate Checking", YOLO_IMG_SIZE, YOLO_IMG_SIZE)
        self.get_logger().info(
            "Node ready. /robot2/start 또는 /robot3/start 대기 중..."
        )

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
        PaddleOCR 인스턴스를 초기화한다. (GCV 실패 시 fallback)

        벤치마크 결과:
          - 원본(101x30)은 오인식. 2x 확대 시 신뢰도 0.97로 정확 인식.
          - GCV 연결 실패 시 자동으로 이 엔진이 사용된다.
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

    def _init_gcv(self):
        '''
        Google Cloud Vision API 클라이언트를 초기화한다. (1순위 OCR)

        벤치마크 결과 GCV는 원본 이미지(101x30)에서도 전처리 없이 "097하0228"을
        완벽하게 인식했으며, 7종 변형 중 6종에서 정확했다.

        인증: GOOGLE_VISION_CRED_PATH 파일을 환경변수로 설정.
        실패해도 노드는 계속 실행되며, OCR 호출 시 PaddleOCR fallback으로 전환된다.
        '''
        import os
        try:
            os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", GOOGLE_VISION_CRED_PATH)
            self.gcv_client = vision.ImageAnnotatorClient()
            self.get_logger().info("Google Cloud Vision initialized.")
        except Exception as e:
            self.get_logger().error(f"GCV init failed (PaddleOCR fallback 사용): {e}")

    def _init_cmd_subscribers(self):
        '''
        robot2 / robot3 의 start 토픽을 각각 별도 콜백으로 구독한다.
        콜백을 분리해야 발신 네임스페이스를 정확히 알 수 있다.
        '''
        self.create_subscription(
            String, TOPIC_CMD_ROBOT2,
            lambda msg: self.cmd_callback(msg, "/robot2"), 10
        )
        self.create_subscription(
            String, TOPIC_CMD_ROBOT3,
            lambda msg: self.cmd_callback(msg, "/robot3"), 10
        )
        self.get_logger().info(f"CMD 구독: {TOPIC_CMD_ROBOT2}, {TOPIC_CMD_ROBOT3}")

    def _activate_robot(self, ns: str):
        '''
        cmd_callback 에서 호출되는 경량 래퍼.
        실제 구독 생성·해제는 executor 와 동일 컨텍스트인 _activation_timer 에서 수행한다.
        콜백 안에서 create_subscription / destroy_subscription 을 직접 호출하면
        executor 의 wait set 재구성과 충돌할 수 있으므로 플래그만 세팅한다.
        '''
        self._pending_ns = ns
        self.get_logger().info(f"[activate] 대기열 등록: ns={ns}")

    def _activation_timer(self):
        '''
        10Hz 타이머 콜백. spin 루프와 동일 컨텍스트에서 실행되므로
        create_subscription / destroy_subscription 이 안전하다.

        두 가지 역할을 순서대로 처리한다:
          1. 카메라 구독 해제:
             mode == None 인데 _cam_sub 이 살아있으면 즉시 해제한다.
             (confirmed 후 image_callback 이 mode = None 을 세팅하므로
              플래그 없이 상태만으로 판단 가능 → 플래그 분실 버그 원천 제거)
          2. 카메라 구독 생성:
             _pending_ns 가 세팅돼 있으면 새 구독·퍼블리셔를 생성한다.
        '''
        # ── 1단계: mode == None 이고 _cam_sub 살아있으면 해제 ──────────────────
        if self.mode is None and self._cam_sub is not None:
            self.destroy_subscription(self._cam_sub)
            self._cam_sub = None
            self.get_logger().info("[activation_timer] 카메라 구독 해제 완료 — 대기 중")

        # ── 2단계: pending_ns 있으면 새 구독 생성 ──────────────────────────────
        if self._pending_ns is None:
            return

        # mode 가 아직 None 이면 start 처리가 덜 된 것 — 다음 틱에서 처리
        if self.mode is None:
            return

        ns = self._pending_ns
        self._pending_ns = None   # 처리 시작 — 중복 실행 방지

        if self.ns == ns and self._cam_sub is not None:
            self.get_logger().info(f"[activation_timer] {ns} 이미 활성 — 스킵")
            return

        topic_rgb   = f"{ns}/oakd/rgb/image_raw/compressed"
        topic_audio = f"{ns}/cmd_audio"
        topic_done  = f"{ns}/capture_done"

        # ── 기존 카메라 구독 해제 (로봇 전환 시) ──
        if self._cam_sub is not None:
            self.destroy_subscription(self._cam_sub)
            self._cam_sub = None
            self.get_logger().info("[activation_timer] 기존 카메라 구독 해제 (로봇 전환)")

        # ── 카메라 구독 (BEST_EFFORT — OAK-D 드라이버가 BEST_EFFORT로 퍼블리시) ──
        cam_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self._cam_sub = self.create_subscription(
            CompressedImage, topic_rgb, self.image_callback, cam_qos
        )

        # ── 퍼블리셔 재생성 ──
        self.audio_pub        = self.create_publisher(AudioNoteVector, topic_audio, 10)
        self.capture_done_pub = self.create_publisher(Bool, topic_done, 10)

        self.ns = ns
        self.get_logger().info(
            f"[activation_timer] 활성화 완료 ns={ns}\n"
            f"  카메라 구독 : {topic_rgb}\n"
            f"  오디오 발행 : {topic_audio}\n"
            f"  완료 발행   : {topic_done}"
        )

    # ── 모드 제어 ────────────────────────────────

    def cmd_callback(self, msg: String, ns: str):
        '''
        std_msgs/String 토픽으로 모드 명령을 수신한다.
        ns: 발신 로봇 네임스페이스 ("/robot2" | "/robot3") — lambda 로 주입됨

        "amr_start"        → 단속 타이머 30초, Santa Claus 알림음 스레드 시작
        "cctv_start"       → 단속 타이머 5초,  알림음 없음
        "cctv_start:<key>" → 위와 동일 + cctv_detections 케이스 키 보관

        메시지에 좌표 등 추가 정보가 붙어 있어도 첫 번째 토큰(모드)만 사용한다.
        ocr_node 는 이미 현장에 도착해 있으므로 좌표는 불필요하다.

        새 start 신호 수신 시 기존 추적 목록을 초기화하고 항상 새 작업을 시작한다.
        '''
        raw = msg.data.strip()
        self.get_logger().info(f"[cmd] 수신: '{raw}' from {ns}")   # ★ 수신 확인용 — 콜백 미호출 vs publish 실패 구분

        # ── 첫 번째 토큰만으로 모드를 결정한다 ──────────────────────────────
        token = raw.split(":")[0]

        if token == "amr_start":
            cmd                = "amr_start"
            self.cctv_case_key = None
        elif token == "cctv_start":
            cmd                = "cctv_start"
            parts              = raw.split(":", 2)
            self.cctv_case_key = parts[1] if len(parts) >= 2 and parts[1] else None
        else:
            self.get_logger().warn(f"알 수 없는 명령: '{raw}'. 'amr_start' 또는 'cctv_start' 만 허용.")
            return

        # ── 해당 로봇의 카메라·퍼블리셔 활성화 ──────────────────────────────
        self._activate_robot(ns)

        # 동일 모드라도 새 start 신호는 항상 새 작업으로 처리한다.
        # (예: cctv_start 가 연속으로 오면 각각 독립된 단속으로 진행)
        # 기존 추적 목록을 초기화해 이전 차량과 섞이지 않도록 한다.
        self.mode             = cmd
        self.tracked_vehicles = []
        self._audio_stop_event.set()

        # 새 start 신호가 오면 이전 amr_start 세션의 미확정 로컬 데이터 삭제
        self._clear_local_temp()

        if cmd == "amr_start":
            self.parking_timeout = PARKING_TIMEOUT_AMR
            self.get_logger().info(f"[MODE] amr_start ({ns}) — 타이머 {PARKING_TIMEOUT_AMR:.0f}초, 알림음 활성")
            self._audio_stop_event.clear()
            threading.Thread(target=self._play_canon, daemon=True).start()
        else:
            self.parking_timeout = PARKING_TIMEOUT_CCTV
            self.get_logger().info(
                f"[MODE] cctv_start ({ns}) — 타이머 {PARKING_TIMEOUT_CCTV:.0f}초, "
                f"케이스 키: {self.cctv_case_key or '없음'}"
            )

    # ── 알림음 ───────────────────────────────────

    def _play_canon(self):
        '''
        캐논 변주곡 D장조 하이라이트(≈32초)를 AudioNoteVector 토픽으로 퍼블리시한다.

        ★ 올바른 방법: 음표 전체를 notes[] 배열에 담아 단 한 번 publish.
          로봇 내부 오디오 매니저가 시퀀싱을 담당하므로
          네트워크 타이밍 지터나 create3_republisher 지연에 영향받지 않는다.

        ✗ 잘못된 방법 (기존): 음표를 1개씩 개별 메시지로 append=False 전송.
          매 메시지가 이전 재생을 강제 중단 → 음표 누락이 구조적으로 발생한다.

        중단 처리: _audio_stop_event 세트 시 빈 메시지(notes=[])를 append=False 로
        발행해 즉시 재생을 정지시킨다.
        '''
        if self._audio_stop_event.is_set():
            return

        # audio_pub 이 아직 _activation_timer 에서 생성되지 않았을 수 있으므로 잠시 대기
        for _ in range(20):   # 최대 2초 대기 (0.1초 × 20)
            if self.audio_pub is not None:
                break
            time.sleep(0.1)

        if self.audio_pub is None:
            self.get_logger().warn("[Audio] audio_pub 미생성 — 캐논 스킵")
            return

        self.get_logger().info("[Audio] 캐논 변주곡 시작 (단일 메시지 방식)")

        # ── 전체 음표 배열을 한 메시지로 구성 ──
        msg        = AudioNoteVector()
        msg.append = False   # 혹시 재생 중인 이전 시퀀스 제거 후 즉시 시작
        msg.notes  = [
            AudioNote(
                frequency   = freq,
                max_runtime = Duration(sec=ns // 1_000_000_000,
                                       nanosec=ns % 1_000_000_000)
            )
            for freq, ns in CANON_NOTES
        ]
        self.audio_pub.publish(msg)   # 단 한 번만 발행

        # ── 전체 재생 시간만큼 대기 (중단 신호 감지 포함) ──
        total_sec = sum(ns for _, ns in CANON_NOTES) / 1_000_000_000
        self._audio_stop_event.wait(timeout=total_sec)

        # ── 중단 신호 수신 시 빈 메시지로 재생 즉시 정지 ──
        if self._audio_stop_event.is_set():
            stop_msg        = AudioNoteVector()
            stop_msg.append = False
            stop_msg.notes  = []
            self.audio_pub.publish(stop_msg)

        self.get_logger().info("[Audio] 캐논 변주곡 종료")

    # ── 메인 파이프라인 ──────────────────────────

    def image_callback(self, msg: CompressedImage):
        '''
        새 프레임 수신 시 호출되는 핵심 파이프라인.

        처리 순서:
          1. CompressedImage → OpenCV 프레임 디코딩
          2. mode가 None이면 화면만 표시하고 탐지 스킵
          3. YOLO로 차량·번호판 탐지
          4. 번호판이 차량 내부에 있는지 Overlap 검증 → 유효한 쌍 생성
          5. 여러 쌍 중 bbox 면적 최대 차량 1대만 선택
          6. IoU 기반 추적 갱신 및 단속 타이머 판정
          7. 모니터링 화면 업데이트
        '''
        frame = cv2.imdecode(np.array(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:   # 이미지 디코딩 실패 시 (손상된 패킷 등) 즉시 반환
            self.get_logger().warn("이미지 디코딩 실패", throttle_duration_sec=5.0)
            return

        if self.mode is None:   # 모드 미설정 시 탐지 스킵, 화면은 표시
            cv2.imshow("Plate Checking", frame)
            cv2.waitKey(1)
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

                # ── 촬영 완료 신호 퍼블리시 (CAPTURE_DONE_REPEAT 회 반복) ────────
                threading.Thread(
                    target=self._publish_capture_done_repeated,
                    daemon=True
                ).start()
                self.get_logger().info(
                    f"[capture_done] True × {CAPTURE_DONE_REPEAT}회 발송 시작"
                )

                # ── mode = None 세팅 → _activation_timer 가 자동으로 구독 해제 ──
                # _activation_timer 는 10Hz 로 "mode is None and _cam_sub is not None"
                # 상태를 감지해 destroy_subscription 을 처리한다.
                # 플래그 방식 대신 상태 직접 감지 방식이므로 분실 버그가 없다.
                self.mode = None
                self.ns   = None
                self.get_logger().info("[confirmed] mode=None 세팅 → 타이머가 구독 해제 예정")

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
                "event":         event,                                         # "initial" 또는 "confirmed"
                "frame":         frame.copy(),                                  # 전체 프레임 (원본 보존을 위해 복사)
                "plate_crop":    plate_crop.copy() if plate_crop.size > 0 else None,  # 번호판 크롭 (없으면 None)
                "car_det":       track.car_det,                                 # 차량 bbox 딕셔너리
                "id_det":        track.id_det,                                  # 번호판 bbox 딕셔너리
                "mode":          self.mode,                                     # 큐 투입 시점 모드 스냅샷
                "cctv_case_key": self.cctv_case_key,                            # cctv_start 케이스 키 스냅샷
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

        cv2.imshow("Plate Checking", frame)
        cv2.waitKey(1)   # 1ms 대기 (GUI 이벤트 처리용, 프레임 블로킹 없음)

    # ── capture_done 반복 발행 ───────────────────

    def _publish_capture_done_repeated(self):
        '''
        capture_done = True 를 CAPTURE_DONE_REPEAT 회 반복 발행한다.
        네트워크 패킷 손실 대비용. 별도 데몬 스레드에서 호출된다.
        '''
        for i in range(CAPTURE_DONE_REPEAT):
            if self.capture_done_pub is None:
                break
            done_msg      = Bool()
            done_msg.data = True
            self.capture_done_pub.publish(done_msg)
            self.get_logger().info(f"[capture_done] {i+1}/{CAPTURE_DONE_REPEAT}")
            if i < CAPTURE_DONE_REPEAT - 1:
                time.sleep(CAPTURE_DONE_INTERVAL)

    # ── 로컬 임시 데이터 정리 ───────────────────

    def _clear_local_temp(self):
        '''
        LOCAL_TEMP_DIR 에 남아 있는 현재 케이스의 미확정 임시 파일을 삭제한다.
        새 start 신호 수신 시, 또는 노드 종료 시 호출된다.
        '''
        if self._local_case_key is None:
            return
        for suffix in ("_initial_frame.jpg", "_initial_plate.jpg"):
            path = os.path.join(LOCAL_TEMP_DIR, self._local_case_key + suffix)
            try:
                if os.path.exists(path):
                    os.remove(path)
                    self.get_logger().info(f"[local] 임시 파일 삭제: {path}")
            except Exception as e:
                self.get_logger().warn(f"[local] 임시 파일 삭제 실패: {e}")
        self._local_case_key = None

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
        OCR 수행과 Firebase 업로드를 처리하는 함수.

        ── amr_start 모드 ──────────────────────────────────────────────────────
          initial  : detections/<타임스탬프> 에 새 케이스 생성, status="watching"
          confirmed: 동일 경로에 evidence_image / plate_image / plate_number 추가

        ── cctv_start 모드 ──────────────────────────────────────────────────────
          initial  : 완전히 무시 (detections 에 아무것도 쓰지 않음)
          confirmed: cctv_detections/<cctv_case_key> 에 3가지만 추가
                       amr_evidence_image, plate_image, plate_number
        detections 경로는 amr_start 전용이며, cctv_start 시에는 절대 건드리지 않는다.
        '''
        if self.db_ref is None:
            return

        event         = item["event"]
        frame         = item["frame"]
        plate_crop    = item["plate_crop"]
        item_mode     = item.get("mode")
        cctv_case_key = item.get("cctv_case_key")
        now_dt        = datetime.datetime.now()
        now_iso       = now_dt.isoformat()

        # ════════════════════════════════════════════════════════════════════
        # [A] cctv_start 모드: confirmed 시에만 cctv_detections 에 삽입
        # ════════════════════════════════════════════════════════════════════
        if item_mode == "cctv_start":
            if event != "confirmed":
                return   # initial 이벤트는 완전히 무시

            if not cctv_case_key:
                self.get_logger().warn("[cctv/confirmed] cctv_case_key 없음. 업로드 스킵.")
                return

            plate_number = "UNKNOWN"
            if plate_crop is not None:
                if self.gcv_client is not None:
                    plate_number = self._ocr_gcv(plate_crop)
                if plate_number == "UNKNOWN" and self.ocr_reader is not None:
                    plate_number = self._ocr_paddle(plate_crop)

            self.db_ref.reference(
                f"{FIREBASE_CCTV_DB_PATH}/{cctv_case_key}"
            ).update({
                "amr_evidence_image": self._to_b64(frame),
                "amr_confirmed_at":   now_iso,
                "plate_image":        self._to_b64(plate_crop) if plate_crop is not None else None,
                "plate_number":       plate_number,
            })
            self.get_logger().info(
                f"[cctv/confirmed] 번호판 {plate_number} → cctv_detections/{cctv_case_key}"
            )
            return   # detections 에는 아무것도 쓰지 않음

        # ════════════════════════════════════════════════════════════════════
        # [B] amr_start 모드: confirmed 시에만 Firebase 업로드
        #     initial  → 로컬 임시 파일로만 저장 (Firebase 미사용)
        #     confirmed → 로컬 파일 + 현재 프레임을 합쳐 Firebase에 한 번에 업로드
        #                 업로드 완료 후 로컬 임시 파일 삭제
        # ════════════════════════════════════════════════════════════════════
        if event == "initial":
            # ── 로컬 임시 저장만 수행, Firebase 미사용 ──
            case_key             = now_dt.strftime("%Y%m%d_%H%M%S_%f")
            self._local_case_key = case_key

            frame_path = os.path.join(LOCAL_TEMP_DIR, case_key + "_initial_frame.jpg")
            plate_path = os.path.join(LOCAL_TEMP_DIR, case_key + "_initial_plate.jpg")

            try:
                cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if plate_crop is not None and plate_crop.size > 0:
                    cv2.imwrite(plate_path, plate_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                self.get_logger().info(f"[initial/local] 임시 저장 완료: {case_key}")
            except Exception as e:
                self.get_logger().error(f"[initial/local] 임시 저장 실패: {e}")

        elif event == "confirmed":
            if self._local_case_key is None:
                self.get_logger().warn("[confirmed] local_case_key 없음. 업로드 스킵.")
                return

            case_key   = self._local_case_key
            frame_path = os.path.join(LOCAL_TEMP_DIR, case_key + "_initial_frame.jpg")
            plate_path = os.path.join(LOCAL_TEMP_DIR, case_key + "_initial_plate.jpg")

            # ── initial 프레임 로컬에서 복원 ──
            initial_frame = cv2.imread(frame_path)
            initial_plate = cv2.imread(plate_path) if os.path.exists(plate_path) else None

            # ── OCR 수행 (confirmed 프레임의 번호판 크롭 사용) ──
            plate_number = "UNKNOWN"
            if plate_crop is not None:
                if self.gcv_client is not None:
                    plate_number = self._ocr_gcv(plate_crop)
                if plate_number == "UNKNOWN" and self.ocr_reader is not None:
                    plate_number = self._ocr_paddle(plate_crop)

            # ── Firebase에 한 번에 업로드 ──
            confirmed_at_iso = now_iso
            detected_at_iso  = datetime.datetime.strptime(
                case_key, "%Y%m%d_%H%M%S_%f"
            ).isoformat()

            case_ref = self.db_ref.reference(f"{FIREBASE_DB_PATH}/{case_key}")
            case_ref.set({
                "status":         "confirmed",
                "detected_at":    detected_at_iso,
                "confirmed_at":   confirmed_at_iso,
                "initial_image":  self._to_b64(initial_frame) if initial_frame is not None else None,
                "evidence_image": self._to_b64(frame),
                "plate_image":    self._to_b64(plate_crop) if plate_crop is not None else None,
                "plate_number":   plate_number,
            })
            self.get_logger().info(
                f"[confirmed] Plate: {plate_number}  key: {case_key}  at {now_dt.strftime('%H:%M:%S')}"
            )

            # ── 로컬 임시 파일 삭제 ──
            for path in (frame_path, plate_path):
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except Exception as e:
                    self.get_logger().warn(f"[confirmed] 임시 파일 삭제 실패: {e}")
            self._local_case_key = None


    def _ocr_gcv(self, img: np.ndarray) -> str:
        '''
        Google Cloud Vision API로 번호판 이미지를 인식한다. (1순위)

        벤치마크 결과: 원본(101x30) 그대로도 전처리 없이 "097하0228" 완벽 인식.
        언어 힌트를 한국어("ko")로 고정해 한글이 영문·숫자로 오인되는 것을 방지한다.
        API 오류 또는 빈 결과 시 "UNKNOWN" 반환 → PaddleOCR fallback으로 이어짐.
        '''
        try:
            _, buf    = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            gcv_image = vision.Image(content=buf.tobytes())
            context   = vision.ImageContext(language_hints=["ko"])
            response  = self.gcv_client.document_text_detection(
                image=gcv_image, image_context=context
            )
            if response.error.message:
                self.get_logger().warn(f"GCV API 오류: {response.error.message}")
                return "UNKNOWN"
            text = response.full_text_annotation.text.replace(" ", "").replace("\n", "")
            result = text if text else "UNKNOWN"
            self.get_logger().info(f"[OCR/GCV] → '{result}'")
            return result
        except Exception as e:
            self.get_logger().warn(f"GCV 호출 실패: {e}")
            return "UNKNOWN"

    def _ocr_paddle(self, img: np.ndarray) -> str:
        '''
        PaddleOCR로 번호판 이미지를 인식한다. (GCV 실패 시 fallback)

        벤치마크 결과:
          - 원본(101x30)은 오인식(0.62). 2x 확대 시 0.97로 정확 인식.
          - 원본과 180° 회전본 모두 시도해 신뢰도 합산이 높은 쪽을 채택한다.
        키 이름: PaddleX OCRResult는 rec_texts / rec_scores (복수형) 사용.
        '''
        
        if self.ocr_reader is None or img is None:
            return "UNKNOWN"
 
        def _run_ocr(image: np.ndarray) -> tuple[list[str], list[float]]:
            '''단일 이미지에 대해 OCR을 실행하고 (texts, scores) 튜플을 반환한다.'''
            results = self.ocr_reader.predict(image)
            if not isinstance(results, list):
                results = list(results)
 
            texts, scores = [], []
            for res in results:
                # ── PaddleX OCRResult: dict-like 객체 (rec_texts / rec_scores, 복수형) ──
                if hasattr(res, "__getitem__") or isinstance(res, dict):
                    try:
                        for t, s in zip(res["rec_texts"], res["rec_scores"]):
                            texts.append(t)
                            scores.append(s)
                        continue
                    except (KeyError, TypeError):
                        pass
 
                # ── PaddleX: get_res() 메서드가 있는 경우 (구형 API) ──
                if hasattr(res, "get_res"):
                    d = res.get_res()
                    for t, s in zip(d.get("rec_texts", d.get("rec_text", [])),
                                    d.get("rec_scores", d.get("rec_score", []))):
                        texts.append(t)
                        scores.append(s)
 
                # ── 구버전: 리스트 형태 [[bbox, [text, score]], ...] ──
                elif isinstance(res, list):
                    for line in res:
                        try:
                            if len(line) >= 2 and isinstance(line[1], (list, tuple)):
                                texts.append(line[1][0])
                                scores.append(line[1][1])
                        except Exception:
                            pass
 
            return texts, scores
 
        try:
            # ── 이미지가 너무 작으면 확대 (height < 60px 기준) ──
            h, w = img.shape[:2]
            if h < 60:
                scale = max(2, 60 // h)
                img   = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
 
            # ── 원본 방향과 180° 회전 모두 시도 → 신뢰도 합산이 높은 쪽 채택 ──
            img_180 = cv2.rotate(img, cv2.ROTATE_180)
 
            texts_orig, scores_orig = _run_ocr(img)
            texts_180,  scores_180  = _run_ocr(img_180)
 
            sum_orig = sum(s for s in scores_orig if s >= 0.6)
            sum_180  = sum(s for s in scores_180  if s >= 0.6)
 
            if sum_180 > sum_orig:
                texts, scores = texts_180, scores_180
                self.get_logger().info("[OCR] 180° 회전본이 신뢰도 높음 → 회전본 사용")
            else:
                texts, scores = texts_orig, scores_orig
 
            # ── 신뢰도 0.6 미만 제거 후 결합 ──
            filtered = [t for t, s in zip(texts, scores) if s >= 0.6]
            result   = "".join(filtered).replace(" ", "")
 
            self.get_logger().info(
                f"[OCR] texts={texts}  scores={[round(s,3) for s in scores]}  "
                f"→ '{result}'"
            )
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
        미확정 로컬 임시 파일도 이 시점에 삭제한다.
        '''
        self._audio_stop_event.set()   # 알림음 스레드 종료 신호 (Santa 루프 즉시 탈출)
        self.save_queue.put(None)      # 업로드 워커 스레드 종료 신호 (Sentinel)
        self._clear_local_temp()       # 미확정 로컬 임시 파일 삭제
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
