#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
[프로젝트: Click Car - AMR 카메라 기반 불법주정차 단속 노드]
- 최종 수정: 2026-03-15

[변경사항 요약 (2026-03-15)]
  - MultiThreadedExecutor 도입: cmd_callback에서 구독 직접 생성/해제 (Lock 보호)
  - _activation_timer 제거: 레이스 컨디션 원천 차단
  - image_callback 2Hz 스로틀: 컴퓨팅 자원 절감 (4노드 동시 실행 고려)
  - cctv_start 모드: case_key 없어도 cctv_detections 최신 케이스 자동 조회 후 덧붙임
  - 로그 강조: start 토픽 수신 시 ANSI 색상으로 강조
  - 이미지 디코딩: np.frombuffer로 메모리 복사 1회 절감

[System Architecture & Role]
1. 수신     : OAK-D RGB 카메라로부터 CompressedImage 수신 (ROS2 토픽)
2. 모드 수신 : std_msgs/String 토픽으로 동작 모드 수신
               - "amr_start"  → AMR 직접 출동 (단속 타이머 30초 + 경보음)
               - "cctv_start" → CCTV 연동 모드 (단속 타이머 5초, 알림음 없음)
3. 탐지     : YOLOv8 기반 차량(car) + 번호판(id) 동시 탐지
4. 필터링   : 화면 내 여러 차량 중 Bounding Box 면적이 가장 큰 차량 1대만 단속 대상
5. 검증     : 번호판이 차량 영역 내부에 있는지 Overlap 비율로 검증
6. 추적     : IoU 기반 동일 차량 식별 + 모드별 타이머 관리
7. 업로드   : Firebase Realtime DB에 케이스별 누적 저장
               - amr_start  모드: confirmed 시에만 업로드 (로컬 임시 저장 → confirmed 후 일괄 업로드)
               - cctv_start 모드: confirmed 시 cctv_detections 최신 케이스에 증거 덧붙임
                 (case_key 없어도 DB에서 최신 키를 자동 조회)
8. 알림음   : amr_start 모드 — 단순 경보음 3회 (피에조 버저 특성상 복잡한 멜로디 불가)

[Firebase 데이터 구조]
  detections/<YYYYMMDD_HHMMSSffffff>/
    ├── status          "confirmed"
    ├── detected_at     최초 감지 ISO 시각
    ├── confirmed_at    확정 ISO 시각
    ├── initial_image   최초 감지 전체 프레임 JPEG → base64
    ├── evidence_image  확정 시 전체 프레임 JPEG → base64
    ├── plate_image     확정 시 번호판 크롭 JPEG → base64
    └── plate_number    OCR 인식 결과

  cctv_detections/<케이스키>/  ← webcam_detector 가 생성, 여기에 AMR 증거 덧붙임
    ├── amr_evidence_image
    ├── amr_confirmed_at
    ├── plate_image
    └── plate_number
'''

import base64
import datetime
import os
import queue
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String, Bool
from irobot_create_msgs.msg import AudioNoteVector, AudioNote
from builtin_interfaces.msg import Duration
from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials, db
from paddleocr import PaddleOCR
from google.cloud import vision


# ──────────────────────────────────────────────
# [CHAPTER 1: 하이퍼파라미터]
# ──────────────────────────────────────────────

MODEL_PATH               = "/home/rokey/click_car/models/amr.pt"
FIREBASE_CRED_PATH       = "/home/rokey/Downloads/iligalstop-firebase-adminsdk-fbsvc-d989ef0f8c.json"
FIREBASE_DB_URL          = "https://iligalstop-default-rtdb.asia-southeast1.firebasedatabase.app"
FIREBASE_DB_PATH         = "detections"

CONF_THRESHOLD           = 0.50
ID_IN_CAR_OVERLAP_THRESH = 0.50
CAR_IOU_THRESH           = 0.30
YOLO_IMG_SIZE            = 704
PARKING_TIMEOUT_AMR      = 30.0
PARKING_TIMEOUT_CCTV     =  5.0
SAVE_QUEUE_MAXSIZE       = 50

# ── 멀티 로봇 지원 ────────────────────────────────────────────────────────────
TOPIC_CMD_ROBOT2  = "/robot2/start"
TOPIC_CMD_ROBOT3  = "/robot3/start"

FIREBASE_CCTV_DB_PATH = "cctv_detections"

GOOGLE_VISION_CRED_PATH = "/home/rokey/Downloads/google_vision.json"

LOCAL_TEMP_DIR        = "/tmp/click_car_amr"

CAPTURE_DONE_REPEAT   = 5
CAPTURE_DONE_INTERVAL = 0.2

# ── 트랙 grace period ─────────────────────────────────────────────────────────
# YOLO가 탐지 실패해도 이 시간(초) 이내라면 트랙을 유지한다.
# 2Hz 추론에서 1프레임 놓침 = 0.5초 공백이므로 2초면 연속 4프레임 탐지 실패도 허용.
# grace 기간 중에는 마지막 bbox 위치를 그대로 유지하고 타이머는 계속 진행된다.
# → 30초 타이머 리셋 방지, 객체 ID 유지
TRACK_GRACE_SEC = 2.0

# ── ANSI 색상 코드 (로그 강조용) ─────────────────────────────────────────────
_GRN  = "\033[92m"   # 밝은 초록 — start 수신 성공
_YEL  = "\033[93m"   # 노란색   — 경고/상태 변화
_RED  = "\033[91m"   # 빨간색   — 오류
_CYN  = "\033[96m"   # 하늘색   — confirmed 이벤트
_RST  = "\033[0m"    # 리셋


# ──────────────────────────────────────────────
# [CHAPTER 2: 멜로디 정의]
# ──────────────────────────────────────────────
# ★ 피에조 버저(Create3 스피커) 공통 주의사항:
#   - 공진 주파수(2~4kHz) 근처에서만 충분한 음량/음질 보장
#   - 원곡 음역(330~880Hz)은 피에조에서 거의 묵음 처리됨
#   - 해결책: 2옥타브 올려서 1760~3520Hz 대역 사용
#   - BPM을 원곡의 절반 이하로 낮춰야 피에조 응답 지연에도 음표 구분 가능
#
# ── 사용 방법 ────────────────────────────────────────────────────────────────
# 1. 기본값: 엘리제를 위하여 모티프 (ELISE_NOTES) — 현재 활성
# 2. 캐논 D 전체 테마 (CANON_NOTES) — 소리가 잘 난다면 아래 주석 교체
#    ALERT_NOTES = ELISE_NOTES  →  ALERT_NOTES = CANON_NOTES

_E = 750_000_000      # 8분음표  750ms  (BPM=40 기준)
_Q = 1_500_000_000    # 4분음표 1500ms
_H = 3_000_000_000    # 2분음표 3000ms
_R =   100_000_000    # 음표 간 묵음 100ms

# ── [A] 엘리제를 위하여 모티프 (기본값) ──────────────────────────────────────
# 원곡: A단조,  BPM ≈ 40 (원곡 BPM 60의 2/3 — 피에조 응답 고려)
# 2옥타브 up:
#   E4(330)→E7(2637)  D#4(311)→D#7(2489)
#   B4(494)→B6(1976)  D5(587)→D7(2349)  C5(523)→C7(2093)  A4(440)→A6(1760)
# 총 재생시간: 약 14.2초

ELISE_NOTES = [
    # ── 모티프 1회 ──────────────────────────────────
    (2637, _E), (0, _R),   # E7
    (2489, _E), (0, _R),   # D#7
    (2637, _E), (0, _R),   # E7
    (2489, _E), (0, _R),   # D#7
    (2637, _E), (0, _R),   # E7
    (1976, _E), (0, _R),   # B6
    (2349, _E), (0, _R),   # D7
    (2093, _E), (0, _R),   # C7
    (1760, _Q), (0, _R),   # A6  (종지)
    # ── 모티프 2회 반복 ─────────────────────────────
    (2637, _E), (0, _R),
    (2489, _E), (0, _R),
    (2637, _E), (0, _R),
    (2489, _E), (0, _R),
    (2637, _E), (0, _R),
    (1976, _E), (0, _R),
    (2349, _E), (0, _R),
    (2093, _E), (0, _R),
    (1760, _Q),             # 마지막 A, 묵음 없이 종료
]

# ── [B] 캐논 D장조 하이라이트 (주석 해제 후 사용) ────────────────────────────
# 원곡: D장조,  BPM ≈ 50 (원곡 BPM 100의 1/2 — 피에조 응답 고려)
# 2옥타브 up:
#   A4(440)→A6(1760)  B4(494)→B6(1976)  C#5(554)→C#7(2217)
#   D5(587)→D7(2349)  E5(659)→E7(2637)  F#5(740)→F#7(2960)
#   G5(784)→G7(3136)  A5(880)→A7(3520)
# 총 재생시간: 약 29.6초
#
# _CN = 800_000_000    # BPM=50 기준 8분음표 (800ms → 원곡 600ms 의 4/3배)
# _CL = 1_600_000_000  # 2배 음표
# _CX = 3_200_000_000  # 4배 음표 (종지)
# _CR = 100_000_000    # 묵음
#
# CANON_NOTES = [
#     # ── 하강 테마 1회 ────────────────────────────
#     (2960, _CN), (0, _CR),   # F#7
#     (2637, _CN), (0, _CR),   # E7
#     (2349, _CN), (0, _CR),   # D7
#     (2217, _CN), (0, _CR),   # C#7
#     (1976, _CN), (0, _CR),   # B6
#     (1760, _CN), (0, _CR),   # A6
#     (1976, _CN), (0, _CR),   # B6
#     (2217, _CN), (0, _CR),   # C#7
#     # ── 하강 테마 2회 ────────────────────────────
#     (2960, _CN), (0, _CR),
#     (2637, _CN), (0, _CR),
#     (2349, _CN), (0, _CR),
#     (2217, _CN), (0, _CR),
#     (1976, _CN), (0, _CR),
#     (1760, _CN), (0, _CR),
#     (1976, _CN), (0, _CR),
#     (2217, _CN), (0, _CR),
#     # ── 상승 응답 ─────────────────────────────────
#     (2349, _CN), (0, _CR),   # D7
#     (2637, _CN), (0, _CR),   # E7
#     (2960, _CN), (0, _CR),   # F#7
#     (3136, _CN), (0, _CR),   # G7
#     (2960, _CN), (0, _CR),   # F#7
#     (2637, _CN), (0, _CR),   # E7
#     (2349, _CN), (0, _CR),   # D7
#     (2217, _CN), (0, _CR),   # C#7
#     # ── 클라이맥스 + 종지 ──────────────────────────
#     (1760, _CN), (0, _CR),   # A6
#     (1976, _CN), (0, _CR),   # B6
#     (2349, _CN), (0, _CR),   # D7
#     (2960, _CN), (0, _CR),   # F#7
#     (3520, _CL),             # A7 (정점)
#     (3136, _CN), (0, _CR),   # G7
#     (2960, _CN), (0, _CR),   # F#7
#     (2637, _CN), (0, _CR),   # E7
#     (2349, _CX),             # D7 (종지)
# ]

# ── 실제 사용할 멜로디 선택 ───────────────────────────────────────────────────
# 엘리제 소리가 잘 나고 캐논을 원한다면:
#   아래 줄을  ALERT_NOTES = CANON_NOTES  로 교체
ALERT_NOTES = ELISE_NOTES

del _E, _Q, _H, _R


# ──────────────────────────────────────────────
# [CHAPTER 3: 차량 추적 상태 컨테이너]
# ──────────────────────────────────────────────

class TrackedVehicle:
    def __init__(self, car_det: dict, id_det: dict):
        now                      = time.monotonic()
        self.first_seen          = now
        self.last_seen           = now
        self.car_det             = car_det
        self.id_det              = id_det
        self.initial_uploaded    = False
        self.confirmed_uploaded  = False

    def elapsed(self) -> float:
        return time.monotonic() - self.first_seen

    def update(self, car_det: dict, id_det: dict):
        self.last_seen = time.monotonic()
        self.car_det   = car_det
        self.id_det    = id_det


# ──────────────────────────────────────────────
# [CHAPTER 4: 메인 ROS2 노드]
# ──────────────────────────────────────────────

class ParkingDetectionNode(Node):
    '''
    카메라 영상을 받아 불법주정차를 단속하는 메인 ROS2 노드.

    [구독 관리 설계 — 정적 구독 방식]
      robot2 / robot3 카메라 구독을 __init__ 에서 모두 생성한다.
      동적 create/destroy 는 ROS2 executor wait set 타이밍 문제를 유발하므로 사용하지 않는다.
      self.ns 플래그로 어느 로봇의 프레임을 처리할지 결정한다.
      퍼블리셔(오디오, capture_done)도 두 로봇 모두 미리 생성한다.
    '''

    def __init__(self):
        super().__init__("parking_detection_node")

        # ── 상태 변수 ──
        self.ns                = None   # 현재 활성 로봇 네임스페이스 (None = 대기)
        self.tracked_vehicles  = []
        self.save_queue        = queue.Queue(maxsize=SAVE_QUEUE_MAXSIZE)
        self.db_ref            = None
        self.ocr_reader        = None
        self.gcv_client        = None
        self.mode              = None
        self.parking_timeout   = None
        self._audio_stop_event = threading.Event()
        self._local_case_key   = None

        # ── 디스플레이 큐 ─────────────────────────────────────────────────────
        # imshow/waitKey 는 반드시 메인 스레드에서 호출해야 한다.
        # 콜백(워커 스레드)은 프레임을 여기에 넣고, main()이 꺼내서 표시한다.
        # maxsize=1: 항상 최신 프레임만 유지 (오래된 프레임 자동 드랍)
        self.display_queue = queue.Queue(maxsize=1)

        # ── Callback group ──
        self._cmd_group = ReentrantCallbackGroup()

        # ── 초기화 ──
        self._load_model()
        self._init_firebase()
        self._init_ocr()
        self._init_gcv()

        # ── 카메라 QoS ──
        cam_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # ── robot2 / robot3 카메라 구독 — __init__ 에서 모두 생성 ─────────────
        # executor 가 완전히 구동되기 전에 생성하므로 wait set 등록이 보장된다.
        # image_callback 내부에서 self.ns 로 필터링하여 활성 로봇 프레임만 처리한다.
        self.create_subscription(
            CompressedImage,
            "/robot2/oakd/rgb/image_raw/compressed",
            lambda msg: self.image_callback(msg, "/robot2"),
            cam_qos
        )
        self.create_subscription(
            CompressedImage,
            "/robot3/oakd/rgb/image_raw/compressed",
            lambda msg: self.image_callback(msg, "/robot3"),
            cam_qos
        )

        # ── CMD 구독 ──
        self.create_subscription(
            String, TOPIC_CMD_ROBOT2,
            lambda msg: self.cmd_callback(msg, "/robot2"),
            10, callback_group=self._cmd_group
        )
        self.create_subscription(
            String, TOPIC_CMD_ROBOT3,
            lambda msg: self.cmd_callback(msg, "/robot3"),
            10, callback_group=self._cmd_group
        )

        # ── 퍼블리셔 — 두 로봇 모두 미리 생성 ───────────────────────────────
        self._pubs = {
            ns: {
                "audio":        self.create_publisher(AudioNoteVector, f"{ns}/cmd_audio",     10),
                "capture_done": self.create_publisher(Bool,             f"{ns}/capture_done",  10),
            }
            for ns in ["/robot2", "/robot3"]
        }

        # 업로드·OCR 전담 데몬 스레드
        threading.Thread(target=self._upload_worker, daemon=True).start()

        os.makedirs(LOCAL_TEMP_DIR, exist_ok=True)

        self.get_logger().info(
            f"{_GRN}[READY] /robot2/start 또는 /robot3/start 대기 중...{_RST}"
        )

    # ── 초기화 메서드들 ──────────────────────────

    def _load_model(self):
        self.model = YOLO(MODEL_PATH)
        self.model.predict(
            source=np.zeros((YOLO_IMG_SIZE, YOLO_IMG_SIZE, 3), dtype=np.uint8),
            imgsz=YOLO_IMG_SIZE,
            verbose=False
        )
        self.get_logger().info("YOLO warm-up complete.")

    def _init_firebase(self):
        try:
            firebase_admin.initialize_app(
                credentials.Certificate(FIREBASE_CRED_PATH),
                {"databaseURL": FIREBASE_DB_URL}
            )
            self.db_ref = db
            self.get_logger().info("Firebase connected.")
        except Exception as e:
            self.get_logger().error(f"{_RED}Firebase init failed: {e}{_RST}")

    def _init_ocr(self):
        try:
            self.ocr_reader = PaddleOCR(
                lang="korean",
                use_textline_orientation=True,
                enable_mkldnn=False
            )
            self.get_logger().info("PaddleOCR initialized.")
        except Exception as e:
            self.get_logger().error(f"{_RED}PaddleOCR init failed: {e}{_RST}")

    def _init_gcv(self):
        try:
            os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", GOOGLE_VISION_CRED_PATH)
            self.gcv_client = vision.ImageAnnotatorClient()
            self.get_logger().info("Google Cloud Vision initialized.")
        except Exception as e:
            self.get_logger().error(f"{_RED}GCV init failed (PaddleOCR fallback 사용): {e}{_RST}")

    # ── 모드 제어 ────────────────────────────────

    def cmd_callback(self, msg: String, ns: str):
        '''
        start 토픽 수신 콜백.

        [설계 변경 — 정적 구독 방식]
          카메라 구독은 __init__ 에서 이미 생성되어 있다.
          여기서는 self.ns / self.mode 플래그만 세팅한다.
          image_callback 이 self.ns 를 보고 해당 로봇 프레임만 처리한다.
          create_subscription / destroy_subscription 은 일절 호출하지 않는다.
        '''
        raw   = msg.data.strip()
        token = raw.split(":")[0]

        # ════════════════════════════════════════════════
        # ★ START 신호 수신 — 강조 로그
        # ════════════════════════════════════════════════
        self.get_logger().info(
            f"{_GRN}{'='*60}{_RST}\n"
            f"{_GRN}[START 수신] 토픽: {TOPIC_CMD_ROBOT2 if ns=='/robot2' else TOPIC_CMD_ROBOT3}{_RST}\n"
            f"{_GRN}  로봇 네임스페이스 : {ns}{_RST}\n"
            f"{_GRN}  수신 메시지      : '{raw}'{_RST}\n"
            f"{_GRN}{'='*60}{_RST}"
        )

        if token == "amr_start":
            cmd = "amr_start"
        elif token == "cctv_start":
            cmd = "cctv_start"
        else:
            self.get_logger().warn(
                f"{_YEL}[CMD] 알 수 없는 명령 '{raw}' — 'amr_start' 또는 'cctv_start' 만 허용{_RST}"
            )
            return

        # ── 상태 초기화 및 플래그 세팅 ─────────────────────────────────────
        self._audio_stop_event.set()   # 이전 알림음 즉시 중단
        self._clear_local_temp()       # 미확정 임시 파일 삭제
        self.tracked_vehicles = []
        self.mode             = cmd
        self.ns               = ns     # ← 이 플래그로 image_callback 이 필터링

        self.get_logger().info(
            f"{_GRN}[ACTIVE] ns={ns}  mode={cmd}{_RST}\n"
            f"  처리할 카메라 : {ns}/oakd/rgb/image_raw/compressed"
        )

        # ── amr_start: 경보음 스레드 시작 ──
        if cmd == "amr_start":
            self.parking_timeout = PARKING_TIMEOUT_AMR
            self._audio_stop_event.clear()
            threading.Thread(target=self._play_alert, daemon=True).start()
            self.get_logger().info(
                f"{_YEL}[MODE] amr_start — 타이머 {PARKING_TIMEOUT_AMR:.0f}초, 경보음 활성{_RST}"
            )
        else:
            self.parking_timeout = PARKING_TIMEOUT_CCTV
            self.get_logger().info(
                f"{_YEL}[MODE] cctv_start — 타이머 {PARKING_TIMEOUT_CCTV:.0f}초{_RST}"
            )

    # ── 알림음 ───────────────────────────────────

    def _play_alert(self):
        '''
        엘리제를 위하여 모티프를 2옥타브 올려 AudioNoteVector 로 퍼블리시한다.

        [피에조 버저 한계와 해결책]
          Create3 스피커는 피에조 버저. 원곡 음역(330~440Hz)은 거의 묵음 처리됨.
          2옥타브 올려 1760~2637Hz 대역을 사용 → 공진 대역에서 음정 인식 가능.
          BPM=40 으로 매우 느리게 → 피에조 응답 지연에도 음표 구분 가능.
        '''
        if self._audio_stop_event.is_set():
            return

        audio_pub = self._pubs.get(self.ns, {}).get("audio")
        if audio_pub is None:
            self.get_logger().warn(f"{_YEL}[Audio] ns={self.ns} 퍼블리셔 없음 — 스킵{_RST}")
            return

        msg        = AudioNoteVector()
        msg.append = False
        msg.notes  = [
            AudioNote(
                frequency   = freq,
                max_runtime = Duration(
                    sec    = dur_ns // 1_000_000_000,
                    nanosec= dur_ns %  1_000_000_000
                )
            )
            for freq, dur_ns in ALERT_NOTES
        ]
        audio_pub.publish(msg)

        total_sec = sum(dur_ns for _, dur_ns in ALERT_NOTES) / 1_000_000_000
        self._audio_stop_event.wait(timeout=total_sec)

        if self._audio_stop_event.is_set():
            stop_msg        = AudioNoteVector()
            stop_msg.append = False
            stop_msg.notes  = []
            audio_pub.publish(stop_msg)

        self.get_logger().info("[Audio] 경보음 종료")

    # ── 메인 파이프라인 ──────────────────────────

    def image_callback(self, msg: CompressedImage, msg_ns: str):
        '''
        새 프레임 수신 콜백. robot2 / robot3 각각 별도 콜백으로 등록되어 있다.

        [네임스페이스 필터링]
          msg_ns 가 self.ns(현재 활성 로봇)와 다르면 즉시 반환한다.
          두 카메라를 모두 상시 구독하지만, 실제 처리는 활성 로봇 것만 한다.
          self.ns == None 이면 대기 상태 — 모든 프레임 스킵.

        [압축 해제]
          CompressedImage.data 는 JPEG 바이트. np.frombuffer → cv2.imdecode.
        '''
        # ── 비활성 로봇 프레임 즉시 드랍 ────────────────────────────────────
        # 활성 로봇이 있는데 다른 로봇 프레임이 오면 드랍
        if self.ns is not None and msg_ns != self.ns:
            return

        # ── JPEG 압축 해제 ──────────────────────────────────────────────────
        frame = cv2.imdecode(
            np.frombuffer(bytes(msg.data), dtype=np.uint8),
            cv2.IMREAD_COLOR
        )
        if frame is None:
            self.get_logger().warn("이미지 디코딩 실패", throttle_duration_sec=5.0)
            return

        # ── 대기 상태(self.ns is None): 화면용 큐에 넣고 YOLO 스킵 ────────────
        if self.ns is None:
            try:
                self.display_queue.put_nowait(frame)
            except queue.Full:
                pass
            return

        # ── YOLO 추론 및 추적 ───────────────────────────────────────────────
        cars, ids       = self._detect(frame)
        validated_pairs = []

        for id_det in ids:
            car = self._find_parent_car(id_det, cars)
            if car is not None:
                validated_pairs.append((car, id_det))

        if validated_pairs:
            validated_pairs.sort(key=lambda pair: pair[0]["area"], reverse=True)
            validated_pairs = [validated_pairs[0]]

        self._update_tracking(frame, validated_pairs)
        self._draw(frame)

    def _detect(self, frame: np.ndarray) -> tuple[list, list]:
        results = self.model.predict(
            source=frame,
            imgsz=YOLO_IMG_SIZE,
            conf=CONF_THRESHOLD,
            verbose=False
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
                "area":       max(0, x2-x1) * max(0, y2-y1),
            }
            (cars if name == "car" else ids).append(det)
        return cars, ids

    def _find_parent_car(self, id_det: dict, cars: list) -> dict | None:
        id_area = max(1, id_det["area"])

        def overlap(car):
            ix = max(0, min(id_det["x2"], car["x2"]) - max(id_det["x1"], car["x1"]))
            iy = max(0, min(id_det["y2"], car["y2"]) - max(id_det["y1"], car["y1"]))
            return (ix * iy) / id_area

        candidates = [(overlap(car), car) for car in cars if overlap(car) >= ID_IN_CAR_OVERLAP_THRESH]
        return max(candidates, key=lambda x: x[0])[1] if candidates else None

    @staticmethod
    def _iou(a: dict, b: dict) -> float:
        ix    = max(0, min(a["x2"], b["x2"]) - max(a["x1"], b["x1"]))
        iy    = max(0, min(a["y2"], b["y2"]) - max(a["y1"], b["y1"]))
        inter = ix * iy
        if inter == 0:
            return 0.0
        return inter / (max(1, a["area"]) + max(1, b["area"]) - inter)

    def _update_tracking(self, frame: np.ndarray, validated_pairs: list):
        matched_track_indices = set()
        matched_pair_indices  = set()

        for t_idx, track in enumerate(self.tracked_vehicles):
            best_iou, best_p_idx = 0.0, -1
            for p_idx, (car_det, _) in enumerate(validated_pairs):
                if p_idx in matched_pair_indices:
                    continue
                iou = self._iou(track.car_det, car_det)
                if iou > best_iou:
                    best_iou, best_p_idx = iou, p_idx

            if best_iou >= CAR_IOU_THRESH:
                car_det, id_det = validated_pairs[best_p_idx]
                track.update(car_det, id_det)
                matched_track_indices.add(t_idx)
                matched_pair_indices.add(best_p_idx)

        # ── Grace period 적용 ────────────────────────────────────────────────
        # 매칭 실패 트랙도 last_seen 이후 TRACK_GRACE_SEC 이내라면 유지한다.
        #
        # [왜 필요한가]
        #   2Hz 추론에서 YOLO가 1프레임 탐지 실패 = 0.5초 공백.
        #   grace 없이 즉시 제거하면 다음 탐지에서 신규 TrackedVehicle 이 생성되어
        #   first_seen 이 갱신 → 30초 타이머 리셋 → confirmed 영원히 못 남.
        #
        # [동작]
        #   matched  : 정상 매칭 → last_seen 갱신됨
        #   unmatched, grace 내 : 마지막 bbox 그대로 유지, 타이머 계속 진행
        #   unmatched, grace 초과 : 제거 (차량이 실제로 사라진 것으로 판단)
        now = time.monotonic()
        visible_tracks = [
            track for i, track in enumerate(self.tracked_vehicles)
            if i in matched_track_indices
            or (now - track.last_seen) < TRACK_GRACE_SEC
        ]
        # grace 로 유지된 트랙(매칭 안 됐지만 살아있는) 수 로그
        grace_kept = sum(
            1 for i, track in enumerate(self.tracked_vehicles)
            if i not in matched_track_indices
            and (now - track.last_seen) < TRACK_GRACE_SEC
        )
        if grace_kept > 0:
            self.get_logger().debug(
                f"[tracking] YOLO 탐지 실패 — grace 유지 {grace_kept}개 트랙 "
                f"(타이머 계속 진행 중)"
            )

        for p_idx, (car_det, id_det) in enumerate(validated_pairs):
            if p_idx not in matched_pair_indices:
                new_track = TrackedVehicle(car_det, id_det)
                visible_tracks.append(new_track)
                self._enqueue(frame, new_track, event="initial")
                new_track.initial_uploaded = True

        next_tracks = []
        for track in visible_tracks:
            if not track.confirmed_uploaded and track.elapsed() >= self.parking_timeout:
                self._enqueue(frame, track, event="confirmed")
                track.confirmed_uploaded = True

                confirmed_ns = self.ns   # 스냅샷 (mode=None 전에 저장)
                threading.Thread(
                    target=self._publish_capture_done_repeated,
                    args=(confirmed_ns,),
                    daemon=True
                ).start()
                self.get_logger().info(
                    f"{_CYN}[CONFIRMED] 번호판 확정 → capture_done ×{CAPTURE_DONE_REPEAT} 발송{_RST}"
                )

                # ── 대기 상태로 복귀 (플래그만 리셋) ───────────────────────
                # 구독은 계속 살아있음 — 다음 start 신호 때까지 ns 필터로 드랍
                self.mode = None
                self.ns   = None
                self.get_logger().info(
                    f"{_YEL}[DONE] confirmed 완료 → 대기 상태 복귀{_RST}"
                )

            next_tracks.append(track)

        self.tracked_vehicles = next_tracks

    def _enqueue(self, frame: np.ndarray, track: 'TrackedVehicle', event: str):
        id_det = track.id_det
        px1, py1, px2, py2 = id_det["x1"], id_det["y1"], id_det["x2"], id_det["y2"]
        plate_crop = frame[py1:py2, px1:px2]

        try:
            self.save_queue.put_nowait({
                "event":      event,
                "frame":      frame.copy(),
                "plate_crop": plate_crop.copy() if plate_crop.size > 0 else None,
                "car_det":    track.car_det,
                "id_det":     track.id_det,
                "mode":       self.mode,
            })
        except queue.Full:
            self.get_logger().warn(f"Queue full. '{event}' image dropped.")

    # ── 시각화 ──────────────────────────────────

    def _draw(self, frame: np.ndarray):
        for track in self.tracked_vehicles:
            c       = track.car_det
            elapsed = track.elapsed()
            ratio   = min(elapsed / self.parking_timeout, 1.0)

            cv2.rectangle(frame, (c["x1"], c["y1"]), (c["x2"], c["y2"]), (0, 255, 0), 2)

            d = track.id_det
            cv2.rectangle(frame, (d["x1"], d["y1"]), (d["x2"], d["y2"]), (0, 0, 255), 2)

            bar_x1   = c["x1"]
            bar_y    = max(0, c["y1"] - 18)
            bar_w    = c["x2"] - c["x1"]
            bar_fill = int(bar_w * ratio)
            cv2.rectangle(frame,
                          (bar_x1, bar_y), (bar_x1 + bar_fill, bar_y + 10),
                          (0, 255, 0), -1)
            cv2.putText(frame,
                        f"Target: {elapsed:.0f}s / {self.parking_timeout:.0f}s",
                        (bar_x1, bar_y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        try:
            self.display_queue.put_nowait(frame)
        except queue.Full:
            pass

    # ── capture_done 반복 발행 ───────────────────

    def _publish_capture_done_repeated(self, ns: str):
        pub = self._pubs.get(ns, {}).get("capture_done")
        if pub is None:
            return
        for i in range(CAPTURE_DONE_REPEAT):
            done_msg      = Bool()
            done_msg.data = True
            pub.publish(done_msg)
            self.get_logger().info(f"[capture_done] {i+1}/{CAPTURE_DONE_REPEAT}")
            if i < CAPTURE_DONE_REPEAT - 1:
                time.sleep(CAPTURE_DONE_INTERVAL)

    # ── 로컬 임시 데이터 정리 ───────────────────

    def _clear_local_temp(self):
        if self._local_case_key is None:
            return
        for suffix in ("_initial_frame.jpg", "_initial_plate.jpg"):
            path = os.path.join(LOCAL_TEMP_DIR, self._local_case_key + suffix)
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception as e:
                self.get_logger().warn(f"[local] 임시 파일 삭제 실패: {e}")
        self._local_case_key = None

    # ── Firebase 업로드 및 OCR 워커 ─────────────

    def _upload_worker(self):
        while True:
            item = self.save_queue.get()
            if item is None:
                break
            try:
                self._upload(item)
            except Exception as e:
                self.get_logger().error(f"{_RED}Upload error: {e}{_RST}")

    def _upload(self, item: dict):
        '''
        OCR 수행과 Firebase 업로드를 처리한다.

        ── amr_start 모드 ──────────────────────────────────────────────────────
          initial  : 로컬 임시 파일에만 저장 (Firebase 미사용)
          confirmed: 로컬 파일 복원 + OCR + Firebase detections/<타임스탬프> 업로드

        ── cctv_start 모드 ──────────────────────────────────────────────────────
          initial  : 무시
          confirmed: cctv_detections 에서 가장 최신 케이스 키를 자동 조회한 뒤
                     전체 프레임 / 번호판 이미지 / OCR 결과 세 가지를 덧붙임.
                     case_key 가 없어도 동작한다.
        '''
        if self.db_ref is None:
            return

        event      = item["event"]
        frame      = item["frame"]
        plate_crop = item["plate_crop"]
        item_mode  = item.get("mode")
        now_dt     = datetime.datetime.now()
        now_iso    = now_dt.isoformat()

        # ════════════════════════════════════════════════════════════════════
        # [A] cctv_start 모드
        # ════════════════════════════════════════════════════════════════════
        if item_mode == "cctv_start":
            if event != "confirmed":
                return

            # ── cctv_detections 에서 가장 최신 케이스 키 자동 조회 ──────────
            latest_key = None
            try:
                snapshot = (
                    self.db_ref.reference(FIREBASE_CCTV_DB_PATH)
                    .order_by_key()
                    .limit_to_last(1)
                    .get()
                )
                if snapshot:
                    latest_key = list(snapshot.keys())[0]
            except Exception as e:
                self.get_logger().warn(f"[cctv] 최신 케이스 조회 실패: {e}")

            if latest_key is None:
                self.get_logger().warn(
                    f"{_YEL}[cctv/confirmed] cctv_detections 에 케이스 없음 → 업로드 스킵{_RST}"
                )
                return

            plate_number = "UNKNOWN"
            if plate_crop is not None:
                if self.gcv_client is not None:
                    plate_number = self._ocr_gcv(plate_crop)
                if plate_number == "UNKNOWN" and self.ocr_reader is not None:
                    plate_number = self._ocr_paddle(plate_crop)

            self.db_ref.reference(
                f"{FIREBASE_CCTV_DB_PATH}/{latest_key}"
            ).update({
                "amr_evidence_image": self._to_b64(frame),
                "amr_confirmed_at":   now_iso,
                "plate_image":        self._to_b64(plate_crop) if plate_crop is not None else None,
                "plate_number":       plate_number,
            })
            self.get_logger().info(
                f"{_CYN}[cctv/confirmed] 번호판 {plate_number} → {FIREBASE_CCTV_DB_PATH}/{latest_key}{_RST}"
            )
            return

        # ════════════════════════════════════════════════════════════════════
        # [B] amr_start 모드
        # ════════════════════════════════════════════════════════════════════
        if event == "initial":
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
                self.get_logger().error(f"{_RED}[initial/local] 임시 저장 실패: {e}{_RST}")

        elif event == "confirmed":
            if self._local_case_key is None:
                self.get_logger().warn(
                    f"{_YEL}[confirmed] local_case_key 없음 → 업로드 스킵{_RST}"
                )
                return

            case_key   = self._local_case_key
            frame_path = os.path.join(LOCAL_TEMP_DIR, case_key + "_initial_frame.jpg")
            plate_path = os.path.join(LOCAL_TEMP_DIR, case_key + "_initial_plate.jpg")

            initial_frame = cv2.imread(frame_path)
            initial_plate = cv2.imread(plate_path) if os.path.exists(plate_path) else None

            plate_number = "UNKNOWN"
            if plate_crop is not None:
                if self.gcv_client is not None:
                    plate_number = self._ocr_gcv(plate_crop)
                if plate_number == "UNKNOWN" and self.ocr_reader is not None:
                    plate_number = self._ocr_paddle(plate_crop)

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
                f"{_CYN}[confirmed] Plate: {plate_number}  key: {case_key}{_RST}"
            )

            for path in (frame_path, plate_path):
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except Exception as e:
                    self.get_logger().warn(f"[confirmed] 임시 파일 삭제 실패: {e}")
            self._local_case_key = None

    # ── OCR ─────────────────────────────────────

    def _ocr_gcv(self, img: np.ndarray) -> str:
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
            text   = response.full_text_annotation.text.replace(" ", "").replace("\n", "")
            result = text if text else "UNKNOWN"
            self.get_logger().info(f"[OCR/GCV] → '{result}'")
            return result
        except Exception as e:
            self.get_logger().warn(f"GCV 호출 실패: {e}")
            return "UNKNOWN"

    def _ocr_paddle(self, img: np.ndarray) -> str:
        if self.ocr_reader is None or img is None:
            return "UNKNOWN"

        def _run_ocr(image: np.ndarray) -> tuple[list[str], list[float]]:
            results = self.ocr_reader.predict(image)
            if not isinstance(results, list):
                results = list(results)
            texts, scores = [], []
            for res in results:
                if hasattr(res, "__getitem__") or isinstance(res, dict):
                    try:
                        for t, s in zip(res["rec_texts"], res["rec_scores"]):
                            texts.append(t)
                            scores.append(s)
                        continue
                    except (KeyError, TypeError):
                        pass
                if hasattr(res, "get_res"):
                    d = res.get_res()
                    for t, s in zip(d.get("rec_texts", d.get("rec_text", [])),
                                    d.get("rec_scores", d.get("rec_score", []))):
                        texts.append(t)
                        scores.append(s)
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
            h, w = img.shape[:2]
            if h < 60:
                scale = max(2, 60 // h)
                img   = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

            img_180 = cv2.rotate(img, cv2.ROTATE_180)
            texts_orig, scores_orig = _run_ocr(img)
            texts_180,  scores_180  = _run_ocr(img_180)

            sum_orig = sum(s for s in scores_orig if s >= 0.6)
            sum_180  = sum(s for s in scores_180  if s >= 0.6)

            if sum_180 > sum_orig:
                texts, scores = texts_180, scores_180
            else:
                texts, scores = texts_orig, scores_orig

            filtered = [t for t, s in zip(texts, scores) if s >= 0.6]
            result   = "".join(filtered).replace(" ", "")
            self.get_logger().info(
                f"[OCR/Paddle] texts={texts}  scores={[round(s,3) for s in scores]}  → '{result}'"
            )
            return result if result else "UNKNOWN"
        except Exception as e:
            self.get_logger().warn(f"PaddleOCR 호출 실패: {e}")
            return "UNKNOWN"

    @staticmethod
    def _to_b64(img: np.ndarray) -> str:
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf).decode("utf-8")

    # ── 노드 종료 ────────────────────────────────

    def destroy_node(self):
        self._audio_stop_event.set()
        self.save_queue.put(None)
        self._clear_local_temp()
        cv2.destroyAllWindows()
        super().destroy_node()


# ──────────────────────────────────────────────
# [CHAPTER 5: 진입점]
# ──────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = ParkingDetectionNode()

    # ── GUI 창 생성 — 반드시 메인 스레드에서 ────────────────────────────────
    cv2.namedWindow("Plate Checking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Plate Checking", YOLO_IMG_SIZE, YOLO_IMG_SIZE)

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    # executor 를 별도 스레드에서 spin → 메인 스레드는 imshow 전담
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        while rclpy.ok():
            # 디스플레이 큐에서 프레임 꺼내 표시 (메인 스레드 전용)
            try:
                frame = node.display_queue.get(timeout=0.05)
                cv2.imshow("Plate Checking", frame)
            except queue.Empty:
                pass
            # waitKey 는 GUI 이벤트 루프 구동 — 메인 스레드에서만 유효
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
