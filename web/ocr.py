#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
[프로젝트: Click Car - AMR 카메라 기반 불법주정차 단속 노드]
- 최종 수정: 2026-03-12

[System Architecture & Role]
1. 수신: OAK-D RGB 카메라로부터 CompressedImage 수신 (ROS2)
2. 탐지: YOLOv8 기반 차량(car) + 번호판(id) 동시 탐지
3. 필터링: 화면 내 여러 차량 중 Bounding Box 면적이 가장 큰 차량 1대만 단속 대상으로 선정
4. 검증: 번호판이 차량 영역 내부에 있는지 Overlap 검증
5. 추적: IoU 기반 동일 차량 식별 + 30초 타이머 관리
6. 송신 1: 최초 탐지 → 차량 전체 이미지 큐 투입
7. 송신 2: 30초 초과 → 번호판 크롭 이미지 큐 투입
8. 워커 스레드: 큐에서 꺼낸 번호판 이미지를 반듯하게 펴고(Unwarp) OCR 수행 후 Firebase 비동기 업로드
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
import easyocr


# ──────────────────────────────────────────────
# [CHAPTER 1: 하이퍼파라미터]
# ──────────────────────────────────────────────
ROBOT_NAMESPACE          = "/robot2"
MODEL_PATH               = "/home/rokey/click_car/models/amr.pt"
FIREBASE_CRED_PATH       = "/home/rokey/click_car/web/click_car.json"
FIREBASE_DB_URL          = "https://iligalstop-default-rtdb.asia-southeast1.firebasedatabase.app"
FIREBASE_DB_PATH         = "detections"

CONF_THRESHOLD           = 0.50
ID_IN_CAR_OVERLAP_THRESH = 0.50
CAR_IOU_THRESH           = 0.30
YOLO_IMG_SIZE            = 704
PARKING_TIMEOUT_SEC      = 30.0
SAVE_QUEUE_MAXSIZE       = 50  

# ── 토픽 경로 ──────────────────────────────────
TOPIC_RGB   = f"{ROBOT_NAMESPACE}/oakd/rgb/image_raw/compressed"


# ──────────────────────────────────────────────
# [CHAPTER 2: 차량 트래킹 상태 컨테이너]
# ──────────────────────────────────────────────

class TrackedVehicle:
    '''
    단일 차량의 타이머 및 최신 탐지 상태를 보관하는 데이터 컨테이너입니다.
    최초 발견 시각(first_seen)을 기준으로 30초 단속 타이머를 계산하며,
    DB 업로드 중복을 막기 위한 플래그(car_uploaded, plate_uploaded)를 관리합니다.
    '''
    def __init__(self, car_det: dict, id_det: dict):
        ''' 최초 객체 생성 시 현재 시간을 기록하여 타이머를 시작합니다. '''
        now                  = time.monotonic()
        self.first_seen      = now
        self.last_seen       = now
        self.car_det         = car_det
        self.id_det          = id_det
        self.car_uploaded    = False
        self.plate_uploaded  = False

    def elapsed(self) -> float:
        ''' 최초 탐지 시점으로부터 경과된 시간(초)을 반환합니다. (단속 기준 판별용) '''
        return time.monotonic() - self.first_seen

    def update(self, car_det: dict, id_det: dict):
        ''' 다음 프레임에서도 동일 차량이 추적되었을 때 위치 좌표와 갱신 시간을 업데이트합니다. '''
        self.last_seen = time.monotonic()
        self.car_det   = car_det
        self.id_det    = id_det


# ──────────────────────────────────────────────
# [CHAPTER 3: 메인 노드]
# ──────────────────────────────────────────────

class ParkingDetectionNode(Node):
    '''
    카메라 영상을 받아 불법주정차 단속을 수행하는 메인 ROS2 노드입니다.
    - 단일 타겟팅: 화면에 여러 대가 있어도 '가장 큰 차량' 1대만 단속 대상으로 추적합니다.
    - 스레드 분리: 메인 스레드는 영상 수신 및 객체 탐지(YOLO)만 수행하고,
                   무거운 OCR 연산과 네트워크 업로드는 워커 스레드가 비동기로 처리합니다.
    '''

    def __init__(self):
        super().__init__("parking_detection_node")

        self.tracked_vehicles = []
        self.save_queue       = queue.Queue(maxsize=SAVE_QUEUE_MAXSIZE)
        self.db_ref           = None
        self.ocr_reader       = None

        self._load_model()
        self._init_firebase()
        self._init_ocr()
        self._init_subscriber()

        # 네트워크 송신 및 OCR 연산으로 인한 프레임 드랍을 막기 위한 백그라운드 스레드 실행
        threading.Thread(target=self._upload_worker, daemon=True).start()
        cv2.namedWindow("Parking Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Parking Detection", YOLO_IMG_SIZE, YOLO_IMG_SIZE)
        self.get_logger().info("Node ready.")

    def _load_model(self):
        ''' YOLOv8 모델을 로드하고 빈 이미지를 넣어 Cold Start 지연을 방지(워밍업)합니다. '''
        self.model = YOLO(MODEL_PATH)
        self.model.predict(
            source=np.zeros((YOLO_IMG_SIZE, YOLO_IMG_SIZE, 3), dtype=np.uint8),
            imgsz=YOLO_IMG_SIZE, verbose=False
        )
        self.get_logger().info("YOLO warm-up complete.")

    def _init_firebase(self):
        ''' Firebase Admin SDK를 초기화하고 Realtime DB 참조 객체를 생성합니다. '''
        try:
            firebase_admin.initialize_app(
                credentials.Certificate(FIREBASE_CRED_PATH),
                {"databaseURL": FIREBASE_DB_URL}
            )
            self.db_ref = db.reference(FIREBASE_DB_PATH)
            self.get_logger().info("Firebase connected.")
        except Exception as e:
            self.get_logger().error(f"Firebase init failed: {e}")

    def _init_ocr(self):
        ''' 한국어와 영어를 지원하는 EasyOCR 모델을 메모리에 로드합니다. '''
        try:
            self.ocr_reader = easyocr.Reader(['ko', 'en'])
            self.get_logger().info("EasyOCR initialized.")
        except Exception as e:
            self.get_logger().error(f"EasyOCR init failed: {e}")

    def _init_subscriber(self):
        ''' 최신 프레임을 지연 없이 받기 위해 QoS를 BEST_EFFORT로 설정하여 카메라 토픽을 구독합니다. '''
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.create_subscription(CompressedImage, TOPIC_RGB, self.image_callback, qos)

    def image_callback(self, msg: CompressedImage):
        '''
        새로운 이미지 프레임이 들어올 때마다 호출되는 핵심 파이프라인.
        디코딩 → YOLO 탐지 → Overlap 기반 차량/번호판 매칭 → 가장 큰 타겟 필터링 → 트래킹 갱신을 수행합니다.
        '''
        frame = cv2.imdecode(np.array(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return

        cars, ids = self._detect(frame)
        validated_pairs = []

        # 번호판이 차량 영역 내부에 겹치는지 검증하여 유효한 차량-번호판 쌍(Pair) 생성
        for id_det in ids:
            car = self._find_parent_car(id_det, cars)
            if car is not None:
                validated_pairs.append((car, id_det))

        # [필터링 로직] 여러 대의 차량이 감지되었을 경우, Bbox 면적(area)이 가장 큰 1대만 타겟팅
        if validated_pairs:
            validated_pairs.sort(key=lambda pair: pair[0]["area"], reverse=True)
            validated_pairs = [validated_pairs[0]]

        self._update_tracking(frame, validated_pairs)
        self._draw(frame)

    def _detect(self, frame: np.ndarray) -> tuple[list, list]:
        '''
        YOLOv8 모델로 이미지를 추론하고, 탐지된 객체 중 'car'와 'id'(번호판)만 분리하여 반환합니다.
        각 요소는 좌표(x1, y1, x2, y2), 신뢰도(conf), 면적(area) 정보를 포함하는 사전(dict)입니다.
        '''
        results = self.model.predict(source=frame, imgsz=YOLO_IMG_SIZE,
                                     conf=CONF_THRESHOLD, verbose=False)
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
                "area":       max(0, x2 - x1) * max(0, y2 - y1),
            }
            (cars if name == "car" else ids).append(det)
        return cars, ids

    def _find_parent_car(self, id_det: dict, cars: list) -> dict | None:
        '''
        번호판 Bounding Box가 어느 차량의 Bounding Box 내부에 속해 있는지 계산합니다.
        교집합 면적이 번호판 면적의 ID_IN_CAR_OVERLAP_THRESH(예: 50%) 이상인 차량 중
        가장 많이 겹치는 차량을 반환합니다.
        '''
        id_area = max(1, id_det["area"])
        def overlap(car):
            ix = max(0, min(id_det["x2"], car["x2"]) - max(id_det["x1"], car["x1"]))
            iy = max(0, min(id_det["y2"], car["y2"]) - max(id_det["y1"], car["y1"]))
            return (ix * iy) / id_area

        candidates = [(overlap(car), car) for car in cars if overlap(car) >= ID_IN_CAR_OVERLAP_THRESH]
        return max(candidates, key=lambda x: x[0])[1] if candidates else None

    @staticmethod
    def _iou(a: dict, b: dict) -> float:
        '''
        두 Bounding Box 간의 IoU(Intersection over Union, 교집합/합집합)를 계산합니다.
        이전 프레임의 차량과 현재 프레임의 차량이 동일한 객체인지 추적(Tracking)할 때 사용됩니다.
        '''
        ix = max(0, min(a["x2"], b["x2"]) - max(a["x1"], b["x1"]))
        iy = max(0, min(a["y2"], b["y2"]) - max(a["y1"], b["y1"]))
        inter = ix * iy
        if inter == 0:
            return 0.0
        area_a = max(1, a["area"])
        area_b = max(1, b["area"])
        return inter / (area_a + area_b - inter)

    def _update_tracking(self, frame: np.ndarray, validated_pairs: list):
        '''
        차량의 추적 상태와 단속 타이머를 관리하는 핵심 로직입니다.
        1. 기존 추적 목록과 현재 프레임의 객체를 IoU 기반으로 매칭하여 위치/시간을 갱신합니다.
        2. 화면에서 사라졌거나 다른 차에 밀린(가장 크지 않은) 차량은 추적을 종료합니다.
        3. 새롭게 등장한 1순위 타겟은 증빙용 차량 전체 이미지를 큐에 즉시 투입하고 추적을 시작합니다.
        4. 추적 시간이 30초(PARKING_TIMEOUT_SEC)를 초과하면 번호판 크롭 이미지를 단속용 큐에 투입합니다.
        '''
        matched_track_indices = set()
        matched_pair_indices  = set()

        for t_idx, track in enumerate(self.tracked_vehicles):
            best_iou   = 0.0
            best_p_idx = -1
            for p_idx, (car_det, _) in enumerate(validated_pairs):
                if p_idx in matched_pair_indices:
                    continue
                iou = self._iou(track.car_det, car_det)
                if iou > best_iou:
                    best_iou   = iou
                    best_p_idx = p_idx

            if best_iou >= CAR_IOU_THRESH:
                car_det, id_det = validated_pairs[best_p_idx]
                track.update(car_det, id_det)
                matched_track_indices.add(t_idx)
                matched_pair_indices.add(best_p_idx)

        visible_tracks = [
            track for i, track in enumerate(self.tracked_vehicles)
            if i in matched_track_indices
        ]

        for p_idx, (car_det, id_det) in enumerate(validated_pairs):
            if p_idx not in matched_pair_indices:
                new_track = TrackedVehicle(car_det, id_det)
                visible_tracks.append(new_track)
                
                self._enqueue_car(frame, new_track)
                new_track.car_uploaded = True

        next_tracks = []
        for track in visible_tracks:
            if track.plate_uploaded:
                continue

            elapsed = track.elapsed()
            if elapsed >= PARKING_TIMEOUT_SEC:
                self._enqueue_plate(frame, track)
                track.plate_uploaded = True
            else:
                next_tracks.append(track)

        self.tracked_vehicles = next_tracks

    def _enqueue_car(self, frame: np.ndarray, track: TrackedVehicle):
        ''' 최초 탐지 시 차량 전체 프레임을 증빙 자료용(type="car")으로 업로드 큐에 비동기 투입합니다. '''
        try:
            self.save_queue.put_nowait({
                "type":    "car",
                "image":   frame.copy(),
                "car_det": track.car_det,
                "id_det":  track.id_det,
            })
        except queue.Full:
            self.get_logger().warn("Queue full. Car image dropped.")

    def _enqueue_plate(self, frame: np.ndarray, track: TrackedVehicle):
        ''' 30초 초과 시 번호판 영역만 잘라내어(Crop) 단속 자료용(type="plate")으로 큐에 비동기 투입합니다. '''
        id_det = track.id_det
        x1, y1, x2, y2 = id_det["x1"], id_det["y1"], id_det["x2"], id_det["y2"]
        crop = frame[y1:y2, x1:x2]

        if crop.size > 0:
            try:
                self.save_queue.put_nowait({
                    "type":    "plate",
                    "image":   crop.copy(),
                    "car_det": track.car_det,
                    "id_det":  track.id_det,
                })
            except queue.Full:
                self.get_logger().warn("Queue full. Plate image dropped.")

    def _draw(self, frame: np.ndarray):
        ''' 모니터링 창에 Bounding Box(차량: 초록, 번호판: 빨강)와 단속 타이머 진행률(Bar)을 덧그립니다. '''
        for track in self.tracked_vehicles:
            c       = track.car_det
            elapsed = track.elapsed()
            ratio   = min(elapsed / PARKING_TIMEOUT_SEC, 1.0)

            cv2.rectangle(frame, (c["x1"], c["y1"]), (c["x2"], c["y2"]), (0, 255, 0), 2)
            d = track.id_det
            cv2.rectangle(frame, (d["x1"], d["y1"]), (d["x2"], d["y2"]), (0, 0, 255), 2)

            bar_x1    = c["x1"]
            bar_y     = max(0, c["y1"] - 18)
            bar_w     = c["x2"] - c["x1"]
            bar_fill  = int(bar_w * ratio)
            cv2.rectangle(frame, (bar_x1, bar_y), (bar_x1 + bar_fill, bar_y + 10), (0, 255, 0), -1)
            cv2.putText(frame, f"Target: {elapsed:.0f}s", (bar_x1, bar_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        cv2.imshow("Parking Detection", frame)
        cv2.waitKey(1)

    # ── [신규] 시점 변환 (Perspective Transform) 유틸리티 ──

    @staticmethod
    def _order_points(pts: np.ndarray) -> np.ndarray:
        ''' 
        찾아낸 번호판의 4개 꼭짓점을 [좌상, 우상, 우하, 좌하]의 일정한 순서로 정렬하여 반환합니다.
        시점 변환 시 이미지가 뒤집히거나 꼬이는 것을 방지하기 위한 필수 전처리입니다.
        '''
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def _unwarp_plate(self, img_crop: np.ndarray) -> np.ndarray:
        '''
        비스듬하게 찍힌 찌그러진 번호판 이미지를 정면에서 본 직사각형 형태로 반듯하게 펴는 함수입니다.
        1. Canny 엣지와 findContours로 외곽선을 추출합니다.
        2. approxPolyDP 알고리즘의 epsilon 허용치(0.06)를 높여, 모서리가 둥글더라도 
           네 변의 연장선이 교차하는 4개의 꼭짓점을 강제로 찾아냅니다.
        3. getPerspectiveTransform을 이용해 수학적으로 이미지를 평평하게 변환합니다.
        '''
        gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blur, 50, 150)

        # 외곽선 검출
        cnts, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return img_crop

        # 면적이 가장 큰 순서대로 정렬 (번호판 자체가 가장 큰 영역일 확률이 높음)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

        for c in cnts:
            peri = cv2.arcLength(c, True)
            # epsilon 값(0.06)을 살짝 높여 둥근 모서리 곡선을 무시하고 직선의 교차점 4개를 찾도록 유도
            approx = cv2.approxPolyDP(c, 0.06 * peri, True)

            if len(approx) == 4:
                pts = approx.reshape(4, 2)
                rect = self._order_points(pts)
                (tl, tr, br, bl) = rect

                # 변환 후 이미지의 최대 가로/세로 길이 계산
                widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
                widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
                maxWidth = max(int(widthA), int(widthB))

                heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
                heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
                maxHeight = max(int(heightA), int(heightB))

                # 새 평면 좌표
                dst = np.array([
                    [0, 0],
                    [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1],
                    [0, maxHeight - 1]], dtype="float32")

                M = cv2.getPerspectiveTransform(rect, dst)
                unwarped = cv2.warpPerspective(img_crop, M, (maxWidth, maxHeight))
                return unwarped

        # 4개 꼭짓점을 찾지 못했을 경우 원본 크롭 반환 (Fallback)
        return img_crop

    # ── Firebase 업로드 및 OCR 워커 ──────────────

    def _upload_worker(self):
        '''
        메인 영상 처리 속도를 늦추지 않기 위해 백그라운드에서 동작하는 데몬 스레드입니다.
        큐(save_queue)에서 데이터를 꺼내 OCR(번호판 인식)과 Firebase 네트워크 업로드를 수행합니다.
        None을 수신하면 안전하게 종료(Sentinel)됩니다.
        '''
        while True:
            item = self.save_queue.get()
            if item is None:
                break
            try:
                self._upload(item)
            except Exception as e:
                self.get_logger().error(f"Upload error: {e}")

    def _upload(self, item: dict):
        '''
        실제 OCR 연산과 Firebase 송신을 처리하는 함수입니다.
        - type이 "car"면 증빙용(전체 이미지 + 번호 추출), "plate"면 단속용(크롭 이미지 + 번호 추출)입니다.
        - _unwarp_plate로 번호판을 반듯하게 편 후, EasyOCR을 구동하여 텍스트를 추출합니다.
        - 이미지를 Base64 문자열로 압축 변환한 뒤, 시간/인식번호/좌표 등과 함께 Firebase에 푸시합니다.
        '''
        if self.db_ref is None:
            return

        upload_type = item["type"]
        image       = item["image"]
        car_det     = item["car_det"]
        id_det      = item["id_det"]

        plate_number = "UNKNOWN"
        if self.ocr_reader is not None:
            # 1. OCR 대상 이미지 준비 (car일 경우 전체 프레임에서 번호판 영역만 임시로 자름)
            if upload_type == "car":
                px1, py1, px2, py2 = id_det["x1"], id_det["y1"], id_det["x2"], id_det["y2"]
                plate_crop = image[py1:py2, px1:px2]
            else:
                plate_crop = image

            if plate_crop.size > 0:
                # 2. 찌그러진 번호판 펴기 (Unwarp 적용)
                unwarped_plate = self._unwarp_plate(plate_crop)

                # 3. 펴진 이미지로 OCR 인식 수행
                ocr_result = self.ocr_reader.readtext(unwarped_plate, detail=0)
                if ocr_result:
                    plate_number = "".join(ocr_result).replace(" ", "")
        
        _, enc = cv2.imencode(".jpg", image)
        b64    = base64.b64encode(enc.tobytes()).decode("utf-8")

        now = datetime.datetime.now()
        
        self.db_ref.child(now.strftime("%Y%m%d_%H%M%S_%f")).set({
            "type":           upload_type,
            "detected_at":    now.isoformat(),
            "plate_number":   plate_number,
            "car_confidence": round(car_det["conf"], 4),
            "id_confidence":  round(id_det["conf"], 4),
            "car_bbox": {"x1": car_det["x1"], "y1": car_det["y1"], "x2": car_det["x2"], "y2": car_det["y2"]},
            "id_bbox":  {"x1": id_det["x1"], "y1": id_det["y1"], "x2": id_det["x2"], "y2": id_det["y2"]},
            "image_base64": b64,
        })
        self.get_logger().info(
            f"Uploaded [{upload_type}] Plate: {plate_number} at {now.strftime('%H:%M:%S')}"
        )

    def destroy_node(self):
        ''' 노드 종료 시 큐에 Sentinel(None)을 넣어 워커 스레드를 종료하고 GUI 자원을 해제합니다. '''
        self.save_queue.put(None)
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    ''' ROS2 노드 실행을 위한 엔트리 포인트 (진입점) 입니다. '''
    rclpy.init(args=args)
    node = ParkingDetectionNode()
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