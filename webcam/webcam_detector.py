import cv2                      # 영상 처리 및 카메라 입력
import numpy as np              # 행렬 연산 (Homography 계산 등)
import json                     # homography.json 파일 읽기
import time                     # 타임스탬프 및 업로드 주기 제어
import threading                # Firebase 업로드를 메인 루프와 분리하기 위한 스레드
from ultralytics import YOLO    # YOLO 모델 로드 및 추론


# ══════════════════════════════════════════════
#  사용자 설정
# ══════════════════════════════════════════════

HOMOGRAPHY_JSON = "homography.json"                                          # 캘리브레이션 결과 JSON 경로
MODEL_PATH      = "/home/rokey/click_car/models/webcam/v2/weights/best.pt"  # YOLO 가중치 파일 경로
CAMERA_INDEX    = 2     # cv2.VideoCapture 에 전달할 카메라 인덱스
CAM_WIDTH       = 640   # 캡처 해상도 너비 (캘리브레이션과 반드시 일치시킬 것)
CAM_HEIGHT      = 480   # 캡처 해상도 높이 (캘리브레이션과 반드시 일치시킬 것)

OFFSET_X_PLUS   =  0.0   # x 좌표에 더할 값 (+x 방향 보정)
OFFSET_X_MINUS  =  0.0   # x 좌표에서 뺄 값 (-x 방향 보정)
OFFSET_Y_PLUS   =  0.0   # y 좌표에 더할 값 (+y 방향 보정, 웹캠이 +y를 향할 때 사용)
OFFSET_Y_MINUS  =  0.0   # y 좌표에서 뺄 값 (-y 방향 보정)

ENABLE_TERMINAL = True   # True 이면 매 프레임 감지 결과를 터미널에 출력
ENABLE_FIREBASE = False  # True 이면 Realtime DB에 업로드 (터미널 검증 후 전환)

FIREBASE_CRED   = "~/Downloads/click-car-2f586-firebase-adminsdk-fbsvc-32cdaa4988.json"              # 서비스 계정 키 파일 경로
FIREBASE_DB_URL = "https://click-car-2f586-default-rtdb.asia-southeast1.firebasedatabase.app"  # Realtime DB URL
DB_PATH         = "webcam"   # DB 내 최상위 경로 (webcam/latest, webcam/cars/... 로 쓰임)
UPLOAD_HZ       = 10         # 초당 최대 업로드 횟수 (이 값의 역수가 최소 업로드 간격)


# ══════════════════════════════════════════════
#  Homography
# ══════════════════════════════════════════════

def load_homography(path: str):
    """
    저장된 JSON 파일에서 Homography 행렬과 실제 기준점 좌표를 불러온다.

    캘리브레이션 단계에서 camera_homography_calibrator.py 가 생성한 JSON을 읽으며,
    3×3 행렬(H)과 실제 좌표계 기준점(world_pts, 단위 m)을 반환한다.
    H는 픽셀 좌표를 실제 좌표(m)로 변환하는 데 사용되고,
    world_pts는 ROI 폴리곤 생성에 사용된다.
    """
    with open(path) as f:
        d = json.load(f)                                        # JSON 파일 전체를 딕셔너리로 파싱
    H         = np.array(d["homography_matrix"], dtype=np.float64)  # 3×3 Homography 행렬로 변환 (float64: 정밀도 확보)
    world_pts = np.array(d["world_points"],      dtype=np.float32)  # 캘리브레이션 기준점 실제 좌표 (ROI 생성에 사용)
    print(f"[Homography] 로드 완료  RMS={d.get('rms_error', 'N/A')}")  # 캘리브레이션 오차 확인용 출력
    return H, world_pts


def pixel_to_world(H, px, py):
    """
    단일 픽셀 좌표 (px, py) 를 Homography 행렬 H 를 통해 실제 좌표(m)로 변환한다.

    cv2.perspectiveTransform 은 동차 좌표계(homogeneous coordinates)를 사용하므로
    입력을 (1,1,2) 형태로 reshape 한 뒤 변환하고, 결과에서 (x, y) 를 추출한다.
    이후 사용자가 설정한 OFFSET 값을 더하거나 빼서 최종 실제 좌표를 반환한다.
    """
    pt = np.array([[[float(px), float(py)]]], dtype=np.float32)  # perspectiveTransform 입력 형식 (1,1,2)으로 변환
    w  = cv2.perspectiveTransform(pt, H)                         # Homography 행렬 H 를 적용해 실제 좌표로 변환
    wx = float(w[0, 0, 0]) + OFFSET_X_PLUS - OFFSET_X_MINUS     # x 변환 결과에 오프셋 적용
    wy = float(w[0, 0, 1]) + OFFSET_Y_PLUS - OFFSET_Y_MINUS     # y 변환 결과에 오프셋 적용
    return wx, wy


# ══════════════════════════════════════════════
#  ROI
# ══════════════════════════════════════════════

def make_roi_polygon(H_inv, world_pts):
    """
    실제 좌표계(m)의 기준점 4개를 역변환(H_inv)하여 픽셀 공간의 ROI 폴리곤을 생성한다.

    캘리브레이션 시 지정한 실제 좌표 4점을 역 Homography(H_inv)로 픽셀 좌표로 되돌려,
    카메라 화면 위에 그려질 관심 영역(ROI) 다각형을 만든다.
    반환된 폴리곤은 cv2.polylines 시각화 및 point_in_roi 내부 판별에 사용된다.
    """
    pts = np.array(world_pts, dtype=np.float32).reshape(-1, 1, 2)  # perspectiveTransform 입력 형식 (N,1,2)으로 변환
    px  = cv2.perspectiveTransform(pts, H_inv)                      # 역 Homography로 실제 좌표 → 픽셀 좌표 역변환
    return px.reshape(-1, 2).astype(np.int32)                       # polylines/pointPolygonTest 에 쓸 수 있는 정수형 (N,2)으로 변환


def point_in_roi(roi_poly, px, py):
    """
    주어진 픽셀 좌표 (px, py) 가 ROI 폴리곤 내부에 있는지 판별한다.

    cv2.pointPolygonTest 의 반환값이 0 이상이면 경계 포함 내부,
    음수이면 외부로 판단한다. ROI 밖의 detection을 필터링하는 데 사용된다.
    """
    return cv2.pointPolygonTest(
        roi_poly.astype(np.float32),   # pointPolygonTest 는 float32 폴리곤을 요구
        (float(px), float(py)),        # 판별할 점 (float 튜플)
        False                          # False = 내외부 판별만 (거리 계산 안 함, 속도 우선)
    ) >= 0                             # 0 이상이면 내부 또는 경계, 음수면 외부


# ══════════════════════════════════════════════
#  Bounding Box → 실제 좌표 4꼭짓점
# ══════════════════════════════════════════════

def bbox_corners_world(H, x1, y1, x2, y2):
    """
    YOLO가 출력한 bounding box의 픽셀 4꼭짓점을 실제 좌표(m)로 변환한다.

    쿼터뷰 환경에서 bounding box가 차량의 실제 점유 면적과 근사하다고 가정하여,
    별도의 차량 크기 추정 없이 bbox 꼭짓점을 그대로 pixel_to_world 로 변환한다.
    반환 순서는 좌상 → 우상 → 우하 → 좌하 이다.
    """
    corners_px = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]           # bbox 4꼭짓점 픽셀 좌표 (좌상, 우상, 우하, 좌하)
    return [pixel_to_world(H, px, py) for px, py in corners_px]      # 각 꼭짓점을 실제 좌표(m)로 변환한 리스트 반환


# ══════════════════════════════════════════════
#  출력 — 터미널
# ══════════════════════════════════════════════

def output_terminal(car_info: list):
    """
    현재 프레임에서 감지된 차량 정보를 터미널에 출력한다.

    감지된 차량이 없으면 아무것도 출력하지 않는다.
    차량별로 ID, 레이블, 신뢰도, 실제 중심 좌표(m),
    그리고 bounding box 4꼭짓점의 실제 좌표(m)를 출력한다.
    Firebase 업로드 전 육안 검증 단계에서 사용한다.
    """
    if not car_info:   # 감지된 차량이 없으면 아무것도 출력하지 않고 즉시 반환
        return
    print(f"\n{'─'*62}")   # 프레임 구분선 출력
    for c in car_info:     # 감지된 차량마다 반복
        # 차량 ID, 클래스명, 신뢰도, 실제 중심 좌표 출력
        print(f"  ID:{c['id']:>3}  {c['label']:<10}  conf:{c['conf']:.2f}"
              f"  center=({c['center_x']:+.3f}, {c['center_y']:+.3f}) m")
        # 영어 키를 한글 방향명으로 매핑
        label_map = {"top_left": "좌상", "top_right": "우상",
                     "bottom_right": "우하", "bottom_left": "좌하"}
        for key, corner in c["corners_m"].items():   # bbox 4꼭짓점 실제 좌표 출력
            print(f"         {label_map[key]}: ({corner['x']:+.4f}, {corner['y']:+.4f}) m")


# ══════════════════════════════════════════════
#  출력 — Firebase
# ══════════════════════════════════════════════

class FirebaseUploader:
    """
    Realtime Database에 차량 위치 정보를 실시간으로 업로드하는 클래스.

    메인 루프의 프레임 처리 속도를 저하시키지 않도록,
    업로드는 별도의 데몬 스레드(_worker)에서 수행된다.
    push() 로 적재된 데이터 중 가장 최신 것만 유지하며(오래된 데이터 자동 폐기),
    UPLOAD_HZ 로 지정한 최대 빈도로 Realtime Database에 기록한다.
    """

    def __init__(self, cred_path: str, db_url: str, db_path: str, upload_hz: int):
        """
        Firebase Admin SDK를 초기화하고 Realtime Database 클라이언트를 생성한다.
        업로드 스레드를 데몬으로 시작하여 메인 프로세스 종료 시 자동 정리된다.
        """
        import firebase_admin                              # Firebase Admin SDK (조건부 import: ENABLE_FIREBASE=False 시 불필요)
        from firebase_admin import credentials, db as rtdb

        cred = credentials.Certificate(cred_path)                        # 서비스 계정 키 파일로 인증 객체 생성
        firebase_admin.initialize_app(cred, {"databaseURL": db_url})     # DB URL을 포함해 Firebase 앱 초기화
        self.db_path      = db_path           # DB 내 루트 경로 ("webcam")
        self.rtdb         = rtdb              # Realtime DB 모듈 참조 저장 (_worker 에서 사용)
        self.min_interval = 1.0 / upload_hz  # 업로드 최소 간격(초) = 1 / UPLOAD_HZ
        self._last_upload = 0.0              # 마지막 업로드 시각 (단조 시계, 초기값 0)
        self._lock        = threading.Lock() # _pending 접근 시 메인 스레드와 워커 스레드 간 충돌 방지
        self._pending     = None             # 다음 업로드 대상 데이터 (None 이면 업로드할 내용 없음)
        self._active_ids  = set()            # 현재 DB에 기록된 차량 doc_id 집합 (삭제 판단용)
        self._thread      = threading.Thread(target=self._worker, daemon=True)  # 업로드 전담 데몬 스레드 생성
        self._thread.start()                 # 워커 스레드 시작
        print(f"[Firebase] Realtime DB 연결 완료  경로: /{db_path}")

    def push(self, snapshot: dict):
        """
        업로드할 데이터를 큐에 적재한다.

        스레드 안전을 위해 Lock으로 보호하며, 항상 최신 데이터만 유지한다.
        이전에 적재되었지만 아직 업로드되지 않은 데이터는 덮어써진다.
        """
        with self._lock:           # Lock 획득: 워커 스레드가 동시에 _pending을 읽지 못하도록 보호
            self._pending = snapshot   # 최신 데이터로 덮어쓰기 (이전 미업로드 데이터는 자동 폐기)

    def _worker(self):
        """
        백그라운드 스레드에서 주기적으로 Realtime Database에 데이터를 업로드한다.

        min_interval 간격을 지켜 과도한 쓰기를 방지한다.
        webcam/latest        → 전체 스냅샷 (set으로 덮어쓰기)
        webcam/cars/car_<id> → 차량별 최신 상태 (감지 중인 차량만 유지)
        사라진 차량은 해당 경로를 delete()로 삭제한다.
        """
        while True:
            time.sleep(0.01)   # 0.01초 간격으로 루프 (CPU 과점유 방지)
            now = time.time()
            if now - self._last_upload < self.min_interval:  # 업로드 최소 간격 미달이면 이번 루프 건너뜀
                continue
            with self._lock:       # Lock 획득: 메인 스레드의 push() 와 동시 접근 방지
                data = self._pending   # 대기 중인 최신 데이터를 꺼냄
                self._pending = None   # 꺼낸 뒤 초기화 (중복 업로드 방지)
            if data is None:       # 업로드할 데이터가 없으면 건너뜀
                continue
            try:
                # webcam/latest 경로에 현재 프레임 전체 스냅샷을 덮어쓰기
                self.rtdb.reference(f"{self.db_path}/latest").set(data)

                # 현재 프레임에서 감지된 차량을 webcam/cars/car_<id> 경로에 각각 갱신
                new_ids = set()   # 이번 프레임 차량 doc_id 집합 (이전 프레임과 비교용)
                for car in data.get("cars", []):              # 감지된 차량 목록 순회
                    doc_id = f"car_{car['id']}"               # 차량 ID로 고유 경로명 생성
                    self.rtdb.reference(f"{self.db_path}/cars/{doc_id}").set(car)  # 해당 경로에 차량 데이터 갱신
                    new_ids.add(doc_id)                       # 현재 프레임 차량 집합에 추가

                # 이전 프레임에는 있었지만 현재 프레임에서 사라진 차량 경로 삭제
                for old_id in self._active_ids - new_ids:
                    self.rtdb.reference(f"{self.db_path}/cars/{old_id}").delete()

                self._active_ids  = new_ids   # 활성 차량 집합을 현재 프레임 기준으로 갱신
                self._last_upload = now        # 마지막 업로드 시각 갱신

            except Exception as e:
                print(f"[Firebase] 업로드 오류: {e}")   # 네트워크 오류 등 예외를 출력하고 루프 유지


# ══════════════════════════════════════════════
#  시각화
# ══════════════════════════════════════════════

def draw_overlay(frame, H_inv, car_info, roi_poly):
    """
    카메라 프레임 위에 ROI 경계와 차량 정보를 시각화한다.

    ROI 폴리곤은 노란 실선으로 표시하고,
    각 차량에 대해 중심점(빨간 원), 실제 좌표로 역변환된 bbox(주황 사각형),
    ID 및 실제 좌표 텍스트를 그린다.
    bbox 역변환은 H_inv(역 Homography)를 통해 실제 좌표(m) → 픽셀로 되돌린다.
    """
    cv2.polylines(frame, [roi_poly], True, (0, 255, 255), 2)   # ROI 경계를 노란 실선(두께 2)으로 그림

    for c in car_info:   # 감지된 차량마다 반복
        cx_px, cy_px = c["pixel_center"]                       # bbox 중심 픽셀 좌표
        corners_m    = list(c["corners_m"].values())           # corners_m 딕셔너리의 값만 리스트로 추출

        cv2.circle(frame, (cx_px, cy_px), 5, (0, 0, 255), -1)   # 중심점을 빨간 원(반지름 5, 채움)으로 표시

        # 실제 좌표(m) 꼭짓점 배열을 perspectiveTransform 입력 형식 (N,1,2)으로 변환
        corners_np = np.array([[p["x"], p["y"]] for p in corners_m],
                               dtype=np.float32).reshape(-1, 1, 2)
        corners_px = cv2.perspectiveTransform(corners_np, H_inv)   # 역 Homography로 실제 좌표 → 픽셀 좌표 역변환
        corners_px = corners_px.reshape(-1, 2).astype(np.int32)    # polylines 에 쓸 수 있는 정수형 (N,2)으로 변환
        cv2.polylines(frame, [corners_px], True, (255, 128, 0), 2) # bbox를 주황 실선(두께 2)으로 그림

        # 차량 ID와 실제 중심 좌표(m)를 bbox 우상단에 텍스트로 표시
        cv2.putText(frame,
                    f"ID:{c['id']} ({c['center_x']:+.2f},{c['center_y']:+.2f})m",
                    (cx_px + 8, cy_px - 6),                 # 텍스트 시작 위치: 중심점 우상단
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)  # 초록, 크기 0.4, 안티앨리어싱


# ══════════════════════════════════════════════
#  메인
# ══════════════════════════════════════════════

def main():
    """
    전체 파이프라인을 초기화하고 프레임 처리 루프를 실행한다.

    초기화 순서: Homography 로드 → YOLO 모델 로드 → 카메라 오픈 → Firebase 연결(선택)
    매 프레임마다:
      1. YOLO ByteTrack으로 차량 탐지 및 ID 추적
      2. bbox 중심점이 ROI 밖이면 skip
      3. bbox 꼭짓점 및 중심을 실제 좌표(m)로 변환
      4. 터미널 출력 및/또는 Firebase 업로드
      5. 시각화 후 화면 표시
    """
    H, world_pts = load_homography(HOMOGRAPHY_JSON)   # JSON에서 Homography 행렬과 기준점 로드
    H_inv        = np.linalg.inv(H)                   # 역 Homography 계산 (실제 좌표 → 픽셀, 시각화용)
    roi_poly     = make_roi_polygon(H_inv, world_pts)  # 기준점을 픽셀로 역변환하여 ROI 폴리곤 생성

    model = YOLO(MODEL_PATH)   # YOLO 모델 로드
    print(f"[YOLO] 모델 로드: {MODEL_PATH}")

    cap = cv2.VideoCapture(CAMERA_INDEX)              # 지정 인덱스 카메라 열기
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)     # 캡처 너비 설정
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)    # 캡처 높이 설정
    print(f"[Camera] index={CAMERA_INDEX}  {CAM_WIDTH}x{CAM_HEIGHT}")

    uploader = None   # Firebase 비활성 시 None 유지
    if ENABLE_FIREBASE:
        uploader = FirebaseUploader(FIREBASE_CRED, FIREBASE_DB_URL, DB_PATH, UPLOAD_HZ)  # DB 연결 및 워커 스레드 시작
    else:
        print("[Firebase] 비활성화 — 터미널 출력 전용 모드")

    print(f"\n[Start]  터미널={ENABLE_TERMINAL}  Firebase={ENABLE_FIREBASE}  [q] 종료\n")

    while True:
        ret, frame = cap.read()   # 카메라에서 프레임 읽기
        if not ret:               # 프레임 읽기 실패 시 (카메라 연결 끊김 등) 루프 종료
            break

        # YOLO ByteTrack 추론: persist=True 로 프레임 간 ID 유지
        results = model.track(
            frame,
            tracker="bytetrack.yaml",  # ByteTrack 알고리즘 사용
            persist=True,              # 이전 프레임 트랙 정보를 유지해 ID 연속성 보장
            conf=0.87,                 # 신뢰도 임계값: 이 값 미만의 detection은 무시
            verbose=False,             # 매 프레임 YOLO 로그 출력 억제
        )

        boxes    = results[0].boxes   # 첫 번째 이미지(단일 프레임)의 bbox 결과
        car_info = []                 # 이번 프레임에서 유효한 차량 정보 목록 초기화

        if boxes is not None and boxes.id is not None:   # 트래킹 ID가 있는 bbox 가 존재할 때만 처리
            for tid, cls, conf, xyxy in zip(
                boxes.id.int().tolist(),    # 트래킹 ID 리스트
                boxes.cls.int().tolist(),   # 클래스 인덱스 리스트
                boxes.conf.tolist(),        # 신뢰도 리스트
                boxes.xyxy.tolist(),        # bbox 좌표 리스트 (x1,y1,x2,y2 형식)
            ):
                x1, y1, x2, y2 = [int(v) for v in xyxy]   # float bbox 좌표를 정수 픽셀로 변환
                cx_px = (x1 + x2) // 2                     # bbox 중심 x 픽셀 (정수 나눗셈)
                cy_px = (y1 + y2) // 2                     # bbox 중심 y 픽셀 (정수 나눗셈)

                if not point_in_roi(roi_poly, cx_px, cy_px):  # 중심점이 ROI 밖이면 이 차량 무시
                    continue

                corners_m  = bbox_corners_world(H, x1, y1, x2, y2)  # bbox 4꼭짓점 → 실제 좌표(m) 변환
                cx_m, cy_m = pixel_to_world(H, cx_px, cy_px)        # bbox 중심 → 실제 좌표(m) 변환

                tags = ["top_left", "top_right", "bottom_right", "bottom_left"]  # 꼭짓점 순서 레이블
                car_info.append({
                    "id":           tid,              # ByteTrack이 부여한 고유 트래킹 ID
                    "label":        model.names[cls], # 클래스 인덱스 → 클래스명 ("car" 등)
                    "conf":         round(conf, 3),   # 신뢰도 (소수점 3자리)
                    "pixel_center": [cx_px, cy_px],   # 중심 픽셀 좌표 — 시각화 전용 (DB 미업로드)
                    "center_x":     round(cx_m, 4),   # 중심 실제 x 좌표 (m, 소수점 4자리)
                    "center_y":     round(cy_m, 4),   # 중심 실제 y 좌표 (m, 소수점 4자리)
                    "corners_m":    {
                        tag: {"x": round(p[0], 4), "y": round(p[1], 4)}  # 꼭짓점별 실제 좌표 딕셔너리
                        for tag, p in zip(tags, corners_m)
                    },
                    "timestamp":    time.time(),       # 이 프레임의 Unix 타임스탬프
                })

        if ENABLE_TERMINAL:   # 터미널 출력이 활성화된 경우에만 출력
            output_terminal(car_info)

        if ENABLE_FIREBASE and uploader:   # Firebase 업로드가 활성화된 경우에만 push
            uploader.push({
                "timestamp": time.time(),   # 스냅샷 생성 시각
                # pixel_center는 시각화 전용이므로 DB 업로드 데이터에서 제외
                "cars":      [{k: v for k, v in c.items() if k != "pixel_center"} for c in car_info],
                "car_count": len(car_info),   # 이번 프레임 감지 차량 수
            })

        annotated = results[0].plot()                           # YOLO 기본 bbox + ID 시각화 적용
        draw_overlay(annotated, H_inv, car_info, roi_poly)      # ROI 및 실제 좌표 오버레이 추가
        cv2.imshow("RC Car Tracker", annotated)                 # 화면에 표시

        if cv2.waitKey(1) & 0xFF == ord('q'):   # 1ms 대기 후 'q' 키 입력 시 루프 종료
            break

    cap.release()             # 카메라 자원 해제
    cv2.destroyAllWindows()   # 모든 OpenCV 창 닫기
    print("[Done] 종료.")


if __name__ == "__main__":
    main()
