
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
from sensor_msgs.msg import CompressedImage, CameraInfo
from std_msgs.msg import String
from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials, db

# ──────────────────────────────────────────────
# [CHAPTER 1: 하이퍼파라미터]
# ──────────────────────────────────────────────
ROBOT_NAMESPACE          = "/robot3"
MODEL_PATH               = "/home/rokey/click_car/models/amr.pt"
FIREBASE_CRED_PATH       = "/home/rokey/click_car/src/web/click_car.json"
FIREBASE_DB_URL          = "https://iligalstop-default-rtdb.asia-southeast1.firebasedatabase.app"
FIREBASE_DB_PATH         = "detections"

CONF_THRESHOLD           = 0.50
ID_IN_CAR_OVERLAP_THRESH = 0.50
CAR_IOU_THRESH           = 0.30
YOLO_IMG_SIZE            = 704
PARKING_TIMEOUT_SEC      = 30.0
SAVE_QUEUE_MAXSIZE       = 10

TOPIC_RGB        = f"{ROBOT_NAMESPACE}/oakd/rgb/image_raw/compressed"
TOPIC_DEPTH      = f"{ROBOT_NAMESPACE}/oakd/stereo/image_raw/compressedDepth"
TOPIC_INFO       = f"{ROBOT_NAMESPACE}/oakd/rgb/camera_info"
TOPIC_AMR_TARGET = "amr_done"   # 자료형: std_msgs/String, payload: "x,y"

# ──────────────────────────────────────────────
# [CHAPTER 2: 차량 트래킹 상태 컨테이너]
# ──────────────────────────────────────────────
class TrackedVehicle:
    def __init__(self, car_det: dict, id_det: dict):
        now                  = time.monotonic()
        self.first_seen      = now
        self.last_seen       = now
        self.car_det         = car_det
        self.id_det          = id_det
        self.car_uploaded    = False

        
        self.plate_uploaded  = False
        self.coord_published = False

    def elapsed(self) -> float:
        return time.monotonic() - self.first_seen

    def update(self, car_det: dict, id_det: dict):
        self.last_seen = time.monotonic()
        self.car_det   = car_det
        self.id_det    = id_det

# ──────────────────────────────────────────────
# [CHAPTER 3: 메인 노드]
# ──────────────────────────────────────────────
class ParkingDetectionNode(Node):
    def __init__(self):
        super().__init__("parking_detection_node")

        self.tracked_vehicles      = []
        self.save_queue            = queue.Queue(maxsize=SAVE_QUEUE_MAXSIZE)
        self.db_ref                = None
        self.latest_depth_frame    = None          # uint16 depth(mm) 저장
        self.camera_info           = None
        self._depth_lock           = threading.Lock()
        self._depth_no_data_warned = False
        self.last_rgb_received     = None

        self._load_model()
        self._init_firebase()
        self._init_subscriber()
        self._init_publisher()

        threading.Thread(target=self._upload_worker, daemon=True).start()

        cv2.namedWindow("Parking Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Parking Detection", YOLO_IMG_SIZE, YOLO_IMG_SIZE)

        self.create_timer(0.5, self._watchdog_timer)

        self.get_logger().info("Node ready.")

    # ── 초기화 ──────────────────────────────────

    def _load_model(self):
        self.model = YOLO(MODEL_PATH)
        self.model.predict(
            source=np.zeros((YOLO_IMG_SIZE, YOLO_IMG_SIZE, 3), dtype=np.uint8),
            imgsz=YOLO_IMG_SIZE,
            verbose=False
        )
        self.get_logger().info(f"[DEBUG-1] Model classes: {self.model.names}")
        self.get_logger().info("YOLO warm-up complete.")

    def _init_firebase(self):
        try:
            if not firebase_admin._apps:
                firebase_admin.initialize_app(
                    credentials.Certificate(FIREBASE_CRED_PATH),
                    {"databaseURL": FIREBASE_DB_URL}
                )
            self.db_ref = db.reference(FIREBASE_DB_PATH)
            self.get_logger().info("Firebase connected.")
        except Exception as e:
            self.get_logger().error(f"[DEBUG-2] Firebase init failed: {e}")

    def _init_subscriber(self):
        qos_be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        qos_rel = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.create_subscription(CompressedImage, TOPIC_RGB,   self.image_callback, qos_be)
        self.create_subscription(CompressedImage, TOPIC_DEPTH, self.depth_callback, qos_be)
        self.create_subscription(CameraInfo,      TOPIC_INFO,  self.info_callback,  qos_rel)

        self.get_logger().info(f"Subscribing RGB  : {TOPIC_RGB}")
        self.get_logger().info(f"Subscribing Depth: {TOPIC_DEPTH}")
        self.get_logger().info(f"Subscribing Info : {TOPIC_INFO}")

    def _init_publisher(self):
        qos_pub = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.amr_target_pub = self.create_publisher(String, TOPIC_AMR_TARGET, qos_pub)
        self.get_logger().info(f"Publishing target: {TOPIC_AMR_TARGET}")

    # ── watchdog ───────────────────────────────

    def _watchdog_timer(self):
        if self.last_rgb_received is None:
            self._draw_waiting_screen()

    def _draw_waiting_screen(self):
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(canvas, "Waiting for RGB topic...", (40, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(canvas, TOPIC_RGB, (40, 255),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        oak_topics_alive = self.last_rgb_received is not None
        status_text = "RGB RECEIVED" if oak_topics_alive else "NO RGB FRAME"
        color = (0, 255, 0) if oak_topics_alive else (0, 0, 255)
        cv2.putText(canvas, status_text, (40, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Parking Detection", canvas)
        cv2.waitKey(1)

    # ── compressedDepth 디코딩 ───────────────────

    def _decode_compressed_depth(self, msg: CompressedImage):
        """
        compressedDepth 메시지를 uint16 depth(mm) 이미지로 복원.
        환경에 따라 앞부분에 compressedDepth 헤더가 붙는 경우가 있어 둘 다 대응함.
        """
        try:
            raw = np.frombuffer(msg.data, dtype=np.uint8)

            # 1) 바로 디코딩 시도
            depth_img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
            if depth_img is not None and depth_img.size > 0:
                return depth_img

            # 2) compressedDepth 헤더 스킵 후 재시도
            idx = msg.data.find(b'PNG')
            if idx != -1:
                depth_img = cv2.imdecode(
                    np.frombuffer(msg.data[idx-1:], dtype=np.uint8),
                    cv2.IMREAD_UNCHANGED
                )
                if depth_img is not None and depth_img.size > 0:
                    return depth_img

            # 3) 더 느슨하게 PNG 시그니처 탐색
            png_sig = b'\x89PNG\r\n\x1a\n'
            idx = msg.data.find(png_sig)
            if idx != -1:
                depth_img = cv2.imdecode(
                    np.frombuffer(msg.data[idx:], dtype=np.uint8),
                    cv2.IMREAD_UNCHANGED
                )
                if depth_img is not None and depth_img.size > 0:
                    return depth_img

            return None

        except Exception as e:
            self.get_logger().warn(f"[DEBUG-D0] compressedDepth decode error: {e}")
            return None

    # ── Depth 콜백 ───────────────────────────────

    def depth_callback(self, msg: CompressedImage):
        try:
            depth_img = self._decode_compressed_depth(msg)

            if depth_img is None or depth_img.size == 0:
                self.get_logger().warn("[DEBUG-D1] Depth decode failed.")
                return

            if depth_img.dtype != np.uint16:
                self.get_logger().warn(
                    f"[DEBUG-D1] Decoded depth dtype is {depth_img.dtype}, expected uint16."
                )

            with self._depth_lock:
                self.latest_depth_frame = depth_img

        except Exception as e:
            self.get_logger().warn(f"[DEBUG-D2] depth_callback error: {e}")

    # ── CameraInfo 콜백 ──────────────────────────

    def info_callback(self, msg: CameraInfo):
        if self.camera_info is not None:
            return

        self.camera_info = {
            "fx":     msg.k[0],
            "fy":     msg.k[4],
            "cx":     msg.k[2],
            "cy":     msg.k[5],
            "width":  msg.width,
            "height": msg.height,
        }
        self.get_logger().info(
            f"[INFO] CameraInfo saved — "
            f"fx={self.camera_info['fx']:.2f}, fy={self.camera_info['fy']:.2f}, "
            f"cx={self.camera_info['cx']:.2f}, cy={self.camera_info['cy']:.2f}, "
            f"res={self.camera_info['width']}×{self.camera_info['height']}"
        )

    # ── 메인 파이프라인 ──────────────────────────

    def image_callback(self, msg: CompressedImage):
        self.last_rgb_received = time.monotonic()

        frame = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            self.get_logger().warn("[DEBUG-3] Frame decode failed. Check topic format.")
            return

        cars, ids = self._detect(frame)
        self.get_logger().info(
            f"[DEBUG-4] Detect result — cars: {len(cars)}, plates: {len(ids)}"
        )

        with self._depth_lock:
            depth_snap = None if self.latest_depth_frame is None else self.latest_depth_frame.copy()

        for det in cars + ids:
            self._log_depth_at_center(det, depth_snap)

        validated_pairs = []
        for id_det in ids:
            car = self._find_parent_car(id_det, cars)
            self.get_logger().info(
                f"[DEBUG-5] Plate conf={id_det['conf']:.2f} "
                f"bbox=({id_det['x1']},{id_det['y1']},{id_det['x2']},{id_det['y2']}) "
                f"→ parent_car={'FOUND' if car else 'NONE'}"
            )
            if car is None:
                if cars:
                    id_area = max(1, id_det["area"])
                    for i, c in enumerate(cars):
                        ix = max(0, min(id_det["x2"], c["x2"]) - max(id_det["x1"], c["x1"]))
                        iy = max(0, min(id_det["y2"], c["y2"]) - max(id_det["y1"], c["y1"]))
                        ov = (ix * iy) / id_area
                        self.get_logger().info(
                            f"[DEBUG-6]   car[{i}] overlap={ov:.3f} / threshold={ID_IN_CAR_OVERLAP_THRESH}"
                        )
                continue
            validated_pairs.append((car, id_det))

        self._update_tracking(frame, depth_snap, validated_pairs)
        self._draw(frame)

    # ── bbox 중심점 depth + 3D 좌표 로그 ────────────

    def _log_depth_at_center(self, det: dict, depth_frame: np.ndarray | None):
        cx  = (det["x1"] + det["x2"]) // 2
        cy  = (det["y1"] + det["y2"]) // 2
        lbl = det["class_name"]

        xyz = self._get_xyz_from_pixel(cx, cy, depth_frame)
        if xyz is None:
            if depth_frame is None:
                if not self._depth_no_data_warned:
                    self.get_logger().warn("[DEPTH] depth frame not yet received. NO_DATA logs suppressed.")
                    self._depth_no_data_warned = True
            else:
                self.get_logger().info(f"[DEPTH] {lbl} pixel=({cx},{cy}) INVALID")
            return

        X, Y, Z = xyz
        self.get_logger().info(
            f"[DEPTH] {lbl} pixel=({cx},{cy}) "
            f"X={X:+.3f}m Y={Y:+.3f}m Z={Z:.3f}m"
        )

    # ── 픽셀 -> 3D 좌표 계산 ───────────────────────

    def _get_xyz_from_pixel(self, u: int, v: int, depth_frame: np.ndarray | None):
        if depth_frame is None:
            return None
        if self.camera_info is None:
            return None

        h, w = depth_frame.shape[:2]
        if not (0 <= v < h and 0 <= u < w):
            return None

        depth_mm = int(depth_frame[v, u])
        if depth_mm <= 0:
            return None

        Z = depth_mm / 1000.0
        X = (u - self.camera_info["cx"]) * Z / self.camera_info["fx"]
        Y = (v - self.camera_info["cy"]) * Z / self.camera_info["fy"]
        return (X, Y, Z)

    def _get_target_xy_from_det(self, det: dict, depth_frame: np.ndarray | None):
        """
        det 중심점의 3D 좌표 계산 후
        AMR 이동용 평면 좌표로 변환해서 (x, y) 반환.
        여기서는 요청대로 string "x,y" 로 보낼 때
        x = 카메라 기준 좌우(X), y = 카메라 기준 전방(Z) 사용.
        """
        cx = (det["x1"] + det["x2"]) // 2
        cy = (det["y1"] + det["y2"]) // 2

        xyz = self._get_xyz_from_pixel(cx, cy, depth_frame)
        if xyz is None:
            return None

        X, _, Z = xyz
        return (X, Z)

    def _publish_target_xy(self, x: float, y: float):
        msg = String()
        msg.data = f"{x:.3f},{y:.3f}"
        self.amr_target_pub.publish(msg)
        self.get_logger().info(f"[AMR_TARGET] Published to {TOPIC_AMR_TARGET}: {msg.data}")

    # ── YOLO 탐지 ───────────────────────────────

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
                "area":       max(0, x2 - x1) * max(0, y2 - y1),
            }
            (cars if name == "car" else ids).append(det)

        return cars, ids

    # ── Overlap 검증 ────────────────────────────

    def _find_parent_car(self, id_det: dict, cars: list) -> dict | None:
        id_area = max(1, id_det["area"])

        def overlap(car):
            ix = max(0, min(id_det["x2"], car["x2"]) - max(id_det["x1"], car["x1"]))
            iy = max(0, min(id_det["y2"], car["y2"]) - max(id_det["y1"], car["y1"]))
            return (ix * iy) / id_area

        candidates = [(overlap(car), car) for car in cars if overlap(car) >= ID_IN_CAR_OVERLAP_THRESH]
        return max(candidates, key=lambda x: x[0])[1] if candidates else None

    # ── IoU (동일 차량 판별) ─────────────────────

    @staticmethod
    def _iou(a: dict, b: dict) -> float:
        ix = max(0, min(a["x2"], b["x2"]) - max(a["x1"], b["x1"]))
        iy = max(0, min(a["y2"], b["y2"]) - max(a["y1"], b["y1"]))
        inter = ix * iy
        if inter == 0:
            return 0.0

        area_a = max(1, a["area"])
        area_b = max(1, b["area"])
        return inter / (area_a + area_b - inter)

    # ── 트래킹 갱신 + 타이머 관리 ──────────────────

    def _update_tracking(self, frame: np.ndarray, depth_snap: np.ndarray | None, validated_pairs: list):
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
                self.get_logger().info("New vehicle tracking started.")
                self._enqueue_car(frame, new_track)
                new_track.car_uploaded = True

        next_tracks = []
        for track in visible_tracks:
            if track.plate_uploaded:
                continue

            elapsed = track.elapsed()
            self.get_logger().info(
                f"Tracking: elapsed={elapsed:.1f}s / {PARKING_TIMEOUT_SEC}s"
            )

            if elapsed >= PARKING_TIMEOUT_SEC:
                # 1) 목표 좌표 발행
                if not track.coord_published:
                    target_xy = self._get_target_xy_from_det(track.car_det, depth_snap)
                    if target_xy is not None:
                        tx, ty = target_xy
                        self._publish_target_xy(tx, ty)
                        track.coord_published = True
                    else:
                        self.get_logger().warn(
                            "[AMR_TARGET] Target publish skipped: depth/camera_info unavailable or invalid depth."
                        )

                # 2) 번호판 크롭 저장
                self._enqueue_plate(frame, track)
                track.plate_uploaded = True
            else:
                next_tracks.append(track)

        self.tracked_vehicles = next_tracks

    # ── Firebase 큐 투입 ─────────────────────────

    def _enqueue_car(self, frame: np.ndarray, track: TrackedVehicle):
        try:
            self.save_queue.put_nowait({
                "type":    "car",
                "image":   frame.copy(),
                "car_det": track.car_det,
                "id_det":  track.id_det,
            })
            self.get_logger().info(
                f"[CAR] Enqueued full frame. queue={self.save_queue.qsize()}/{SAVE_QUEUE_MAXSIZE}"
            )
        except queue.Full:
            self.get_logger().warn("Queue full. Car image dropped.")

    def _enqueue_plate(self, frame: np.ndarray, track: TrackedVehicle):
        id_det = track.id_det
        x1, y1, x2, y2 = id_det["x1"], id_det["y1"], id_det["x2"], id_det["y2"]
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            self.get_logger().warn(
                f"Empty crop. id_bbox=({x1},{y1},{x2},{y2}), frame={frame.shape}"
            )
            return

        try:
            self.save_queue.put_nowait({
                "type":    "plate",
                "image":   crop.copy(),
                "car_det": track.car_det,
                "id_det":  track.id_det,
            })
            self.get_logger().info(
                f"[PLATE] Enqueued crop. elapsed={track.elapsed():.1f}s "
                f"queue={self.save_queue.qsize()}/{SAVE_QUEUE_MAXSIZE}"
            )
        except queue.Full:
            self.get_logger().warn("Queue full. Plate image dropped.")

    # ── 시각화 ──────────────────────────────────

    def _draw(self, frame: np.ndarray):
        for track in self.tracked_vehicles:
            c       = track.car_det
            elapsed = track.elapsed()
            ratio   = min(elapsed / PARKING_TIMEOUT_SEC, 1.0)

            cv2.rectangle(frame, (c["x1"], c["y1"]), (c["x2"], c["y2"]), (0, 255, 0), 2)
            cv2.putText(frame, f"car {c['conf']:.2f}",
                        (c["x1"], max(25, c["y1"] - 30)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            d = track.id_det
            cv2.rectangle(frame, (d["x1"], d["y1"]), (d["x2"], d["y2"]), (0, 0, 255), 2)
            cv2.putText(frame, f"id {d['conf']:.2f}",
                        (d["x1"], max(25, d["y1"] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            bar_x1    = c["x1"]
            bar_y     = max(0, c["y1"] - 18)
            bar_w     = c["x2"] - c["x1"]
            bar_fill  = int(bar_w * ratio)
            bar_color = (0, 255, 0) if ratio < 0.5 else (0, 165, 255) if ratio < 0.85 else (0, 0, 255)

            cv2.rectangle(frame, (bar_x1, bar_y), (bar_x1 + bar_w, bar_y + 10), (80, 80, 80), -1)
            cv2.rectangle(frame, (bar_x1, bar_y), (bar_x1 + bar_fill, bar_y + 10), bar_color, -1)
            cv2.putText(frame, f"{elapsed:.0f}s/{PARKING_TIMEOUT_SEC:.0f}s",
                        (bar_x1, bar_y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, bar_color, 1)

        q_size  = self.save_queue.qsize()
        q_ratio = q_size / max(1, SAVE_QUEUE_MAXSIZE)
        q_color = (0, 255, 0) if q_ratio < 0.5 else (0, 165, 255) if q_ratio < 1.0 else (0, 0, 255)
        q_text  = f"Queue: {q_size}/{SAVE_QUEUE_MAXSIZE}"
        (tw, _), _ = cv2.getTextSize(q_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

        cv2.putText(frame, q_text, (frame.shape[1] - tw - 10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, q_color, 2)
        cv2.putText(frame, f"Tracking: {len(self.tracked_vehicles)}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("Parking Detection", frame)
        cv2.waitKey(1)

    # ── Firebase 업로드 워커 ─────────────────────

    def _upload_worker(self):
        while True:
            item = self.save_queue.get()
            if item is None:
                break
            try:
                self._upload(item)
            except Exception as e:
                self.get_logger().error(f"Upload error: {e}")

    def _upload(self, item: dict):
        if self.db_ref is None:
            self.get_logger().error(
                "[DEBUG-7] db_ref is None — Firebase 미연결. [DEBUG-2] 확인 필요."
            )
            return

        upload_type = item["type"]
        image       = item["image"]
        car_det     = item["car_det"]
        id_det      = item["id_det"]

        ok, enc = cv2.imencode(".jpg", image)
        if not ok:
            self.get_logger().error("JPEG encoding failed.")
            return

        b64 = base64.b64encode(enc.tobytes()).decode("utf-8")

        now = datetime.datetime.now()
        self.db_ref.child(now.strftime("%Y%m%d_%H%M%S_%f")).set({
            "type":           upload_type,
            "detected_at":    now.isoformat(),
            "car_confidence": round(car_det["conf"], 4),
            "id_confidence":  round(id_det["conf"], 4),
            "car_bbox": {"x1": car_det["x1"], "y1": car_det["y1"],
                         "x2": car_det["x2"], "y2": car_det["y2"]},
            "id_bbox":  {"x1": id_det["x1"], "y1": id_det["y1"],
                         "x2": id_det["x2"], "y2": id_det["y2"]},
            "image_base64": b64,
        })
        self.get_logger().info(
            f"Uploaded [{upload_type}]: {now.strftime('%Y%m%d_%H%M%S_%f')}"
        )

    # ── 종료 ────────────────────────────────────

    def destroy_node(self):
        self.save_queue.put(None)
        cv2.destroyAllWindows()
        super().destroy_node()

# ──────────────────────────────────────────────
# [CHAPTER 4: 엔트리 포인트]
# ──────────────────────────────────────────────
def main(args=None):
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