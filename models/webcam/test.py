import cv2
from ultralytics import YOLO

model = YOLO("best.pt")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(
        frame,
        tracker="bytetrack.yaml",
        persist=True,      # 프레임 간 ID 유지 (필수!)
        conf=0.87,
        verbose=False      # 터미널 출력 억제
    )

    annotated = results[0].plot()  # bbox + ID 시각화

    # 트래킹 정보 출력 (디버깅용)
    boxes = results[0].boxes
    if boxes.id is not None:
        for tid, cls, conf in zip(
            boxes.id.int().tolist(),
            boxes.cls.int().tolist(),
            boxes.conf.tolist()
        ):
            label = model.names[cls]
            print(f"ID: {tid} | {label} | conf: {conf:.2f}")

    cv2.imshow("RC Car Tracker", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()