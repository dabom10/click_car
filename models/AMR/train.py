from ultralytics import YOLO

# 모델 로드 (nano/small/medium/large/xlarge)
model = YOLO("yolo26n.pt")  # n → s → m → l → x 순으로 무거워짐

# 학습
results = model.train(
    data="/home/rokey/click_car/models/webcam/dataset/argu_no/data.yaml",
    
    # ── 기본 설정 ──────────────────────────
    epochs=100,
    imgsz=640,
    batch=16,            # GPU 메모리 부족 시 8로 낮추기
    workers=4,
    device=0,            # GPU 없으면 device="cpu"
    project="/home/rokey/click_car/models/webcam",
    name="v3",
    
    # ── Optimizer ──────────────────────────
    optimizer="Adam",     # SGD / Adam / AdamW
    lr0=0.001,           # 초기 learning rate (0.01 → 0.001, gradient exploding 방지)
    lrf=0.1,             # 최종 lr = lr0 * lrf → 0.0001 (0.01이면 0.00001로 너무 작음)
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    
    # ── Augmentation (이미 했으니 끄거나 약하게) ──
    augment=False,       # 추가 augmentation 비활성화
    mosaic=0.0,          # 0.0 ~ 1.0
    mixup=0.0,
    
    # ── 학습 전략 ──────────────────────────
    patience=20,         # Early stopping (20 epoch 동안 개선 없으면 종료)
    save_period=10,      # 10 epoch마다 체크포인트 저장
    val=True,
)