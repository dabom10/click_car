from ultralytics import YOLO

DATA_PATH = "/home/rokey/click_car/models/AMR/dataset/argu_no/data.yaml"
PROJECT   = "/home/rokey/click_car/models/AMR"

# ============================================================
# 즉시 적용 (재학습 없음)
# → conf threshold만 0.43~0.45로 낮추면 됨
# → 아래 predict/detect 호출할 때 conf=0.43 으로 넘기면 끝
#
# model = YOLO("best.pt")
# results = model.predict(source=0, conf=0.43)
# ============================================================


# ============================================================
# 실험 A+C
# 변경 사항:
#   - mosaic=1.0, copy_paste=0.3  (작은 객체 대응)
#   - lr0: 0.01 → 0.005           (초반 진동 감소)
#   - lrf: 0.01 → 0.1             (끝까지 학습 유지)
#   - warmup_epochs: 3 → 5
#   - epochs: 100 → 150
#   - patience: 20 → 30
#   - batch: 32 → 16              (v1 기준이 더 좋았으므로)
# ============================================================

def train_v5():
    model = YOLO("yolov8n.pt")
    model.train(
        data=DATA_PATH,

        # ── 기본 설정 ──────────────────────────
        epochs=150,
        imgsz=704,
        batch=16,
        workers=4,
        device=0,
        project="/home/rokey/click_car/models/AMR",
        name="v5_box",

        # ── Optimizer ──────────────────────────
        optimizer="Adam",
        lr0=0.005,
        lrf=0.1,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,

        # ── Augmentation ───────────────────────
        augment=False,
        mosaic=1.0,
        copy_paste=0.1,   # ✏️ 0.3 → 0.1 (bbox 안정화)
        mixup=0.0,

        # ── Loss 가중치 ─────────────────────────
        box=10.0,         # ✏️ 기본 7.5 → 10.0 (bbox 정밀도 향상 목표)

        # ── 학습 전략 ──────────────────────────
        patience=30,
        save_period=10,
        val=True,
    )

# ============================================================
# 실험 A+B+C
# 변경 사항 (A+C 대비 추가):
#   - cls=2.5  (id 클래스 가중치 높이기 → val/cls_loss 개선 목표)
#
# ※ data.yaml에서 cls_weights 직접 지원 안 되는 경우를 대비해
#    YOLO train의 cls 파라미터로 전체 cls loss scale을 높이는 방식 사용
#    → car보다 id(번호판) loss 기여도를 높이려면 데이터 레벨에서
#      id 샘플을 오버샘플링하거나, 아래처럼 cls 가중치를 올려서 대응
# ============================================================
def train_ABC():
    model = YOLO("yolov8n.pt")
    model.train(
        data=DATA_PATH,

        # ── 기본 설정 ──────────────────────────
        epochs=150,
        imgsz=704,
        batch=16,
        workers=4,
        device=0,
        project="/home/rokey/click_car/models/AMR",
        name="v4_exp_ABC",

        # ── Optimizer ──────────────────────────
        optimizer="Adam",
        lr0=0.005,
        lrf=0.1,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,

        # ── Augmentation ───────────────────────
        augment=False,
        mosaic=1.0,
        copy_paste=0.3,
        mixup=0.0,

        # ── Loss 가중치 ─────────────────────────
        cls=2.5,          # ✏️ 추가 (기본값 0.5 → 2.5, 번호판 오인식 줄이기)

        # ── 학습 전략 ──────────────────────────
        patience=30,
        save_period=10,
        val=True,
    )


# ============================================================
# 실행
# 원하는 실험 함수만 주석 해제해서 실행
# ============================================================
if __name__ == "__main__":

    print("=" * 50)
    print("실험 A+C 시작")
    print("=" * 50)
    train_v5()

    # print("=" * 50)
    # print("실험 A+B+C 시작")
    # print("=" * 50)
    # train_ABC()