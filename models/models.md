# YOLO 모델 비교 분석 보고서
## 불법 주정차 단속 시스템 — AMR용 & Webcam용

> **작성일**: 2026.03  
> **시스템**: AMR 기반 불법 주정차 단속 멀티로봇 시스템  
> **분석 대상**: AMR용 2개 모델 + Webcam용 4개 모델 (총 6개)

---

## 목차

1. [시스템 개요 및 모델 구성](#1-시스템-개요-및-모델-구성)
2. [분석 기준 — 왜 Recall이 최우선인가](#2-분석-기준--왜-recall이-최우선인가)
3. [AMR용 모델 비교 분석](#3-amr용-모델-비교-분석)
4. [Webcam용 모델 비교 분석](#4-webcam용-모델-비교-분석)
5. [전체 종합 정리 및 최종 권장 설정](#5-전체-종합-정리-및-최종-권장-설정)

---

## 1. 시스템 개요 및 모델 구성

본 시스템은 고정식 CCTV(Webcam)와 2대의 자율 이동 로봇(AMR)이 연계하여 불법 주정차 차량을 탐지·단속하는 멀티로봇 솔루션이다.

| 구분 | 역할 | YOLO 입력 | 탐지 클래스 |
|------|------|-----------|------------|
| **Webcam (CCTV)** | 5분 초과 불법 주정차 1차 감지 → AMR 출동 좌표 발행 | 도로 고정 화각 | `car` |
| **AMR1 (순찰봇)** | 지정 구역 순찰, 차량 탐지 및 번호판 촬영 | 로봇 탑재 카메라 | `car`, `id` (번호판) |
| **AMR2 (단속봇)** | 좌표 수신 후 현장 출동, 번호판 정밀 채증 | 로봇 탑재 카메라 | `car`, `id` (번호판) |

### 학습 모델 구성

**AMR용 (2개 모델)**

| 모델 | 아키텍처 | Batch | 증강 | 클래스 |
|------|---------|-------|------|--------|
| AMR-v1 | YOLOv8n | 16 | 없음 | car, id |
| AMR-v2 | YOLOv8n | 32 | 없음 | car, id |

**Webcam용 (4개 모델)**

| 모델 | 아키텍처 | Batch | 증강 | 클래스 |
|------|---------|-------|------|--------|
| WC-v1 | YOLOv8n | 16 | **있음** | car |
| WC-v2 | YOLOv8n | 16 | 없음 | car |
| WC-v3 | YOLOv11n | 16 | 없음 | car |
| WC-v4 | YOLO26n | 16 | 없음 | car |

> **공통 학습 설정**: epochs=100, imgsz=640, optimizer=Adam, lr0=0.01, lrf=0.01, patience=20

---

## 2. 분석 기준 — 왜 Recall이 최우선인가

본 시스템의 핵심 로직은 아래와 같다.

```
차량 탐지 → 최초 탐지 timestamp 기록 → 300초(5분) 카운트 시작
→ 300초 초과 시 불법 주정차 판정 → AMR 출동
```

이 구조에서 **탐지를 한 번이라도 놓치면 타이머가 리셋**된다.  
즉, **False Negative(있는데 없다고 판단)가 치명적**이다.

반면 False Positive(없는데 있다고 판단)는 AMR2가 현장 도착 후 **차량 존재 여부를 재확인**하는 구조로 자연스럽게 걸러진다.

| 지표 | 우선순위 | 이유 |
|------|---------|------|
| **Recall** | ★★★ 최우선 | 놓치면 타이머 리셋 → 단속 불가 |
| **Precision** | ★★☆ 중요 | 낮으면 AMR 헛출동 증가 (오단속 3% 미만 요구사항) |
| **mAP@50-95** | ★★☆ 중요 | bbox 정밀도 → 번호판 촬영 품질에 직결 |
| **mAP@50** | ★☆☆ 참고 | 종합 지표로 참고용 |

> **수락 기준 (요구사항)**: 탐지율 95% 이상, 오단속 3% 미만

---

## 3. AMR용 모델 비교 분석

### 3-1. 핵심 지표 요약

| 지표 | AMR-v1 (batch=16) | AMR-v2 (batch=32) | 승자 |
|------|:-----------------:|:-----------------:|:----:|
| **Recall** (최종) | **0.9898** | 0.9801 | 🏆 v1 |
| Precision (최종) | 0.9528 | 0.9529 | — 동일 |
| mAP@50 (최종) | 0.9834 | **0.9840** | v2 (미미) |
| **mAP@50-95** (최종) | **0.7946** | 0.7842 | 🏆 v1 |
| val/box_loss (최종) | **0.787** | 0.834 | 🏆 v1 |
| val/cls_loss (최종) | **0.398** | 0.412 | 🏆 v1 |
| Recall 0.95 최초 달성 | **epoch 24** | epoch 31 | 🏆 v1 |
| 초반 30ep Recall 표준편차 | 0.319 | **0.250** | v2 |
| 총 학습 시간 | 196.6s | **189.0s** | v2 |

### 3-2. F1-Confidence Curve

| AMR-v1 (batch=16) | AMR-v2 (batch=32) |
|:-----------------:|:-----------------:|
| ![AMR-v1 F1 Curve](v1_BoxF1_curve.png) | ![AMR-v2 F1 Curve](v2_BoxF1_curve.png) |
| **최적 conf: 0.438** at F1=0.97 | **최적 conf: 0.614** at F1=0.97 |

### 3-3. Precision-Recall Curve

| AMR-v1 | AMR-v2 |
|:------:|:------:|
| ![AMR-v1 PR Curve](v1_BoxPR_curve.png) | ![AMR-v2 PR Curve](v2_BoxPR_curve.png) |
| car=0.994 / **id=0.972** / all=0.983 | car=0.994 / id=0.960 / all=0.977 |

### 3-4. Confusion Matrix

| AMR-v1 | AMR-v2 |
|:------:|:------:|
| ![AMR-v1 Confusion Matrix](v1_confusion_matrix.png) | ![AMR-v2 Confusion Matrix](v2_confusion_matrix.png) |
| car FN=**4**, id FN=6, background FP=1+1 | car FN=**5**, id FN=6, background FP=2+1 |

### 3-5. 항목별 상세 해석

#### ① Recall — v1이 확실히 우세

AMR-v1이 0.9898로 AMR-v2(0.9801)보다 약 1% 높다.  
Confusion Matrix 기준 AMR-v1은 car FN=4개, AMR-v2는 FN=5개로, **100번 중 1번 더 차량을 놓친다**.  
단속 타이머 리셋 관점에서 의미 있는 차이다.

#### ② PR Curve — `id` 클래스(번호판)가 약점

두 모델 모두 `car` 클래스 AP=0.994로 거의 완벽하지만,  
`id`(번호판) 클래스에서 **AMR-v1=0.972, AMR-v2=0.960**으로 v1이 앞선다.  
번호판 인식은 단속 법적 증거에 직결되므로 이 차이가 실제 운영에서 중요하다.

#### ③ F1 Curve — 최적 conf threshold 차이

- AMR-v1: **conf=0.438** at F1=0.97
- AMR-v2: **conf=0.614** at F1=0.97

AMR-v2는 최적 threshold가 0.614로 훨씬 높다.  
conf를 낮게 설정하면 성능이 급격히 저하될 수 있어, AMR-v1이 **더 넓은 conf 범위에서 안정적**으로 동작한다.

> ⚠️ **현재 CCTV 노드 conf=0.87은 지나치게 높다.**  
> F1 curve 기준 AMR-v1의 최적값(0.438)과 비교하면 0.87은 Recall을 스스로 깎는 설정이다.  
> **conf를 0.4~0.5 범위로 낮추는 것을 강력히 권장**한다.

#### ④ 학습 안정성 — 초반은 v2가 더 안정적, 수렴은 v1이 빠름

초반 30 epoch Recall 표준편차: AMR-v1=0.319, AMR-v2=0.250  
→ 배치가 클수록 gradient 추정이 안정되는 일반적 현상.

수렴 속도(Recall≥0.95 최초 달성): AMR-v1=epoch 24, AMR-v2=epoch 31  
→ AMR-v1이 7 epoch 빠르게 수렴.

### 3-6. ✅ AMR 최종 추천: **AMR-v1 (batch=16)**

Recall 최우선 기준에서 AMR-v1이 전반적으로 우세하다.  
mAP@50은 AMR-v2가 0.0006 높지만 사실상 무의미한 차이이며,  
Recall·mAP@50-95·val loss 모두 AMR-v1이 앞선다.

| 권장 설정 | 값 |
|----------|----|
| 사용 모델 | AMR-v1 (YOLOv8n, batch=16) |
| conf threshold | **0.43~0.45** (F1 최적점 기준) |
| 비고 | 현재 0.87 설정은 Recall 저하 유발, 즉시 변경 필요 |

---

## 4. Webcam용 모델 비교 분석

### 4-1. 핵심 지표 요약

| 지표 | WC-v1 YOLOv8n+증강 | WC-v2 YOLOv8n | WC-v3 YOLOv11n | WC-v4 YOLO26n |
|------|:-----------------:|:-------------:|:--------------:|:-------------:|
| **Recall** (최종) | 0.9811 | **1.0000** | 0.6339 | 0.9901 |
| Precision (최종) | **0.9910** | 0.9981 | 0.7735 | 0.9823 |
| mAP@50 (최종) | 0.9947 | 0.9950 | 0.7849 | **0.9945** |
| **mAP@50-95** (최종) | 0.5796 | 0.6452 | 0.3929 | **0.6858** |
| val/box_loss (최종) | 1.4316 | 1.2795 | 1.7332 | **1.1501** |
| val/cls_loss (최종) | 0.5474 | 0.4414 | 1.6776 | **0.3522** |
| Recall 0.95 최초 달성 | epoch 16 | **epoch 2** | epoch 1 ⚠️ | **epoch 1** |
| 초반 30ep Recall 표준편차 | 0.349 | 0.320 | 0.302 | **0.305** |
| 총 학습 시간 | 169s | **91s** | **53s** ⚠️ | 153s |

> ⚠️ WC-v3는 epoch 4부터 val_cls_loss=`inf`/`nan` 발산, 실사용 불가

### 4-2. F1-Confidence Curve

| WC-v1 (YOLOv8n+증강) | WC-v2 (YOLOv8n) |
|:-------------------:|:---------------:|
| ![WC-v1 F1 Curve](v1_BoxF1_curve.png) | ![WC-v2 F1 Curve](v2_BoxF1_curve.png) |
| all classes **1.00 at 0.644** | all classes **0.99 at 0.769** |

| WC-v3 (YOLOv11n) ⚠️ | WC-v4 (YOLO26n) |
|:-------------------:|:---------------:|
| ![WC-v3 F1 Curve](v3_BoxF1_curve.png) | ![WC-v4 F1 Curve](v4_BoxF1_curve.png) |
| all classes **0.04 at 0.000** — 학습 실패 | all classes **0.99 at 0.608** |

### 4-3. Precision-Recall Curve

| WC-v1 | WC-v2 |
|:-----:|:-----:|
| ![WC-v1 PR Curve](v1_BoxPR_curve.png) | ![WC-v2 PR Curve](v2_BoxPR_curve.png) |
| car=**0.995** / all=0.995 | car=**0.995** / all=0.995 |

| WC-v3 ⚠️ | WC-v4 |
|:--------:|:-----:|
| ![WC-v3 PR Curve](v3_BoxPR_curve.png) | ![WC-v4 PR Curve](v4_BoxPR_curve.png) |
| car=0.868 — 우상단 미도달 | car=**0.994** / all=0.994 |

### 4-4. Confusion Matrix

| WC-v1 | WC-v2 |
|:-----:|:-----:|
| ![WC-v1 Confusion Matrix](v1_confusion_matrix.png) | ![WC-v2 Confusion Matrix](v2_confusion_matrix.png) |
| car=112✅, background FP=**5** | car=112✅, background FP=**12** |

| WC-v3 ⚠️ | WC-v4 |
|:--------:|:-----:|
| ![WC-v3 Confusion Matrix](v3_confusion_matrix.png) | ![WC-v4 Confusion Matrix](v4_confusion_matrix.png) |
| car 112개 **전부 background로 오분류** | car=112✅, background FP=**10** |

### 4-5. 항목별 상세 해석

#### WC-v1 (YOLOv8n + 이미지 증강)

증강을 적용했음에도 WC-v2보다 Recall이 낮다(0.9811 vs 1.0000).  
이 데이터셋에서는 **증강이 오히려 성능을 저하**시켰다.  
학습 시간도 169s로 4개 중 가장 길며, Webcam용으로는 비효율적이다.

- Confusion Matrix: FP=5개로 가장 낮아 **오탐은 가장 적음**
- 증강 효과가 없다면 WC-v2 대비 채택 이유 없음

#### WC-v2 (YOLOv8n, 증강 없음)

Recall 1.0000, mAP@50 0.9950으로 **수치상 최고**이나 주의가 필요하다.

- Confusion Matrix: background FP=**12개**로 4개 모델 중 가장 많음
- Recall 1.0은 검증셋이 작을 경우 **과적합 신호**일 수 있음
- epoch 2에서 Recall 0.95 달성 → 매우 빠른 수렴이나 불안정한 초반 학습의 반증

#### WC-v3 (YOLOv11n) — ⛔ 사용 불가

```
epoch 4: val_cls_loss = inf
epoch 5: val_cls_loss = inf  
epoch 6: val_cls_loss = nan  ← 완전 발산
```

Confusion Matrix에서 차량 112개를 **전부 background로 예측** (완전 실패).  
F1 curve 최대값 0.04로 사실상 랜덤 수준이다.

> **원인 분석**: YOLOv11n은 아키텍처 개선으로 더 낮은 learning rate가 필요하다.  
> `lr0=0.01`이 YOLOv11n에게 과도하게 높아 gradient exploding이 발생한 것으로 판단된다.  
> **재학습 시 `lr0=0.001`로 10배 낮춰서 시도 권장.**

#### WC-v4 (YOLO26n) — 가장 균형 잡힌 모델

- Recall 0.9901, Precision 0.9823으로 **균형 최우수**
- **mAP@50-95=0.6858** — 4개 중 최고, bbox를 가장 정밀하게 잡음
- val/box_loss=1.1501, val/cls_loss=0.3522 — **두 loss 모두 최저**
- Confusion Matrix FP=10개로 WC-v2(12개)보다 깔끔

mAP@50-95가 높다는 것은 IoU threshold를 엄격하게 적용해도 검출 품질이 유지된다는 의미로,  
**CCTV 좌표 추출의 정확도**에 직결된다.

### 4-6. ✅ Webcam 최종 추천: **WC-v4 (YOLO26n)**

| 비교 항목 | WC-v2 (YOLOv8n) | **WC-v4 (YOLO26n)** |
|----------|:---------------:|:-------------------:|
| Recall | 1.0000 (과적합 의심) | 0.9901 (안정적) |
| Precision | 0.9981 | 0.9823 |
| mAP@50-95 | 0.6452 | **0.6858** |
| val/box_loss | 1.2795 | **1.1501** |
| background FP | **12개** | 10개 |
| 신뢰도 | 과적합 가능성 있음 | 안정적 수렴 |

| 권장 설정 | 값 |
|----------|----|
| 사용 모델 | WC-v4 (YOLO26n, batch=16) |
| conf threshold | **0.60~0.65** (F1 최적점 0.608 기준) |
| 비고 | WC-v3 재학습 시 lr0=0.001로 시도 권장 |

---

## 5. 전체 종합 정리 및 최종 권장 설정

### 5-1. 최종 선택 모델 요약

| 용도 | 선택 모델 | 핵심 이유 |
|------|---------|---------|
| **AMR (순찰봇/단속봇)** | **AMR-v1** (YOLOv8n, batch=16) | Recall 0.9898 최고, id 클래스 AP 0.972, conf 범위 안정 |
| **Webcam (CCTV)** | **WC-v4** (YOLO26n, batch=16) | mAP@50-95 최고(0.6858), val loss 최저, 균형 우수 |

### 5-2. 시스템 배포 권장 conf threshold

| 노드 | 현재 설정 | 권장 설정 | 변경 이유 |
|------|----------|----------|---------|
| CCTV (Webcam) | `0.87` | **`0.60~0.65`** | F1 최적 0.608, 현재 설정은 Recall 저하 |
| AMR1/AMR2 | 미정 | **`0.43~0.45`** | F1 최적 0.438, Recall 최대화 |

> ⚠️ **현재 CCTV conf=0.87은 즉시 변경이 필요하다.**  
> F1 curve 기준 최적값(0.608)보다 훨씬 높아, Recall을 스스로 0.87 이하로 제한하는 셋팅이다.  
> 이는 시스템 요구사항인 탐지율 95% 이상을 충족하지 못할 위험이 있다.

### 5-3. 향후 개선 권장 사항

**단기**
- CCTV conf threshold 0.87 → 0.60~0.65로 즉시 변경
- AMR conf threshold 0.43~0.45로 설정

**중기**
- WC-v3 (YOLOv11n) `lr0=0.001`로 재학습 — 아키텍처 자체는 우수하므로 시도 가치 있음
- AMR용 YOLOv8n 데이터 증강 효과 재검증 (webcam에서는 역효과였음)

**장기**
- 실제 도로 환경 데이터 추가 수집 (야간, 우천, 버스·트럭 가림 케이스)
- AMR 탑재 카메라의 실시간 FPS 측정 → 10FPS 요구사항 충족 여부 검증

---

*본 분석은 제공된 학습 결과 파일(results.csv, confusion matrix, PR/F1 curve)을 기반으로 작성되었습니다.*