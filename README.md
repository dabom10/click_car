# CLICK CAR - 불법 주차 자동 단속 시스템

> TurtleBot4(ROS2) 기반 다중 AMR을 활용한 자율 순찰 및 불법 주차 단속 시스템

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [시스템 아키텍처](#2-시스템-아키텍처)
3. [기술 스택](#3-기술-스택)
4. [디렉토리 구조](#4-디렉토리-구조)
5. [핵심 노드 상세](#5-핵심-노드-상세)
   - [AMR 순찰·단속 제어](#51-amr-순찰단속-제어-amr1py--amr2py)
   - [차량 감지 & 좌표 추적](#52-차량-감지--좌표-추적-depth_coor_amr1py--depth_coor_amr2py)
   - [번호판 OCR 인식](#53-번호판-ocr-인식-ocr_nodepy)
   - [웹 대시보드 & Firebase 브릿지](#54-웹-대시보드--firebase-브릿지)
6. [Firebase DB 구조](#6-firebase-db-구조)
7. [전체 통신 흐름](#7-전체-통신-흐름)
8. [실행 방법](#8-실행-방법)
9. [주요 파라미터](#9-주요-파라미터)
10. [트러블슈팅 & 설계 결정](#10-트러블슈팅--설계-결정)

---

## 1. 프로젝트 개요

불법 주차 단속을 자동화하기 위해 TurtleBot4 AMR 2대가 실내 지도를 자율 순찰합니다.
차량이 불법 구역에 감지되면 AMR이 해당 위치로 이동하여 증거를 촬영하고, OCR로 번호판을 인식한 뒤 단속 기록을 Firebase에 자동 저장합니다.
관리자는 웹 대시보드에서 로봇 상태 모니터링 및 원격 제어가 가능합니다.

**단속 경로는 두 가지:**
- **AMR 자체 감지**: 온보드 OAK-D 카메라로 직접 불법 차량 탐지
- **CCTV 연동**: CCTV가 좌표를 전달하면 AMR이 해당 위치로 이동

---

## 2. 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                        웹 대시보드 (ui_v6.html)              │
│              배터리·상태 모니터링 / 순찰 시작·정지            │
└───────────────────────┬─────────────────────────────────────┘
                        │ Firebase Realtime DB
         ┌──────────────┴──────────────┐
         │   robot_two.py              │   robot_three.py
         │   (robot2 브릿지)           │   (robot3 브릿지)
         └──────────┬──────────────────┘
                    │ ROS2 Topic
         ┌──────────┴────────────────────────────────┐
         │              AMR 순찰·단속 제어             │
         │        amr1.py (robot3)                   │
         │        amr2.py (robot2)                   │
         └───┬─────────────────────────┬─────────────┘
             │                         │
    ┌────────┴────────┐       ┌────────┴────────────┐
    │  depth_coor     │       │    ocr_node.py       │
    │  (차량 감지 +   │       │  (번호판 촬영 + OCR) │
    │   맵 좌표 변환) │       │   → Firebase 업로드  │
    └────────┬────────┘       └─────────────────────┘
             │ OAK-D Stereo Camera
    ┌────────┴────────┐
    │  YOLOv8 + EKF3D │
    │  불법 구역 판별  │
    └─────────────────┘
```

---

## 3. 기술 스택

| 분류 | 기술 | 용도 |
|------|------|------|
| 로봇 플랫폼 | TurtleBot4 | 실제 구동 하드웨어 |
| 미들웨어 | ROS2 (Humble) | 노드 간 통신 |
| 자율주행 | Nav2, SLAM, AMCL | 지도 생성·위치 추정·경로 계획 |
| 객체 인식 | YOLOv8 (커스텀 학습) | 차량·번호판 감지 |
| 3D 추적 | EKF3D (Extended Kalman Filter) | 차량 위치 안정적 추적 |
| 번호판 OCR | Google Cloud Vision API | 1차 OCR (고정밀) |
| 번호판 OCR | PaddleOCR (한국어) | 2차 폴백 OCR |
| 카메라 | OAK-D Stereo (RGB-D) | 깊이 포함 차량 감지 |
| 데이터베이스 | Firebase Realtime Database | 단속 기록·로봇 상태 저장 |
| 웹 UI | HTML / CSS / JavaScript | 관리자 대시보드 |

---

## 4. 디렉토리 구조

```
click_car/
├── src/
│   ├── crackdown/                  # ROS2 패키지 - AMR 순찰·단속
│   │   ├── package.xml
│   │   ├── setup.py                # entry_points: amr1, amr2, cctv, dataget
│   │   └── crackdown/
│   │       ├── amr1.py             # robot3 순찰·단속 메인 노드
│   │       ├── amr2.py             # robot2 순찰·단속 메인 노드
│   │       └── cctv.py             # CCTV 연동 모듈
│   ├── amr_detect/                 # 차량 감지 & 좌표 변환
│   │   ├── depth_coor_amr1.py      # robot3용 감지 노드
│   │   └── depth_coor_amr2.py      # robot2용 감지 노드
│   └── webcam/
│       └── ocr_node.py             # 번호판 OCR 및 Firebase 업로드 노드
├── web/
│   ├── ui_v6.html                  # 관리자 웹 대시보드
│   ├── robot_two.py                # robot2 Firebase ↔ ROS2 브릿지
│   ├── robot_three.py              # robot3 Firebase ↔ ROS2 브릿지
│   └── click_car.json              # Firebase 서비스 계정 키
└── models/
    ├── amr.pt                      # AMR용 YOLOv8 모델 (차량+번호판)
    └── webcam.pt                   # 웹캠용 YOLOv8 모델
```

---

## 5. 핵심 노드 상세

### 5.1 AMR 순찰·단속 제어 (`amr1.py` / `amr2.py`)

robot3는 amr1, robot2는 amr2가 담당하며 구조는 동일합니다.

#### 웨이포인트 & 구역 구성

맵 전체를 **9개 웨이포인트 / 5개 구역(Zone)** 으로 분할합니다.
위반 좌표가 수신되면 해당 좌표가 속한 Zone을 판별하고, 그 Zone의 첫 번째 웨이포인트로 이동합니다.

```
웨이포인트 순서 (시계 반대 방향)
WP1(-0.725, -0.2) → WP2(-0.725, 1.9) → WP3(-2.2, 2.2) → WP4(-2.15, -0.3)
→ WP5(-2.3, -2.0) → WP6(-2.4, -4.0) → WP7(2.1, -4.0)
→ WP8(1.97, -2.5) → WP9(-1.5, -2.2)

Zone 구성
Zone 1: WP1, WP2  |  Zone 2: WP3, WP4  |  Zone 3: WP5
Zone 4: WP6, WP7  |  Zone 5: WP8, WP9
```

#### 상태 머신

```
[대기 - 충전 중]
     │ patrol_command: "start" 수신
     ▼
[MODE_PATROL] ── 웨이포인트 순서대로 순찰
     │ cctv_done / amr_done 수신 (위반 좌표)
     ▼
[MODE_ROUTE_TO_ZONE] ── 해당 Zone 첫 웨이포인트로 이동
     │ 웨이포인트 도착
     ▼
[MODE_ENFORCEMENT] ── 차량 방향 계산 → 0.75m 앞 정지 → 촬영 신호 발행
     │ capture_done 수신
     ▼
[복귀] ── 남은 웨이포인트 → Pre-dock 위치 → 도킹
     │
     └─── 다시 대기 (무한 반복)
```

#### 단속 이동 알고리즘

```
1. AMCL로 현재 로봇 위치 (rx, ry) 읽기
2. 목표 차량 방향: yaw = atan2(ty - ry, tx - rx)
3. 정지 좌표: stop = target - cos/sin(yaw) × 0.75m
4. Nav2로 stop 좌표 이동
5. 도착 후 capture_done 플래그 초기화
6. amr_start 또는 cctv_start 신호 발행
7. capture_done = True 수신까지 대기
```

#### ROS2 토픽

| 방향 | 토픽 | 타입 | 설명 |
|------|------|------|------|
| 구독 | `/{ns}/patrol_command` | String | "start" / "stop" |
| 구독 | `/{ns}/cctv_done` | String | CCTV 위반 좌표 "x,y" |
| 구독 | `/{ns}/amr_done` | String | AMR 카메라 위반 좌표 "x,y" |
| 구독 | `/{ns}/capture_done` | Bool | 촬영 완료 신호 |
| 구독 | `/{ns}/battery_state` | BatteryState | 배터리 잔량 |
| 구독 | `/{ns}/amcl_pose` | PoseWithCovarianceStamped | 현재 위치 |
| 발행 | `/{ns}/start` | String | "cctv_start" / "amr_start" |
| 발행 | `/{ns}_status` | String | patrol / enforce / returning / charging |

---

### 5.2 차량 감지 & 좌표 추적 (`depth_coor_amr1.py` / `depth_coor_amr2.py`)

OAK-D Stereo 카메라로 차량을 감지하고 맵 좌표로 변환, 불법 구역 내 차량만 발행합니다.

#### 감지 파이프라인

```
OAK-D RGB 프레임 (동기화된 Depth 포함)
    │
    ▼
YOLOv8 차량 감지
- 클래스: "car"
- 신뢰도 임계값: 0.70
- 이미지 크기: 704×704
- ROI: 상단 30% 제거 (하늘 영역 필터링)
    │
    ▼
깊이값 추출
- 바운딩박스 내 하위 10 퍼센타일 깊이 사용 (노이즈 억제)
    │
    ▼
EKF3D 추적 (3축 칼만 필터: px, py, pz)
- 정지 차량 모델 가정 (F = Identity)
- 깊이 Z에 비례한 동적 관측 노이즈 R
- Chi-squared gating (임계값 7.815)으로 이상값 제거
- Smoothing 윈도우: 5프레임
- 트랙 TTL: 1.0초 (미감지 유예)
    │
    ▼
좌표 변환
카메라 프레임
  → 로봇 베이스 (CAM_OFFSET: X=-0.10m, Z=+0.25m)
  → 오도메트리 프레임 (로봇 회전/이동 반영)
  → 맵 프레임 (TF2 map→odom 변환)
    │
    ▼
불법 구역 판별 (point_in_zone)
Zone 1: x ∈ [-2.04, -0.694], y ∈ [0.0, 1.850]
Zone 2: x ∈ [-1.240, 0.504], y ∈ [-5.000, -4.300]
    │ 구역 내 차량만
    ▼
/{ns}/amr_done 발행 (최대 10회, 0.2초 간격)
```

#### 회전 중 감지 억제

로봇이 회전 중일 때(`angular_z > 0.10 rad/s`)는 좌표 변환 오차가 크므로 감지 발행을 건너뜁니다.

---

### 5.3 번호판 OCR 인식 (`ocr_node.py`)

AMR 또는 CCTV로부터 단속 신호를 받아 번호판을 촬영·인식하고 Firebase에 기록합니다.

#### 동작 모드

| 모드 | 트리거 | 촬영 시간 | 경고음 |
|------|--------|-----------|--------|
| `amr_start` | AMR 온보드 카메라 단독 단속 | 30초 | Elise 멜로디 (2637~750 Hz) |
| `cctv_start` | CCTV 연동 단속 | 5초 | 없음 |

#### 감지·추적 로직

```
YOLOv8 (차량 + 번호판 동시 감지)
    │
    ├── 가장 큰 차량 바운딩박스 선택
    ├── 번호판이 차량 bbox와 50% 이상 겹치는지 검증
    └── IoU 기반 트래킹 (임계값 0.30, grace period 2초)
```

#### OCR 처리

```
번호판 이미지 크롭
    │
    ├─ [1차] Google Cloud Vision API
    │       정상 방향 + 180° 회전 양방향 시도
    │       신뢰도 ≥ 0.6 결과 채택
    │
    └─ [2차, 폴백] PaddleOCR (한국어)
            정상 방향 + 180° 회전 양방향 시도
```

#### Firebase 업로드

- **amr_start**: 감지 즉시 로컬 임시 저장 → `capture_done` 수신 후 `detections/{timestamp}` 경로에 업로드
- **cctv_start**: 기존 CCTV 케이스(`cctv_detections/{latest_key}`)에 AMR 증거 이미지 추가

---

### 5.4 웹 대시보드 & Firebase 브릿지

#### 웹 대시보드 (`web/ui_v6.html`)

관리자용 실시간 모니터링 및 제어 화면

- **로봇 상태 카드**: 배터리 %, 현재 상태(patrol / enforce / returning / charging) 실시간 표시
- **원격 제어**: 순찰 시작·정지 버튼 (Firebase 경유 → ROS2)
- **단속 기록**: 감지 시각, 번호판, 초기 감지 사진 + 확정 증거 사진
- **반응형 디자인**: 모바일 지원

#### Firebase 브릿지 노드 (`robot_two.py` / `robot_three.py`)

웹 UI와 ROS2 사이의 중간 계층 역할

```
[웹 UI] → Firebase robot_command/{id}/patrol_command
                        ↓ (Firebase 리스너)
              robot_two/three.py
              → ROS2 /{ns}/patrol_command 발행 (7회 반복, 신뢰성 확보)

[ROS2] ← /{ns}/battery_state, /{ns}_status
                        ↓ (구독)
              robot_two/three.py
              → Firebase robot_status/{id}/battery, status 갱신
                        ↓
                    [웹 UI 실시간 반영]
```

---

## 6. Firebase DB 구조

```
root/
├── robot_status/
│   ├── robot2/
│   │   ├── battery: <0~100>
│   │   └── status: "patrol" | "enforce" | "returning" | "charging"
│   └── robot3/
│       └── (동일 구조)
│
├── robot_command/
│   ├── robot2/
│   │   └── patrol_command: "start" | "stop"
│   └── robot3/
│       └── (동일 구조)
│
├── detections/
│   └── {YYYYMMDD_HHMMSS_ffffff}/
│       ├── status: "confirmed"
│       ├── detected_at: <ISO 8601 타임스탬프>
│       ├── confirmed_at: <ISO 8601 타임스탬프>
│       ├── initial_image: <base64 JPEG>    # 최초 감지 프레임
│       ├── evidence_image: <base64 JPEG>   # 확정 증거 프레임
│       ├── plate_image: <base64 JPEG>      # 번호판 크롭 이미지
│       └── plate_number: "12가 3456"
│
└── cctv_detections/
    └── {case_key}/
        ├── (CCTV 노드가 생성한 초기 케이스 데이터)
        ├── amr_evidence_image: <base64>    # AMR이 추가한 2차 증거
        ├── amr_confirmed_at: <ISO 8601>
        ├── plate_image: <base64>
        └── plate_number: "12가 3456"
```

---

## 7. 전체 통신 흐름

```
[관리자]
   │ 브라우저
   ▼
[웹 대시보드 ui_v6.html]
   │ Firebase Realtime DB 읽기/쓰기
   ▼
[Firebase]
   │ robot_command 변경 감지
   ▼
[robot_two.py / robot_three.py]  ←── /{ns}/battery_state, /{ns}_status (ROS2 수신)
   │ /{ns}/patrol_command 발행         │
   ▼                                   │
[amr1.py / amr2.py] ────────────────┘
   │ 순찰 중
   │
   ├── [AMR 자체 감지 경로]
   │     depth_coor_amr1/2.py
   │       OAK-D → YOLOv8 → EKF3D → 맵 좌표
   │       → /{ns}/amr_done 발행
   │         ↓
   │     amr1/2.py MODE_ENFORCEMENT 진입
   │         ↓
   │     /{ns}/start ("amr_start") 발행
   │
   └── [CCTV 연동 경로]
         cctv.py → /{ns}/cctv_done 발행
           ↓
         amr1/2.py MODE_ENFORCEMENT 진입
           ↓
         /{ns}/start ("cctv_start") 발행

[ocr_node.py]  ← /{ns}/start 수신
   │ YOLOv8 번호판 감지 → OCR
   │ → /{ns}/capture_done 발행
   │ → Firebase 단속 기록 업로드
   ▼
[Firebase] → [웹 대시보드 실시간 갱신]
```

---

## 8. 실행 방법

### 환경 요구사항

- ROS2 Humble
- TurtleBot4 Navigation 패키지
- Python 3.10+
- Firebase Admin SDK (`pip install firebase-admin`)
- PaddleOCR (`pip install paddleocr`)
- Ultralytics YOLOv8 (`pip install ultralytics`)
- DepthAI (OAK-D SDK, `pip install depthai`)

### 빌드

```bash
cd ~/click_car
colcon build
source install/setup.bash
```

### 노드 실행

```bash
# 1. AMR 순찰·단속 노드 (각 로봇에서 실행)
ros2 run crackdown amr1   # robot3
ros2 run crackdown amr2   # robot2

# 2. 차량 감지 노드 (각 로봇에서 실행)
python3 src/amr_detect/depth_coor_amr1.py   # robot3
python3 src/amr_detect/depth_coor_amr2.py   # robot2

# 3. OCR 노드 (각 로봇에서 실행)
python3 src/webcam/ocr_node.py

# 4. Firebase 브릿지 노드 (서버 또는 별도 PC에서 실행)
python3 web/robot_three.py &
python3 web/robot_two.py &

# 5. 웹 대시보드
# web/ui_v6.html 을 브라우저에서 열거나 웹서버로 서빙
```

---

## 9. 주요 파라미터

| 파라미터 | robot3 (amr1) | robot2 (amr2) | 설명 |
|----------|--------------|--------------|------|
| `ENFORCEMENT_STOP_DIST` | 0.75 m | 0.60 m | 차량으로부터 정지 거리 |
| `BATTERY_LOW_THRESHOLD` | 25 % | 25 % | 자동 복귀 배터리 임계값 |
| `ENFORCEMENT_WAIT_AMR` | 30 s | 30 s | amr_start 촬영 대기 시간 |
| `ENFORCEMENT_WAIT_CCTV` | 10 s | 10 s | cctv_start 촬영 대기 시간 |
| `CONF_THRESHOLD` (YOLO) | 0.70 | 0.70 | 차량 감지 최소 신뢰도 |
| `DEPTH_PERCENTILE` | 10 % | 10 % | 깊이 노이즈 억제 퍼센타일 |
| `MAX_PUBLISH_COUNT` | 10 회 | 10 회 | 동일 차량 좌표 발행 최대 횟수 |
| `PUBLISH_INTERVAL` | 0.2 s | 0.2 s | 발행 스로틀 간격 |
| `TRACK_GRACE_SEC` | 2.0 s | 2.0 s | 미감지 트랙 유예 시간 |
| `ID_IN_CAR_OVERLAP_THRESH` | 50 % | 50 % | 번호판-차량 겹침 최소 비율 |

---

## 10. 트러블슈팅 & 설계 결정

### Zone 경계 겹침 버그

**증상**: 특정 좌표 수신 시 잘못된 구역으로 이동
**원인**: Zone 1과 Zone 2의 경계 X좌표가 동일하게 설정되어 두 구역이 중복
**해결**: 경계값을 측정값 기반으로 재산정, 두 구역이 X=-1.37을 기준으로 명확히 분리되도록 수정

### `capture_done` 잔류값 버그

**증상**: 연속 단속 시 두 번째 단속에서 촬영 없이 즉시 완료 처리
**원인**: 이전 단속의 `capture_done = True` 값이 초기화되지 않은 채 잔류
**해결**: 촬영 신호 발행 직전에 `capture_done = False` 초기화 추가

### MultiThreadedExecutor 충돌

**증상**: 시작 시 현재 위치를 읽으려고 임시 rclpy 노드를 생성했더니 무한 대기 발생
**원인**: `executor.spin()` 실행 중 동일 프로세스에서 별도 `rclpy.spin_once()` 호출 → 스핀 충돌
**해결**: 임시 노드 방식 제거, AMRNode 클래스 멤버로 `/amcl_pose` 구독 추가

---

> Firebase 서비스 계정 키(`web/click_car.json`)는 공개 저장소에 포함하지 마십시오.
