-- 1. 웹캠 차량 감지 기록
CREATE TABLE vehicle_detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_path TEXT NOT NULL,
    detected_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    location TEXT,
    status TEXT DEFAULT 'waiting' 
    -- waiting / dispatched / completed
);

-- 2. 터틀봇4 출동 명령 로그
CREATE TABLE robot_dispatch_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    detection_id INTEGER,
    dispatch_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    robot_id TEXT,
    status TEXT DEFAULT 'sent',
    FOREIGN KEY (detection_id) REFERENCES vehicle_detections(id)
);

-- 3. 터틀봇4 번호판 인식 데이터
CREATE TABLE license_plate_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    detection_id INTEGER,
    plate_number TEXT,
    image_path TEXT,
    captured_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (detection_id) REFERENCES vehicle_detections(id)
);