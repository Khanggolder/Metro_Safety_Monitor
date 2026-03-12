# Hệ thống Giám sát An toàn Metro bằng AI (v2 — NCKH_UPDATE)

Phiên bản nâng cấp của hệ thống giám sát an toàn ga metro, sử dụng camera kết hợp AI (**YOLO Pose Estimation** + **ResNet18**) để phát hiện các tình huống nguy hiểm trên sân ga theo thời gian thực: **ngã (fall)**, **xâm nhập vùng nguy hiểm (intrusion)**, **cảnh báo sớm trước khi ngã (pre-fall)**, và **đánh giá rủi ro liên tục (risk scoring)**.

So với phiên bản gốc (NCKH v1), v2 bổ sung: **xử lý đa camera song song**, **background engine chạy ngầm**, **pre-fall detection**, **dynamic risk scoring**, **adaptive danger zone**, **demo viewer OpenCV**, **metrics tracking**, và **SQLite database**.

---

## Yêu cầu hệ thống

- **Hệ điều hành:** Windows 10/11
- **Python:** 3.9 trở lên
- **GPU:** NVIDIA GPU hỗ trợ CUDA (khuyến nghị — hệ thống vẫn chạy được trên CPU nhưng FPS sẽ thấp)

### Thư viện cần cài đặt

```bash
pip install ultralytics opencv-python torch torchvision shapely streamlit pygame psutil pandas
```

> Nếu dùng GPU, hãy đảm bảo cài đúng phiên bản PyTorch có hỗ trợ CUDA. Xem hướng dẫn tại [pytorch.org](https://pytorch.org/get-started/locally/).

---

## ⚠️ Cài đặt bắt buộc trước khi chạy

Sau khi tải về, **bắt buộc phải sửa đường dẫn** trong 2 file sau cho đúng với vị trí thư mục trên máy của bạn:

### 1. Sửa `config.py` — Đường dẫn ROOT

Mở file `config.py`, sửa dòng 2 — thay đường dẫn **tuyệt đối** trỏ tới thư mục `NCKH_UPDATE` trên máy bạn:

```python
# ❌ Đường dẫn cũ (sẽ lỗi trên máy khác):
ROOT = r"C:\Users\ad\Downloads\codepython\project\NCKH"

# ✅ Sửa thành đường dẫn tới thư mục NCKH_UPDATE trên máy bạn:
ROOT = r"<ĐƯỜNG_DẪN_TỚI_THƯ_MỤC_NCKH_UPDATE>"
```

**Ví dụ:**
```python
ROOT = r"D:\MyProject\Metro_Safety_Monitor\NCKH_UPDATE"
```

> **Quan trọng:** `ROOT` được dùng để xác định đường dẫn video (`data/`), model AI (`yolo26n-pose.pt`, `best_model_v1.pth`), thư mục cảnh báo (`alerts/`), và file âm thanh (`alarm.mp3`). Nếu sai, hệ thống sẽ không tìm được file và báo lỗi.

### 2. Sửa `db_manager.py` — Đường dẫn Database

Mở file `db_manager.py`, sửa dòng 7–9 — thay đường dẫn SQLite database:

```python
# ❌ Đường dẫn cũ (sẽ lỗi trên máy khác):
DB_PATH = os.path.join(
    r"D:\Metro_Safety_Monitor\NCKH_UPDATE", "metro_ai.db"
)

# ✅ Sửa thành (tự động lấy từ thư mục hiện tại):
DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "metro_ai.db"
)
```

> **Gợi ý:** Cách dùng `os.path.dirname(os.path.abspath(__file__))` sẽ tự động trỏ tới thư mục chứa file `db_manager.py`, không cần sửa lại khi chuyển máy.

### Tóm tắt các file cần sửa

| File | Dòng | Biến | Sửa thành |
|------|------|------|-----------|
| `config.py` | 2 | `ROOT` | Đường dẫn tuyệt đối tới thư mục `NCKH_UPDATE` |
| `db_manager.py` | 7–9 | `DB_PATH` | Dùng `os.path.dirname(os.path.abspath(__file__))` hoặc đường dẫn tuyệt đối |

> **Lưu ý:** Tất cả các module khác (`background_engine.py`, `demo_viewer_from_engine.py`, `main.py`...) đều lấy đường dẫn từ `ROOT` trong `config.py`, nên chỉ cần sửa 2 file trên là đủ.

---

## Cấu trúc thư mục

```
NCKH_UPDATE/
├── config.py                    # ⚠️ CẦN SỬA ROOT — cấu hình camera, polygon
├── camera_manager.py            # Wrapper cv2.VideoCapture với property polygon
├── door_engine.py               # Nhận diện trạng thái cửa (ResNet18, Dropout 0.5)
├── pose_engine.py               # Phát hiện ngã/pre-fall/xâm nhập + risk scoring (386 dòng)
├── background_engine.py         # Engine đa camera chạy ngầm, mỗi camera 1 thread (216 dòng)
├── metrics_manager.py           # Singleton quản lý metrics thread-safe (130 dòng)
├── db_manager.py                # ⚠️ CẦN SỬA DB_PATH — SQLite WAL, 2 bảng (133 dòng)
├── main.py                      # Dashboard Streamlit 3 tab: Tổng quan, Phân tích, Lịch sử (146 dòng)
├── demo_viewer_from_engine.py   # Viewer OpenCV real-time + âm thanh cảnh báo 2 mức (147 dòng)
│
├── best_model.pth               # Model ResNet18 phân loại cửa (phiên bản cũ)
├── best_model_v1.pth            # Model ResNet18 phân loại cửa (phiên bản mới — đang dùng)
├── yolo26n-pose.pt              # Model YOLO Pose nano (mặc định)
├── yolo11n-pose.pt              # Model YOLO Pose v11 nano
├── yolo11s-pose.pt              # Model YOLO Pose v11 small
├── yolo26s-pose.pt              # Model YOLO Pose v26 small
├── alarm.mp3                    # Âm thanh cảnh báo mặc định
├── metro_ai.db                  # Database SQLite (tự tạo khi chạy)
│
├── data/                        # Video mẫu để demo
│   ├── te_2.mp4                 # Video test té ngã
│   ├── xam_nhap.mp4             # Video test xâm nhập
│   └── back_ground_1.mp4       # Video background
│
└── alerts/                      # Ảnh cảnh báo (tự tạo khi phát hiện sự cố)
    ├── falls/                   # Ảnh phát hiện ngã (ID_<track_id>_<timestamp>.jpg)
    └── intrusions/              # Ảnh phát hiện xâm nhập
```

---

## Cấu hình camera

Mở file `config.py` để chỉnh sửa danh sách camera và các vùng polygon:

```python
#config.py
ROOT = r"<ĐƯỜNG_DẪN_TỚI_THƯ_MỤC_NCKH_UPDATE>"  # ← SỬA DÒNG NÀY

CAMERAS = {
    "Cam 1": {
        "video": ROOT + r"\data\te_2.mp4",
        "door_zone": [[741, 503], [753, 738], [277, 872], [255, 513]],
        "danger_zone": [[2, 1029], [1402, 584], [1399, 565], [2, 945]]
    },
    # ... thêm camera khác ở đây
}
```

- **video**: Đường dẫn video file hoặc URL stream RTSP
- **door_zone**: Polygon vùng cửa (dùng cho ResNet phân loại đóng/mở)
- **danger_zone**: Polygon vùng nguy hiểm (dùng cho phát hiện xâm nhập + tính risk score)

> Tọa độ polygon lấy bằng cách mở video trong tool vẽ polygon (VD: [Roboflow PolygonZone](https://polygonzone.roboflow.com/) hoặc script OpenCV đơn giản), sau đó copy tọa độ các đỉnh vào config.

---

## Cách chạy

### Cách 1: Dashboard Streamlit + Viewer OpenCV (khuyến nghị)

Mở **hai terminal** riêng biệt:

**Terminal 1** — Khởi động engine + Dashboard:

```bash
cd Metro_Safety_Monitor/NCKH_UPDATE
streamlit run main.py
```

Dashboard sẽ mở trên trình duyệt (`localhost:8501`), hiển thị tổng quan hệ thống: FPS, cảnh báo, lịch sử, phân tích theo giờ. Đồng thời, `BackgroundEngine` tự khởi động và xử lý tất cả camera ngầm.

**Terminal 2** — Mở cửa sổ xem video real-time:

```bash
cd Metro_Safety_Monitor/NCKH_UPDATE
python demo_viewer_from_engine.py
```

Viewer lấy frame đã xử lý từ engine (không chạy YOLO/ResNet lần 2), hiển thị bằng OpenCV kèm âm thanh cảnh báo.

### Cách 2: Chỉ chạy Dashboard

```bash
streamlit run main.py
```

Engine vẫn chạy xử lý ngầm, dữ liệu cảnh báo được lưu vào database. Tuy nhiên sẽ không có cửa sổ video real-time.

---

## Phím tắt trong Viewer

| Phím | Chức năng |
|------|-----------|
| `1` – `9` | Chuyển sang camera tương ứng |
| `n` | Camera tiếp theo |
| `p` | Camera trước đó |
| `q` | Thoát viewer |

---

## Mô tả chi tiết từng module

### `config.py` — Cấu hình hệ thống

- Biến `ROOT` — đường dẫn gốc project
- Dict `CAMERAS` — 3 camera mặc định (Cam 1: `te_2.mp4`, Cam 2: `xam_nhap.mp4`, Cam 3: `back_ground_1.mp4`)
- Mỗi camera có `video`, `door_zone`, `danger_zone`

### `camera_manager.py` — Quản lý nguồn video

Class `CameraSystem`:
- `__init__(cam_name)` — khởi tạo `cv2.VideoCapture` từ config
- `read()` → `(ret, frame)` — đọc frame
- `release()` — giải phóng camera
- Property `door_zone`, `danger_zone` — trả tọa độ polygon

### `door_engine.py` — Nhận diện cửa (ResNet18)

Class `DoorEngine` — phân loại trạng thái cửa metro đóng/mở:

**Kiến trúc:**
- ResNet18 (pretrained=None) → thay FC layer bằng `Dropout(0.5) → Linear(512, 2)`
- Transform: `ToPILImage → Resize(224) → ToTensor → Normalize(ImageNet)`
- Device: CUDA nếu có, fallback CPU

**Phương thức:**
- `crop_polygon(frame, polygon=None)` — crop vùng cửa bằng mask + bounding rect, hỗ trợ polygon tùy chỉnh (khác v1: v1 chỉ dùng polygon cố định từ constructor)
- `predict(frame, polygon=None)` → `DOOR_OPEN (1)` hoặc `DOOR_CLOSE (0)`, tự động dọn CUDA cache sau inference

**Khác biệt so với NCKH v1:**
- Dropout tăng từ 0.3 → **0.5** (regularization mạnh hơn)
- Hỗ trợ **polygon parameter** linh hoạt (có thể truyền polygon khác vào `predict()`)
- Tự động **dọn CUDA cache** (`torch.cuda.empty_cache()`) sau mỗi lần predict

### `pose_engine.py` — Phát hiện ngã / pre-fall / xâm nhập + Risk Scoring (386 dòng)

Class `PoseEngine` — core engine phân tích tư thế, là module phức tạp nhất.

#### Khởi tạo

```python
PoseEngine(model_path, danger_polygon, enable_prefall=True, enable_adaptive_danger=True)
```

- Load YOLO Pose model, fuse layers, chuyển sang device
- Tạo `Shapely.Polygon` từ `danger_polygon`
- Khởi tạo dictionaries theo dõi: `prev_y_coords`, `fall_streak`, `intrude_streak`, `prefall_streak`, `prev_feet`, `dwell_seconds`, `prev_boundary_dist`
- Xây dựng **micro-zone** (dải mép vùng nguy hiểm, rộng 40px)
- Có thể bật/tắt pre-fall và adaptive danger zone qua flag

#### YOLO Inference

```python
model.track(frame, persist=True, conf=0.3, iou=0.6, imgsz=512,
            half=True, tracker="bytetrack.yaml")
```

- Chạy mỗi 2 frame (`SKIP=2`)
- Output: bounding boxes (xywh), keypoints (17 điểm), track IDs

#### 1. Phát hiện ngã (Fall Detection)

**4 tiêu chí kích hoạt (raw_falling = True):**

| # | Tiêu chí | Ngưỡng | Ý nghĩa |
|---|----------|--------|----------|
| 1 | `velocity_y` | > 12 | Vận tốc rơi nhanh theo trục Y |
| 2 | `nose_y > hip_y + 50` hoặc `shoulder_y > hip_y + 40` | - | Đầu/vai thấp hơn hông (lộn/nghiêng) |
| 3 | `keypoint_ratio` | < 0.70 | Keypoints co cụm (người cuộn tròn) |
| 4 | `ratio` (w/h) | > 1.1 | Bounding box nằm ngang |

**2 tiêu chí phủ định (override False):**

| # | Tiêu chí | Ngưỡng | Ý nghĩa |
|---|----------|--------|----------|
| 5 | `nose_y < hip_y - 50` hoặc `shoulder_y < hip_y - 40` | - | Người đứng thẳng bình thường |
| 6 | `velocity_y` | < -5 | Người đang đứng lên (di chuyển lên) |

Cần **4 frame liên tiếp** (`N_FALL=4`) phát hiện raw_falling = True mới kích hoạt cảnh báo. Label: **"EMERGENCY: FALL"** (màu vàng).

#### 2. Phát hiện xâm nhập (Intrusion Detection)

- Tính vị trí chân: trung bình 2 mắt cá chân (`kp[15]`, `kp[16]`)
- Dùng `Shapely.Polygon.contains(Point(feet))` để kiểm tra chân trong vùng nguy hiểm
- Chỉ kích hoạt khi cửa **đóng** (`door_state != 1`)
- Cần **3 frame liên tiếp** (`N_INTRUDE=3`)
- Label: **"DANGER: INTRUSION"** (màu đỏ)

#### 3. Cảnh báo sớm trước khi ngã (Pre-fall Warning) — MỚI trong v2

Phát hiện dấu hiệu bất thường **nhẹ hơn** ngưỡng ngã, chỉ khi có chuyển động:

**Kiểm tra chuyển động (motion gate):**
```
has_motion = (|velocity_y| + vel_x_feet) > 3.0 OR speed_feet > 3.0
```
→ Tránh false positive khi người đứng yên, cúi nhặt đồ.

**4 tiêu chí kích hoạt pre-fall (ngưỡng mềm hơn fall):**

| # | Tiêu chí | Ngưỡng Pre-fall | So sánh Fall |
|---|----------|-----------------|-------------|
| 1 | `velocity_y` | > 6 | > 12 |
| 2 | `nose_y > hip_y` | + 20 | + 50 |
| 3 | `keypoint_ratio` | < 0.82 | < 0.70 |
| 4 | `ratio` (w/h) | > 0.85 | > 1.1 |

**3 tiêu chí phủ định:**
- `velocity_y < -3` → đang đứng lên
- `nose_y < hip_y - 20` → đứng thẳng
- `shoulder_y < hip_y - 15` → đứng thẳng

Cần **3 frame liên tiếp** (`N_PREFALL=3`). Label: **"WARNING: PREFALL"** (màu cam).
Pre-fall chỉ chạy khi người **chưa bị phát hiện ngã** (`not is_this_fall`).

#### 4. Đánh giá rủi ro liên tục (Dynamic Risk Score) — MỚI trong v2

Mỗi người được theo dõi một chỉ số rủi ro (0.0 → 1.0) dựa trên 4 yếu tố:

```python
risk = 0.40 * d_risk + 0.25 * v_risk + 0.25 * dwell_risk + 0.10 * speed_risk
```

| Yếu tố | Trọng số | Công thức | Ý nghĩa |
|--------|---------|-----------|----------|
| `d_risk` | 40% | `exp(-distance / 80)` | Khoảng cách tới mép vùng nguy hiểm (càng gần → càng cao) |
| `v_risk` | 25% | `1.0 nếu (prev_d - d) > 3` | Đang tiến lại gần mép (hướng di chuyển) |
| `dwell_risk` | 25% | `sigmoid((dwell_time - 1.0) / 0.7)` | Thời gian đứng trong micro-zone (dải mép 40px) |
| `speed_risk` | 10% | `sigmoid((speed - 8.0) / 6.0)` | Tốc độ di chuyển (chạy nhanh = nguy hiểm hơn) |

**Các mức rủi ro:**

| Mức | Score | Label hiển thị |
|-----|-------|---------------|
| OK | < 0.30 | Không hiển thị |
| WARN | 0.30 – 0.55 | `[WARN 0.42]` |
| DANGER | 0.55 – 0.80 | `[DANGER 0.67]` |
| EMERGENCY | ≥ 0.80 | `[EMERGENCY 0.85]` |

Risk score chỉ hiển thị khi cửa **đóng** và risk level ≠ OK.

#### 5. Adaptive Danger Zone (micro-zone) — MỚI trong v2

- Tự động tạo dải mép vùng nguy hiểm rộng `BAND_PX = 40` pixel bằng `polygon.buffer(-40).difference(polygon)`
- Dùng cho tính toán `dwell_risk` (thời gian đứng trong dải mép)
- Nếu buffer không hợp lệ (polygon quá nhỏ), fallback về polygon gốc

#### 6. Stale Track Cleanup

Sau mỗi frame, xóa dữ liệu tracking của người đã rời khỏi frame (không còn trong `active_ids`), giải phóng memory: `prev_y_coords`, `fall_streak`, `intrude_streak`, `prefall_streak`, `prev_feet`, `prev_boundary_dist`, `dwell_seconds`, `last_seen_ts`, `_prev_risk_level`.

#### Output format

```python
(frame, any_fall, any_intrude, fall_trigger_ids, intrude_trigger_ids, active_ids, extra)
```

`extra` dict chứa: `prefall_trigger_ids`, `prefall_active_ids`, `risk_score_by_tid`, `risk_level_by_tid`, `danger_trigger_ids`.

### `background_engine.py` — Engine đa camera chạy ngầm (216 dòng)

**`CameraWorker`** — worker cho từng camera:

- Mỗi camera chạy trên **1 daemon thread** riêng biệt
- Vòng lặp: đọc frame → DoorEngine (mỗi 35 frame) → PoseEngine → overlay → lưu alert → cập nhật metrics
- `latest_frame` được bảo vệ bằng `threading.Lock` (thread-safe)
- `latest_alarm` riêng biệt (HIGH/PREFALL) với lock riêng
- **Auto reconnect**: Khi video kết thúc (`ret == False`), tự giải phóng và tạo lại `CameraSystem` sau 2 giây
- Ghi system stats vào DB mỗi 30 giây (`STATS_LOG_INTERVAL`)
- Phát hiện intrusion count realtime → cập nhật metrics

**`BackgroundEngine`** — orchestrator:

- Khởi tạo `MetricsManager`, `DBManager`, tạo `CameraWorker` cho mỗi camera
- `start()` — khởi động tất cả workers
- `stop()` — dừng tất cả workers
- Model paths: `yolo26n-pose.pt` (YOLO) + `best_model_v1.pth` (ResNet)

**`get_engine()`** — singleton factory:

- Global lock đảm bảo chỉ tạo 1 instance
- Tự start nếu chưa chạy
- Được gọi từ cả `main.py` (Streamlit) và `demo_viewer_from_engine.py`

### `metrics_manager.py` — Quản lý metrics (130 dòng)

Class `MetricsManager` — **singleton, thread-safe**:

**Các metric được track:**

| Metric | Phương thức update | Phương thức get |
|--------|-------------------|-----------------|
| FPS per camera | `update_fps()` | `get_fps()`, `get_avg_fps()` |
| YOLO inference (ms) | `update_yolo_infer()` | `get_yolo_infer()` |
| ResNet inference (ms) | `update_resnet_infer()` | `get_resnet_infer()` |
| Intrusion count | `update_intrusion_count()` | `get_intrusion_count()` |
| Camera status | `update_camera_status()` | `get_camera_status()`, `get_all_camera_status()` |
| CPU usage | (static) | `get_cpu_usage()` (via `psutil`) |
| GPU VRAM (MB) | (static) | `get_gpu_memory_mb()` (via `torch.cuda`) |

**History management:**
- `push_history()` — lưu dữ liệu performance theo thời gian (tối đa 500 records)
- Cleanup tự động mỗi 1 giờ (`_cleanup_interval = 3600`), xóa dữ liệu cũ hơn 1 giờ
- `snapshot()` — trả dict tổng hợp tất cả metrics

### `db_manager.py` — Database SQLite (133 dòng)

Class `DBManager` — **singleton, WAL mode, thread-safe**:

**Database**: `metro_ai.db` với 2 bảng:

**Bảng `alerts`:**
| Column | Type | Mô tả |
|--------|------|-------|
| id | INTEGER PK | Auto increment |
| timestamp | TEXT | "YYYY-MM-DD HH:MM:SS" |
| type | TEXT | "fall" hoặc "intrusion" |
| camera_name | TEXT | Tên camera (VD: "Cam 1") |
| image_path | TEXT | Đường dẫn ảnh JPG |

**Bảng `system_stats`:**
| Column | Type | Mô tả |
|--------|------|-------|
| timestamp | TEXT | "YYYY-MM-DD HH:MM:SS" |
| camera_name | TEXT | Tên camera |
| fps | REAL | FPS tại thời điểm |
| yolo_infer_ms | REAL | YOLO inference time (ms) |
| resnet_infer_ms | REAL | ResNet inference time (ms) |

**Các query hỗ trợ:**
- `get_alerts_today()` — tất cả cảnh báo hôm nay
- `get_alerts_count_today()` — đếm cảnh báo hôm nay
- `get_alerts_by_hour_today()` — phân bổ cảnh báo theo giờ (cho biểu đồ)
- `get_alerts_by_type_today()` — phân loại fall vs intrusion
- `get_latest_alerts(limit=20)` — N cảnh báo mới nhất
- `get_latest_stats(limit=100)` — N bản ghi stats mới nhất

### `main.py` — Dashboard Streamlit (146 dòng)

Dashboard 3 tab, auto-refresh mỗi 0.5 giây:

**Sidebar:**
- Trạng thái từng camera (🟢 Live / 🔴 Dead + FPS)
- CPU usage, GPU VRAM

**Tab 1 — 📋 Tổng quan:**
- 4 metrics: Cảnh báo hôm nay, FPS trung bình, Latency trung bình, CPU Usage
- GPU Memory (nếu có)
- Chi tiết từng camera: FPS, YOLO ms, ResNet ms, Intrusion count
- Toast notification khi có cảnh báo mới

**Tab 2 — 📈 Phân tích:**
- Biểu đồ cột: Cảnh báo theo giờ (24h)
- Biểu đồ + bảng: Fall vs Intrusion

**Tab 3 — 🕐 Lịch sử cảnh báo:**
- 20 cảnh báo mới nhất (expandable)
- Hiển thị: ID, loại, camera, thời gian, ảnh chụp

### `demo_viewer_from_engine.py` — Viewer OpenCV (147 dòng)

Cửa sổ OpenCV hiển thị video real-time, **không chạy inference**:

- Lấy frame đã xử lý từ `CameraWorker.get_latest_frame()` (đã có bounding box, skeleton, label)
- **HUD overlay** (semi-transparent): Camera name, Engine FPS, YOLO ms, ResNet ms, phím tắt
- Hỗ trợ chuyển camera bằng phím (`1-9`, `n`, `p`)

**Hệ thống âm thanh 2 mức:**

| Mức | File mặc định | Cooldown | Ưu tiên |
|-----|--------------|----------|---------|
| **HIGH** | `data/alarm_high.mp3` (fallback: `alarm.mp3`) | 3 giây | Cao nhất |
| **PREFALL** | `data/alarm_prefall.mp3` (fallback: `alarm.mp3`) | 2 giây | Thấp hơn HIGH |

- Backend: **pygame** (ưu tiên) → **winsound.Beep** (fallback) → không âm thanh
- Khi HIGH đang trong cooldown, PREFALL bị bỏ qua (tránh chồng tiếng)
- Alarm chỉ phát cho camera đang xem
- **Consumed tracking**: Mỗi alarm chỉ phát 1 lần, theo dõi bằng `last_consumed_alarm_ts`

---

## Bảng tổng hợp tham số

### PoseEngine

| Tham số | Giá trị | Mô tả |
|---------|---------|-------|
| `SKIP` | 2 | Chạy YOLO mỗi N frame |
| `N_FALL` | 4 | Streak liên tiếp để xác nhận ngã |
| `N_INTRUDE` | 3 | Streak liên tiếp để xác nhận xâm nhập |
| `N_PREFALL` | 3 | Streak liên tiếp để xác nhận pre-fall |
| `BAND_PX` | 40 | Độ rộng micro-zone (pixel) |
| `SIGMA_DIST` | 80.0 | Hệ số phân rã khoảng cách (d_risk) |
| `V_MARGIN` | 3.0 | Ngưỡng tiến gần mép (v_risk) |
| `DWELL_DECAY` | 0.3 | Tốc độ giảm dwell khi rời micro-zone |
| `conf` | 0.3 | Ngưỡng confidence YOLO |
| `iou` | 0.6 | Ngưỡng IoU cho NMS |
| `imgsz` | 512 | Kích thước input YOLO |

### BackgroundEngine

| Tham số | Giá trị | Mô tả |
|---------|---------|-------|
| `DOOR_SKIP` | 35 | Predict cửa mỗi N frame |
| `ALERT_COOLDOWN` | 3s | Cooldown giữa cảnh báo (ảnh + DB) |
| `STATS_LOG_INTERVAL` | 30s | Ghi thống kê vào DB mỗi N giây |

### Risk Score Thresholds

| Level | Score range | Hiển thị |
|-------|-----------|----------|
| OK | < 0.30 | Ẩn |
| WARN | 0.30 – 0.55 | `[WARN 0.42]` (vàng) |
| DANGER | 0.55 – 0.80 | `[DANGER 0.67]` (cam) |
| EMERGENCY | ≥ 0.80 | `[EMERGENCY 0.85]` (đỏ) |

---

## Kiến trúc hệ thống

```
┌─────────────────────────────────────────────────────┐
│                  BackgroundEngine (singleton)         │
│                                                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
│  │ Worker 1 │  │ Worker 2 │  │ Worker 3 │  ...       │
│  │ (Thread) │  │ (Thread) │  │ (Thread) │           │
│  │          │  │          │  │          │           │
│  │ Camera   │  │ Camera   │  │ Camera   │           │
│  │ YOLO     │  │ YOLO     │  │ YOLO     │           │
│  │ ResNet   │  │ ResNet   │  │ ResNet   │           │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘           │
│       │              │              │                 │
│       ▼              ▼              ▼                 │
│  ┌──────────────────────────────────────────┐        │
│  │         MetricsManager (singleton)        │        │
│  │  FPS, YOLO ms, ResNet ms, CPU, GPU        │        │
│  └──────────────────────────────────────────┘        │
│  ┌──────────────────────────────────────────┐        │
│  │          DBManager (singleton, WAL)       │        │
│  │  alerts table + system_stats table        │        │
│  └──────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────┘
          │                              │
          ▼                              ▼
┌──────────────────┐          ┌────────────────────┐
│  main.py         │          │ demo_viewer.py     │
│  (Streamlit)     │          │ (OpenCV + Audio)   │
│  Dashboard UI    │          │ Real-time Viewer   │
│  3 tabs + sidebar│          │ HUD overlay        │
└──────────────────┘          └────────────────────┘
```

---

## Dữ liệu cảnh báo

Mỗi khi phát hiện ngã hoặc xâm nhập, hệ thống sẽ:

1. **Lưu ảnh** vào `alerts/falls/` hoặc `alerts/intrusions/` (tên file: `ID_<track_id>_<timestamp>.jpg`)
2. **Ghi vào database** SQLite (`metro_ai.db`) kèm timestamp, loại cảnh báo, tên camera, đường dẫn ảnh

Dữ liệu này hiển thị trên **Dashboard Streamlit**:
- Tab **Tổng quan**: tổng cảnh báo hôm nay
- Tab **Phân tích**: biểu đồ cảnh báo theo giờ, tỷ lệ fall/intrusion
- Tab **Lịch sử**: danh sách expandable với ảnh

---

## Xử lý sự cố

| Vấn đề | Cách khắc phục |
|--------|---------------|
| Viewer hiện "Waiting for frames..." | Đảm bảo đã chạy `streamlit run main.py` trước (engine cần khởi động) |
| FPS thấp (< 10) | Kiểm tra GPU CUDA đã cài đúng. Giảm `imgsz` hoặc dùng model nano |
| Không có âm thanh | Cài `pygame`: `pip install pygame`. Fallback: `winsound.Beep` (Windows) |
| Lỗi import module | Chạy từ đúng thư mục `NCKH_UPDATE/`, không chạy từ thư mục khác |
| Video kết thúc, viewer đen | Worker tự reconnect sau 2 giây (status chuyển "Reconnecting" → "Live") |
| Database locked | SQLite đã dùng WAL mode. Kiểm tra không có process khác lock file |
| GPU out of memory | Giảm `imgsz`, dùng model nano thay small, hoặc giảm số camera |

---

## Ghi chú kỹ thuật

- **YOLO Pose**: `yolo26n-pose.pt` — YOLO v26 nano pose, tracking bằng ByteTrack, FP16 trên CUDA
- **Door classification**: ResNet18 fine-tuned, Dropout 0.5, model `best_model_v1.pth` (từ `classification_door_new`)
- **Database**: SQLite WAL mode, thread-safe via lock, non-blocking write
- **Threading**: Mỗi camera 1 daemon thread, singleton engine + metrics + DB
- **Memory**: Stale tracks tự động cleanup, metrics history capped 500 records, hourly cleanup
- **Precision**: `torch.set_float32_matmul_precision("high")`, `cudnn.benchmark = True`
