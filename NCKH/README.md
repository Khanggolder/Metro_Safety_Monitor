# NCKH — Hệ thống Giám sát An toàn Metro (Phiên bản gốc v1)

Phiên bản đầu tiên của hệ thống giám sát an toàn ga metro, sử dụng **YOLO Pose Estimation** + **ResNet18** để phát hiện **ngã** và **xâm nhập vùng nguy hiểm** theo thời gian thực trên giao diện **Streamlit**.

---

## Cấu trúc thư mục

```
NCKH/
├── config.py              # Cấu hình 3 camera, polygon vùng cửa + vùng nguy hiểm
├── camera_manager.py      # Wrapper cv2.VideoCapture, trả polygon từ config
├── door_engine.py         # Nhận diện cửa đóng/mở (ResNet18, Dropout 0.3)
├── pose_engine.py         # Phát hiện ngã + xâm nhập (YOLO Pose + ByteTrack)
├── main.py                # Ứng dụng Streamlit (UI chính, xử lý single-camera)
│
├── best_model.pth         # Model ResNet18 cho phân loại cửa
├── yolo26n-pose.pt        # Model YOLO Pose (nano, mặc định)
├── yolo11n-pose.pt        # Model YOLO Pose v11 (nano)
├── yolo11s-pose.pt        # Model YOLO Pose v11 (small)
├── yolo26s-pose.pt        # Model YOLO Pose v26 (small)
├── alarm.mp3              # Âm thanh cảnh báo
│
├── data/                  # Video mẫu để demo
│   ├── te_ngang_010.mp4   # Video test té ngã
│   ├── vung_cam_001.mp4   # Video test xâm nhập vùng cấm
│   └── back_ground_004.mp4 # Video background
│
├── alerts/                # Ảnh cảnh báo (tự tạo khi có sự cố)
│   ├── falls/             # Ảnh phát hiện ngã
│   └── intrusions/        # Ảnh phát hiện xâm nhập
│
└── storage/               # Dữ liệu lưu trữ khác
```

---

## Cách chạy

```bash
cd Metro_Safety_Monitor/NCKH
streamlit run main.py
```

Giao diện Streamlit sẽ mở trên trình duyệt tại `localhost:8501`. Chọn camera từ sidebar để bắt đầu giám sát.

---

## Các module chi tiết

### `config.py` — Cấu hình camera

Định nghĩa 3 camera mặc định với đường dẫn video và tọa độ polygon:

```python
CAMERAS = {
    "Cam 1": {
        "video": ROOT + r"\data\te_ngang_010.mp4",
        "door_zone": [[1413,662], [1416,810], [1618,930], [1622,701]],
        "danger_zone": [[1673,1076], [1008,592], [1008,577], [1913,754], [1912,1074]]
    },
    # "Cam 2", "Cam 3" tương tự...
}
```

- **door_zone**: Polygon vùng cửa (dùng cho ResNet phân loại đóng/mở)
- **danger_zone**: Polygon vùng nguy hiểm (dùng cho phát hiện xâm nhập)
- Hỗ trợ video file hoặc URL stream RTSP

> Tọa độ polygon lấy bằng tool vẽ polygon (VD: [Roboflow PolygonZone](https://polygonzone.roboflow.com/) hoặc script OpenCV).

### `camera_manager.py` — Quản lý video

Class `CameraSystem` đóng gói `cv2.VideoCapture`, cung cấp:
- `read()` — đọc frame
- `release()` — giải phóng camera
- Property `door_zone`, `danger_zone` — trả tọa độ polygon từ config

### `door_engine.py` — Nhận diện cửa

Class `DoorEngine` sử dụng **ResNet18** fine-tuned:

1. **Khởi tạo**: Load ResNet18 (weights=None), thay FC layer bằng `Dropout(0.3) → Linear(512, 2)`, load pretrained weights
2. **crop_polygon()**: Crop vùng cửa từ frame theo polygon bằng mask + bounding rect
3. **predict()**: Transform crop (resize 224×224, normalize ImageNet), inference → `DOOR_OPEN (1)` hoặc `DOOR_CLOSE (0)`

Được gọi mỗi 10 frame (`DOOR_SKIP`) trong `main.py` để tiết kiệm tài nguyên.

### `pose_engine.py` — Phát hiện ngã & xâm nhập

Class `PoseEngine` sử dụng **YOLO Pose + ByteTrack** (185 dòng):

**Cấu hình YOLO inference:**
- `conf=0.3`, `iou=0.6`, `imgsz=512`
- `half=True` nếu dùng CUDA (FP16)
- Tracker: `bytetrack.yaml`
- Chạy mỗi 2 frame (`SKIP=2`)

**Phát hiện ngã (4 tiêu chí):**

| Tiêu chí | Ngưỡng | Mô tả |
|----------|--------|-------|
| `ratio` | > 1.1 | Tỷ lệ w/h bounding box (nằm ngang) |
| `velocity_y` | > 12 | Vận tốc rơi theo trục Y (pixel/frame) |
| `nose_y > hip_y + 50` | - | Đầu thấp hơn hông (lộn ngược) |
| `keypoint_ratio` | < 0.70 | Keypoint co cụm (người cuộn tròn) |

**Phủ định (override False):**
- `nose_y < hip_y - 50` hoặc `shoulder_y < hip_y - 40` → người đang đứng thẳng
- `velocity_y < -5` → người đang đứng lên

**Phát hiện xâm nhập:**
- Tính vị trí trung bình 2 mắt cá chân (ankle) → `feet`
- Dùng `Shapely.Polygon.contains(Point(feet))` để kiểm tra
- Chỉ cảnh báo khi cửa **đóng** (`door_state == 0`)
- Cần `N_INTRUDE = 3` frame liên tiếp để xác nhận

**Streak-based confirmation:** Cả ngã (`N_FALL=4`) và xâm nhập (`N_INTRUDE=3`) đều dùng cơ chế đếm liên tiếp — phải phát hiện N frame liên tục mới kích hoạt cảnh báo, reset về 0 nếu gián đoạn → giảm false positive.

**Bảo vệ riêng tư:** Khi phát hiện ngã hoặc xâm nhập, khuôn mặt tự động làm mờ bằng `GaussianBlur(151×151, σ=30)` dựa trên 5 face keypoints (nose, eyes, ears) mở rộng 20px.

**Skeleton rendering:** Vẽ 8 đường xương (skeleton) nối vai-hông-đầu gối-mắt cá chân.

### `main.py` — Giao diện Streamlit

Ứng dụng chính (157 dòng):

- **Page config**: title "Metro AI Monitor", wide layout
- **Camera selection**: Sidebar selectbox, khởi tạo lại engine khi chuyển camera
- **Video loop**: Đọc frame → DoorEngine mỗi 10 frame → PoseEngine mỗi frame → overlay
- **Overlay**: Polygon cửa (xanh nếu mở, đỏ nếu đóng), polygon vùng nguy hiểm (đỏ)
- **Alert**: Lưu ảnh JPG vào `alerts/falls/` hoặc `alerts/intrusions/` kèm timestamp + track ID
- **Sidebar tabs**: Hiển thị 10 ảnh cảnh báo mới nhất (falls + intrusions)
- **Audio**: `pygame.mixer` phát `alarm.mp3`, cooldown 3 giây
- **Display**: Cập nhật UI mỗi 3 frame (`DISPLAY_SKIP=3`)

---

## Bảng tổng hợp tham số

| Tham số | Giá trị | Mô tả |
|---------|---------|-------|
| `DOOR_SKIP` | 10 | Predict cửa mỗi N frame |
| `ALERT_COOLDOWN` | 3s | Cooldown giữa các cảnh báo âm thanh |
| `DISPLAY_SKIP` | 3 | Cập nhật UI mỗi N frame |
| `PoseEngine.SKIP` | 2 | Chạy YOLO mỗi N frame |
| `N_FALL` | 4 | Streak liên tiếp để xác nhận ngã |
| `N_INTRUDE` | 3 | Streak liên tiếp để xác nhận xâm nhập |
| `conf` | 0.3 | Ngưỡng confidence YOLO |
| `iou` | 0.6 | Ngưỡng IoU cho NMS |
| `imgsz` | 512 | Kích thước input YOLO |
| Dropout | 0.3 | Dropout trong DoorEngine FC layer |

---

## Yêu cầu

```bash
pip install ultralytics opencv-python torch torchvision shapely streamlit pygame
```

- Python 3.9+
- NVIDIA GPU có CUDA (khuyến nghị)

---

## Hạn chế của v1

- Chỉ xử lý **một camera** tại một thời điểm (single-threaded)
- Xử lý video gắn liền với vòng lặp UI Streamlit — không có background engine
- Không lưu cảnh báo vào database — chỉ lưu ảnh JPG
- Chưa có **pre-fall detection** và **risk scoring**
- Không có demo viewer độc lập (phải mở Streamlit để xem video)
- Không có metrics tracking (FPS, latency)
- Không tự reconnect khi video kết thúc

> Phiên bản nâng cấp **NCKH_UPDATE** giải quyết tất cả các hạn chế trên. Xem [../NCKH_UPDATE/README.md](../NCKH_UPDATE/README.md).
