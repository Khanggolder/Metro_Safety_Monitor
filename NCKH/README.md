# NCKH — Hệ thống Giám sát An toàn Metro (Phiên bản gốc v1)

Phiên bản đầu tiên của hệ thống giám sát an toàn ga metro, sử dụng **YOLO Pose Estimation** + **ResNet18** để phát hiện **ngã** và **xâm nhập vùng nguy hiểm** theo thời gian thực trên giao diện **Streamlit**.

---

## Cấu trúc thư mục

```
NCKH/
├── config.py              # Cấu hình camera, polygon vùng cửa + vùng nguy hiểm
├── camera_manager.py      # Quản lý nguồn video (đọc frame, trả polygon)
├── door_engine.py         # Nhận diện trạng thái cửa đóng/mở (ResNet18)
├── pose_engine.py         # Phát hiện ngã + xâm nhập (YOLO Pose + ByteTrack)
├── main.py                # Ứng dụng Streamlit (UI chính)
│
├── best_model.pth         # Model ResNet18 cho phân loại cửa
├── yolo26n-pose.pt        # Model YOLO Pose Estimation (nano)
├── yolo11n-pose.pt        # Model YOLO Pose thay thế
├── yolo11s-pose.pt        # Model YOLO Pose (small)
├── yolo26s-pose.pt        # Model YOLO Pose (small v2)
├── alarm.mp3              # Âm thanh cảnh báo
│
├── data/                  # Video mẫu để demo
│   ├── te_ngang_010.mp4
│   ├── vung_cam_001.mp4
│   └── back_ground_004.mp4
│
├── alerts/                # Ảnh cảnh báo (tự tạo khi có sự cố)
│   ├── falls/
│   └── intrusions/
│
└── storage/               # Dữ liệu lưu trữ khác
```

---

## Cách chạy

```bash
cd Metro_Safety_Monitor/NCKH
streamlit run main.py
```

Giao diện Streamlit sẽ mở trên trình duyệt. Chọn camera từ sidebar để bắt đầu giám sát.

---

## Các module chính

### `config.py` — Cấu hình camera

Định nghĩa danh sách camera với đường dẫn video và tọa độ polygon:

```python
CAMERAS = {
    "Cam 1": {
        "video": ROOT + r"\data\te_ngang_010.mp4",
        "door_zone": [[1413,662], [1416,810], [1618,930], [1622,701]],
        "danger_zone": [[1673,1076], [1008,592], ...]
    },
    # ...
}
```

- **door_zone**: Polygon vùng cửa (dùng cho ResNet phân loại)
- **danger_zone**: Polygon vùng nguy hiểm (dùng cho phát hiện xâm nhập)

### `camera_manager.py` — Quản lý video

Class `CameraSystem` đóng gói `cv2.VideoCapture`, cung cấp property truy cập `door_zone` và `danger_zone` từ config.

### `door_engine.py` — Nhận diện cửa

Class `DoorEngine` sử dụng **ResNet18** fine-tuned:
1. Crop vùng cửa từ frame theo polygon
2. Transform ảnh (resize 224×224, normalize)
3. Predict: `DOOR_OPEN (1)` hoặc `DOOR_CLOSE (0)`

Được gọi mỗi 10 frame (`DOOR_SKIP`) để tiết kiệm tài nguyên.

### `pose_engine.py` — Phát hiện ngã & xâm nhập

Class `PoseEngine` sử dụng **YOLO Pose + ByteTrack**:

**Phát hiện ngã** dựa trên:
- Tỷ lệ bounding box w/h > 1.1
- Vận tốc rơi (velocity_y > 12)
- Vị trí bất thường: đầu thấp hơn hông
- Keypoint co cụm (keypoint_ratio < 0.70)

**Phát hiện xâm nhập:**
- Kiểm tra vị trí chân trong polygon `danger_zone`
- Chỉ cảnh báo khi cửa **đóng** (door_state == 0)
- Cần N frame liên tiếp (`N_INTRUDE = 3`) để xác nhận

**Streak-based confirmation:** Cả ngã và xâm nhập đều dùng cơ chế streak — phải phát hiện liên tiếp N frame mới kích hoạt cảnh báo, giảm false positive.

**Bảo vệ riêng tư:** Khuôn mặt tự động được làm mờ (GaussianBlur) khi phát hiện ngã hoặc xâm nhập.

### `main.py` — Giao diện Streamlit

Giao diện chính với:
- Chọn camera từ sidebar
- Hiển thị video real-time
- Overlay: polygon cửa (xanh/đỏ theo trạng thái), polygon vùng nguy hiểm
- Sidebar: danh sách ảnh cảnh báo (falls + intrusions)
- Âm thanh cảnh báo qua pygame
- Cooldown 3 giây giữa các cảnh báo

---

## Tham số quan trọng

| Tham số | Giá trị | Mô tả |
|---------|---------|-------|
| `DOOR_SKIP` | 10 | Predict cửa mỗi N frame |
| `ALERT_COOLDOWN` | 3 (giây) | Cooldown giữa các cảnh báo |
| `DISPLAY_SKIP` | 3 | Hiển thị UI mỗi N frame |
| `PoseEngine.SKIP` | 2 | Chạy YOLO mỗi N frame |
| `N_FALL` | 4 | Số frame liên tiếp để xác nhận ngã |
| `N_INTRUDE` | 3 | Số frame liên tiếp để xác nhận xâm nhập |
| `conf` | 0.3 | Ngưỡng confidence YOLO |
| `iou` | 0.6 | Ngưỡng IoU cho NMS |
| `imgsz` | 512 | Kích thước input YOLO |

---

## Yêu cầu

```bash
pip install ultralytics opencv-python torch torchvision shapely streamlit pygame
```

- Python 3.9+
- NVIDIA GPU có CUDA (khuyến nghị)

---

## Hạn chế của v1

- Chỉ xử lý **một camera** tại một thời điểm
- Không có background engine — xử lý gắn liền với UI Streamlit
- Không lưu cảnh báo vào database
- Chưa có pre-fall detection và risk scoring
- Không có demo viewer độc lập

> Phiên bản nâng cấp **NCKH_UPDATE** giải quyết tất cả các hạn chế trên. Xem [../NCKH_UPDATE/README.md](../NCKH_UPDATE/README.md).
