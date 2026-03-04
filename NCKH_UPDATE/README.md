# Hệ thống Giám sát An toàn Metro bằng AI

Hệ thống sử dụng camera giám sát kết hợp AI (YOLO Pose Estimation + ResNet) để phát hiện các tình huống nguy hiểm trên sân ga metro theo thời gian thực: **ngã**, **xâm nhập vùng nguy hiểm**, và **cảnh báo sớm trước khi ngã (pre-fall)**.

---

## Yêu cầu hệ thống

- **Hệ điều hành:** Windows 10/11
- **Python:** 3.9 trở lên
- **GPU:** NVIDIA GPU hỗ trợ CUDA (khuyến nghị, hệ thống vẫn chạy được trên CPU nhưng FPS sẽ thấp)

### Thư viện cần cài đặt

```bash
pip install ultralytics opencv-python torch torchvision shapely streamlit pygame psutil
```

> Nếu dùng GPU, hãy đảm bảo cài đúng phiên bản PyTorch có hỗ trợ CUDA. Xem hướng dẫn tại [pytorch.org](https://pytorch.org/get-started/locally/).

---

## Cấu trúc thư mục

```
NCKH_UPDATE/
├── config.py                    # Cấu hình camera, polygon vùng cửa + vùng nguy hiểm
├── camera_manager.py            # Quản lý nguồn video
├── door_engine.py               # Nhận diện trạng thái cửa (ResNet18)
├── pose_engine.py               # Phát hiện ngã, pre-fall, xâm nhập (YOLO Pose)
├── background_engine.py         # Engine xử lý đa camera chạy ngầm
├── metrics_manager.py           # Quản lý metrics (FPS, latency, ...)
├── db_manager.py                # Lưu trữ cảnh báo vào SQLite
├── main.py                      # Dashboard Streamlit
├── demo_viewer_from_engine.py   # Xem video real-time qua OpenCV (có âm thanh cảnh báo)
│
├── best_model.pth               # Model ResNet18 phân loại cửa đóng/mở
├── yolo26n-pose.pt              # Model YOLO Pose Estimation
├── alarm.mp3                    # Âm thanh cảnh báo
├── metro_ai.db                  # Database SQLite (tự tạo khi chạy)
│
├── data/                        # Video mẫu để demo
│   ├── te_ngang_010.mp4
│   ├── vung_cam_001.mp4
│   └── back_ground_004.mp4
│
└── alerts/                      # Ảnh cảnh báo (tự tạo khi phát hiện sự cố)
    ├── falls/
    └── intrusions/
```

---

## Cấu hình camera

Mở file `config.py` để chỉnh sửa danh sách camera và các vùng polygon:

```python
CAMERAS = {
    "Cam 1": {
        "video": ROOT + r"\data\te_ngang_010.mp4",
        "door_zone": [[1413,662], [1416,810], [1618,930], [1622,701]],
        "danger_zone": [[1673,1076], [1008,592], [1008,577], [1913,754], [1912,1074]]
    },
    # ... thêm camera khác ở đây
}
```

- **video**: đường dẫn video hoặc URL stream RTSP
- **door_zone**: polygon vùng cửa (để nhận diện cửa đóng/mở)
- **danger_zone**: polygon vùng nguy hiểm (để phát hiện xâm nhập)

> Tọa độ polygon lấy bằng cách mở video trong một tool vẽ polygon (VD: [Roboflow](https://polygonzone.roboflow.com/) hoặc một script OpenCV đơn giản), sau đó copy tọa độ các đỉnh vào config.

---

## Cách chạy

### Cách 1: Dashboard Streamlit + Viewer OpenCV (khuyến nghị)

Mở **hai terminal** riêng biệt:

**Terminal 1** — Khởi động hệ thống + Dashboard:

```bash
cd C:\Users\ad\Downloads\codepython\project\NCKH_UPDATE
streamlit run main.py
```

Dashboard sẽ mở trên trình duyệt, hiển thị tổng quan hệ thống: FPS, cảnh báo, lịch sử, phân tích theo giờ.

**Terminal 2** — Mở cửa sổ xem video real-time:

```bash
cd C:\Users\ad\Downloads\codepython\project\NCKH_UPDATE
python demo_viewer_from_engine.py
```

Viewer lấy frame đã xử lý từ engine (không chạy YOLO/ResNet lần 2), hiển thị bằng OpenCV kèm âm thanh cảnh báo.

### Cách 2: Chỉ chạy Dashboard

```bash
streamlit run main.py
```

Hệ thống vẫn chạy xử lý ngầm, dữ liệu cảnh báo được lưu vào database. Tuy nhiên sẽ không có cửa sổ video real-time.

---

## Phím tắt trong Viewer

| Phím | Chức năng |
|------|-----------|
| `1` – `9` | Chuyển sang camera tương ứng |
| `n` | Camera tiếp theo |
| `p` | Camera trước đó |
| `q` | Thoát viewer |

---

## Các tính năng phát hiện

### Phát hiện ngã (Fall Detection)

Hệ thống phân tích tư thế cơ thể qua YOLO Pose để nhận diện ngã dựa trên:
- Tỷ lệ chiều rộng/chiều cao bounding box
- Vận tốc rơi (theo trục Y)
- Vị trí tương đối giữa đầu, vai, hông
- Độ co cụm keypoint

Khi phát hiện ngã, trên video hiển thị label **"EMERGENCY: FALL"** (màu vàng) và khuôn mặt tự động được làm mờ.

### Cảnh báo sớm trước khi ngã (Pre-fall Warning)

Hệ thống theo dõi các dấu hiệu bất thường nhẹ hơn ngưỡng ngã — ví dụ: cơ thể bắt đầu nghiêng, vận tốc rơi tăng nhưng chưa đạt mức ngã. Khi phát hiện, hiển thị **"WARNING: PREFALL"** (màu cam).

Pre-fall chỉ kích hoạt khi người đang di chuyển (tránh báo nhầm khi cúi nhặt đồ).

### Phát hiện xâm nhập vùng nguy hiểm (Intrusion Detection)

Khi cửa tàu **đóng**, nếu chân người rơi vào vùng nguy hiểm (danger_zone) liên tiếp nhiều frame, hệ thống cảnh báo **"DANGER: INTRUSION"** (màu đỏ).

Khi cửa tàu **mở**, xâm nhập tự động được tắt — hành khách lên xuống bình thường.

### Đánh giá rủi ro liên tục (Risk Score)

Mỗi người được theo dõi một chỉ số rủi ro (0 → 1) dựa trên:
- Khoảng cách tới mép vùng nguy hiểm
- Hướng di chuyển (đang tiến gần mép hay rời xa)
- Thời gian đứng gần mép
- Tốc độ di chuyển

Chỉ số này hiển thị dạng `[WARN 0.42]` hoặc `[DANGER 0.67]` trên label (chỉ khi cửa đóng).

---

## Âm thanh cảnh báo

Viewer tự động phát âm thanh khi phát hiện sự cố:

| Mức | Khi nào phát | Cooldown |
|-----|-------------|----------|
| **HIGH** | Ngã hoặc xâm nhập | 3 giây |
| **PREFALL** | Cảnh báo sớm trước ngã | 2 giây |

- Nếu đang phát mức HIGH, PREFALL sẽ bị bỏ qua (tránh chồng tiếng)
- Âm thanh chỉ phát cho camera đang chọn trong viewer

**Tùy chỉnh âm thanh:** Đặt file `data/alarm_high.mp3` và `data/alarm_prefall.mp3` để dùng âm khác nhau. Nếu không có, hệ thống sẽ dùng file `alarm.mp3` mặc định.

---

## Dữ liệu cảnh báo

Mỗi khi phát hiện ngã hoặc xâm nhập, hệ thống sẽ:

1. **Lưu ảnh** vào thư mục `alerts/falls/` hoặc `alerts/intrusions/`
2. **Ghi vào database** SQLite (`metro_ai.db`) kèm timestamp, loại cảnh báo, tên camera

Dữ liệu này hiển thị trên Dashboard Streamlit ở tab **Lịch sử cảnh báo** và **Phân tích**.

---

## Xử lý sự cố

| Vấn đề | Cách khắc phục |
|--------|---------------|
| Viewer hiện "Waiting for frames..." | Đảm bảo đã chạy `streamlit run main.py` trước |
| FPS thấp (< 10) | Kiểm tra GPU CUDA đã cài đúng chưa, hoặc giảm `imgsz` |
| Không có âm thanh | Cài `pygame`: `pip install pygame` |
| Lỗi import module | Chạy từ đúng thư mục `NCKH_UPDATE/`, không chạy từ thư mục khác |
| Video kết thúc, viewer đen | Viewer engine sẽ tự reconnect; demo standalone sẽ tự loop |

---

## Ghi chú kỹ thuật

- YOLO Pose model: `yolo26n-pose.pt` (YOLOv26 nano pose, tracking bằng ByteTrack)
- Door classification: ResNet18 fine-tuned trên dữ liệu cửa metro
- Database: SQLite WAL mode, ghi non-blocking
- Hệ thống chạy đa luồng (mỗi camera 1 thread), engine độc lập với UI
