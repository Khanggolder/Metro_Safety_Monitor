# Metro Safety Monitor

Hệ thống giám sát an toàn ga metro bằng AI — sử dụng camera kết hợp **YOLO Pose Estimation** và **ResNet18** để phát hiện các tình huống nguy hiểm trên sân ga theo thời gian thực: **ngã (fall)**, **xâm nhập vùng nguy hiểm (intrusion)**, và **cảnh báo sớm trước khi ngã (pre-fall)**.

---

## Tổng quan kiến trúc

```
Metro_Safety_Monitor/
├── NCKH/                    # Phiên bản gốc (v1) — Streamlit UI đơn camera
├── NCKH_UPDATE/             # Phiên bản nâng cấp (v2) — đa camera, background engine
├── classification_door/     # Huấn luyện model phân loại trạng thái cửa
└── .gitignore
```

---

## Các thành phần

### NCKH — Phiên bản gốc (v1)

Phiên bản đầu tiên của hệ thống, xử lý **một camera** tại một thời điểm thông qua giao diện **Streamlit**.

**Tính năng chính:**
- Phát hiện ngã dựa trên tư thế (bounding box ratio, vận tốc rơi, vị trí keypoint)
- Phát hiện xâm nhập vùng nguy hiểm (khi cửa đóng)
- Nhận diện trạng thái cửa đóng/mở bằng ResNet18
- Làm mờ khuôn mặt khi phát hiện sự cố (bảo vệ quyền riêng tư)
- Lưu ảnh cảnh báo vào thư mục `alerts/`
- Âm thanh cảnh báo qua pygame

**Cách chạy:**
```bash
cd Metro_Safety_Monitor/NCKH
streamlit run main.py
```

> Xem chi tiết trong [NCKH/README.md](NCKH/README.md).

---

### NCKH_UPDATE — Phiên bản nâng cấp (v2)

Phiên bản mở rộng với **xử lý đa camera song song**, chạy engine ngầm tách biệt khỏi UI.

**Tính năng bổ sung so với v1:**
- **Background Engine** — xử lý đa camera đa luồng (mỗi camera 1 thread)
- **Pre-fall Detection** — cảnh báo sớm trước khi ngã
- **Dynamic Risk Score** — đánh giá rủi ro liên tục (0 → 1) cho mỗi người
- **Adaptive Danger Zone** — micro-zone mở rộng tự động dựa theo vị trí vùng nguy hiểm
- **Demo Viewer** — xem video real-time qua OpenCV, không cần Streamlit
- **Metrics Manager** — theo dõi FPS, latency YOLO/ResNet, CPU/GPU
- **SQLite Database** — lưu trữ cảnh báo và thống kê hệ thống

**Cách chạy:**
```bash
# Terminal 1: Dashboard
cd Metro_Safety_Monitor/NCKH_UPDATE
streamlit run main.py

# Terminal 2: Viewer real-time (tùy chọn)
python demo_viewer_from_engine.py
```

> Xem chi tiết trong [NCKH_UPDATE/README.md](NCKH_UPDATE/README.md).

---

### classification_door — Huấn luyện model cửa

Module huấn luyện model **ResNet18** để phân loại trạng thái cửa metro (đóng/mở).

**Nội dung:**
- `model/datasets.py` — Xử lý dữ liệu đầu vào, tạo dataset từ ảnh crop vùng cửa
- `model/model_option1.py` — Kịch bản huấn luyện option 1
- `model/model_option2.py` — Kịch bản huấn luyện option 2
- `ket_qua_option_1/` — Kết quả huấn luyện option 1 (log, biểu đồ)
- `ket_qua_option2/` — Kết quả huấn luyện option 2 (log, biểu đồ)
- `data/` — Dữ liệu huấn luyện
- `requirements.txt` — Thư viện cần thiết

**Model output:** File `best_model.pth` (ResNet18 fine-tuned) được sử dụng bởi `DoorEngine` trong NCKH và NCKH_UPDATE.

---

## Yêu cầu hệ thống

| Thành phần | Yêu cầu |
|-----------|----------|
| Hệ điều hành | Windows 10/11 |
| Python | 3.9+ |
| GPU | NVIDIA GPU hỗ trợ CUDA (khuyến nghị) |

### Cài đặt thư viện

```bash
pip install ultralytics opencv-python torch torchvision shapely streamlit pygame psutil
```

> Nếu dùng GPU, cài đúng phiên bản PyTorch có CUDA tại [pytorch.org](https://pytorch.org/get-started/locally/).

---

## Công nghệ sử dụng

| Công nghệ | Mục đích |
|-----------|----------|
| **YOLO Pose (v8/v11)** | Ước lượng tư thế, phát hiện ngã/pre-fall |
| **ByteTrack** | Theo dõi đối tượng (tracking) |
| **ResNet18** | Phân loại trạng thái cửa đóng/mở |
| **Shapely** | Kiểm tra điểm trong polygon (vùng nguy hiểm) |
| **Streamlit** | Dashboard giám sát |
| **OpenCV** | Xử lý video, hiển thị real-time |
| **SQLite** | Lưu trữ cảnh báo và thống kê |
| **PyTorch** | Inference model AI |

---

## Luồng xử lý chính

```
Camera → Frame → YOLO Pose → Keypoints + Tracking
                                  │
                    ┌─────────────┼──────────────┐
                    ▼             ▼               ▼
               Fall Detection  Intrusion      Pre-fall
               (tư thế, vận   Detection      Warning
                tốc, ratio)   (feet in zone)  (dấu hiệu sớm)
                    │             │               │
                    └─────────────┼──────────────┘
                                  ▼
                          Alert System
                    (ảnh, DB, âm thanh, UI)
```

Đồng thời, **ResNet18** phân loại trạng thái cửa mỗi N frame — khi cửa mở, phát hiện xâm nhập tự động tắt.

---

## Giấy phép

Dự án nghiên cứu khoa học (NCKH).
