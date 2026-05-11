<img width="2560" height="1783" alt="z7812219730552_b7a39694fb8b4de0f583e46075d0f03d" src="https://github.com/user-attachments/assets/69c856c0-814d-489b-928c-5ecd157b3e2d" />
# Metro Safety Monitor
Hệ thống giám sát an toàn ga metro bằng AI — sử dụng camera kết hợp **YOLO Pose Estimation** và **ResNet18** để phát hiện các tình huống nguy hiểm trên sân ga theo thời gian thực: **ngã (fall)**, **xâm nhập vùng nguy hiểm (intrusion)**, và **cảnh báo sớm trước khi ngã (pre-fall)**.

https://github.com/user-attachments/assets/cc76a6a9-88fb-4d70-b9bc-b2ae32957534



---

## Tổng quan kiến trúc

```
Metro_Safety_Monitor/
├── NCKH/                        # Phiên bản gốc (v1) — Streamlit UI, xử lý đơn camera
├── NCKH_UPDATE/                 # Phiên bản nâng cấp (v2) — đa camera, background engine
├── classification_door/         # Huấn luyện model phân loại cửa (phiên bản cũ)
├── classification_door_new/     # Huấn luyện model phân loại cửa (phiên bản mới, 4 options)
└── .gitignore
```

---

## Các thành phần

### 📁 NCKH — Phiên bản gốc (v1)

Phiên bản đầu tiên của hệ thống, xử lý **một camera** tại một thời điểm thông qua giao diện **Streamlit**.

**Tính năng chính:**
- Phát hiện ngã dựa trên phân tích tư thế (bounding box ratio, vận tốc rơi, vị trí keypoint)
- Phát hiện xâm nhập vùng nguy hiểm (khi cửa đóng)
- Nhận diện trạng thái cửa đóng/mở bằng ResNet18 (Dropout 0.3)
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

### 📁 NCKH_UPDATE — Phiên bản nâng cấp (v2)

Phiên bản mở rộng với **xử lý đa camera song song**, chạy engine ngầm (background) tách biệt khỏi UI.

**Tính năng bổ sung so với v1:**
- **Background Engine** — xử lý đa camera đa luồng (mỗi camera 1 thread), singleton pattern
- **Pre-fall Detection** — cảnh báo sớm trước khi ngã (ngưỡng mềm + kiểm tra chuyển động)
- **Dynamic Risk Score** — đánh giá rủi ro liên tục (0 → 1) cho mỗi người, dựa trên khoảng cách, hướng di chuyển, thời gian đứng gần mép, tốc độ
- **Adaptive Danger Zone** — micro-zone (dải mép) tự động tạo bằng polygon buffer
- **Demo Viewer** — xem video real-time qua OpenCV, lấy frame từ engine (không chạy inference lần 2)
- **Metrics Manager** — theo dõi FPS, latency YOLO/ResNet, CPU/GPU (singleton, thread-safe)
- **SQLite Database** — lưu trữ cảnh báo và thống kê hệ thống (WAL mode, singleton)
- **Dashboard 3 tab** — Tổng quan, Phân tích theo giờ, Lịch sử cảnh báo

**Cách chạy:**
```bash
# Terminal 1: Dashboard + khởi động engine
cd Metro_Safety_Monitor/NCKH_UPDATE
streamlit run main.py

# Terminal 2: Viewer real-time (tùy chọn)
python demo_viewer_from_engine.py
```

> Xem chi tiết trong [NCKH_UPDATE/README.md](NCKH_UPDATE/README.md).

---

### 📁 classification_door — Huấn luyện model cửa (phiên bản cũ)

Module huấn luyện model **ResNet18** để phân loại trạng thái cửa metro (đóng/mở) — phiên bản ban đầu.

| Nội dung | Mô tả |
|----------|-------|
| `model/datasets.py` | Custom dataset: đọc ảnh theo thư mục con, map class name → index |
| `model/model_option1.py` | Transfer learning ResNet18, freeze backbone, train 8 epoch, batch 32, lr=3e-4, CosineAnnealing |
| `model/model_option2.py` | Biến thể huấn luyện với cấu hình khác |
| `ket_qua_option_1/` | Kết quả option 1 (model, biểu đồ accuracy/precision/recall/F1, TensorBoard log) |
| `ket_qua_option2/` | Kết quả option 2 |
| `data/` | Dữ liệu huấn luyện (thư mục train/test) |
| `requirements.txt` | Thư viện cần thiết |

**Model output:** File `best_model.pth` — dùng bởi `DoorEngine` trong NCKH (v1).

---

### 📁 classification_door_new — Huấn luyện model cửa (phiên bản mới)

Phiên bản cải tiến với **4 kịch bản huấn luyện** và cơ chế **early stopping**.

| Nội dung | Mô tả |
|----------|-------|
| `models/datasets.py` | Custom dataset (giống phiên bản cũ) |
| `models/option_1.py` | ResNet18, freeze backbone, batch 8, lr=3e-4, 20 epoch, early stopping patience=2 |
| `models/option_2.py` | Biến thể cấu hình #2 |
| `models/option_3.py` | Biến thể cấu hình #3 |
| `models/option_4.py` | Biến thể cấu hình #4 |
| `models/__init__.py` | Package init |
| `test.ipynb` | Notebook kiểm thử model |
| `ket_qua_option1/` → `ket_qua_option4/` | Kết quả 4 options |
| `data/` | Dữ liệu huấn luyện |

**Cải tiến so với `classification_door`:**
- Thêm 2 option huấn luyện mới (tổng 4 options)
- Early stopping (patience=2) để tránh overfitting
- Batch size nhỏ hơn (8 vs 32) cho dữ liệu nhỏ
- Max epoch tăng (20 vs 8) kết hợp early stopping
- Notebook test để đánh giá nhanh

**Model output:** File `best_model_v1.pth` — dùng bởi `DoorEngine` trong NCKH_UPDATE (v2).

---

## Yêu cầu hệ thống

| Thành phần | Yêu cầu |
|-----------|----------|
| Hệ điều hành | Windows 10/11 |
| Python | 3.9+ |
| GPU | NVIDIA GPU hỗ trợ CUDA (khuyến nghị — hệ thống vẫn chạy trên CPU nhưng FPS thấp) |

### Cài đặt thư viện

```bash
pip install ultralytics opencv-python torch torchvision shapely streamlit pygame psutil pandas
```

> Nếu dùng GPU, cài đúng phiên bản PyTorch có CUDA tại [pytorch.org](https://pytorch.org/get-started/locally/).

---

## Công nghệ sử dụng

| Công nghệ | Mục đích |
|-----------|----------|
| **YOLO Pose (v11/v26)** | Ước lượng tư thế 17 keypoints, phát hiện ngã/pre-fall |
| **ByteTrack** | Theo dõi đối tượng liên frame (tracking) |
| **ResNet18** | Phân loại trạng thái cửa đóng/mở (transfer learning) |
| **Shapely** | Kiểm tra điểm trong polygon, tính khoảng cách tới boundary |
| **Streamlit** | Dashboard giám sát (UI) |
| **OpenCV** | Xử lý video, vẽ overlay, hiển thị real-time |
| **SQLite (WAL mode)** | Lưu trữ cảnh báo và thống kê hệ thống |
| **PyTorch** | Inference model AI (CUDA + half precision) |
| **pygame** | Phát âm thanh cảnh báo |
| **psutil** | Theo dõi CPU usage |

---

## Luồng xử lý chính

```
Camera → Frame → YOLO Pose → Keypoints + Tracking (ByteTrack)
                                  │
                    ┌─────────────┼──────────────┐
                    ▼             ▼               ▼
               Fall Detection  Intrusion      Pre-fall
               (tư thế, vận   Detection      Warning
                tốc, ratio)   (feet in zone)  (dấu hiệu sớm)
                    │             │               │
                    ▼             ▼               ▼
               Streak ≥ 4    Streak ≥ 3      Streak ≥ 3
                    │             │               │
                    └─────────────┼──────────────┘
                                  ▼
                       ┌──────────────────┐
                       │   Alert System   │
                       ├──────────────────┤
                       │ • Lưu ảnh JPG    │
                       │ • Ghi SQLite DB  │
                       │ • Âm thanh alarm │
                       │ • UI notification│
                       └──────────────────┘
```

Song song: **ResNet18** phân loại trạng thái cửa mỗi N frame → khi cửa **mở**, intrusion detection tự động tắt.

---

## Giấy phép

Dự án nghiên cứu khoa học (NCKH).
<img width="2560" height="1783" alt="z7812219730552_b7a39694fb8b4de0f583e46075d0f03d" src="https://github.com/user-attachments/assets/3b499c16-d5ad-4943-81a5-86afa493aa55" />

