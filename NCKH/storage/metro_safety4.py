import cv2
import numpy as np
import torch
from ultralytics import YOLO
from shapely.geometry import Point, Polygon
import streamlit as st
import os
from datetime import datetime
import glob
import time
import pygame
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

#Mở terminal run lệnh:
#streamlit run D:\Metro_Safety_Monitor\NCKH\storage\metro_safety4.py

device = "cuda:0" if torch.cuda.is_available() else "cpu"

door_model = models.resnet18(weights=None)
num_feats = door_model.fc.in_features
door_model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(num_feats, 2)
)
door_model.load_state_dict(torch.load(
    r"D:\Metro_Safety_Monitor\NCKH\best_model.pth",
    map_location=device
))

door_model.to(device)
door_model.eval()

DOOR_OPEN = 1
DOOR_CLOSE = 0

door_state = DOOR_CLOSE
DOOR_SKIP = 10

# door_zone_pts = [[1053, 362], [1051, 777], [1539, 1077], [1573, 1077], [1580, 479]] #back_ground_004
# door_zone_pts = [[1074, 612], [1082, 912], [1551, 1077], [1759, 1076], [1751, 685]]#vung_cam_001
# door_zone_pts = [[948, 599], [948, 599], [950, 864], [1496, 1076], [1582, 1076], [1582, 1076], [1579, 675]]#vung_cam_002
#door_zone_pts = [[1413, 662], [1416, 810], [1618, 930], [1622, 701]] #te_ngang_010
door_zone_pts = [[422, 463], [439, 715], [139, 818], [90, 474]] #te_ngang_010

door_polygon = Polygon(door_zone_pts)

def crop_polygon(frame, polygon_pts):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    pts = np.array(polygon_pts, dtype = np.int32)
    cv2.fillPoly(mask, [pts], 255)

    res = cv2.bitwise_and(frame, frame, mask=mask)

    x,y,w,h = cv2.boundingRect(pts)
    crop = res[y:y+h, x:x+w]
    return crop

door_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
def predict_door(frame):
    crop = crop_polygon(frame, door_zone_pts)

    if crop is None or crop.size == 0:
        return DOOR_CLOSE

    # BGR -> RGB
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    # transform
    input_tensor = door_transform(crop).unsqueeze(0).to(device)

    with torch.no_grad():
        output = door_model(input_tensor)
        pred = output.argmax(1).item()

    return pred

# --- TẠO THƯ MỤC LƯU CẢNH BÁO ---
ALERT_ROOT = r"D:\Metro_Safety_Monitor\NCKH\alerts"
# --- GIAO DIỆN STREAMLIT ---
st.set_page_config(page_title="Metro Monitor", layout="wide")
st.title("Hệ thống Giám sát Metro")
frame_placeholder = st.empty()

# --- SIDEBAR: Xem danh sách ảnh cảnh báo ---
with st.sidebar:
    st.header("Cảnh báo đã lưu")

    # Tabs cho 2 loại cảnh báo
    tab_fall, tab_intrude = st.tabs(["Té ngã (falls)", "Xâm nhập (intrusions)"])


    def get_alert_images(folder):
        path = os.path.join(ALERT_ROOT, folder, "*.jpg")
        files = sorted(glob.glob(path), key=os.path.getmtime, reverse=True)
        return files[:10]  # chỉ hiển thị 10 ảnh mới nhất


    with tab_fall:
        fall_images = get_alert_images("falls")
        if fall_images:
            for img_path in fall_images:
                st.image(img_path, caption=os.path.basename(img_path), width="stretch")
        else:
            st.info("Chưa có cảnh báo té ngã")

    with tab_intrude:
        intrude_images = get_alert_images("intrusions")
        if intrude_images:
            for img_path in intrude_images:
                st.image(img_path, caption=os.path.basename(img_path), width="stretch")
        else:
            st.info("Chưa có cảnh báo xâm nhập")


def setup_folders():
    for subfolder in ["falls", "intrusions"]:
        full_path = os.path.join(ALERT_ROOT, subfolder)
        if not os.path.exists(full_path):
            os.makedirs(full_path)
setup_folders()


pygame.mixer.init()
pygame.mixer.music.load(r"D:\Metro_Safety_Monitor\NCKH\alarm.mp3")
def play_alarm():
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.play()

model = YOLO(r"D:\Metro_Safety_Monitor\NCKH_UPDATE\yolo26s-pose.pt")

model.to(device)
model.fuse()

# danger_zone_pts = [[1130, 1076], [228, 289], [246, 286], [1549, 1077]] #back_ground_004
# danger_zone_pts = [[1193, 1076], [208, 562], [208, 524], [1710, 698], [1705, 1076]] #vung_cam_001
# danger_zone_pts = [[1246, 1076], [211, 529], [211, 508], [1711, 711], [1711, 1076]] #vung_cam_002
#danger_zone_pts = [[1673, 1076], [1008, 592], [1008, 577], [1913, 754], [1912, 1074]] #te_ngang_010
danger_zone_pts = [[123, 884], [62, 816], [994, 542], [1014, 585]]
danger_polygon = Polygon(danger_zone_pts)

prev_y_coords = {}
fall_streak = {}
intrude_streak = {}

N_FALL = 4
N_INTRUDE = 3

NOSE = 0
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12

# --- Keypoints mặt (COCO 17) để che mặt ---
FACE_POINTS = [0, 1, 2, 3, 4]  # nose, left_eye, right_eye, left_ear, right_ear


def check_intrusion(feet_point):
    return danger_polygon.contains(Point(feet_point))


def update_streak(streaks, track_id, flag):
    streaks[track_id] = streaks.get(track_id, 0) + 1 if flag else 0
    return streaks[track_id]


def save_alert(frame, folder_name, track_id):
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"ID_{track_id}_{time_str}.jpg"
    final_path = os.path.join(ALERT_ROOT, folder_name, file_name)
    cv2.imwrite(final_path, frame)


def blur_face(frame, kpts):
    valid_face_pts = [kpts[i] for i in FACE_POINTS if kpts[i][0] > 0 and kpts[i][1] > 0]
    if len(valid_face_pts) < 3:
        return frame

    # Tính bounding box quanh các điểm mặt
    face_pts = np.array(valid_face_pts, dtype=np.int32)
    x_min, y_min = np.min(face_pts, axis=0)
    x_max, y_max = np.max(face_pts, axis=0)

    expand = 20
    x1 = max(0, x_min - expand)
    y1 = max(0, y_min - expand)
    x2 = min(frame.shape[1], x_max + expand)
    y2 = min(frame.shape[0], y_max + expand)

    # Blur vùng mặt (GaussianBlur)
    face_roi = frame[y1:y2, x1:x2]
    blurred = cv2.GaussianBlur(face_roi, (151, 151), 30)
    frame[y1:y2, x1:x2] = blurred

    return frame

cap = cv2.VideoCapture(r"D:\Metro_Safety_Monitor\NCKH\data\ngã chong rung\v2.mp4", cv2.CAP_FFMPEG)
frame_count = 0
SKIP = 2
DISPLAY_SKIP = 10
last_alert_time = 0
ALERT_COOLDOWN = 3
global_alert = True
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    if frame_count % DOOR_SKIP == 0:
        door_state = predict_door(frame)
    is_falling = False
    is_intruding = False
    display_frame = frame.copy()
    if frame_count % SKIP == 0:
        # start = time.time()
        results = model.track(
            frame,
            persist=True,
            verbose=False,
            device=device,
            conf=0.3,
            iou=0.6,
            imgsz=512,
            half=(device == "cuda:0"),
            tracker="bytetrack.yaml"
        )
        # fps = 1 / (time.time() - start)
        # print(f"YOLO FPS: {fps:.2f}")
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()
            keypoints = results[0].keypoints.xy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, kpts, track_id in zip(boxes, keypoints, track_ids):
                x, y, w, h = box

                # 1. Logic Xâm nhập
                left_ankle = kpts[15]
                right_ankle = kpts[16]
                feet_center = ((left_ankle[0] + right_ankle[0]) / 2, (left_ankle[1] + right_ankle[1]) / 2)
                raw_intruding = check_intrusion(feet_center)
                if door_state == DOOR_OPEN:
                    raw_intruding = False
                intrude_n = update_streak(intrude_streak, track_id, raw_intruding)
                is_intruding = intrude_n >= N_INTRUDE

                # 2. Logic Té ngã
                left_shoulder = kpts[LEFT_SHOULDER]
                right_shoulder = kpts[RIGHT_SHOULDER]
                left_hip = kpts[LEFT_HIP]
                right_hip = kpts[RIGHT_HIP]
                nose = kpts[NOSE]

                shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2 if left_shoulder[0] > 0 and right_shoulder[
                    0] > 0 else \
                    left_shoulder[1] if left_shoulder[0] > 0 else right_shoulder[1] if right_shoulder[0] > 0 else 0

                hip_y = (left_hip[1] + right_hip[1]) / 2 if left_hip[0] > 0 and right_hip[0] > 0 else \
                    left_hip[1] if left_hip[0] > 0 else right_hip[1] if right_hip[0] > 0 else shoulder_y

                nose_y = nose[1] if nose[0] > 0 else hip_y
                current_y = shoulder_y if shoulder_y > 0 else hip_y
                prev_y = prev_y_coords.get(track_id, current_y)
                velocity_y = current_y - prev_y if prev_y is not None else 0
                prev_y_coords[track_id] = current_y

                ratio = w / h if h > 0 else 0
                valid_ys = [kp[1] for kp in kpts if kp[0] > 0 and kp[1] > 0]
                keypoint_span = (max(valid_ys) - min(valid_ys)) if valid_ys else h
                keypoint_ratio = keypoint_span / h if h > 0 else 1.0

                raw_falling = False
                if velocity_y > 12: raw_falling = True
                if (nose_y > hip_y + 50) or (shoulder_y > hip_y + 40): raw_falling = True
                if keypoint_ratio < 0.70: raw_falling = True
                if ratio > 1.1: raw_falling = True
                if (nose_y < hip_y - 50) or (shoulder_y < hip_y - 40): raw_falling = False
                if velocity_y < -5: raw_falling = False

                fall_n = update_streak(fall_streak, track_id, raw_falling)
                is_falling = fall_n >= N_FALL

                # Lưu ảnh cảnh báo
                if is_falling:
                    color, label = (0, 255, 255), "EMERGENCY: FALL"
                    if fall_n == N_FALL:
                        save_alert(display_frame, "falls", track_id)
                        play_alarm()
                elif is_intruding:
                    color, label = (0, 0, 255), "DANGER: INTRUSION"
                    if intrude_n == N_INTRUDE:
                        save_alert(display_frame, "intrusions", track_id)
                        play_alarm()
                else:
                    color, label = (0, 255, 0), f"ID:{track_id} OK"

                # --- Che mặt nếu đang cảnh báo (fall hoặc intrusion) ---
                if is_falling or is_intruding:
                    display_frame = blur_face(display_frame, kpts)

                # Vẽ Box & Skeleton
                cv2.rectangle(display_frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), color, 1)
                cv2.putText(display_frame, label, (int(x - w / 2), int(y - h / 2) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                skeleton = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11),
                            (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
                for a, b in skeleton:
                    if all(kpts[a]) and all(kpts[b]):
                        cv2.line(display_frame, tuple(map(int, kpts[a])), tuple(map(int, kpts[b])), (0, 255, 255), 1)
    pts_door = np.array(door_zone_pts, np.int32).reshape((-1, 1, 2))
    color = (0, 255, 0) if door_state == DOOR_OPEN else (0, 0, 255)
    cv2.polylines(display_frame, [pts_door], True, color, 2)

    text = "DOOR OPEN" if door_state == DOOR_OPEN else "DOOR CLOSE"
    cv2.putText(display_frame, text, (pts_door[0][0][0], pts_door[0][0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    # Vẽ vùng nguy hiểm
    pts_poly = np.array(danger_zone_pts, np.int32).reshape((-1, 1, 2))
    cv2.polylines(display_frame, [pts_poly], isClosed=True, color=(0, 0, 255), thickness=1)

    current_time = time.time()

    if is_falling and (current_time - last_alert_time > ALERT_COOLDOWN):
        st.toast("🚨 EMERGENCY: Có người té ngã!", icon="⚠️")
        last_alert_time = current_time

    elif is_intruding and (current_time - last_alert_time > ALERT_COOLDOWN):
        st.toast("🚨 DANGER: Xâm nhập vùng cấm!", icon="🚫")
        last_alert_time = current_time

    if frame_count % DISPLAY_SKIP == 0:
        frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", width="stretch")
cap.release()