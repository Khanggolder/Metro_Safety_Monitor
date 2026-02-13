import streamlit as st
import cv2
import time
import pygame
import glob
import os
import numpy as np
from datetime import datetime
import torch
from config import CAMERAS
from camera_manager import CameraSystem
from door_engine import DoorEngine
from pose_engine import PoseEngine
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")
from shapely.geometry import Polygon

#streamlit run C:\Users\ad\Downloads\codepython\project\NCKH\main.py

st.set_page_config(page_title="Metro AI Monitor", layout="wide")
st.title("Hệ thống Giám sát An toàn Metro")

if "sound_init" not in st.session_state:
    pygame.mixer.init()
    pygame.mixer.music.load(r"C:\Users\ad\Downloads\codepython\project\NCKH\alarm.mp3")
    st.session_state.sound_init = True

ALERT_ROOT = r"C:\Users\ad\Downloads\codepython\project\NCKH\alerts"
os.makedirs(os.path.join(ALERT_ROOT, "falls"), exist_ok=True)
os.makedirs(os.path.join(ALERT_ROOT, "intrusions"), exist_ok=True)

DOOR_SKIP = 10
ALERT_COOLDOWN = 3


def save_alert(frame, folder_name, track_id=None):
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ID_{track_id}_{time_str}.jpg" if track_id else f"{time_str}.jpg"
    path = os.path.join(ALERT_ROOT, folder_name, filename)
    cv2.imwrite(path, frame)

cam_name = st.sidebar.selectbox("Chọn Camera", list(CAMERAS.keys()))
cfg = CAMERAS[cam_name]

if "cam" not in st.session_state or "cam_name" not in st.session_state or st.session_state.cam_name != cam_name:

    if "cam" in st.session_state:
        st.session_state.cam.release()

    st.session_state.cam_name = cam_name
    st.session_state.cam = CameraSystem(cam_name)

if "door" not in st.session_state:
    st.session_state.door = DoorEngine(
        r"C:\Users\ad\Downloads\codepython\project\NCKH\best_model.pth",
        cfg["door_zone"]
    )
else:
    st.session_state.door.polygon = cfg["door_zone"]

if "pose" not in st.session_state:
    st.session_state.pose = PoseEngine(
        r"C:\Users\ad\Downloads\codepython\project\NCKH\yolo26n-pose.pt",
        cfg["danger_zone"]
    )
else:
    st.session_state.pose.danger_polygon = Polygon(cfg["danger_zone"])

# alert time init
if "last_alert_time" not in st.session_state:
    st.session_state.last_alert_time = 0

cam = st.session_state.cam
door = st.session_state.door
pose = st.session_state.pose
last_alert_time = st.session_state.last_alert_time

frame_holder = st.empty()

frame_count = 0
door_state = 0
DISPLAY_SKIP = 5

with st.sidebar:
    st.header("Cảnh báo đã lưu")
    tab_fall, tab_intrude = st.tabs(["Té ngã (falls)", "Xâm nhập (intrusions)"])

    def get_alert_images(folder):
        path = os.path.join(ALERT_ROOT, folder, "*.jpg")
        files = sorted(glob.glob(path), key=os.path.getmtime, reverse=True)
        return files[:10]

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

run = st.session_state.get("run", True)
while run:
    ret, frame = cam.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % DOOR_SKIP == 0:
        door_state = door.predict(frame)

    processed_frame, is_fall, is_intrude, trigger_ids = pose.process(frame, door_state)

    current_time = time.time()

    if (is_fall or is_intrude) and (current_time - last_alert_time > ALERT_COOLDOWN):

        if not pygame.mixer.music.get_busy():
            pygame.mixer.music.play()

        folder = "falls" if is_fall else "intrusions"

        for tid in trigger_ids:
            save_alert(processed_frame, folder, tid)

        if is_fall:
            st.toast("🚨 EMERGENCY: Có người té ngã!", icon="⚠️")
        else:
            st.toast("🚨 DANGER: Xâm nhập vùng cấm!", icon="🚫")

        last_alert_time = current_time
        st.session_state.last_alert_time = last_alert_time

    door_pts = np.array(cam.door_zone, np.int32).reshape((-1, 1, 2))
    danger_pts = np.array(cam.danger_zone, np.int32).reshape((-1, 1, 2))

    door_color = (0, 255, 0) if door_state == 1 else (0, 0, 255)
    cv2.polylines(processed_frame, [door_pts], True, door_color, 2)
    cv2.polylines(processed_frame, [danger_pts], True, (0, 0, 255), 1)
    text = "DOOR OPEN" if door_state == 1 else "DOOR CLOSE"
    cv2.putText(processed_frame, text,
                (door_pts[0][0][0], door_pts[0][0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, door_color, 2)

    if frame_count % DISPLAY_SKIP == 0:
        frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        frame_holder.image(frame_rgb, channels="RGB", width="stretch")

cam.release()