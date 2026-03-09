#main.py
import streamlit as st
import time
import os
import pandas as pd
from datetime import datetime

from background_engine import get_engine
from config import CAMERAS

#streamlit run D:\Metro_Safety_Monitor\NCKH_UPDATE\main.py
st.set_page_config(page_title="Metro AI Dashboard", layout="wide")

engine = get_engine()

with st.sidebar:
    st.title("Metro AI")
    st.caption("Hệ thống Giám sát An toàn Metro")
    st.divider()
    st.subheader("Trạng thái Camera")
    metrics = engine.metrics
    for cname in CAMERAS:
        status = metrics.get_camera_status(cname)
        fps = metrics.get_fps(cname)
        icon = "🟢" if status == "Live" else "🔴"
        st.text(f"{icon} {cname}: {status} | {fps:.1f} FPS")

    st.divider()
    st.subheader("Hệ thống")
    st.text(f"CPU: {metrics.get_cpu_usage():.1f}%")
    gpu_mb = metrics.get_gpu_memory_mb()
    if gpu_mb > 0:
        st.text(f"GPU VRAM: {gpu_mb:.0f} MB")
    else:
        st.text("GPU: N/A")

st.title("📊 Dashboard Giám sát Metro AI")

tab_overview, tab_analysis, tab_history = st.tabs(
    ["📋 Tổng quan", "📈 Phân tích", "🕐 Lịch sử cảnh báo"]
)

db = engine.db

with tab_overview:
    col1, col2, col3, col4 = st.columns(4)

    alerts_today = db.get_alerts_count_today()
    avg_fps = metrics.get_avg_fps()
    avg_latency = metrics.get_avg_latency()
    cpu = metrics.get_cpu_usage()
    gpu = metrics.get_gpu_memory_mb()

    col1.metric("🚨 Cảnh báo hôm nay", alerts_today)
    col2.metric("⚡ FPS trung bình", f"{avg_fps:.1f}")
    col3.metric("🕐 Latency trung bình", f"{avg_latency:.1f} ms")
    col4.metric("💻 CPU Usage", f"{cpu:.1f}%")

    if gpu > 0:
        st.metric("🎮 GPU Memory", f"{gpu:.0f} MB")

    if "last_alert_count" not in st.session_state:
        st.session_state.last_alert_count = alerts_today

    if alerts_today > st.session_state.last_alert_count:
        st.toast("Cảnh báo mới!", icon="🚨")
        st.session_state.last_alert_count = alerts_today

    st.divider()
    st.subheader("Chi tiết từng Camera")

    cam_cols = st.columns(len(CAMERAS))
    for i, cname in enumerate(CAMERAS):
        with cam_cols[i]:
            status = metrics.get_camera_status(cname)
            fps = metrics.get_fps(cname)
            yolo = metrics.get_yolo_infer(cname)
            resnet = metrics.get_resnet_infer(cname)
            intrude = metrics.get_intrusion_count(cname)

            icon = "🟢" if status == "Live" else "🔴"
            st.markdown(f"### {icon} {cname}")
            st.text(f"FPS: {fps:.1f}")
            st.text(f"YOLO: {yolo:.1f} ms")
            st.text(f"ResNet: {resnet:.1f} ms")
            st.text(f"Intrusions: {intrude}")

with tab_analysis:
    st.subheader("📊 Cảnh báo theo giờ (hôm nay)")

    hourly_data = db.get_alerts_by_hour_today()
    if hourly_data:
        hour_map = {h: c for h, c in hourly_data}
        hours = list(range(24))
        counts = [hour_map.get(h, 0) for h in hours]
        df_hourly = pd.DataFrame({
            "Giờ": [f"{h:02d}:00" for h in hours],
            "Số cảnh báo": counts,
        })
        st.bar_chart(df_hourly.set_index("Giờ"))
    else:
        st.info("Chưa có cảnh báo nào hôm nay.")

    st.divider()
    st.subheader("🔄 Fall vs Intrusion (hôm nay)")

    type_data = db.get_alerts_by_type_today()
    if type_data:
        df_type = pd.DataFrame({
            "Loại": list(type_data.keys()),
            "Số lượng": list(type_data.values()),
        })
        col_chart, col_table = st.columns(2)
        with col_chart:
            st.bar_chart(df_type.set_index("Loại"))
        with col_table:
            st.dataframe(df_type, width='stretch', hide_index=True)
    else:
        st.info("Chưa có dữ liệu phân loại.")

with tab_history:
    st.subheader("🕐 Cảnh báo mới nhất")

    latest_alerts = db.get_latest_alerts(limit=20)
    if latest_alerts:
        for alert in latest_alerts:
            with st.expander(
                f"{'🔴' if alert['type'] == 'fall' else '🟡'} "
                f"[{alert['type'].upper()}] {alert['camera_name']} — {alert['timestamp']}"
            ):
                st.text(f"ID: {alert['id']}")
                st.text(f"Loại: {alert['type']}")
                st.text(f"Camera: {alert['camera_name']}")
                st.text(f"Thời gian: {alert['timestamp']}")

                img_path = alert.get("image_path", "")
                if img_path and os.path.exists(img_path):
                    st.image(img_path, caption=os.path.basename(img_path),
                             width='stretch')
                else:
                    st.warning(f"Không tìm thấy ảnh: {img_path}")
    else:
        st.info("Chưa có cảnh báo nào.")

time.sleep(0.5)
st.rerun()