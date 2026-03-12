#background_engine.py
import threading
import time
import cv2
import os
import numpy as np
from datetime import datetime

from config import CAMERAS, ROOT
from camera_manager import CameraSystem
from pose_engine import PoseEngine
from door_engine import DoorEngine
from metrics_manager import MetricsManager
from db_manager import DBManager

ALERT_ROOT = os.path.join(ROOT, "alerts")
os.makedirs(os.path.join(ALERT_ROOT, "falls"), exist_ok=True)
os.makedirs(os.path.join(ALERT_ROOT, "intrusions"), exist_ok=True)

DOOR_SKIP = 35
ALERT_COOLDOWN = 3
STATS_LOG_INTERVAL = 30


class CameraWorker:
    def __init__(self, cam_name: str, yolo_path: str, resnet_path: str,
                 metrics: MetricsManager, db: DBManager):
        self.cam_name = cam_name
        self.cfg = CAMERAS[cam_name]
        self.metrics = metrics
        self.db = db

        self.pose = PoseEngine(yolo_path, self.cfg["danger_zone"])
        self.door = DoorEngine(resnet_path, self.cfg["door_zone"])

        self.door_zone = self.cfg["door_zone"]
        self.danger_zone = self.cfg["danger_zone"]

        self._running = False
        self._thread = None
        self.last_alert_time = 0
        self.latest_frame = None
        self._frame_lock = threading.Lock()

        self.latest_alarm = None
        self.latest_alarm_ts = 0.0
        self._alarm_lock = threading.Lock()

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name=f"cam-{self.cam_name}")
        self._thread.start()

    def stop(self):
        self._running = False

    def get_latest_frame(self):
        with self._frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def get_latest_alarm(self):
        with self._alarm_lock:
            return self.latest_alarm, self.latest_alarm_ts

    def _loop(self):
        cam = CameraSystem(self.cam_name)
        self.metrics.update_camera_status(self.cam_name, "Live")

        frame_count = 0
        door_state = 0
        last_stats_log = time.time()
        current_fps = 0.0
        current_yolo_ms = 0.0
        current_resnet_ms = 0.0

        while self._running:
            t0 = time.time()
            ret, frame = cam.read()
            if not ret:
                self.metrics.update_camera_status(self.cam_name, "Reconnecting")
                cam.release()
                time.sleep(2)
                cam = CameraSystem(self.cam_name)
                continue

            frame_count += 1

            if frame_count % DOOR_SKIP == 0:
                t_r = time.time()
                door_state = self.door.predict(frame, self.door_zone)
                current_resnet_ms = (time.time() - t_r) * 1000
                self.metrics.update_resnet_infer(self.cam_name, current_resnet_ms)

            t_y = time.time()
            processed_frame, any_fall, any_intrude, \
                fall_trigger_ids, intrude_trigger_ids, active_tids, extra = \
                self.pose.process(frame, door_state)
            pose_elapsed = (time.time() - t_y) * 1000

            if self.pose.frame_count % self.pose.SKIP == 0:
                current_yolo_ms = pose_elapsed
                self.metrics.update_yolo_infer(self.cam_name, current_yolo_ms)

            alarm = None
            if fall_trigger_ids or intrude_trigger_ids:
                alarm = "HIGH"
            elif extra.get("prefall_trigger_ids"):
                alarm = "PREFALL"
            if alarm is not None:
                with self._alarm_lock:
                    self.latest_alarm = alarm
                    self.latest_alarm_ts = time.time()

            door_pts = np.array(self.door_zone, np.int32).reshape((-1, 1, 2))
            danger_pts = np.array(self.danger_zone, np.int32).reshape((-1, 1, 2))
            door_color = (0, 255, 0) if door_state == 1 else (0, 0, 255)
            cv2.polylines(processed_frame, [door_pts], True, door_color, 2)
            cv2.polylines(processed_frame, [danger_pts], True, (0, 0, 255), 2)
            text = "DOOR OPEN" if door_state == 1 else "DOOR CLOSE"
            cv2.putText(processed_frame, text,
                        (door_pts[0][0][0], door_pts[0][0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, door_color, 2)

            current_time = time.time()
            if (any_fall or any_intrude) and \
                    (current_time - self.last_alert_time > ALERT_COOLDOWN):
                for tid in fall_trigger_ids:
                    img_path = self._save_alert(processed_frame, "falls", tid)
                    self.db.insert_alert("fall", self.cam_name, img_path)
                for tid in intrude_trigger_ids:
                    img_path = self._save_alert(processed_frame, "intrusions", tid)
                    self.db.insert_alert("intrusion", self.cam_name, img_path)
                self.last_alert_time = current_time

            intrude_count = sum(1 for v in self.pose.intrude_streak.values()
                                if v >= self.pose.N_INTRUDE)
            self.metrics.update_intrusion_count(self.cam_name, intrude_count)

            with self._frame_lock:
                self.latest_frame = processed_frame

            elapsed = time.time() - t0
            current_fps = 1.0 / elapsed if elapsed > 0 else 0.0
            self.metrics.update_fps(self.cam_name, current_fps)

            if time.time() - last_stats_log > STATS_LOG_INTERVAL:
                self.db.insert_stats(self.cam_name, current_fps,
                                     current_yolo_ms, current_resnet_ms)
                self.metrics.push_history(self.cam_name, current_fps,
                                          current_yolo_ms, current_resnet_ms)
                last_stats_log = time.time()

        cam.release()
        self.metrics.update_camera_status(self.cam_name, "Dead")

    @staticmethod
    def _save_alert(frame, folder_name, track_id=None):
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ID_{track_id}_{time_str}.jpg" if track_id else f"{time_str}.jpg"
        path = os.path.join(ALERT_ROOT, folder_name, filename)
        cv2.imwrite(path, frame)
        return path

_engine_instance = None
_engine_lock = threading.Lock()


class BackgroundEngine:

    def __init__(self):
        self._started = False
        self.workers = {}
        self.metrics = MetricsManager()
        self.db = DBManager()

        self.yolo_path = os.path.join(ROOT, "yolo26n-pose.pt")
        self.resnet_path = os.path.join(ROOT,"best_model_v1.pth")

    def start(self):
        if self._started:
            return
        self._started = True

        for cam_name in CAMERAS:
            worker = CameraWorker(
                cam_name=cam_name,
                yolo_path=self.yolo_path,
                resnet_path=self.resnet_path,
                metrics=self.metrics,
                db=self.db,
            )
            self.workers[cam_name] = worker
            worker.start()

    def stop(self):
        for w in self.workers.values():
            w.stop()
        self._started = False

    @property
    def is_running(self):
        return self._started


def get_engine() -> BackgroundEngine:
    global _engine_instance
    with _engine_lock:
        if _engine_instance is None:
            _engine_instance = BackgroundEngine()
        if not _engine_instance.is_running:
            _engine_instance.start()
        return _engine_instance
