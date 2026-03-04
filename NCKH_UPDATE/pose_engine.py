#pose_engine.py
import cv2
import math
import time
from ultralytics import YOLO
import torch
import numpy as np
from shapely.geometry import Point, Polygon

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

RISK_OK       = "OK"
RISK_WARN     = "WARN"
RISK_DANGER   = "DANGER"
RISK_EMERGENCY = "EMERGENCY"

_RISK_THRESHOLDS = (0.30, 0.55, 0.80)

def _risk_level(score):
    if score < _RISK_THRESHOLDS[0]:
        return RISK_OK
    if score < _RISK_THRESHOLDS[1]:
        return RISK_WARN
    if score < _RISK_THRESHOLDS[2]:
        return RISK_DANGER
    return RISK_EMERGENCY

def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


class PoseEngine:
    def __init__(self, model_path, danger_polygon,
                 enable_prefall=True, enable_adaptive_danger=True):
        self.model = YOLO(model_path)
        self.model.to(device)
        self.model.fuse()
        self.danger_polygon = Polygon(danger_polygon)

        self.prev_y_coords = {}
        self.fall_streak = {}
        self.intrude_streak = {}

        self.frame_count = 0
        self.SKIP = 2

        self.N_FALL = 4
        self.N_INTRUDE = 3

        self.NOSE = 0
        self.LEFT_SHOULDER = 5
        self.RIGHT_SHOULDER = 6
        self.LEFT_HIP = 11
        self.RIGHT_HIP = 12

        self.FACE_POINTS = [0, 1, 2, 3, 4]

        self.enable_prefall = enable_prefall
        self.enable_adaptive_danger = enable_adaptive_danger

        self.N_PREFALL = 3
        self.prefall_streak = {}
        self.prev_feet = {}

        self.BAND_PX = 40
        self.SIGMA_DIST = 80.0
        self.V_MARGIN = 3.0
        self.DWELL_DECAY = 0.3
        self._micro_zone = self._build_micro_zone()
        self.prev_boundary_dist = {}
        self.dwell_seconds = {}
        self.last_seen_ts = {}

    def _build_micro_zone(self):
        inner = self.danger_polygon.buffer(-self.BAND_PX)
        if inner.is_empty or not inner.is_valid:
            return self.danger_polygon          # fallback
        return self.danger_polygon.difference(inner)

    def set_scene_polygons(self, danger_polygon):
        self.danger_polygon = Polygon(danger_polygon)
        self._micro_zone = self._build_micro_zone()

    def reset_tracks(self):
        self.prev_y_coords.clear()
        self.fall_streak.clear()
        self.intrude_streak.clear()
        self.prefall_streak.clear()
        self.prev_feet.clear()
        self.prev_boundary_dist.clear()
        self.dwell_seconds.clear()
        self.last_seen_ts.clear()
        self.frame_count = 0

    def update_streak(self, streaks, tid, flag):
        streaks[tid] = streaks.get(tid, 0) + 1 if flag else 0
        return streaks[tid]

    def check_intrusion(self, pt):
        if isinstance(pt, Point):
            return self.danger_polygon.contains(pt)
        return self.danger_polygon.contains(Point(pt))

    def blur_face(self, frame, kpts):
        valid_face_pts = [kpts[i] for i in self.FACE_POINTS if kpts[i][0] > 0 and kpts[i][1] > 0]
        if len(valid_face_pts) < 3:
            return frame
        face_pts = np.array(valid_face_pts, dtype=np.int32)
        x_min, y_min = np.min(face_pts, axis=0)
        x_max, y_max = np.max(face_pts, axis=0)
        expand = 20
        x1 = max(0, x_min - expand)
        y1 = max(0, y_min - expand)
        x2 = min(frame.shape[1], x_max + expand)
        y2 = min(frame.shape[0], y_max + expand)
        face_roi = frame[y1:y2, x1:x2]
        blurred = cv2.GaussianBlur(face_roi, (151, 151), 30)
        frame[y1:y2, x1:x2] = blurred
        return frame

    def _compute_risk(self, tid, feet, now):
        pt = Point(feet)
        d = self.danger_polygon.exterior.distance(pt)

        d_risk = math.exp(-d / self.SIGMA_DIST)

        prev_d = self.prev_boundary_dist.get(tid, d)
        v_risk = 1.0 if (prev_d - d) > self.V_MARGIN else 0.0
        self.prev_boundary_dist[tid] = d

        in_band = self._micro_zone.contains(pt)
        prev_ts = self.last_seen_ts.get(tid, now)
        dt = max(now - prev_ts, 0.0)
        self.last_seen_ts[tid] = now

        cur_dwell = self.dwell_seconds.get(tid, 0.0)
        if in_band:
            cur_dwell += dt
        else:
            cur_dwell = max(0.0, cur_dwell - self.DWELL_DECAY * dt)
        self.dwell_seconds[tid] = cur_dwell
        dwell_risk = _sigmoid((cur_dwell - 1.0) / 0.7)

        prev_ft = self.prev_feet.get(tid, feet)
        dx = feet[0] - prev_ft[0]
        dy = feet[1] - prev_ft[1]
        speed = math.hypot(dx, dy)
        speed_risk = _sigmoid((speed - 8.0) / 6.0)

        risk = 0.40 * d_risk + 0.25 * v_risk + 0.25 * dwell_risk + 0.10 * speed_risk
        return float(np.clip(risk, 0.0, 1.0))

    def process(self, frame, door_state):
        self.frame_count += 1
        run_yolo = self.frame_count % self.SKIP == 0

        _empty_extra = {
            "prefall_trigger_ids": [],
            "prefall_active_ids": set(),
            "risk_score_by_tid": {},
            "risk_level_by_tid": {},
            "danger_trigger_ids": [],
        }

        if not run_yolo:
            return frame, False, False, [], [], set(), _empty_extra

        with torch.no_grad():
            results = self.model.track(
                frame,
                persist=True,
                verbose=False,
                device=device,
                conf=0.3,
                iou=0.6,
                imgsz=512,
                half=(device.startswith("cuda")),
                tracker="bytetrack.yaml"
            )

        any_fall = False
        any_intrude = False
        fall_trigger_ids = []
        intrude_trigger_ids = []
        prefall_trigger_ids = []
        prefall_active_ids = set()
        risk_score_by_tid = {}
        risk_level_by_tid = {}
        danger_trigger_ids = []

        if results[0].boxes.id is None:
            return frame, False, False, [], [], set(), _empty_extra

        boxes = results[0].boxes.xywh.cpu().numpy()
        kpts = results[0].keypoints.xy.cpu().numpy()
        tids = results[0].boxes.id.int().cpu().tolist()

        now = time.time()
        prev_levels = getattr(self, "_prev_risk_level", {})

        for box, kp, tid in zip(boxes, kpts, tids):
            x, y, w, h = box

            is_this_fall = False
            is_this_intrude = False
            is_this_prefall = False

            left_ankle = kp[15]
            right_ankle = kp[16]
            feet = ((left_ankle[0] + right_ankle[0]) / 2,
                    (left_ankle[1] + right_ankle[1]) / 2)

            if door_state == 1:
                raw_intrude = False
            else:
                raw_intrude = self.check_intrusion(feet)

            intrude_n = self.update_streak(self.intrude_streak, tid, raw_intrude)
            if intrude_n >= self.N_INTRUDE:
                is_this_intrude = True
                any_intrude = True
                if intrude_n == self.N_INTRUDE:
                    intrude_trigger_ids.append(tid)

            left_shoulder = kp[self.LEFT_SHOULDER]
            right_shoulder = kp[self.RIGHT_SHOULDER]
            left_hip = kp[self.LEFT_HIP]
            right_hip = kp[self.RIGHT_HIP]
            nose = kp[self.NOSE]

            shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2 \
                if left_shoulder[0] > 0 and right_shoulder[0] > 0 else \
                left_shoulder[1] if left_shoulder[0] > 0 else \
                right_shoulder[1] if right_shoulder[0] > 0 else 0

            hip_y = (left_hip[1] + right_hip[1]) / 2 \
                if left_hip[0] > 0 and right_hip[0] > 0 else \
                left_hip[1] if left_hip[0] > 0 else \
                right_hip[1] if right_hip[0] > 0 else shoulder_y

            nose_y = nose[1] if nose[0] > 0 else hip_y
            current_y = shoulder_y if shoulder_y > 0 else hip_y

            prev_y = self.prev_y_coords.get(tid, current_y)
            velocity_y = current_y - prev_y
            self.prev_y_coords[tid] = current_y

            ratio = w / h if h > 0 else 0

            valid_ys = [p[1] for p in kp if p[0] > 0 and p[1] > 0]
            keypoint_span = (max(valid_ys) - min(valid_ys)) if valid_ys else h
            keypoint_ratio = keypoint_span / h if h > 0 else 1.0

            raw_falling = False
            if velocity_y > 12:
                raw_falling = True
            if (nose_y > hip_y + 50) or (shoulder_y > hip_y + 40):
                raw_falling = True
            if keypoint_ratio < 0.70:
                raw_falling = True
            if ratio > 1.1:
                raw_falling = True
            if (nose_y < hip_y - 50) or (shoulder_y < hip_y - 40):
                raw_falling = False
            if velocity_y < -5:
                raw_falling = False

            fall_n = self.update_streak(self.fall_streak, tid, raw_falling)
            if fall_n >= self.N_FALL:
                is_this_fall = True
                any_fall = True
                if fall_n == self.N_FALL:
                    fall_trigger_ids.append(tid)

            if self.enable_prefall and not is_this_fall:
                prev_ft = self.prev_feet.get(tid, feet)
                vel_x_feet = abs(feet[0] - prev_ft[0])
                vel_y_feet = abs(feet[1] - prev_ft[1])
                speed_feet = math.hypot(vel_x_feet, vel_y_feet)

                has_motion = (abs(velocity_y) + vel_x_feet) > 3.0 or speed_feet > 3.0

                raw_prefall = False
                if has_motion:
                    if velocity_y > 6:
                        raw_prefall = True
                    if (nose_y > hip_y + 20) or (shoulder_y > hip_y + 15):
                        raw_prefall = True
                    if keypoint_ratio < 0.82:
                        raw_prefall = True
                    if ratio > 0.85:
                        raw_prefall = True

                if velocity_y < -3:
                    raw_prefall = False
                if nose_y < hip_y - 20:
                    raw_prefall = False
                if shoulder_y < hip_y - 15:
                    raw_prefall = False

                pf_n = self.update_streak(self.prefall_streak, tid, raw_prefall)
                if pf_n >= self.N_PREFALL:
                    is_this_prefall = True
                    prefall_active_ids.add(tid)
                    if pf_n == self.N_PREFALL:
                        prefall_trigger_ids.append(tid)
            elif not self.enable_prefall:
                pass

            if self.enable_adaptive_danger:
                risk = self._compute_risk(tid, feet, now)
                risk_score_by_tid[tid] = risk
                level = _risk_level(risk)
                risk_level_by_tid[tid] = level
                if level in (RISK_DANGER, RISK_EMERGENCY):
                    prev_risk = prev_levels.get(tid, RISK_OK)
                    if prev_risk not in (RISK_DANGER, RISK_EMERGENCY):
                        danger_trigger_ids.append(tid)

            self.prev_feet[tid] = feet

            if is_this_fall:
                color = (0, 255, 255)
                label = "EMERGENCY: FALL"
            elif is_this_prefall:
                color = (0, 165, 255)
                label = "WARNING: PREFALL"
            elif is_this_intrude:
                color = (0, 0, 255)
                label = "DANGER: INTRUSION"
            else:
                color = (0, 255, 0)
                label = f"ID:{tid} OK"

            if self.enable_adaptive_danger and not is_this_fall and door_state != 1:
                rl = risk_level_by_tid.get(tid, RISK_OK)
                rs = risk_score_by_tid.get(tid, 0.0)
                if rl != RISK_OK:
                    label += f" [{rl} {rs:.2f}]"

            if is_this_fall or is_this_intrude:
                frame = self.blur_face(frame, kp)

            cv2.rectangle(frame,
                          (int(x - w/2), int(y - h/2)),
                          (int(x + w/2), int(y + h/2)),
                          color, 2)
            cv2.putText(frame, label,
                        (int(x - w/2), int(y - h/2) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            skeleton = [(5,6), (5,11), (6,12), (11,12), (11,13), (13,15), (12,14), (14,16)]
            for a, b in skeleton:
                if kp[a][0] > 0 and kp[b][0] > 0:
                    cv2.line(frame, tuple(map(int, kp[a])), tuple(map(int, kp[b])),
                             (0, 255, 255), 2)

        if self.enable_adaptive_danger:
            self._prev_risk_level = dict(risk_level_by_tid)

        active_ids = set(tids)
        stale = [k for k in self.prev_y_coords if k not in active_ids]
        for k in stale:
            del self.prev_y_coords[k]
            self.fall_streak.pop(k, None)
            self.intrude_streak.pop(k, None)
            self.prefall_streak.pop(k, None)
            self.prev_feet.pop(k, None)
            self.prev_boundary_dist.pop(k, None)
            self.dwell_seconds.pop(k, None)
            self.last_seen_ts.pop(k, None)
            if hasattr(self, '_prev_risk_level'):
                self._prev_risk_level.pop(k, None)

        extra = {
            "prefall_trigger_ids": prefall_trigger_ids,
            "prefall_active_ids": prefall_active_ids,
            "risk_score_by_tid": risk_score_by_tid,
            "risk_level_by_tid": risk_level_by_tid,
            "danger_trigger_ids": danger_trigger_ids,
        }

        return frame, any_fall, any_intrude, fall_trigger_ids, intrude_trigger_ids, active_ids, extra
