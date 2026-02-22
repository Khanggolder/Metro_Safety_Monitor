import cv2
from ultralytics import YOLO
import torch
import numpy as np
from shapely.geometry import Point, Polygon

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

class PoseEngine:
    def __init__(self, model_path, danger_polygon):
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

    def update_streak(self, streaks, tid, flag):
        streaks[tid] = streaks.get(tid, 0) + 1 if flag else 0
        return streaks[tid]

    def check_intrusion(self, pt):
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

    def process(self, frame, door_state):
        self.frame_count += 1
        run_yolo = self.frame_count % self.SKIP == 0
        if not run_yolo:
            return frame, False, False, []

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

        is_fall = False
        is_intrude = False
        trigger_ids = []

        if results[0].boxes.id is None:
            return frame, is_fall, is_intrude, []

        boxes = results[0].boxes.xywh.cpu().numpy()
        kpts = results[0].keypoints.xy.cpu().numpy()
        tids = results[0].boxes.id.int().cpu().tolist()

        for box, kp, tid in zip(boxes, kpts, tids):
            person_fall = False
            person_intrude = False
            x, y, w, h = box

            left_ankle = kp[15]
            right_ankle = kp[16]
            feet = ((left_ankle[0] + right_ankle[0]) / 2, (left_ankle[1] + right_ankle[1]) / 2)
            raw_intrude = self.check_intrusion(feet)
            if door_state == 1:
                raw_intrude = False
            intrude_n = self.update_streak(self.intrude_streak, tid, raw_intrude)
            if intrude_n >= self.N_INTRUDE:
                person_intrude = True
                is_intrude = True
                if intrude_n == self.N_INTRUDE:
                    trigger_ids.append(tid)

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
                person_fall = True
                is_fall = True
                if fall_n == self.N_FALL:
                    trigger_ids.append(tid)

            if person_fall:
                color = (0, 255, 255)
                label = f"ID:{tid} FALL"
            elif person_intrude:
                color = (0, 0, 255)
                label = f"ID:{tid} INTRUDE"
            else:
                color = (0, 255, 0)
                label = f"ID:{tid} OK"

            if person_fall or person_intrude:
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

        return frame, is_fall, is_intrude, trigger_ids
