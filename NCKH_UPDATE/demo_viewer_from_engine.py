# demo_viewer_from_engine.py
import cv2
import time
import os
import numpy as np

from config import CAMERAS, ROOT
from background_engine import get_engine

WINDOW_NAME = "Metro Safety Viewer (Engine)"
cam_names = list(CAMERAS.keys())

AUDIO_DIR = os.path.join(ROOT, "data")
_FALLBACK_ALARM = os.path.join(ROOT, "alarm.mp3")
ALARM_HIGH_FILE = os.path.join(AUDIO_DIR, "alarm_high.mp3")
ALARM_PREFALL_FILE = os.path.join(AUDIO_DIR, "alarm_prefall.mp3")
if not os.path.isfile(ALARM_HIGH_FILE):
    ALARM_HIGH_FILE = _FALLBACK_ALARM
if not os.path.isfile(ALARM_PREFALL_FILE):
    ALARM_PREFALL_FILE = _FALLBACK_ALARM

COOLDOWN_HIGH = 3.0
COOLDOWN_PREFALL = 2.0

_use_pygame = False
_use_winsound = False

try:
    import pygame
    pygame.mixer.init()
    _snd_high = None
    _snd_prefall = None
    if os.path.isfile(ALARM_HIGH_FILE):
        _snd_high = pygame.mixer.Sound(ALARM_HIGH_FILE)
    if os.path.isfile(ALARM_PREFALL_FILE):
        _snd_prefall = pygame.mixer.Sound(ALARM_PREFALL_FILE)
    _use_pygame = True
    print("[audio] pygame mixer OK")
except Exception:
    try:
        import winsound
        _use_winsound = True
        print("[audio] fallback to winsound.Beep")
    except ImportError:
        print("[audio] no audio backend available")


def _play_alarm(level):
    if _use_pygame:
        snd = _snd_high if level == "HIGH" else _snd_prefall
        if snd is not None:
            snd.play()
            return
    if _use_winsound:
        import winsound
        if level == "HIGH":
            winsound.Beep(1000, 500)
        else:
            winsound.Beep(700, 300)

def main():
    engine = get_engine()
    cam_index = 0

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    last_high_ts = 0.0
    last_prefall_ts = 0.0
    last_consumed_alarm_ts = {}

    print(f"[viewer] Started — viewing: {cam_names[cam_index]}")

    while True:
        cam_name = cam_names[cam_index]
        worker = engine.workers.get(cam_name)

        frame = None
        if worker is not None:
            frame = worker.get_latest_frame()

        if frame is None:
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(frame, "Waiting for frames...",
                        (400, 360), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (100, 100, 100), 2)
            cv2.putText(frame, f"Camera: {cam_name}",
                        (400, 410), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (80, 80, 80), 1)

        now = time.time()
        if worker is not None:
            alarm, alarm_ts = worker.get_latest_alarm()
            prev_consumed = last_consumed_alarm_ts.get(cam_name, 0.0)
            if alarm is not None and alarm_ts > prev_consumed:
                last_consumed_alarm_ts[cam_name] = alarm_ts
                if alarm == "HIGH" and (now - last_high_ts) > COOLDOWN_HIGH:
                    _play_alarm("HIGH")
                    last_high_ts = now
                elif alarm == "PREFALL" \
                        and (now - last_prefall_ts) > COOLDOWN_PREFALL \
                        and (now - last_high_ts) > COOLDOWN_HIGH:
                    _play_alarm("PREFALL")
                    last_prefall_ts = now

        fps_engine = engine.metrics.get_fps(cam_name)
        yolo_ms = engine.metrics.get_yolo_infer(cam_name)
        resnet_ms = engine.metrics.get_resnet_infer(cam_name)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (420, 125), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"Camera: {cam_name}", (10, 25),
                    font, 0.6, (0, 255, 255), 1)
        cv2.putText(frame, f"Engine FPS: {fps_engine:.1f}", (10, 50),
                    font, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"YOLO: {yolo_ms:.1f} ms | ResNet: {resnet_ms:.1f} ms",
                    (10, 75), font, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "[1-9] cam | [n/p] next/prev | [q] quit",
                    (10, 100), font, 0.5, (200, 200, 200), 1)

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            cam_index = (cam_index + 1) % len(cam_names)
            print(f"[viewer] Switched to: {cam_names[cam_index]}")
        elif key == ord('p'):
            cam_index = (cam_index - 1) % len(cam_names)
            print(f"[viewer] Switched to: {cam_names[cam_index]}")
        elif ord('1') <= key <= ord('9'):
            idx = key - ord('1')
            if idx < len(cam_names):
                cam_index = idx
                print(f"[viewer] Switched to: {cam_names[cam_index]}")

    cv2.destroyAllWindows()
    print("[viewer] Exited.")


if __name__ == "__main__":
    main()
