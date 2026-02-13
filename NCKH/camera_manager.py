import cv2
from config import CAMERAS

class CameraSystem:
    def __init__(self, cam_name):
        self.cfg = CAMERAS[cam_name]
        self.cap = cv2.VideoCapture(self.cfg["video"])

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()

    @property
    def door_zone(self):
        return self.cfg["door_zone"]

    @property
    def danger_zone(self):
        return self.cfg["danger_zone"]