#metric_mangager.py
import threading
import time
import psutil
import torch
from collections import defaultdict


class MetricsManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self._data_lock = threading.Lock()

        self._fps = {}
        self._yolo_infer_ms = {}
        self._resnet_infer_ms = {}
        self._intrusion_count = {}
        self._camera_status = {}

        self._history = []

        self._cleanup_interval = 3600
        self._last_cleanup = time.time()

    def update_fps(self, cam_name: str, fps: float):
        with self._data_lock:
            self._fps[cam_name] = fps

    def update_yolo_infer(self, cam_name: str, ms: float):
        with self._data_lock:
            self._yolo_infer_ms[cam_name] = ms

    def update_resnet_infer(self, cam_name: str, ms: float):
        with self._data_lock:
            self._resnet_infer_ms[cam_name] = ms

    def update_intrusion_count(self, cam_name: str, count: int):
        with self._data_lock:
            self._intrusion_count[cam_name] = count

    def update_camera_status(self, cam_name: str, status: str):
        with self._data_lock:
            self._camera_status[cam_name] = status

    def push_history(self, cam_name: str, fps: float, yolo_ms: float, resnet_ms: float):
        now = time.time()
        with self._data_lock:
            self._history.append((now, cam_name, fps, yolo_ms, resnet_ms))

            if len(self._history) > 500:
                self._history = self._history[-500:]

            if now - self._last_cleanup > self._cleanup_interval:
                cutoff = now - self._cleanup_interval
                self._history = [h for h in self._history if h[0] > cutoff]
                self._last_cleanup = now

    def get_fps(self, cam_name: str) -> float:
        with self._data_lock:
            return self._fps.get(cam_name, 0.0)

    def get_avg_fps(self) -> float:
        with self._data_lock:
            vals = list(self._fps.values())
            return sum(vals) / len(vals) if vals else 0.0

    def get_yolo_infer(self, cam_name: str) -> float:
        with self._data_lock:
            return self._yolo_infer_ms.get(cam_name, 0.0)

    def get_resnet_infer(self, cam_name: str) -> float:
        with self._data_lock:
            return self._resnet_infer_ms.get(cam_name, 0.0)

    def get_avg_latency(self) -> float:
        with self._data_lock:
            totals = []
            for cam in self._yolo_infer_ms:
                y = self._yolo_infer_ms.get(cam, 0.0)
                r = self._resnet_infer_ms.get(cam, 0.0)
                totals.append(y + r)
            return sum(totals) / len(totals) if totals else 0.0

    def get_intrusion_count(self, cam_name: str) -> int:
        with self._data_lock:
            return self._intrusion_count.get(cam_name, 0)

    def get_camera_status(self, cam_name: str) -> str:
        with self._data_lock:
            return self._camera_status.get(cam_name, "Dead")

    def get_all_camera_status(self) -> dict:
        with self._data_lock:
            return dict(self._camera_status)

    @staticmethod
    def get_cpu_usage() -> float:
        return psutil.cpu_percent(interval=0)

    @staticmethod
    def get_gpu_memory_mb() -> float:
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
        return 0.0

    def snapshot(self) -> dict:
        with self._data_lock:
            return {
                "fps": dict(self._fps),
                "yolo_infer_ms": dict(self._yolo_infer_ms),
                "resnet_infer_ms": dict(self._resnet_infer_ms),
                "intrusion_count": dict(self._intrusion_count),
                "camera_status": dict(self._camera_status),
                "cpu_percent": self.get_cpu_usage(),
                "gpu_memory_mb": self.get_gpu_memory_mb(),
            }
