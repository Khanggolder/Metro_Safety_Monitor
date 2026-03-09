#db_manager.py
import sqlite3
import os
import threading
from datetime import datetime, timedelta

DB_PATH = os.path.join(
    r"D:\Metro_Safety_Monitor\NCKH_UPDATE", "metro_ai.db"
)


class DBManager:
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
        self._conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.row_factory = sqlite3.Row
        self._db_lock = threading.Lock()
        self._create_tables()

    def _create_tables(self):
        with self._db_lock:
            cur = self._conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    type TEXT NOT NULL,
                    camera_name TEXT NOT NULL,
                    image_path TEXT NOT NULL
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS system_stats (
                    timestamp TEXT NOT NULL,
                    camera_name TEXT NOT NULL,
                    fps REAL,
                    yolo_infer_ms REAL,
                    resnet_infer_ms REAL
                )
            """)
            self._conn.commit()


    def insert_alert(self, alert_type: str, camera_name: str, image_path: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._db_lock:
            self._conn.execute(
                "INSERT INTO alerts (timestamp, type, camera_name, image_path) VALUES (?, ?, ?, ?)",
                (ts, alert_type, camera_name, image_path),
            )
            self._conn.commit()

    def insert_stats(self, camera_name: str, fps: float, yolo_ms: float, resnet_ms: float):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._db_lock:
            self._conn.execute(
                "INSERT INTO system_stats (timestamp, camera_name, fps, yolo_infer_ms, resnet_infer_ms) VALUES (?, ?, ?, ?, ?)",
                (ts, camera_name, fps, yolo_ms, resnet_ms),
            )
            self._conn.commit()


    def get_alerts_today(self) -> list:
        today = datetime.now().strftime("%Y-%m-%d")
        with self._db_lock:
            cur = self._conn.execute(
                "SELECT * FROM alerts WHERE timestamp LIKE ? ORDER BY timestamp DESC",
                (f"{today}%",),
            )
            return [dict(r) for r in cur.fetchall()]

    def get_alerts_count_today(self) -> int:
        today = datetime.now().strftime("%Y-%m-%d")
        with self._db_lock:
            cur = self._conn.execute(
                "SELECT COUNT(*) FROM alerts WHERE timestamp LIKE ?",
                (f"{today}%",),
            )
            return cur.fetchone()[0]

    def get_alerts_by_hour_today(self) -> list:
        today = datetime.now().strftime("%Y-%m-%d")
        with self._db_lock:
            cur = self._conn.execute(
                """
                SELECT CAST(SUBSTR(timestamp, 12, 2) AS INTEGER) as hour, COUNT(*) as cnt
                FROM alerts
                WHERE timestamp LIKE ?
                GROUP BY hour
                ORDER BY hour
                """,
                (f"{today}%",),
            )
            return [(r[0], r[1]) for r in cur.fetchall()]

    def get_alerts_by_type_today(self) -> dict:
        today = datetime.now().strftime("%Y-%m-%d")
        with self._db_lock:
            cur = self._conn.execute(
                "SELECT type, COUNT(*) FROM alerts WHERE timestamp LIKE ? GROUP BY type",
                (f"{today}%",),
            )
            return {r[0]: r[1] for r in cur.fetchall()}

    def get_latest_alerts(self, limit: int = 20) -> list:
        with self._db_lock:
            cur = self._conn.execute(
                "SELECT * FROM alerts ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            )
            return [dict(r) for r in cur.fetchall()]

    def get_latest_stats(self, limit: int = 100) -> list:
        with self._db_lock:
            cur = self._conn.execute(
                "SELECT * FROM system_stats ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            )
            return [dict(r) for r in cur.fetchall()]
