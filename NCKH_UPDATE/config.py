#config.py
ROOT = r"C:\Users\ad\Downloads\codepython\project\NCKH"

CAMERAS = {
    "Cam 1": {
        "video": ROOT + r"\data\te_ngang_010.mp4",
        "door_zone": [[1413,662],[1416,810],[1618,930],[1622,701]],
        "danger_zone": [[1673,1076],[1008,592],[1008,577],[1913,754],[1912,1074]]
    },
    "Cam 2": {
        "video": ROOT + r"\data\vung_cam_001.mp4",
        "door_zone": [[1074,612],[1082,912],[1551,1077],[1759,1076],[1751,685]],
        "danger_zone": [[1193,1076],[208,562],[208,524],[1710,698],[1705,1076]]
    },
    "Cam 3": {
        "video": ROOT + r"\data\back_ground_004.mp4",
        "door_zone": [[1053, 362], [1051, 777], [1539, 1077], [1573, 1077], [1580, 479]],
        "danger_zone": [[1130, 1076], [228, 289], [246, 286], [1549, 1077]]
    }
}
