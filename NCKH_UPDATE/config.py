#config.py
ROOT = r"C:\Users\ad\Downloads\codepython\project\NCKH"

CAMERAS = {
    "Cam 1": {
        "video": ROOT + r"\data\te_2.mp4",
        "door_zone": [[741, 503], [753, 738], [277, 872], [255, 513]],
        "danger_zone": [[2, 1029], [1402, 584], [1399, 565], [2, 945]]
    },
    "Cam 2": {
        "video": ROOT + r"\data\xam_nhap.mp4",
        "door_zone": [[508, 559], [517, 955], [817, 778], [812, 546]],
        "danger_zone": [[257, 1076], [1201, 556], [1211, 556], [610, 1077]]
    },
    "Cam 3": {
        "video": ROOT + r"\data\back_ground_1.mp4",
        "door_zone": [[933, 532], [920, 791], [1317, 996], [1339, 595]],
        "danger_zone": [[1152, 1074], [453, 574], [514, 574], [514, 574], [1461, 1074], [1461, 1074]]
    }
}
