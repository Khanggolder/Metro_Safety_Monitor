#door_engine.py
import cv2
import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
import numpy as np

device = "cuda:0" if torch.cuda.is_available() else "cpu"

DOOR_OPEN = 1
DOOR_CLOSE = 0

class DoorEngine:
    def __init__(self, model_path, polygon):
        self.polygon = polygon
        self.model = models.resnet18(weights=None)
        num_feats = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_feats, 2)
        )
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()
        torch.backends.cudnn.benchmark = True
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def crop_polygon(self, frame, polygon=None):
        poly = polygon if polygon is not None else self.polygon
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        pts = np.array(poly, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
        res = cv2.bitwise_and(frame, frame, mask=mask)
        x, y, w, h = cv2.boundingRect(pts)
        crop = res[y:y+h, x:x+w]
        return crop

    def predict(self, frame, polygon=None):
        crop = self.crop_polygon(frame, polygon)
        if crop is None or crop.size == 0:
            return DOOR_CLOSE

        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = self.transform(crop).unsqueeze(0).to(device)
        with torch.no_grad():
            out = self.model(tensor)
            pred = out.argmax(1).item()
        del tensor, out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return pred