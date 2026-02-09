from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image

class MyDataset(Dataset):
    def __init__(self, root_dir, transform=None, class_to_idx=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = []
        self.labels = []

        IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

        if class_to_idx is None:
            self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
            self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        else:
            self.class_to_idx = class_to_idx
            self.classes = list(class_to_idx.keys())

        for cls, idx in self.class_to_idx.items():
            cls_dir = self.root_dir / cls
            for img_path in cls_dir.iterdir():
                if img_path.suffix.lower() in IMAGE_EXTS:
                    self.image_paths.append(img_path)
                    self.labels.append(idx)

        print(f"[{self.root_dir.name}] images:", len(self.image_paths))
        print(f"[{self.root_dir.name}] class_to_idx:", self.class_to_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label
