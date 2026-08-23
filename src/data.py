from __future__ import annotations

from pathlib import Path
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def make_transform(size: int, train: bool):
    ops = [transforms.Resize((size, size))]
    if train:
        ops += [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15), transforms.ColorJitter(.1, .1, .1, .03)]
    ops += [transforms.ToTensor(),
            transforms.Normalize([0.485, .456, .406], [.229, .224, .225])]
    return transforms.Compose(ops)


class PatchDataset(Dataset):
    def __init__(self, csv_or_frame, patch_size=224, train=False, image_root=None):
        self.df = pd.read_csv(csv_or_frame) if not isinstance(csv_or_frame, pd.DataFrame) else csv_or_frame.reset_index(drop=True)
        self.patch_size = int(patch_size)
        self.transform = make_transform(self.patch_size, train)
        self.image_root = Path(image_root) if image_root else None
        needed = {"uuid", "patient_id", "image_path", "x", "y", "label"}
        if missing := needed - set(self.df):
            raise ValueError(f"Missing columns: {sorted(missing)}")

    def __len__(self): return len(self.df)

    def resolve(self, raw):
        p = Path(str(raw))
        return self.image_root / p.name if self.image_root else p

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        path = self.resolve(r.image_path)
        if not path.exists():
            raise FileNotFoundError(f"Missing image: {path}")
        with Image.open(path) as im:
            im = im.convert("RGB")
            x, y = int(r.x), int(r.y)
            patch = im.crop((x, y, x+self.patch_size, y+self.patch_size))
        return self.transform(patch), int(r.label), idx

