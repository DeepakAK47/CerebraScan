import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

def build_pairs(patient_list, base):
    pairs = []
    for p in patient_list:
        p_path = os.path.join(base, p)
        for f in sorted(os.listdir(p_path)):
            if f.endswith(".tif") and "_mask" not in f:
                img_path  = os.path.join(p_path, f)
                mask_path = img_path.replace(".tif", "_mask.tif")
                pairs.append((img_path, mask_path))
    return pairs

def get_transforms(img_size=256):
    train_tf = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    val_tf = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    return train_tf, val_tf

class BrainMRIDataset(Dataset):
    def __init__(self, pairs, transform=None):
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        image = np.array(Image.open(img_path).convert("RGB"))
        mask  = np.array(Image.open(mask_path).convert("L"))
        mask  = (mask > 127).astype(np.float32)
        if self.transform:
            aug = self.transform(image=image, mask=mask)
            image = aug["image"]
            mask  = aug["mask"].unsqueeze(0)
        return image, mask
