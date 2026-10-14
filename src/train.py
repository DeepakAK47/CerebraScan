"""
Training script for CerebraScan (ResU-Net brain tumor segmentation).

Usage:
    python -m src.train

Works on both GPU (Kaggle) and CPU (local).
"""

# NOTE: Training was done on Kaggle.
# This script is provided for reproducibility.
# To train locally, place dataset at data/raw/kaggle_3m/


import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# Local imports
from models.unet import build_model
from src.utils import CombinedLoss, dice_coef, iou_score
from src.dataset import build_pairs, get_transforms, BrainMRIDataset


# ---------- CONFIG ----------
BASE_DATA_PATH  = "data/raw/kaggle_3m"      # change for Kaggle  
SPLITS_DIR      = "data/splits"
WEIGHTS_DIR     = "models/weights"
BATCH_SIZE      = 16
EPOCHS          = 30
LR              = 1e-4
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
# ----------------------------


def load_split(path):
    with open(path) as f:
        return f.read().splitlines()


def build_loaders():
    train_patients = load_split(os.path.join(SPLITS_DIR, "train.txt"))
    val_patients   = load_split(os.path.join(SPLITS_DIR, "val.txt"))

    train_pairs = build_pairs(train_patients, BASE_DATA_PATH)
    val_pairs   = build_pairs(val_patients,   BASE_DATA_PATH)

    train_tf, val_tf = get_transforms()
    train_ds = BrainMRIDataset(train_pairs, transform=train_tf)
    val_ds   = BrainMRIDataset(val_pairs,   transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum, dice_sum = 0.0, 0.0
    for images, masks in tqdm(loader, desc="Train"):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        dice_sum += dice_coef(outputs, masks).item()
    n = len(loader)
    return loss_sum / n, dice_sum / n


def validate(model, loader, criterion, device):
    model.eval()
    loss_sum, dice_sum, iou_sum = 0.0, 0.0, 0.0
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Val"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss_sum += loss.item()
            dice_sum += dice_coef(outputs, masks).item()
            iou_sum  += iou_score(outputs, masks).item()
    n = len(loader)
    return loss_sum / n, dice_sum / n, iou_sum / n


def main():
    print("Device:", DEVICE)
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    train_loader, val_loader = build_loaders()
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = build_model().to(DEVICE)
    criterion = CombinedLoss().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    best_dice = 0.0
    history = {"train_loss": [], "train_dice": [],
               "val_loss": [], "val_dice": [], "val_iou": []}

    for epoch in range(EPOCHS):
        print(f"\n===== Epoch {epoch+1}/{EPOCHS} =====")

        tr_loss, tr_dice = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        vl_loss, vl_dice, vl_iou = validate(model, val_loader, criterion, DEVICE)
        scheduler.step(vl_dice)

        history["train_loss"].append(tr_loss)
        history["train_dice"].append(tr_dice)
        history["val_loss"].append(vl_loss)
        history["val_dice"].append(vl_dice)
        history["val_iou"].append(vl_iou)

        print(f"Train Loss: {tr_loss:.4f} | Dice: {tr_dice:.4f}")
        print(f"Val   Loss: {vl_loss:.4f} | Dice: {vl_dice:.4f} | IoU: {vl_iou:.4f}")

        if vl_dice > best_dice:
            best_dice = vl_dice
            torch.save(model.state_dict(),
                       os.path.join(WEIGHTS_DIR, "best_model.pth"))
            print(f"✅ Best model saved! Dice: {best_dice:.4f}")

    print(f"\nDone. Best Val Dice: {best_dice:.4f}")

    # Plot curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train")
    plt.plot(history["val_loss"], label="Val")
    plt.title("Loss"); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["train_dice"], label="Train")
    plt.plot(history["val_dice"], label="Val")
    plt.title("Dice"); plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(WEIGHTS_DIR, "curves.png"))
    plt.show()


if __name__ == "__main__":
    main()