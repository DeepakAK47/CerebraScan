"""Inference script for CerebraScan."""

from pathlib import Path

import torch
import numpy as np

from models.unet import build_model
from src.utils import dice_coef, iou_score


def load_trained_model(weights_path, device):
    """Build the model and load a state dict from ``weights_path``."""
    weights_path = Path(weights_path)
    if not weights_path.is_file():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    model = build_model(encoder_weights=None).to(device)
    checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("The model checkpoint must contain a state dictionary.")
    model.load_state_dict(checkpoint)
    model.eval()
    return model


def predict_single(image_tensor, model, device, threshold=0.5):
    """image_tensor: (3, 256, 256) — tensor."""
    if image_tensor.ndim != 3:
        raise ValueError("Expected an image tensor with shape (channels, height, width).")
    if not 0.0 < threshold < 1.0:
        raise ValueError("Prediction threshold must be between 0 and 1.")

    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0).to(device))
        probabilities = torch.sigmoid(output).squeeze().cpu().numpy()
    return (probabilities >= threshold).astype(np.uint8), probabilities


def evaluate_test_set(model, test_loader, device):
    dices, ious = [], []
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            dices.append(dice_coef(outputs, masks).item())
            ious.append(iou_score(outputs, masks).item())
    return np.mean(dices), np.mean(ious)