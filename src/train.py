"""
src/train.py

Property condition classification training module:
- Pretrained vision backbone (ResNet18 / EfficientNet-B0)
- Three experimental arms:
  Arm A: Clean data only
  Arm B: Clean + Classical augmentation
  Arm C: Clean + Classical + Verified Generative augmentation
- Stratified 5-Fold Cross Validation across multiple seeds (42, 123, 999)
- Designed to run seamlessly either locally or on Kaggle GPU
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
import pandas as pd

CLASS_TO_IDX = {"GOOD": 0, "FAIR": 1, "POOR": 2}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}

class PropertyConditionDataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]], transform=None):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img_path = rec["filepath"]
        label = rec["condition_label"]
        with Image.open(img_path) as img:
            img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)
        target = CLASS_TO_IDX[label]
        return img, target

def get_transforms():
    train_tf = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    eval_tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_tf, eval_tf

def create_model(num_classes: int = 3, pretrained: bool = True) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    # Fine-tune the classifier head (and layer4)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )
    return model

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for imgs, targets in loader:
        imgs, targets = imgs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(targets)
        preds = outputs.argmax(dim=-1)
        correct += (preds == targets).sum().item()
        total += len(targets)
    return total_loss / max(1, total), correct / max(1, total)

def evaluate_model(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())
    return np.array(all_preds), np.array(all_targets)

