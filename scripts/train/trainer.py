import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Tuple, List
from sklearn.metrics import classification_report, confusion_matrix


@torch.no_grad()
def predict_logits(model, loader, device):
    model.eval()
    chunks = []
    for xb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        chunks.append(model(xb).cpu().numpy())
    return np.vstack(chunks)


def evaluate(model, loader, device, classes: List[str]):
    logits = predict_logits(model, loader, device)
    preds = logits.argmax(1)

    ys = []
    for _, yb in loader:
        ys.append(np.asarray(yb))
    y_true = np.concatenate(ys)

    acc = float((preds == y_true).mean())
    report = classification_report(
        y_true, preds, target_names=classes, output_dict=True)
    cm = confusion_matrix(y_true, preds).tolist()
    return acc, report, cm


def train_epoch(model, loader, optimizer, criterion, device, use_amp, accum_steps):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    optimizer.zero_grad(set_to_none=True)

    running = 0
    for step, (xb, yb) in enumerate(loader, 1):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(xb)
            loss = criterion(logits, yb) / accum_steps

        scaler.scale(loss).backward()

        if step % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running += loss.item()

    return running / max(1, len(loader))
