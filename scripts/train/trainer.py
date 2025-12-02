import torch
import numpy as np
from typing import List
from sklearn.metrics import classification_report, confusion_matrix


# ============================================================
# Forward-only inference
# ============================================================
@torch.no_grad()
def predict_logits(model, loader, device) -> np.ndarray:
    model.eval()
    out = []

    for xb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        out.append(logits.cpu().numpy())

    return np.vstack(out) if out else np.empty((0, 0), dtype=np.float32)


# ============================================================
# Evaluation
# ============================================================
def evaluate(model, loader, device, classes: List[str]):
    logits = predict_logits(model, loader, device)
    preds = logits.argmax(axis=1)

    y_true = np.concatenate([np.asarray(yb) for _, yb in loader])

    acc = float((preds == y_true).mean())

    rep = classification_report(
        y_true, preds, target_names=classes, output_dict=True
    )
    cm = confusion_matrix(y_true, preds).tolist()

    return acc, rep, cm


# ============================================================
# Training Epoch with AMP + Gradient Accumulation
# ============================================================
def train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    use_amp: bool,
    accum_steps: int,
) -> float:
    model.train()

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    tot_loss = 0.0
    steps = 0

    optimizer.zero_grad(set_to_none=True)

    for step, (xb, yb) in enumerate(loader, start=1):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, dtype=torch.long, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(xb)
            loss = criterion(logits, yb)

        loss_scaled = loss / accum_steps
        scaler.scale(loss_scaled).backward()

        tot_loss += loss.item()
        steps += 1

        if step % accum_steps == 0:
            scaler.step(optimizer)
            optimizer.zero_grad(set_to_none=True)
            scaler.update()

    # flush last partial accumulation
    if steps % accum_steps != 0:
        scaler.step(optimizer)
        optimizer.zero_grad(set_to_none=True)
        scaler.update()

    return tot_loss / max(1, steps)
