import torch
import numpy as np
from typing import List, Tuple
from sklearn.metrics import classification_report, confusion_matrix


@torch.no_grad()
def predict_logits(model, loader, device):
    model.eval()
    out = []
    for xb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        out.append(model(xb).cpu().numpy())
    return np.vstack(out)


def evaluate(model, loader, device, classes: List[str]):
    """
    Return acc, sklearn report dict, confusion matrix list[list].
    """
    logits = predict_logits(model, loader, device)
    preds = logits.argmax(axis=1)

    # Reconstruct y_true safely
    ys = [np.asarray(yb) for _, yb in loader]
    y_true = np.concatenate(ys)

    acc = float((preds == y_true).mean())
    rep = classification_report(
        y_true, preds, target_names=classes, output_dict=True
    )
    cm = confusion_matrix(y_true, preds).tolist()

    return acc, rep, cm


def train_epoch(model, loader, optimizer, criterion,
                device, use_amp: bool, accum_steps: int) -> float:
    """
    AMP + gradient accumulation train loop.
    """
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    optimizer.zero_grad(set_to_none=True)

    running = 0
    total_steps = 0

    for step, (xb, yb) in enumerate(loader, start=1):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, dtype=torch.long, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(xb)
            loss = criterion(logits, yb) / accum_steps

        scaler.scale(loss).backward()
        running += loss.item()
        total_steps += 1

        # periodic step
        if step % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

    # flush final remainder
    if total_steps % accum_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return running / max(1, len(loader))
