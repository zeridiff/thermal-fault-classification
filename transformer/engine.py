"""
engine.py
=========
The training loop, the evaluation loop, and early stopping.
Kept separate from train.py so both the CNN script and the feature-extraction
script can reuse it.
"""

import copy
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

import config


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_optimizer(model, lr_backbone=None, lr_head=None):
    lr_backbone = config.LR_BACKBONE if lr_backbone is None else lr_backbone
    lr_head = config.LR_HEAD if lr_head is None else lr_head
    groups = model.param_groups(lr_backbone, lr_head)
    if config.OPTIMIZER.lower() == "sgd":
        return torch.optim.SGD(groups, momentum=config.MOMENTUM,
                               weight_decay=config.WEIGHT_DECAY, nesterov=True)
    return torch.optim.AdamW(groups, weight_decay=config.WEIGHT_DECAY)


# --------------------------------------------------------------------------- #
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return running_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Returns (loss, accuracy, y_true, y_pred, y_prob)."""
    model.eval()
    running_loss, total = 0.0, 0
    ys, preds, probs = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        if criterion is not None:
            running_loss += criterion(out, y).item() * x.size(0)
        p = torch.softmax(out, dim=1)
        ys.append(y.cpu().numpy())
        preds.append(out.argmax(1).cpu().numpy())
        probs.append(p.cpu().numpy())
        total += x.size(0)

    y_true = np.concatenate(ys)
    y_pred = np.concatenate(preds)
    y_prob = np.concatenate(probs)
    acc = float((y_true == y_pred).mean()) if total else 0.0
    return running_loss / max(total, 1), acc, y_true, y_pred, y_prob


# --------------------------------------------------------------------------- #
def fit(model, train_loader, val_loader, device, class_weight=None,
        epochs=None, verbose=True, tag=""):
    """
    Train with early stopping on validation macro-F1.

    macro-F1 rather than accuracy is the stopping criterion because classes are
    imbalanced (22 to 40 images); accuracy would let the model coast on the
    larger severity classes.

    Returns (best_model, history dict).
    """
    epochs = epochs or config.EPOCHS
    criterion = nn.CrossEntropyLoss(
        weight=None if class_weight is None else class_weight.to(device),
        label_smoothing=config.LABEL_SMOOTHING,
    )
    optimizer = make_optimizer(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_f1, best_state, patience = -1.0, copy.deepcopy(model.state_dict()), 0
    t0 = time.time()

    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc, y_true, y_pred, _ = evaluate(model, val_loader, criterion, device)
        va_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["val_f1"].append(va_f1)

        improved = va_f1 > best_f1 + 1e-4
        if improved:
            best_f1, patience = va_f1, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience += 1

        if verbose:
            print(f"  {tag}epoch {ep:>3}/{epochs} | "
                  f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
                  f"val loss {va_loss:.4f} acc {va_acc:.3f} f1 {va_f1:.3f}"
                  f"{'  <-- best' if improved else ''}")

        if patience >= config.EARLY_STOP_PATIENCE:
            if verbose:
                print(f"  {tag}early stop at epoch {ep} (no val F1 gain for "
                      f"{config.EARLY_STOP_PATIENCE} epochs)")
            break

    model.load_state_dict(best_state)
    history["train_time_s"] = time.time() - t0
    history["best_val_f1"] = best_f1
    return model, history
