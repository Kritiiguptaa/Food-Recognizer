# ===============================================================
# South Indian Food Classifier
# EfficientNet-B3 
# ===============================================================

import os
import copy
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

from torch.amp import autocast, GradScaler
import matplotlib.pyplot as plt



ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data" / "SouthIndian_Split_Data"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True, parents=True)


BATCH_SIZE = 16                 # fits 4GB VRAM
IMG_SIZE = 256
HEAD_EPOCHS = 4
FT_EPOCHS = 40
LR_HEAD = 2e-3
LR_FT = 2e-4
WD = 1e-4
LABEL_SMOOTH = 0.05
CUTMIX_PROB = 0.10              # VERY small, safe
ALPHA = 0.8                     # soft blending
PATIENCE = 10
SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def set_seed(seed=SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy_top1(out, y):
    _, pred = out.max(1)
    correct = (pred == y).sum().item()
    return correct / y.size(0)


def apply_cutmix(x, y):
    if np.random.rand() > CUTMIX_PROB:
        return x, y, None, None, False

    lam = np.random.beta(ALPHA, ALPHA)
    B, C, H, W = x.size()
    idx = torch.randperm(B, device=x.device)

    cx, cy = np.random.randint(W), np.random.randint(H)
    cut_w, cut_h = int(W * np.sqrt(1 - lam)), int(H * np.sqrt(1 - lam))

    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]

    lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
    return x, y, y[idx], lam, True


# ---------------- DATA ----------------

def get_transforms():
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.20, 0.20, 0.20),
        transforms.ToTensor(),
        normalize,
    ])

    val_tfms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])

    return train_tfms, val_tfms


def get_loaders():
    train_tfms, val_tfms = get_transforms()

    train_data = datasets.ImageFolder(DATA_ROOT / "train", train_tfms)
    val_data = datasets.ImageFolder(DATA_ROOT / "val", val_tfms)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    return train_data, train_loader, val_loader


# ---------------- MODEL ----------------

def build_model(num_classes):
    weights = EfficientNet_B3_Weights.IMAGENET1K_V1
    model = efficientnet_b3(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


# ---------------- TRAIN LOOPS ----------------

def train_one_epoch(model, loader, opt, criterion, scaler):
    model.train()
    total_correct = 0
    total_samples = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        x_cm, y1, y2, lam, used = apply_cutmix(x, y)

        opt.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=True):
            out = model(x_cm)
            if used:
                loss = lam * criterion(out, y1) + (1 - lam) * criterion(out, y2)
            else:
                loss = criterion(out, y)

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        total_correct += (out.argmax(1) == y).sum().item()
        total_samples += y.size(0)

    return total_correct / total_samples


def validate(model, loader):
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            total_correct += (out.argmax(1) == y).sum().item()
            total_samples += y.size(0)

    return total_correct / total_samples


# ---------------- MAIN TRAIN ----------------

def train():
    set_seed()

    train_data, train_loader, val_loader = get_loaders()
    num_classes = len(train_data.classes)

    with open(MODEL_DIR / "classes.txt", "w") as f:
        f.write("\n".join(train_data.classes))

    model = build_model(num_classes).to(device)
    best_model = copy.deepcopy(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    scaler = GradScaler()

    # ---------- PHASE 1: HEAD ONLY ----------
    print("\n🚀 Phase 1: Warmup head")
    for p in model.features.parameters():
        p.requires_grad = False

    opt = optim.AdamW(model.classifier.parameters(), lr=LR_HEAD, weight_decay=WD)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=HEAD_EPOCHS)

    best_val = 0
    no_improve = 0
    train_hist, val_hist = [], []

    for e in range(1, HEAD_EPOCHS + 1):
        tr_acc = train_one_epoch(model, train_loader, opt, criterion, scaler)
        val_acc = validate(model, val_loader)

        train_hist.append(tr_acc)
        val_hist.append(val_acc)

        print(f"[Head {e}/{HEAD_EPOCHS}] Train={tr_acc:.3f} | Val={val_acc:.3f}")
        scheduler.step()

        if val_acc > best_val:
            best_val = val_acc
            best_model = copy.deepcopy(model)
            no_improve = 0
            torch.save(best_model.state_dict(), MODEL_DIR / "best_b3.pth")
        else:
            no_improve += 1

    # ---------- PHASE 2: FULL TRAIN ----------
    print("\n🚀 Phase 2: Fine-tuning all layers")
    for p in model.features.parameters():
        p.requires_grad = True

    opt = optim.AdamW(model.parameters(), lr=LR_FT, weight_decay=WD)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=FT_EPOCHS)

    for e in range(1, FT_EPOCHS + 1):
        tr_acc = train_one_epoch(model, train_loader, opt, criterion, scaler)
        val_acc = validate(model, val_loader)

        train_hist.append(tr_acc)
        val_hist.append(val_acc)

        print(f"[FT {e}/{FT_EPOCHS}] Train={tr_acc:.3f} | Val={val_acc:.3f}")
        scheduler.step()

        if val_acc > best_val:
            best_val = val_acc
            best_model = copy.deepcopy(model)
            no_improve = 0
            torch.save(best_model.state_dict(), MODEL_DIR / "best_b3.pth")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"⏹ Early stopping patience = {PATIENCE}")
                break

    print("\n🏆 BEST VAL ACCURACY:", best_val)

    # Plot
    plt.plot(train_hist, label="Train")
    plt.plot(val_hist, label="Val")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    train()















# # ============================================
# # South Indian Food Classifier - Simplified
# # EfficientNet-B2, 224x224, no CutMix/EMA
# # ============================================

# import os
# import copy
# from pathlib import Path

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms
# from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
# from torch.amp import GradScaler, autocast
# import matplotlib.pyplot as plt

# # ---------- Paths (match your structure) ----------
# ROOT = Path(__file__).resolve().parent
# DATA_ROOT = ROOT / "data" / "SouthIndian_Split_Data"
# MODEL_DIR = ROOT / "models"
# MODEL_DIR.mkdir(parents=True, exist_ok=True)

# # ---------- Config ----------
# BATCH_SIZE = 32
# EPOCHS = 50
# LR = 3e-4
# WD = 1e-4
# LABEL_SMOOTH = 0.05
# PATIENCE = 10
# SEED = 42

# IMG_SIZE = 224

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# # ---------- Utils ----------
# def set_seed(seed=SEED):
#     import random
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)


# def accuracy_topk(output, target, topk=(1,)):
#     with torch.no_grad():
#         maxk = max(topk)
#         batch = target.size(0)
#         _, pred = output.topk(maxk, 1, True, True)
#         pred = pred.t()
#         correct = pred.eq(target.view(1, -1).expand_as(pred))
#         res = []
#         for k in topk:
#             correct_k = correct[:k].reshape(-1).float().sum(0)
#             res.append(float(correct_k / batch))
#         return res


# # ---------- Data ----------
# def get_transforms():
#     normalize = transforms.Normalize([0.485, 0.456, 0.406],
#                                      [0.229, 0.224, 0.225])

#     train_tfms = transforms.Compose([
#         transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomRotation(15),
#         transforms.ColorJitter(0.25, 0.25, 0.25),
#         transforms.ToTensor(),
#         normalize,
#     ])

#     val_tfms = transforms.Compose([
#         transforms.Resize((IMG_SIZE, IMG_SIZE)),
#         transforms.ToTensor(),
#         normalize,
#     ])

#     return train_tfms, val_tfms


# def get_loaders():
#     train_tfms, val_tfms = get_transforms()

#     train_dir = DATA_ROOT / "train"
#     val_dir = DATA_ROOT / "val"

#     train_data = datasets.ImageFolder(train_dir, transform=train_tfms)
#     val_data = datasets.ImageFolder(val_dir, transform=val_tfms)

#     train_loader = DataLoader(train_data, batch_size=BATCH_SIZE,
#                               shuffle=True, num_workers=0)
#     val_loader = DataLoader(val_data, batch_size=BATCH_SIZE,
#                             shuffle=False, num_workers=0)

#     return train_data, train_loader, val_loader


# # ---------- Model ----------
# def build_model(num_classes):
#     weights = EfficientNet_B2_Weights.IMAGENET1K_V1
#     model = efficientnet_b2(weights=weights)
#     in_features = model.classifier[1].in_features
#     model.classifier[1] = nn.Linear(in_features, num_classes)
#     return model


# # ---------- Train / Val loops ----------
# def train_one_epoch(model, loader, optimizer, criterion, scaler):
#     model.train()
#     total, correct1 = 0, 0

#     for x, y in loader:
#         x, y = x.to(device), y.to(device)
#         optimizer.zero_grad(set_to_none=True)

#         use_amp = (device.type == "cuda")
#         with autocast(device_type=device.type, enabled=use_amp):
#             out = model(x)
#             loss = criterion(out, y)

#         if use_amp:
#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
#         else:
#             loss.backward()
#             optimizer.step()

#         acc1 = accuracy_topk(out, y, (1,))[0]
#         correct1 += acc1 * x.size(0)
#         total += x.size(0)

#     return correct1 / total


# def validate(model, loader, criterion):
#     model.eval()
#     total, correct1 = 0, 0

#     with torch.no_grad():
#         for x, y in loader:
#             x, y = x.to(device), y.to(device)
#             out = model(x)
#             acc1 = accuracy_topk(out, y, (1,))[0]
#             correct1 += acc1 * x.size(0)
#             total += x.size(0)

#     return correct1 / total


# # ---------- Main train ----------
# def train():
#     set_seed()

#     print("📌 Loading data from:", DATA_ROOT)
#     train_data, train_loader, val_loader = get_loaders()
#     num_classes = len(train_data.classes)
#     print("📚 Classes:", num_classes)

#     # also save class names for webapp
#     with open(MODEL_DIR / "classes.txt", "w", encoding="utf-8") as f:
#         f.write("\n".join(train_data.classes))

#     model = build_model(num_classes).to(device)
#     best_model = copy.deepcopy(model)

#     criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
#     optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
#     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
#     scaler = GradScaler(enabled=(device.type == "cuda"))

#     best_val = 0.0
#     no_improve = 0
#     train_hist, val_hist = [], []

#     for epoch in range(1, EPOCHS + 1):
#         tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, scaler)
#         val_acc = validate(model, val_loader, criterion)
#         scheduler.step()

#         train_hist.append(tr_acc)
#         val_hist.append(val_acc)

#         print(f"Epoch {epoch:02d}/{EPOCHS} | "
#               f"Train: {tr_acc:.3f} | Val: {val_acc:.3f}")

#         if val_acc > best_val:
#             best_val = val_acc
#             best_model = copy.deepcopy(model)
#             no_improve = 0
#             torch.save(best_model.state_dict(), MODEL_DIR / "best_efficientnet_b2_simple.pth")
#         else:
#             no_improve += 1
#             if no_improve >= PATIENCE:
#                 print(f"⏹ Early stopping (no improvement for {PATIENCE} epochs)")
#                 break

#     print(f"\n🏆 Best Val Accuracy: {best_val:.3f}")
#     print(f"💾 Saved to: {MODEL_DIR / 'best_efficientnet_b2_simple.pth'}")

#     # Plot curve
#     plt.figure()
#     plt.plot(train_hist, label="Train")
#     plt.plot(val_hist, label="Val")
#     plt.xlabel("Epoch")
#     plt.ylabel("Accuracy")
#     plt.title("Training Curve")
#     plt.legend()
#     plt.tight_layout()
#     plt.show()


# if __name__ == "__main__":
#     train()
