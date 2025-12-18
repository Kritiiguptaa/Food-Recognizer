
import os
import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import matplotlib.pyplot as plt
import numpy as np
from torch.amp import GradScaler, autocast


DATA_ROOT = os.path.join("data", "Punjabi_Split_Data")   # must contain train/val/test
BATCH_SIZE = 32
NUM_WORKERS = 4                    # set to 0 if Windows complains
HEAD_EPOCHS = 5
FT_EPOCHS = 25
WD = 1e-4                          # AdamW weight decay
LR_HEAD = 3e-3                     # head-only phase
LR_FT_FEATS = 3e-4                 # features during fine-tuning
LR_FT_HEAD  = 8e-4                 # head during fine-tuning
UNFREEZE_LAST_N_BLOCKS = 3         # EfficientNet blocks to unfreeze in phase 2
USE_SAMPLER = False                # set True only if classes are imbalanced
LABEL_SMOOTH = 0.1
PATIENCE = 7                       # early-stopping patience
SEED = 42


def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def count_per_class(dataset, num_classes):
    counts = torch.zeros(num_classes, dtype=torch.long)
    for _, y in dataset.samples:
        counts[y] += 1
    return counts

def build_loaders(data_root, device, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, use_sampler=USE_SAMPLER):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
        normalize,
        transforms.RandomErasing(p=0.20)
    ])
    test_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    train_dir = os.path.join(data_root, "train")
    val_dir   = os.path.join(data_root, "val")
    test_dir  = os.path.join(data_root, "test")

    train_data = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_data   = datasets.ImageFolder(val_dir,   transform=test_tfms)
    test_data  = datasets.ImageFolder(test_dir,  transform=test_tfms)

    num_classes = len(train_data.classes)

    if use_sampler:
        counts = count_per_class(train_data, num_classes).float().clamp(min=1)
        weights = 1.0 / counts
        sample_weights = [weights[y].item() for _, y in train_data.samples]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    pin = (device.type == "cuda")
    persistent = (num_workers > 0) and pin

    train_loader = DataLoader(train_data, batch_size=batch_size,
                              shuffle=shuffle, sampler=sampler,
                              num_workers=num_workers, pin_memory=pin,
                              persistent_workers=persistent)
    val_loader   = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin,
                              persistent_workers=persistent)
    test_loader  = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin,
                              persistent_workers=persistent)
    return train_data, val_data, test_data, train_loader, val_loader, test_loader, num_classes

def create_model(num_classes):
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_feats = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_feats, num_classes)
    return model

def freeze_all_features(model):
    for p in model.features.parameters():
        p.requires_grad = False

def unfreeze_last_n_blocks(model, n=UNFREEZE_LAST_N_BLOCKS):
    freeze_all_features(model)
    blocks = list(model.features.children())
    for b in blocks[-n:]:
        for p in b.parameters():
            p.requires_grad = True

def accuracy_topk(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append((correct_k.item() / batch_size))
        return res


def run_epoch(model, loader, optimizer, criterion, device, scaler=None, train=True):
    if train: model.train()
    else: model.eval()

    running_loss, total, correct1, correct5 = 0.0, 0, 0.0, 0.0

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        use_amp = (scaler is not None and device.type == "cuda")
        with autocast(device_type=device.type, enabled=use_amp):
            out = model(x)
            loss = criterion(out, y)

        if train:
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * x.size(0)
        acc1, acc5 = accuracy_topk(out, y, topk=(1, 5))
        total += x.size(0)
        correct1 += acc1 * x.size(0)
        correct5 += acc5 * x.size(0)

    epoch_loss = running_loss / total
    top1 = correct1 / total
    top5 = correct5 / total
    return epoch_loss, top1, top5

def validate_with_tta(model, loader, device, criterion, tta=5):
    """Lightweight TTA: horizontal flip augmentation on tensors."""
    model.eval()
    total, correct1, correct5, running_loss = 0, 0.0, 0.0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = 0
            for i in range(tta):
                if i % 2 == 1:
                    x_aug = torch.flip(x, dims=[3])  # horizontal flip
                else:
                    x_aug = x
                logits = logits + model(x_aug)
            logits = logits / float(tta)
            loss = criterion(logits, y)
            acc1, acc5 = accuracy_topk(logits, y, topk=(1,5))
            running_loss += loss.item() * x.size(0)
            total += x.size(0)
            correct1 += acc1 * x.size(0)
            correct5 += acc5 * x.size(0)
    return running_loss/total, correct1/total, correct5/total

def train(eval_test=False, tta=1):
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🟢 Using device: {device}")

    Path("models").mkdir(parents=True, exist_ok=True)

    train_data, val_data, test_data, train_loader, val_loader, test_loader, num_classes = \
        build_loaders(DATA_ROOT, device)
    print(f"📚 Number of classes: {num_classes}")

    # save class names for inference
    with open("models/classes.txt","w",encoding="utf-8") as f:
        f.write("\n".join(train_data.classes))

    model = create_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

    # ---------------- Phase 1: head-only ----------------
    freeze_all_features(model)
    model.features.eval()  # keep BN stable

    opt = optim.AdamW(model.classifier.parameters(), lr=LR_HEAD, weight_decay=WD)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=HEAD_EPOCHS)
    scaler = GradScaler(device if device.type == "cuda" else "cpu",
                        enabled=(device.type == "cuda"))

    train_hist, val_hist = [], []
    best_val_top1 = 0.0
    best_path = "models/best_efficientnet_b0.pth"
    epochs_no_improve = 0

    print("\n=== Phase 1: Head warm-up ===")
    for e in range(1, HEAD_EPOCHS + 1):
        tr_loss, tr_top1, tr_top5 = run_epoch(model, train_loader, opt, criterion, device, scaler, train=True)
        va_loss, va_top1, va_top5 = run_epoch(model, val_loader,   opt, criterion, device, scaler=None, train=False)
        sched.step()
        train_hist.append(tr_top1); val_hist.append(va_top1)

        if va_top1 > best_val_top1:
            best_val_top1 = va_top1; epochs_no_improve = 0
            torch.save(model.state_dict(), best_path)
        else:
            epochs_no_improve += 1

        print(f"[Head] Epoch {e}/{HEAD_EPOCHS} | "
              f"train Top1 {tr_top1:.3f} Top5 {tr_top5:.3f} | "
              f"val Top1 {va_top1:.3f} Top5 {va_top5:.3f}")

    # ---------------- Phase 2: fine-tune last N blocks ----------------
    print("\n=== Phase 2: Fine-tune last blocks ===")
    unfreeze_last_n_blocks(model, UNFREEZE_LAST_N_BLOCKS)

    opt = optim.AdamW([
        {"params": filter(lambda p: p.requires_grad, model.features.parameters()), "lr": LR_FT_FEATS},
        {"params": model.classifier.parameters(), "lr": LR_FT_HEAD},
    ], weight_decay=WD)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=FT_EPOCHS)

    for e in range(1, FT_EPOCHS + 1):
        tr_loss, tr_top1, tr_top5 = run_epoch(model, train_loader, opt, criterion, device, scaler, train=True)
        va_loss, va_top1, va_top5 = run_epoch(model, val_loader,   opt, criterion, device, scaler=None, train=False)
        sched.step()
        train_hist.append(tr_top1); val_hist.append(va_top1)

        if va_top1 > best_val_top1:
            best_val_top1 = va_top1; epochs_no_improve = 0
            torch.save(model.state_dict(), best_path)
            improved = " (↑ best)"
        else:
            epochs_no_improve += 1
            improved = ""

        print(f"[FT  ] Epoch {e}/{FT_EPOCHS} | "
              f"train Top1 {tr_top1:.3f} Top5 {tr_top5:.3f} | "
              f"val Top1 {va_top1:.3f} Top5 {va_top5:.3f}{improved}")

        if epochs_no_improve >= PATIENCE:
            print(f"⏹️ Early stopping triggered (no improvement for {PATIENCE} epochs).")
            break

    print(f"\n✅ Best val Top-1: {best_val_top1:.3f} (saved to {best_path})")

    # Plot training curve
    plt.figure()
    plt.plot(train_hist, label="Train Acc")
    plt.plot(val_hist, label="Val Acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Training Progress")
    plt.legend(); plt.tight_layout(); plt.show()

    # Optional: evaluate test with TTA
    if eval_test:
        print("\n=== Test evaluation ===")
        model.load_state_dict(torch.load(best_path, map_location=device))
        test_loss, test_top1, test_top5 = validate_with_tta(model, test_loader, device, criterion, tta=max(1, tta))
        print(f"🧪 Test Top-1: {test_top1:.3f} | Top-5: {test_top5:.3f}")

        # Confusion matrix (requires scikit-learn)
        try:
            from sklearn.metrics import confusion_matrix, classification_report
            model.eval()
            y_true, y_pred = [], []
            with torch.no_grad():
                for x, y in test_loader:
                    x = x.to(device)
                    out = model(x)
                    y_true.extend(y.numpy().tolist())
                    y_pred.extend(out.argmax(1).cpu().numpy().tolist())
            cm = confusion_matrix(y_true, y_pred)
            np.savetxt("models/confusion_matrix.csv", cm, fmt="%d", delimiter=",")
            print("📄 Saved models/confusion_matrix.csv")
            print(classification_report(y_true, y_pred, target_names=test_data.classes))
        except Exception as e:
            print("ℹ️ Skipped classification report (install scikit-learn to enable).", e)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-test", action="store_true", help="Evaluate on test split after training")
    p.add_argument("--tta", type=int, default=1, help="Num TTA passes for test eval (>=1)")
    return p.parse_args()

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)  # Windows-safe
    args = parse_args()
    train(eval_test=args.eval_test, tta=args.tta)






