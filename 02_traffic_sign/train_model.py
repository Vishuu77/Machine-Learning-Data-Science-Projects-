# -*- coding: utf-8 -*-
"""
train_model.py
==============
Trains a small, CPU-friendly CNN to classify German traffic signs and
exports the artifacts that app.py consumes.

Dataset
-------
Public GTSRB subset re-hosted on Hugging Face (no login / no auth):

    https://huggingface.co/datasets/tanganke/gtsrb

(label id == official GTSRB class id 0..42; we use a balanced 12-class subset)

Pipeline
--------
1. download / cache  HF dataset split="train"
2. balanced sub-sample  -> 12 iconic classes x 240 images (190 train / 50 val)
3. train 3-conv-layer CNN on CPU (a few epochs)
4. export artifacts in this folder:
      model.pt    TorchScript CNN that returns class probabilities
      meta.json   classes, labels, German meanings, validation accuracy
      samples/    4 demo images (cropped from the validation subset)

Run:
    python train_model.py
"""
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from PIL import Image

from sign_info import CLASS_INFO

# ----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent

DATASET_ID = "tanganke/gtsrb"
DATASET_URL = "https://huggingface.co/datasets/tanganke/gtsrb"

TARGET_CLASSES = [1, 2, 3, 12, 13, 14, 17, 25, 28, 31, 35, 40]
SAMPLE_CLASSES = [14, 25, 28, 40]      # demo images for app.py -> samples/

IMG_SIZE = 48
N_TRAIN = 190                          # train images per class
N_VAL = 50                             # validation images per class
EPOCHS = 10
PATIENCE = 3
BATCH_SIZE = 128
LR = 1e-3
SEED = 42
DEVICE = torch.device("cpu")

SLUG = {  # readable file names for the demo samples
    1: "speed_limit_30", 2: "speed_limit_50", 3: "speed_limit_60",
    12: "priority_road", 13: "yield", 14: "stop", 17: "no_entry",
    25: "road_work", 28: "children_crossing", 31: "wild_animals",
    35: "ahead_only", 40: "roundabout",
}


# ----------------------------------------------------------------------------
class ConvNet(nn.Module):
    """Tiny CNN: 48x48 RGB image -> class logits (seconds/epoch on CPU)."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128 * 6 * 6, 256), nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class ProbNet(nn.Module):
    """Wraps ConvNet so the exported TorchScript directly returns probabilities."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x):
        return torch.softmax(self.net(x), dim=1)


# ----------------------------------------------------------------------------
def to_rgb(im: Image.Image) -> Image.Image:
    return im if im.mode == "RGB" else im.convert("RGB")


def to_array(im: Image.Image) -> np.ndarray:
    """H,W,3 float32 in [0,1] at IMG_SIZE."""
    a = np.asarray(im.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR), dtype=np.float32)
    return a / 255.0


def build_tensors(imgs_map, id2idx):
    X, Y = [], []
    for t, lst in imgs_map.items():
        for im in lst:
            X.append(to_array(im))
            Y.append(id2idx[t])          # compact label 0..C-1
    X = np.stack(X).transpose(0, 3, 1, 2)  # NCHW
    return torch.from_numpy(X), torch.tensor(Y, dtype=torch.long)


def prepare_dataset():
    """Fetch (cached) dataset and pick a balanced train/val subset."""
    print(f"[1/5] Loading HF dataset '{DATASET_ID}' (split='train', cached after first download) ...")
    ds = load_dataset(DATASET_ID, split="train")
    labels = ds["label"]
    n = len(ds)

    per_class = {}
    for t in TARGET_CLASSES:
        idxs = [i for i in range(n) if labels[i] == t]
        need = N_TRAIN + N_VAL
        if len(idxs) < need:
            raise SystemExit(f"class {t} has only {len(idxs)} images, need {need} — pick other classes")
        per_class[t] = idxs

    rng = random.Random(SEED)
    train_imgs, val_imgs = {}, {}
    for t in TARGET_CLASSES:
        pick = rng.sample(per_class[t], N_TRAIN + N_VAL)
        train_idx, val_idx = pick[:N_TRAIN], pick[N_TRAIN:]
        train_imgs[t] = [to_rgb(ds[i]["image"]) for i in train_idx]
        val_imgs[t] = [to_rgb(ds[i]["image"]) for i in val_idx]
        print(f"   class {t:>2}  {CLASS_INFO[t]['en']:<32} train={len(train_idx):>3}  val={len(val_idx):>3}")
    return train_imgs, val_imgs


# ----------------------------------------------------------------------------
def main():
    t_start = time.time()
    train_imgs, val_imgs = prepare_dataset()

    print("[2/5] Building tensors ...")
    id2idx = {c: i for i, c in enumerate(TARGET_CLASSES)}
    Xtr, ytr = build_tensors(train_imgs, id2idx)
    Xva, yva = build_tensors(val_imgs, id2idx)
    print(f"       train {tuple(Xtr.shape)}  val {tuple(Xva.shape)}  ({time.time()-t_start:.0f}s)")

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xtr, ytr), batch_size=BATCH_SIZE, shuffle=True
    )
    model = ConvNet(len(TARGET_CLASSES)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    print(f"[3/5] Training {len(TARGET_CLASSES)} classes on CPU "
          f"(input {IMG_SIZE}x{IMG_SIZE} RGB, max {EPOCHS} epochs) ...")
    best_acc, best_state, bad_epochs = 0.0, None, 0
    for ep in range(1, EPOCHS + 1):
        model.train()
        total_loss, nb = 0.0, 0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            nb += 1
        model.eval()
        with torch.no_grad():
            pred_tr = model(Xtr).argmax(1)
            pred_va = model(Xva).argmax(1)
            acc_tr = (pred_tr == ytr).float().mean().item() * 100.0
            acc_va = (pred_va == yva).float().mean().item() * 100.0
        print(f"   epoch {ep:>2}/{EPOCHS}  loss={total_loss/nb:.4f}  "
              f"train_acc={acc_tr:5.1f}%  val_acc={acc_va:5.1f}%   ({time.time()-t_start:.0f}s)")
        if acc_va > best_acc:
            best_acc = acc_va
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"   early stopping (val_acc did not improve for {PATIENCE} epochs)")
                break

    model.load_state_dict(best_state)
    model.eval()

    # per-class validation accuracy (for the console report)
    with torch.no_grad():
        pred_va = np.array(TARGET_CLASSES)[model(Xva).argmax(1).numpy()]  # compact -> GTSRB id
    yva_np = np.array(TARGET_CLASSES)[yva.numpy()]  # compact -> GTSRB id
    per_class_acc = {}
    for t in TARGET_CLASSES:
        m = yva_np == t
        per_class_acc[str(t)] = round(float((pred_va[m] == t).mean() * 100.0), 2)
    print("[4/5] Validation per-class accuracy:")
    for t in TARGET_CLASSES:
        print(f"       class {t:>2} {CLASS_INFO[t]['en']:<32} {per_class_acc[str(t)]:>6.2f}%")
    print(f"       OVERALL validation accuracy: {best_acc:.2f}%")

    # export TorchScript probability model
    prob_net = ProbNet(model).eval()
    example = torch.rand(1, 3, IMG_SIZE, IMG_SIZE)
    traced = torch.jit.trace(prob_net, example)
    traced.save(str(HERE / "model.pt"))

    meta = {
        "model_file": "model.pt",
        "dataset": DATASET_ID,
        "dataset_url": DATASET_URL,
        "classes": TARGET_CLASSES,
        "class_labels": {str(t): CLASS_INFO[t]["en"] for t in TARGET_CLASSES},
        "german": {str(t): CLASS_INFO[t]["de"] for t in TARGET_CLASSES},
        "meaning": {str(t): CLASS_INFO[t]["note"] for t in TARGET_CLASSES},
        "img_size": IMG_SIZE,
        "val_acc_pct": round(best_acc, 2),
        "per_class_acc": per_class_acc,
        "n_train": int(len(Xtr)),
        "n_val": int(len(Xva)),
        "epochs_run": ep,
        "seed": SEED,
        "device": "cpu",
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (HERE / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), "utf-8")

    # demo images for the Streamlit app (from the validation subset)
    samples_dir = HERE / "samples"
    samples_dir.mkdir(exist_ok=True)
    for p in samples_dir.glob("*.png"):
        p.unlink()
    sample_files = []
    for t in SAMPLE_CLASSES:
        im = val_imgs[t][0]
        fn = samples_dir / f"sample_{SLUG.get(t, str(t))}_{t}.png"
        im.save(fn)
        sample_files.append(fn.name)
    meta["sample_files"] = sample_files
    (HERE / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), "utf-8")

    # reload check
    chk = torch.jit.load(str(HERE / "model.pt"), map_location="cpu")
    with torch.no_grad():
        out = chk(example)
    assert tuple(out.shape) == (1, len(TARGET_CLASSES)), "reload sanity check failed"
    assert float(out.sum()) > 0.99, "softmax sanity check failed"

    print(f"[5/5] Done in {time.time()-t_start:.0f}s. Artifacts in {HERE}:")
    print("       model.pt  (TorchScript CNN -> softmax probabilities)")
    print("       meta.json (classes, meanings, accuracy)")
    for f in sample_files:
        print(f"       samples/{f}")
    print(f"\nBest validation accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
