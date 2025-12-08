""" Linear probing script for CXR-CLIP ResNet50-M on MIMIC-CXR (multi-label). """

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_score, recall_score, accuracy_score)
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Allow imports from src/core
REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.append(str(REPO_ROOT / "src"))
from core.models.cxr_clip import CXRCLIPLinearProbe  # noqa: E402


def load_config(path):
    import yaml
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


class MultiLabelImageDataset(Dataset):
    """Simple multi-label dataset reader driven by a metadata CSV."""
    def __init__(self, df, image_root, path_column, label_columns, transform):
        self.df = df.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.path_column = path_column
        self.label_columns = label_columns
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.image_root / row[self.path_column]
        image = Image.open(img_path).convert("RGB")
        x = self.transform(image)
        y = torch.tensor(row[self.label_columns].values.astype(np.float32))
        return x, y


def build_transforms(model_cfg):
    return transforms.Compose([
        transforms.Resize(model_cfg["image_size"]),
        transforms.CenterCrop(model_cfg["image_size"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=model_cfg["mean"], std=model_cfg["std"]),
    ])


def build_dataloaders(cfg):
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

    transform = build_transforms(model_cfg)

    # Support either a single CSV with split column, or per-split CSVs in a dict.
    metadata = data_cfg["metadata_csv"]
    if isinstance(metadata, dict):
        def load_split(name):
            if name not in metadata:
                raise ValueError(f"metadata_csv missing split '{name}'")
            return pd.read_csv(metadata[name])
        train_df = load_split("train")
        val_df = load_split("val")
        test_df = load_split("test")
    else:
        df = pd.read_csv(metadata)
        split_col = data_cfg.get("split_column")
        split_map = data_cfg.get("splits", {})

        def subset(split_key):
            split_value = split_map.get(split_key, split_key)
            if split_col is None or split_col not in df.columns:
                raise ValueError("split_column missing; provide per-split metadata_csv or a split column.")
            return df[df[split_col] == split_value]

        train_df = subset("train")
        val_df = subset("val")
        test_df = subset("test")

    label_cols = data_cfg["label_columns"]

    train_ds = MultiLabelImageDataset(train_df, data_cfg["image_root"],
                                      data_cfg["path_column"], label_cols, transform)
    val_ds = MultiLabelImageDataset(val_df, data_cfg["image_root"],
                                    data_cfg["path_column"], label_cols, transform)
    test_ds = MultiLabelImageDataset(test_df, data_cfg["image_root"],
                                     data_cfg["path_column"], label_cols, transform)

    def make_loader(ds, shuffle):
        return DataLoader(
            ds,
            batch_size=cfg["training"]["batch_size"],
            shuffle=shuffle,
            num_workers=data_cfg["num_workers"],
            pin_memory=True,
        )

    return make_loader(train_ds, True), make_loader(val_ds, False), make_loader(test_ds, False), len(label_cols)


def evaluate(model, loader, device):
    model.eval()
    all_targets, all_probs = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(y.cpu().numpy())

    y_true = np.concatenate(all_targets, axis=0)
    y_score = np.concatenate(all_probs, axis=0)
    y_pred = (y_score >= 0.5).astype(np.float32)

    metrics = {}

    # Macro metrics over labels; guard against degenerate cases.
    try:
        metrics["macro_auroc"] = roc_auc_score(y_true, y_score, average="macro")
    except Exception:
        metrics["macro_auroc"] = float("nan")

    try:
        metrics["macro_auprc"] = average_precision_score(y_true, y_score, average="macro")
    except Exception:
        metrics["macro_auprc"] = float("nan")

    metrics["macro_precision"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["macro_recall"] = recall_score(y_true, y_pred, average="macro", zero_division=0)

    # Per-label accuracy then macro average.
    per_label_acc = (y_pred == y_true).mean(axis=0)
    metrics["macro_accuracy"] = float(per_label_acc.mean())

    # Subset accuracy (exact match) for reference.
    metrics["subset_accuracy"] = accuracy_score(y_true, y_pred)

    return metrics


def train(cfg):
    os.makedirs(cfg["logging"]["output_dir"], exist_ok=True)

    train_loader, val_loader, test_loader, num_classes = build_dataloaders(cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CXRCLIPLinearProbe(
        num_classes=num_classes,
        checkpoint_path=cfg["model"]["checkpoint_path"],
        freeze_encoder=cfg["model"]["freeze_encoder"],
        dropout=cfg["model"]["classifier_dropout"],
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )

    scaler = torch.cuda.amp.GradScaler(enabled=cfg["training"].get("use_fp16", False))

    best_val = -float("inf")
    ckpt_path = Path(cfg["logging"]["output_dir"]) / "model.best.pt"
    log_path = Path(cfg["logging"]["output_dir"]) / "metrics.jsonl"

    global_step = 0
    for epoch in range(cfg["training"]["num_epochs"]):
        model.train()
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=cfg["training"].get("use_fp16", False)):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            if global_step % cfg["logging"]["log_every"] == 0:
                print(f"Epoch {epoch} Batch {batch_idx} Loss {loss.item():.4f}")

        if (epoch + 1) % cfg["logging"]["eval_every"] == 0:
            val_metrics = evaluate(model, val_loader, device)
            test_metrics = evaluate(model, test_loader, device)

            record = {
                "epoch": epoch,
                "val": val_metrics,
                "test": test_metrics,
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(record) + "\n")

            print(f"[Eval] Epoch {epoch} Val AUROC {val_metrics['macro_auroc']:.3f} "
                  f"AUPRC {val_metrics['macro_auprc']:.3f}")

            # Track best by macro AUROC.
            if val_metrics["macro_auroc"] > best_val:
                best_val = val_metrics["macro_auroc"]
                if cfg["logging"]["save_ckpt"]:
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state": model.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "val_metrics": val_metrics,
                        },
                        ckpt_path,
                    )
                    print(f"[Checkpoint] Saved best model to {ckpt_path}")


def main():
    parser = argparse.ArgumentParser(description="Linear probing CXR-CLIP on MIMIC-CXR (multi-label)")
    parser.add_argument("--config", type=str,
                        default="tasks/classification/linear_probing/MIMIC-CXR/CXR-CLIP/config.yaml",
                        help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
