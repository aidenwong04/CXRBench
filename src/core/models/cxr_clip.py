""" Wrapper to load a CXR-CLIP ResNet50 encoder for linear probing."""

import torch
import torch.nn as nn
import torchvision.models as tv
from pathlib import Path


def _strip_module_prefix(state_dict):
    """Remove 'module.' prefixes if checkpoints were saved via DataParallel."""
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


class CXRCLIPLinearProbe(nn.Module):
    """
    freezes encoder and adds a linear layer, multilabel classifier
    """
    def __init__(self, num_classes, checkpoint_path=None, freeze_encoder=True, dropout=0.0):
        super().__init__()
        backbone = tv.resnet50(weights=None)

        # Load checkpoint if provided; tolerate partial state dicts.
        if checkpoint_path:
            ckpt_path = Path(checkpoint_path)
            if ckpt_path.is_file():
                state = torch.load(ckpt_path, map_location="cpu")
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                state = _strip_module_prefix(state)
                missing, unexpected = backbone.load_state_dict(state, strict=False)
                if missing:
                    print(f"[CXRCLIP] Missing keys when loading checkpoint: {missing}")
                if unexpected:
                    print(f"[CXRCLIP] Unexpected keys when loading checkpoint: {unexpected}")
            else:
                print(f"[CXRCLIP] Checkpoint not found at {ckpt_path}, using random init.")

        # Keep encoder up to global pooling; replace classifier with a linear head.
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.feature_dim = backbone.fc.in_features
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, x):
        feats = self.encoder(x)
        feats = torch.flatten(feats, 1)
        logits = self.classifier(self.dropout(feats))
        return logits
