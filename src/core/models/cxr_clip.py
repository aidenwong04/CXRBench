""" Wrapper to load a CXR-CLIP ResNet50 encoder for linear probing."""

import torch
import torch.nn as nn
import torchvision.models as tv
from pathlib import Path
import pickle
try:
    import pickle5  # type: ignore
except ImportError:
    pickle5 = None


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
        # Support older torchvision (no weights arg). We load weights via checkpoint anyway.
        try:
            backbone = tv.resnet50(weights=None)
        except TypeError:
            backbone = tv.resnet50(pretrained=False)

        # Load checkpoint if provided; tolerate partial state dicts.
        if checkpoint_path:
            ckpt_path = Path(checkpoint_path)
            if ckpt_path.is_file():
                state = None
                errors = []
                loaders = []
                loaders.append(("torch.load(default)", lambda p: torch.load(p, map_location="cpu")))
                if pickle5 is not None:
                    loaders.append(("torch.load(pickle5)", lambda p: torch.load(p, map_location="cpu", pickle_module=pickle5)))
                # Plain pickle as a last resort (often fails on torch checkpoints).
                loaders.append(("pickle.load", lambda p: pickle.load(open(p, "rb"))))

                for name, loader in loaders:
                    try:
                        state = loader(ckpt_path)
                        print(f"[CXRCLIP] Loaded checkpoint with {name}")
                        break
                    except Exception as e:
                        errors.append((name, str(e)))
                        continue

                if state is None:
                    raise RuntimeError(f"Failed to load checkpoint {ckpt_path}. Errors: {errors}")

                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                state = _strip_module_prefix(state)

                # Retain only image encoder weights; map CXR-CLIP prefixes to torchvision resnet keys.
                mapped_state = {}
                for k, v in state.items():
                    if k.startswith("image_encoder.resnet."):
                        new_k = k.replace("image_encoder.resnet.", "")
                        mapped_state[new_k] = v
                    elif k.startswith("image_encoder."):
                        # Skip other image_encoder params we don't use.
                        continue
                    elif k.startswith("text_encoder.") or k.startswith("text_projection.") or k.startswith("image_projection."):
                        continue
                    else:
                        mapped_state[k] = v

                missing, unexpected = backbone.load_state_dict(mapped_state, strict=False)
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
