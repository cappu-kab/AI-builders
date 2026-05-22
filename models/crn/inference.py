"""
inference.py
============
Minimal inference helpers for CRN: load a checkpoint and enhance a single
1-D waveform. Used by evaluate.py and external scripts.

Public API:
    model, ckpt = load_model(ckpt_path, device)
    enhanced    = enhance_waveform(model, noisy_np, device, sr=16000)
"""
from __future__ import annotations

from typing import Tuple, Dict, Any
from pathlib import Path

import numpy as np
import torch

from model import CRN


def _model_kwargs_from_args(saved_args: Dict[str, Any]) -> Dict[str, Any]:
    keys = ("n_fft", "hop_length", "rnn_hidden", "rnn_layers", "rnn_type",
            "bottleneck_dim", "mask_bound", "dropout_p", "sample_rate")
    out = {k: saved_args[k] for k in keys if k in saved_args}
    if "no_se" in saved_args:
        out["use_se"] = not saved_args["no_se"]
    return out


def load_model(ckpt_path: str | Path, device: torch.device,
               set_eval: bool = True) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Load a CRN checkpoint. Reconstructs the model from saved args when possible.

    Backwards-compat: the FreqAttention bottleneck block was added AFTER some
    checkpoints were trained.  We auto-detect its presence in the state_dict
    and construct CRN(use_freq_attn=...) accordingly so old and new
    checkpoints both load with zero missing/unexpected keys.
    """
    ckpt = torch.load(str(ckpt_path), map_location=device)
    saved_args = ckpt.get("args", {}) or {}
    state = ckpt["model"] if "model" in ckpt else ckpt
    kwargs = _model_kwargs_from_args(saved_args)
    # Old checkpoints (pre-FreqAttention) have no `freq_attn.*` keys — turn
    # the block OFF so the architecture exactly matches what was trained.
    kwargs["use_freq_attn"] = any(k.startswith("freq_attn.") for k in state.keys())
    model = CRN(**kwargs).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[inference] {len(missing)} missing keys (init from scratch).")
    if unexpected:
        print(f"[inference] {len(unexpected)} unexpected keys ignored.")
    if set_eval:
        model.eval()
    return model, ckpt


@torch.inference_mode()
def enhance_waveform(model: torch.nn.Module, noisy: np.ndarray,
                     device: torch.device, sr: int = 16_000) -> np.ndarray:
    """Run model on a single 1-D float32 waveform; return 1-D enhanced waveform."""
    if noisy.ndim > 1:
        noisy = noisy.mean(axis=0)
    x = torch.from_numpy(noisy.astype(np.float32)).unsqueeze(0).to(device)
    y = model(x)
    return y.squeeze(0).detach().cpu().numpy().astype(np.float32)
