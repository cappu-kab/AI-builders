"""
inference.py
============
Minimal inference helpers for AdvancedUNetSE.

Public API:
    model, ckpt = load_model(ckpt_path, device)
    enhanced    = enhance_waveform(model, noisy_np, device, sr=16000)
"""
from __future__ import annotations

from typing import Tuple, Dict, Any
from pathlib import Path

import numpy as np
import torch

from model_unet_advanced import AdvancedUNetSE


def _model_kwargs_from_args(saved_args: Dict[str, Any]) -> Dict[str, Any]:
    keys = ("n_fft", "hop_length", "win_length", "base_channels",
            "depth", "mask_bound", "sample_rate")
    out = {k: saved_args[k] for k in keys if k in saved_args}
    if "no_wide_bottleneck" in saved_args:
        out["wide_bottleneck"] = not saved_args["no_wide_bottleneck"]
    return out


def load_model(ckpt_path: str | Path, device: torch.device,
               set_inference: bool = True) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    ckpt = torch.load(str(ckpt_path), map_location=device)
    saved_args = ckpt.get("args", {}) or {}
    kwargs = _model_kwargs_from_args(saved_args)
    model = AdvancedUNetSE(**kwargs).to(device)
    state = ckpt["model"] if "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[inference] {len(missing)} missing keys (init from scratch).")
    if unexpected:
        print(f"[inference] {len(unexpected)} unexpected keys ignored.")
    if set_inference:
        model.train(False)
    return model, ckpt


@torch.inference_mode()
def enhance_waveform(model: torch.nn.Module, noisy: np.ndarray,
                     device: torch.device, sr: int = 16_000) -> np.ndarray:
    if noisy.ndim > 1:
        noisy = noisy.mean(axis=0)
    x = torch.from_numpy(noisy.astype(np.float32)).unsqueeze(0).to(device)
    y = model(x)
    return y.squeeze(0).detach().cpu().numpy().astype(np.float32)
