#!/usr/bin/env python3
"""
ANC Phase 2 — Phase-Alignment Evaluator
========================================
A prediction that is 180° out-of-phase with the actual future noise will
*amplify* the engine sound when fed to the ANC cancellation path (constructive
interference instead of destructive).  This script audits the predictions
from predictor.py and reports whether each backend is SAFE to use.

Metrics
-------
  • Time-domain MSE
  • Normalized cross-correlation peak  → exact sample lag
  • Phase shift at the dominant frequency (degrees, wrapped to (-180, 180])
  • SAFE / MARGINAL / DANGEROUS verdict

Decision rule
-------------
  SAFE        : corr >= +0.85  AND  |phase| <= 30°
  DANGEROUS   : corr <= -0.30  OR   |phase| >= 135°
  MARGINAL    : everything in between

Usage
-----
  python predictor_phase_eval.py        # loads tcn_predictor.pt if present,
                                        # else trains a fresh TCN on the fly
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.signal import correlate

from predictor import (
    SAMPLE_RATE, INPUT_LEN, PREDICT_LEN, LPC_ORDER, DEVICE,
    LightweightTCN, predict_lpc, predict_tcn,
    generate_engine_noise, make_training_pairs, train_tcn,
)


# ══════════════════════════════════════════════════════════════════════════════
# Phase-alignment metric
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_phase(predicted: np.ndarray,
                   actual:    np.ndarray,
                   sr:        int = SAMPLE_RATE) -> dict:
    """
    Compare a single (predicted, actual) future window.  Returns a dict with:
        mse           Time-domain mean-squared error.
        lag_samples   Sample shift of `predicted` that maximises |corr| with
                      `actual`.  Positive = prediction lags; negative = leads.
        lag_ms        Same as above in milliseconds.
        peak_corr     Signed normalised cross-correlation at that lag.
                      ≈ +1 → in-phase, ≈ -1 → anti-phase.
        dom_freq_hz   Dominant spectral component of `actual`.
        phase_deg     Phase shift in degrees at dom_freq, wrapped (-180, 180].
        verdict       'SAFE' / 'MARGINAL' / 'DANGEROUS'.
        rationale     One-line plain-English explanation.
    """
    pred = predicted.astype(np.float64)
    act  = actual.astype(np.float64)
    n    = len(act)
    if len(pred) != n:
        raise ValueError(f"length mismatch: pred={len(pred)} actual={n}")

    # ── Time-domain MSE ──────────────────────────────────────────────────────
    mse = float(np.mean((pred - act) ** 2))

    # ── Normalised cross-correlation (peak |xc| over all integer lags) ───────
    pred_n = (pred - pred.mean()) / (pred.std() + 1e-12)
    act_n  = (act  - act.mean())  / (act.std()  + 1e-12)
    xc     = correlate(act_n, pred_n, mode='full') / n
    lags   = np.arange(-n + 1, n)
    peak_idx  = int(np.argmax(np.abs(xc)))
    lag       = int(lags[peak_idx])
    peak_corr = float(xc[peak_idx])

    # ── Dominant frequency of the actual signal (excluding DC) ───────────────
    win    = np.hanning(n)
    spec   = np.fft.rfft(act * win)
    spec[0] = 0.0                            # ignore DC
    freqs  = np.fft.rfftfreq(n, 1.0 / sr)
    dom_idx  = int(np.argmax(np.abs(spec)))
    dom_freq = float(freqs[dom_idx])

    # ── Phase shift at dominant frequency, wrapped to (-180, 180] ────────────
    if dom_freq > 0:
        cycle_smp = sr / dom_freq            # samples per full cycle
        phase_deg = (lag / cycle_smp) * 360.0
        phase_deg = ((phase_deg + 180.0) % 360.0) - 180.0
    else:
        phase_deg = 0.0

    # ── Verdict ──────────────────────────────────────────────────────────────
    abs_phase = abs(phase_deg)
    if peak_corr >= 0.85 and abs_phase <= 30.0:
        verdict   = "SAFE"
        rationale = ("strong positive correlation with near-zero phase shift — "
                     "ANC will subtract destructively.")
    elif peak_corr <= -0.30 or abs_phase >= 135.0:
        verdict   = "DANGEROUS"
        rationale = ("prediction is roughly INVERTED relative to actual — "
                     "feeding this to the cancellation path would AMPLIFY noise.")
    else:
        verdict   = "MARGINAL"
        rationale = ("partial phase alignment — usable but ANC suppression "
                     "will be degraded; lengthen history or retrain.")

    return dict(
        mse         = mse,
        lag_samples = lag,
        lag_ms      = lag / sr * 1000.0,
        peak_corr   = peak_corr,
        dom_freq_hz = dom_freq,
        phase_deg   = phase_deg,
        verdict     = verdict,
        rationale   = rationale,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Reporting + plotting
# ══════════════════════════════════════════════════════════════════════════════

def print_phase_report(name: str, r: dict) -> None:
    bar = "-" * 60
    print(f"\n  {name}")
    print(f"  {bar}")
    print(f"    Time-domain MSE        : {r['mse']:.6f}")
    print(f"    Peak cross-correlation : {r['peak_corr']:+.3f}"
          f"   ({'in-phase' if r['peak_corr'] > 0 else 'anti-phase'})")
    print(f"    Sample lag at peak     : {r['lag_samples']:+d} smp"
          f"   ({r['lag_ms']:+.2f} ms)")
    print(f"    Dominant frequency     : {r['dom_freq_hz']:7.1f} Hz")
    print(f"    Phase shift @ f_dom    : {r['phase_deg']:+7.1f}°")
    print(f"    VERDICT                : {r['verdict']}")
    print(f"    {r['rationale']}")


def _verdict_color(v: str) -> str:
    return {'SAFE': 'green', 'MARGINAL': 'darkorange', 'DANGEROUS': 'red'}[v]


def plot_phase_alignment(actual:   np.ndarray,
                         pred_lpc: np.ndarray,
                         pred_tcn: np.ndarray,
                         r_lpc:    dict,
                         r_tcn:    dict,
                         save_path: Path | None = None) -> None:
    """Two-panel overlay so phase agreement is visually unmistakable."""
    n = len(actual)
    t = np.arange(n) / SAMPLE_RATE * 1000.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11.0, 6.8), sharex=True)

    # ── LPC ──────────────────────────────────────────────────────────────────
    ax1.plot(t, actual,   color='black',    lw=2.0,            label='Actual future')
    ax1.plot(t, pred_lpc, color='tab:blue', lw=1.4, ls='--',   label='LPC prediction')
    ax1.fill_between(t, actual, pred_lpc, color='tab:blue', alpha=0.10)
    ax1.set_ylabel('Amplitude')
    ax1.set_title(
        f"LPC  —  lag {r_lpc['lag_samples']:+d} smp   "
        f"corr {r_lpc['peak_corr']:+.3f}   "
        f"phase {r_lpc['phase_deg']:+.0f}°   →   {r_lpc['verdict']}",
        color=_verdict_color(r_lpc['verdict']),
    )
    ax1.legend(loc='upper right'); ax1.grid(alpha=0.3)

    # ── TCN ──────────────────────────────────────────────────────────────────
    ax2.plot(t, actual,   color='black',   lw=2.0,            label='Actual future')
    ax2.plot(t, pred_tcn, color='tab:red', lw=1.4, ls='-.',   label='TCN prediction')
    ax2.fill_between(t, actual, pred_tcn, color='tab:red', alpha=0.10)
    ax2.set_xlabel('Time within prediction window (ms)')
    ax2.set_ylabel('Amplitude')
    ax2.set_title(
        f"TCN  —  lag {r_tcn['lag_samples']:+d} smp   "
        f"corr {r_tcn['peak_corr']:+.3f}   "
        f"phase {r_tcn['phase_deg']:+.0f}°   →   {r_tcn['verdict']}",
        color=_verdict_color(r_tcn['verdict']),
    )
    ax2.legend(loc='upper right'); ax2.grid(alpha=0.3)

    fig.suptitle(
        f"ANC predictor — phase-alignment audit "
        f"({PREDICT_LEN} smp / {PREDICT_LEN / SAMPLE_RATE * 1000:.0f} ms ahead)",
        fontsize=12,
    )
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=120)
        print(f"  saved plot -> {save_path}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# TCN bootstrap (reuse predictor.py's checkpoint, or train a fresh one)
# ══════════════════════════════════════════════════════════════════════════════

def _load_or_train_tcn() -> LightweightTCN:
    ckpt  = Path(__file__).resolve().parent / "tcn_predictor.pt"
    model = LightweightTCN().to(DEVICE)
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        model.eval()
        print(f"  loaded TCN checkpoint  -> {ckpt}")
    else:
        print("  no TCN checkpoint found — training a fresh one (~30 epochs) ...")
        train_sig = generate_engine_noise(duration_s=30.0)
        X, Y      = make_training_pairs(train_sig, hop=64)
        train_tcn(model, X, Y, epochs=30, verbose=False)
        torch.save(model.state_dict(), ckpt)
        print(f"  trained + saved        -> {ckpt}")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 72)
    print("ANC predictor  |  phase-alignment audit")
    print(f"window: {PREDICT_LEN} smp ({PREDICT_LEN / SAMPLE_RATE * 1000:.0f} ms)"
          f"   |   SR: {SAMPLE_RATE} Hz   |   device: {DEVICE}")
    print("=" * 72)

    model = _load_or_train_tcn()

    # Independent test signal (different seed than predictor.py uses)
    test_sig   = generate_engine_noise(duration_s=5.0, seed=999)
    eval_start = INPUT_LEN
    history    = test_sig[eval_start - INPUT_LEN : eval_start]
    actual     = test_sig[eval_start : eval_start + PREDICT_LEN]

    pred_lpc = predict_lpc(history)
    pred_tcn = predict_tcn(model, history)

    r_lpc = evaluate_phase(pred_lpc, actual)
    r_tcn = evaluate_phase(pred_tcn, actual)

    print_phase_report(f"LPC predictor (order={LPC_ORDER})", r_lpc)
    print_phase_report("Lightweight Causal TCN",             r_tcn)

    print("\n  Decision rule")
    print("  " + "-" * 60)
    print("    SAFE      : corr >= +0.85  AND  |phase| <= 30°")
    print("    DANGEROUS : corr <= -0.30   OR  |phase| >= 135°")
    print("    MARGINAL  : everything in between")

    # ── Final at-a-glance summary ────────────────────────────────────────────
    print("\n  Bottom line")
    print("  " + "-" * 60)
    for name, r in (("LPC", r_lpc), ("TCN", r_tcn)):
        tag = {"SAFE": "[OK]    ", "MARGINAL": "[WARN]  ", "DANGEROUS": "[STOP]  "}[r['verdict']]
        print(f"    {tag}{name:<6s}  →  {r['verdict']}   "
              f"(corr={r['peak_corr']:+.2f}, phase={r['phase_deg']:+.0f}°)")

    plot_phase_alignment(actual, pred_lpc, pred_tcn, r_lpc, r_tcn)


if __name__ == "__main__":
    main()
