"""
losses.py  —  5-term CRN loss.

  Term 1 : SI-SDR (tanh-smoothed, always float32)
  Term 2 : Multi-resolution Smooth-LF compressed MagMSE on SPEECH output
  Term 3 : Multi-resolution STFT Aux — spectral convergence + log-mag L1
           (auxiliary; explicitly preserves speech spectral structure)
  Term 4 : LF-weighted noise prediction MagL1 — direct gradient pressure on
           the BiLSTM bottleneck to correctly identify low-frequency noise
           (penalises noise leakage where it is most audible: <200 Hz)
  Term 5 : LF speech-floor penalty — relu(E_lf[clean] - E_lf[enhanced]) in
           the 80-300 Hz band, protecting voiced speech fundamentals from
           over-suppression while the model pursues hum/rumble elimination

Parts dict always contains:
  sisnr        — tanh-smoothed SI-SDR loss (scalar, for back-prop bookkeeping)
  sisnr_db     — true mean SI-SDR in dB  (detached, for accurate history logging)
  mag          — spectral mag term        (detached)
  aux_stft     — auxiliary SC + log-mag term (detached, for monitoring)
  lf_noise     — LF noise prediction spectral L1 (detached)
  speech_floor — LF speech over-suppression penalty (detached)
"""

from __future__ import annotations
import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

_LF_SPEECH_LO_HZ  = 80.0    # lower edge of voiced fundamentals (bass male voices)
_LF_SPEECH_HI_HZ  = 300.0   # upper edge of voiced fundamentals (female / child voices)


# ---------------------------------------------------------------------------
def _stft_mag(x: torch.Tensor, n_fft: int, hop: int, win: int,
              window: torch.Tensor) -> torch.Tensor:
    spec = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=win,
                      window=window, return_complex=True, center=True)
    return torch.abs(spec) + 1e-7


def _si_sdr(pred: torch.Tensor, target: torch.Tensor):
    """SI-SDR loss + monitoring value.

    Always runs in float32 to avoid bf16/fp16 accumulation errors over
    long sequences (48 000-sample dot-products in 16-bit are ±2 dB noisy).

    Returns:
        loss     — −tanh(SI-SDR/20), differentiable
        si_sdr_b — SI-SDR in dB per batch item (detached, for logging)
    """
    eps = 1e-8
    pred   = pred.float()    # ← force fp32 regardless of AMP context
    target = target.float()

    pred   = pred   - pred.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)

    # Guard near-silent targets: SI-SDR is undefined → clip contribution to 0.
    tgt_pwr = target.pow(2).sum(dim=-1)                        # (B,)
    silent  = tgt_pwr < eps

    dot    = (pred * target).sum(dim=-1, keepdim=True)
    s_tgt  = dot * target / (tgt_pwr.unsqueeze(-1) + eps)
    e_ns   = pred - s_tgt

    si_sdr_b = 10.0 * torch.log10(
        s_tgt.pow(2).sum(dim=-1) / (e_ns.pow(2).sum(dim=-1) + eps) + eps
    )                                                           # (B,)
    si_sdr_b = si_sdr_b.masked_fill(silent, 0.0)              # zero out silent

    loss = -torch.tanh(si_sdr_b / 20.0).mean()
    return loss, si_sdr_b.detach()


# ---------------------------------------------------------------------------
def _build_smooth_lf_weight(n_fft: int, lf_peak: float,
                             lf_cutoff_hz: float, transition_hz: float,
                             sample_rate: int) -> torch.Tensor:
    """Cosine-taper LF weight mask, mean-normalised.

    W(f) = lf_peak                   for f ≤ lf_cutoff_hz
           cosine taper → 1          for lf_cutoff_hz < f ≤ lf_cutoff_hz + transition_hz
           1.0                        above
    """
    F_bins = n_fft // 2 + 1
    freqs  = torch.linspace(0.0, sample_rate / 2.0, F_bins)
    lo, hi = lf_cutoff_hz, lf_cutoff_hz + transition_hz
    t          = ((freqs - lo) / (hi - lo)).clamp(0.0, 1.0)
    cos_factor = 0.5 * (1.0 + torch.cos(math.pi * t))
    in_ramp    = (freqs <= hi).float()
    w = 1.0 + (lf_peak - 1.0) * cos_factor * in_ramp
    return w / w.mean()


# ---------------------------------------------------------------------------
class SmoothLFMagMSELoss(nn.Module):
    def __init__(self, n_fft: int = 512, hop: int = 128, win: int = 512,
                 compress: float = 0.3, lf_peak: float = 5.0,
                 lf_cutoff_hz: float = 200.0, transition_hz: float = 100.0,
                 sample_rate: int = 16_000) -> None:
        super().__init__()
        self.n_fft = n_fft; self.hop = hop; self.win = win
        self.compress = compress; self.lf_peak = lf_peak
        self.lf_cutoff_hz = lf_cutoff_hz; self.transition_hz = transition_hz
        self.sample_rate = sample_rate
        self.register_buffer("window", torch.hann_window(win), persistent=False)
        w = _build_smooth_lf_weight(n_fft, lf_peak, lf_cutoff_hz, transition_hz, sample_rate)
        self.register_buffer("_weight", w.view(-1, 1), persistent=False)

    def rebuild_weight(self, new_peak: float) -> None:
        self.lf_peak = new_peak
        w = _build_smooth_lf_weight(
            self.n_fft, new_peak, self.lf_cutoff_hz,
            self.transition_hz, self.sample_rate)
        self._weight.data.copy_(w.view(-1, 1).to(self._weight.device))

    def forward(self, enhanced: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
        # Always float32 for spectral computations
        m_enh = _stft_mag(enhanced.float(), self.n_fft, self.hop, self.win, self.window)
        m_cln = _stft_mag(clean.float(),    self.n_fft, self.hop, self.win, self.window)
        c_enh = m_enh.pow(self.compress)
        c_cln = m_cln.pow(self.compress)
        return ((c_enh - c_cln).pow(2) * self._weight).mean()


class MultiResSmoothLFMSELoss(nn.Module):
    def __init__(self,
                 fft_sizes:     Sequence[int] = (256, 512, 1024),
                 hop_sizes:     Sequence[int] = (64,  128, 256),
                 win_sizes:     Sequence[int] = (240, 480, 960),
                 compress:      float = 0.3,
                 lf_peak:       float = 5.0,
                 lf_cutoff_hz:  float = 200.0,
                 transition_hz: float = 100.0,
                 sample_rate:   int   = 16_000) -> None:
        super().__init__()
        self.losses = nn.ModuleList([
            SmoothLFMagMSELoss(f, h, w, compress, lf_peak,
                                lf_cutoff_hz, transition_hz, sample_rate)
            for f, h, w in zip(fft_sizes, hop_sizes, win_sizes)
        ])

    def forward(self, enhanced: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
        return sum(L(enhanced, clean) for L in self.losses) / len(self.losses)


# ---------------------------------------------------------------------------
class _STFTAuxLoss(nn.Module):
    """Single-resolution spectral convergence + log-magnitude L1.

    SC  = ||M_clean - M_enhanced||_F / (||M_clean||_F + eps)
    LM  = mean |log M_enhanced - log M_clean|

    SC measures relative spectral shape distortion (unitless ratio).
    LM catches per-bin level errors in log-frequency space.
    Together they explicitly penalise any spectral distortion the model
    introduces into the speech signal.
    """
    def __init__(self, n_fft: int, hop: int, win: int) -> None:
        super().__init__()
        self.n_fft, self.hop, self.win = n_fft, hop, win
        self.register_buffer("window", torch.hann_window(win), persistent=False)

    def forward(self, enhanced: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
        # Always float32 — matches the fp32 guard in _si_sdr and SmoothLFMagMSELoss.
        m_enh = _stft_mag(enhanced.float(), self.n_fft, self.hop, self.win, self.window)
        m_cln = _stft_mag(clean.float(),    self.n_fft, self.hop, self.win, self.window)
        sc = torch.norm(m_cln - m_enh, p="fro") / (torch.norm(m_cln, p="fro") + 1e-8)
        lm = F.l1_loss(torch.log(m_enh), torch.log(m_cln))
        return sc + lm


class MultiResSTFTAux(nn.Module):
    """Multi-resolution auxiliary STFT loss (spectral convergence + log-mag L1).

    Averaged over three time-frequency resolutions so no single window size
    dominates.  Added as Term 3 in DenoiseLoss to satisfy the professor's
    requirement for an explicit speech-preservation auxiliary loss.
    """
    def __init__(self,
                 fft_sizes: Sequence[int] = (256, 512, 1024),
                 hop_sizes: Sequence[int] = (64,  128, 256),
                 win_sizes: Sequence[int] = (240, 480, 960)) -> None:
        super().__init__()
        self.losses = nn.ModuleList([
            _STFTAuxLoss(f, h, w)
            for f, h, w in zip(fft_sizes, hop_sizes, win_sizes)
        ])

    def forward(self, enhanced: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
        return sum(L(enhanced, clean) for L in self.losses) / len(self.losses)


# ---------------------------------------------------------------------------
def _lf_band_energy(x: torch.Tensor, lo_hz: float, hi_hz: float,
                     sample_rate: int, n_fft: int = 512) -> torch.Tensor:
    """Per-sample mean spectral energy in [lo_hz, hi_hz].  Returns shape (B,).

    Used for the speech-floor penalty: if enhanced_lf << clean_lf, the model
    is over-suppressing voiced fundamentals.
    """
    win = torch.hann_window(n_fft, device=x.device, dtype=torch.float32)
    spec = torch.stft(x.float(), n_fft=n_fft, hop_length=n_fft // 4,
                      win_length=n_fft, window=win,
                      return_complex=True, center=True)              # (B, F, T)
    mag_sq = spec.abs().pow(2)
    freqs  = torch.linspace(0.0, sample_rate / 2.0,
                             n_fft // 2 + 1, device=x.device)
    mask   = ((freqs >= lo_hz) & (freqs <= hi_hz)).float().view(1, -1, 1)
    n_bins = mask.sum().clamp_min(1.0)
    return (mag_sq * mask).sum(dim=(-1, -2)) / (n_bins * mag_sq.shape[-1] + 1e-8)


class LFWeightedNoiseMSE(nn.Module):
    """Single-resolution frequency-weighted L1 loss on the noise prediction.

    Measures  mean( W(f) * |STFT(noise_pred)(f)| - W(f) * |STFT(noise_tgt)(f)| )

    W(f) peaks at `lf_peak` for f ≤ lf_cutoff_hz with a cosine taper to 1.0
    over the transition band.  This creates direct gradient pressure on the
    BiLSTM bottleneck specifically at the frequencies where the ANC speaker
    must cancel noise: HVAC rumble, fan harmonics, 50/60 Hz hum.

    Unlike the mr_mag term (which supervises the speech OUTPUT), this term
    supervises the NOISE ESTIMATE directly — any LF energy that should have
    been subtracted but wasn't contributes a proportionally large gradient.
    """
    def __init__(self, n_fft: int = 512, hop: int = 128, win: int = 512,
                 lf_peak: float = 8.0,
                 lf_cutoff_hz: float = 200.0, transition_hz: float = 100.0,
                 sample_rate: int = 16_000) -> None:
        super().__init__()
        self.n_fft, self.hop, self.win = n_fft, hop, win
        self.register_buffer("window", torch.hann_window(win), persistent=False)
        w = _build_smooth_lf_weight(n_fft, lf_peak, lf_cutoff_hz, transition_hz, sample_rate)
        self.register_buffer("_weight", w.view(-1, 1), persistent=False)

    def forward(self, noise_pred: torch.Tensor, noise_tgt: torch.Tensor) -> torch.Tensor:
        m_pred = _stft_mag(noise_pred.float(), self.n_fft, self.hop, self.win, self.window)
        m_tgt  = _stft_mag(noise_tgt.float(),  self.n_fft, self.hop, self.win, self.window)
        return F.l1_loss(m_pred * self._weight, m_tgt * self._weight)


# ---------------------------------------------------------------------------
class DenoiseLoss(nn.Module):
    """
    5-term CRN denoising loss.

        total = w_sisnr        * SI-SDR_loss
              + w_mag          * MultiRes_SmoothLF_MagMSE       (speech output)
              + w_aux          * MultiRes_STFT_Aux               (speech preservation)
              + w_lf_noise_stft * LFWeightedNoiseMSE             (noise path, LF-heavy)
              + w_speech_floor * relu(E_lf[clean] - E_lf[enhanced])  (80-300 Hz floor)

    Term 4 (lf_noise_stft) directly supervises the noise-prediction path with
    heavy LF weighting, forcing BiLSTM gradients to fix sub-200 Hz leakage
    before any other spectral band.

    Term 5 (speech_floor) computes the energy deficit in the 80-300 Hz voiced
    fundamental band and penalises it with relu(), so the model can suppress
    pure hum/rumble below 80 Hz while still preserving speech vowel energy.

    parts dict keys:
        sisnr        — tanh-smoothed SI-SDR loss value (for bookkeeping)
        sisnr_db     — TRUE mean SI-SDR in dB (detached; use this for plotting)
        mag          — spectral mag term (detached)
        aux_stft     — auxiliary SC + log-mag term (detached; monitor for distortion)
        lf_noise     — LF noise prediction L1 (detached)
        speech_floor — LF speech over-suppression penalty (detached)
    """

    def __init__(self,
                 w_sisnr:         float = 1.0,
                 w_mag:           float = 0.5,
                 w_aux:           float = 0.3,
                 w_lf_noise_stft: float = 0.5,
                 w_speech_floor:  float = 0.2,
                 lf_peak:         float = 6.0,
                 lf_cutoff_hz:    float = 200.0,
                 transition_hz:   float = 100.0,
                 compress:        float = 0.3,
                 sample_rate:     int   = 16_000,
                 **_legacy) -> None:
        super().__init__()
        self.w_sisnr         = w_sisnr
        self.w_mag           = w_mag
        self.w_aux           = w_aux
        self.w_lf_noise_stft = w_lf_noise_stft
        self.w_speech_floor  = w_speech_floor
        self.sample_rate     = sample_rate
        self.mr_mag  = MultiResSmoothLFMSELoss(
            lf_peak=lf_peak, lf_cutoff_hz=lf_cutoff_hz,
            transition_hz=transition_hz, compress=compress,
            sample_rate=sample_rate,
        )
        self.mr_aux     = MultiResSTFTAux()
        self.lf_noise_mse = LFWeightedNoiseMSE(
            lf_peak=lf_peak, lf_cutoff_hz=lf_cutoff_hz,
            transition_hz=transition_hz, sample_rate=sample_rate,
        )
        if _legacy:
            print(f"[DenoiseLoss] ignoring legacy kwargs: {sorted(_legacy)}")

    def forward(self, enhanced: torch.Tensor, clean: torch.Tensor,
                noisy: torch.Tensor):
        # enhanced = model output  (generated / predicted speech)
        # clean    = ground-truth  (target speech)
        sisnr_loss, sisnr_db = _si_sdr(enhanced, clean)
        mag  = self.mr_mag(enhanced, clean)
        aux  = self.mr_aux(enhanced, clean)

        # Term 4: LF noise prediction supervision
        # noise_tgt is detached — it's a fixed target derived from clean ground-truth.
        noise_pred = noisy - enhanced
        noise_tgt  = (noisy - clean).detach()
        lf_noise   = self.lf_noise_mse(noise_pred, noise_tgt)

        # Term 5: protect voiced speech fundamentals from over-suppression.
        # relu() means we only penalise when enhanced_lf < clean_lf (under-delivery),
        # never when enhanced_lf > clean_lf (residual — other terms handle that).
        clean_lf    = _lf_band_energy(clean,    _LF_SPEECH_LO_HZ, _LF_SPEECH_HI_HZ, self.sample_rate)
        enhanced_lf = _lf_band_energy(enhanced, _LF_SPEECH_LO_HZ, _LF_SPEECH_HI_HZ, self.sample_rate)
        speech_floor = F.relu(clean_lf - enhanced_lf).mean()

        total = (self.w_sisnr         * sisnr_loss
               + self.w_mag           * mag
               + self.w_aux           * aux
               + self.w_lf_noise_stft * lf_noise
               + self.w_speech_floor  * speech_floor)

        return total, {
            "sisnr":        sisnr_loss.detach(),
            "sisnr_db":     sisnr_db.mean(),
            "mag":          mag.detach(),
            "aux_stft":     aux.detach(),
            "lf_noise":     lf_noise.detach(),
            "speech_floor": speech_floor.detach(),
        }
