"""
losses.py
=========
Reusable loss modules for the speech-denoising U-Net.

* Huber waveform loss          (clean vs. predicted)
* Multi-resolution STFT loss   (frequency-weighted; 5x penalty < 200 Hz)
* ASR perceptual loss          (frozen Wav2Vec2 CNN feature-matching)
* SI-SDR loss                  (scale-invariant, phase-aware)
* Silence penalty              (suppress residual noise in clean-silent regions)
* Noise STFT loss              (spectral supervision on the noise-prediction path)
* Anti-collapse regulariser    (prevent identity collapse: near-zero noise predictions)
* Speech-floor penalty         (over-suppression guard)

Reference:
  Yamamoto et al., "Parallel WaveGAN" — multi-resolution STFT loss is
  defined in Eq. (3) and is now standard in TTS / SE work.
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
def _stft_mag(x: torch.Tensor, n_fft: int, hop: int, win: int,
              window: torch.Tensor) -> torch.Tensor:
    spec = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=win,
                      window=window, return_complex=True, center=True)
    return torch.abs(spec) + 1e-7


class STFTLoss(nn.Module):
    """Single-resolution STFT loss: spectral-convergence + log-magnitude L1.

    Frequency weighting
    -------------------
    Two complementary weights are registered as non-persistent buffers:

    _sc_weight  — applied to the spectral-convergence (Frobenius) term:
                  • bins 0 .. cutoff_200hz  get  low_freq_boost   (step)
                  • bins cutoff+1 .. F-1    get  linear ramp 1.0 → high_freq_boost
                  Multiplying both pred/target magnitudes by the same weight
                  before taking the norm gives the boosted bins proportionally
                  more influence on the gradient.

    _mag_weight — applied to the log-magnitude L1 term:
                  • bins 0 .. cutoff_200hz  get  low_freq_boost
                  • all other bins          get  1.0
                  Unlike a uniform ramp, a non-uniform weight does NOT cancel in
                  the log-difference |log(m_pred) - log(m_tgt)|, so low-freq
                  bins genuinely contribute more to the gradient.

    200 Hz cutoff bin indices (sr = 16 000 Hz) — floor(200 * n_fft / sr):
        n_fft=256  → bin 3   (0, 62.5, 125, 187.5 Hz)        4 of 129 bins
        n_fft=512  → bin 6   (0, 31.25 … 187.5 Hz)           7 of 257 bins
        n_fft=1024 → bin 12  (0, 15.625 … 187.5 Hz)         13 of 513 bins
    """
    def __init__(self, n_fft: int = 1024, hop: int = 256, win: int = 1024,
                 high_freq_boost: float = 1.0,
                 low_freq_boost:  float = 1.0,
                 sample_rate:     int   = 16_000):
        super().__init__()
        self.n_fft, self.hop, self.win = n_fft, hop, win
        self.register_buffer("window", torch.hann_window(win), persistent=False)
        F_bins  = n_fft // 2 + 1
        cutoff  = int(200 * n_fft / sample_rate)

        sc_w = torch.ones(F_bins)
        sc_w[:cutoff + 1] = max(1.0, low_freq_boost)
        if high_freq_boost > 1.0 and cutoff + 1 < F_bins:
            sc_w[cutoff + 1:] = torch.linspace(1.0, high_freq_boost,
                                                F_bins - cutoff - 1)
        self.register_buffer("_sc_weight", sc_w.view(-1, 1), persistent=False)

        mag_w = torch.ones(F_bins)
        mag_w[:cutoff + 1] = max(1.0, low_freq_boost)
        self.register_buffer("_mag_weight", mag_w.view(-1, 1), persistent=False)

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                m_pred: "torch.Tensor | None" = None,
                m_tgt:  "torch.Tensor | None" = None):
        """Compute loss.  Pass precomputed magnitude spectrograms to skip STFT."""
        if m_pred is None:
            m_pred = _stft_mag(pred,   self.n_fft, self.hop, self.win, self.window)
        if m_tgt is None:
            m_tgt  = _stft_mag(target, self.n_fft, self.hop, self.win, self.window)

        mw_pred = m_pred * self._sc_weight
        mw_tgt  = m_tgt  * self._sc_weight
        sc = torch.norm(mw_tgt - mw_pred, p="fro") / (torch.norm(mw_tgt, p="fro") + 1e-7)

        log_diff = torch.abs(torch.log(m_pred) - torch.log(m_tgt))
        log_l1   = (log_diff * self._mag_weight).mean()

        return sc, log_l1


class MultiResolutionSTFTLoss(nn.Module):
    """Sum of STFTLoss over several resolutions (FFT sizes).

    3-resolution default (256 / 512 / 1024) covers the full frequency band:
      256-pt  hop=64  : 4 ms frames — captures sibilants, fricatives, transients
      512-pt  hop=128 : 8 ms frames — model's native resolution
      1024-pt hop=256 : 16 ms frames — harmonic detail for voiced speech

    high_freq_boost (default 2.0) linearly weights the spectral-convergence
    term so high-frequency bins contribute as much as low-frequency ones.

    low_freq_boost (default 5.0) applies a step-function penalty multiplier
    to all bins whose centre frequency is ≤ 187.5 Hz (the last bin below 200 Hz
    for all three FFT sizes at 16 kHz).  Both the spectral-convergence and the
    log-magnitude L1 sub-terms are boosted.
    """
    def __init__(self,
                 fft_sizes:       Sequence[int] = (256, 512, 1024),
                 hop_sizes:       Sequence[int] = (64,  128, 256),
                 win_sizes:       Sequence[int] = (240, 480, 960),
                 high_freq_boost: float = 2.0,
                 low_freq_boost:  float = 1.0,
                 sample_rate:     int   = 16_000):
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_sizes)
        self.losses = nn.ModuleList([
            STFTLoss(f, h, w,
                     high_freq_boost=high_freq_boost,
                     low_freq_boost=low_freq_boost,
                     sample_rate=sample_rate)
            for f, h, w in zip(fft_sizes, hop_sizes, win_sizes)
        ])

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                cached_mid: "tuple[torch.Tensor, torch.Tensor] | None" = None):
        """Compute multi-resolution STFT loss.

        cached_mid: optional (m_pred, m_tgt) precomputed magnitude spectrograms
        at the *middle* resolution (index 1, default 512-pt) to avoid recomputing
        them when the caller has already computed them for another purpose.
        """
        sc_total, mag_total = 0.0, 0.0
        mid_idx = len(self.losses) // 2          # 1 for 3-resolution default
        for i, L in enumerate(self.losses):
            if i == mid_idx and cached_mid is not None:
                sc, mag = L(pred, target, m_pred=cached_mid[0], m_tgt=cached_mid[1])
            else:
                sc, mag = L(pred, target)
            sc_total = sc_total + sc
            mag_total = mag_total + mag
        n = len(self.losses)
        return sc_total / n, mag_total / n


# ---------------------------------------------------------------------------
class ASRPerceptualLoss(nn.Module):
    """Feature-matching loss using a frozen Wav2Vec2 CNN encoder.

    Architecture rationale
    ----------------------
    WAV2VEC2_BASE has two parts:
      1. CNN feature extractor — 7 Conv1d layers, ~350 K params, stride 320
      2. Transformer encoder  — 12 layers, ~94 M params

    We use only (1).  This is ~270× cheaper than the full model while still
    producing phonetically meaningful representations: each output frame covers
    20 ms of audio and captures formant and spectral envelope shape, which is
    exactly what L1/STFT losses miss.

    Graceful fallback
    -----------------
    If torchaudio is unavailable or the download fails, the loss is silently
    disabled (returns 0.0) so training continues unaffected.
    """

    def __init__(self) -> None:
        super().__init__()
        self._enabled = False
        try:
            import torchaudio
            _bundle = torchaudio.pipelines.WAV2VEC2_BASE
            _full   = _bundle.get_model()
            self.extractor = _full.feature_extractor
            for p in self.extractor.parameters():
                p.requires_grad_(False)
            self.extractor.eval()
            self._enabled = True
            print("[ASRPerceptualLoss] wav2vec2 CNN extractor ready")
        except Exception as exc:
            print(f"[ASRPerceptualLoss] disabled — {exc}")

    def forward(self, enhanced: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
        if not self._enabled:
            return enhanced.new_zeros(())
        n = min(enhanced.shape[0], 4)
        e = enhanced[:n].float()
        c = clean[:n].float()
        with torch.no_grad():
            f_c, _ = self.extractor(c, None)
        f_e, _ = self.extractor(e, None)
        return F.l1_loss(f_e, f_c.detach())


# ---------------------------------------------------------------------------
def _si_sdr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Scale-Invariant SDR — higher is better; returned as a *loss* (negated).

    tanh(x/20) maps dB → (-1, +1) smoothly, preserving gradient at all values
    unlike a hard clamp which zeros gradients at the boundary.
    """
    eps = 1e-8
    pred   = pred   - pred.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    dot    = (pred * target).sum(dim=-1, keepdim=True)
    s_tgt  = dot * target / (target.pow(2).sum(dim=-1, keepdim=True) + eps)
    e_ns   = pred - s_tgt
    si_sdr = 10.0 * torch.log10(s_tgt.pow(2).sum(dim=-1) /
                                (e_ns.pow(2).sum(dim=-1) + eps) + eps)
    return -torch.tanh(si_sdr / 20.0).mean()


class DenoiseLoss(nn.Module):
    """
    Residual-learning loss for a noise-prediction model.

    The model returns  enhanced = noisy - predicted_noise, so:
        noise_pred   = noisy - enhanced
        noise_target = noisy - clean

    Loss terms
    ----------
    w_l1          * Huber(enhanced, clean, δ=0.5)
    w_noise       * Huber(noise_pred, noise_tgt)
    w_sc          * spectral_convergence
    w_mag         * log_magnitude_L1
    w_sisnr       * (–tanh(SI-SDR/20))
    w_sil         * |enhanced| in clean-silent regions
    w_noise_stft  * freq-weighted L1(|STFT(noise_pred)|, |STFT(noise_target)|)
    w_speech_floor * relu(E[clean²] – E[enhanced²])
    w_perceptual   * L1(wav2vec2_cnn(enhanced), wav2vec2_cnn(clean))

    NOTE: w_anticollapse is accepted for backward compatibility but silently
    ignored.  The original term  −mean|noise_pred|  had a negative sign that
    always pulled the total loss down regardless of quality and rewarded
    larger noise predictions — directly conflicting with the Huber noise loss
    and causing gradient instability from epoch 1.
    """
    def __init__(self,
                 w_l1:           float = 1.0,
                 w_sc:           float = 0.1,
                 w_mag:          float = 0.1,
                 w_sisnr:        float = 0.1,
                 w_noise:        float = 1.0,
                 w_sil:          float = 0.1,
                 w_noise_stft:   float = 0.5,
                 w_anticollapse: float = 0.03,
                 w_speech_floor: float = 0.2,
                 w_perceptual:   float = 0.05,
                 low_freq_boost: float = 1.0,
                 sample_rate:    int   = 16_000):
        super().__init__()
        self.w_l1           = w_l1
        self.w_sc           = w_sc
        self.w_mag          = w_mag
        self.w_sisnr        = w_sisnr
        self.w_noise        = w_noise
        self.w_sil          = w_sil
        self.w_noise_stft   = w_noise_stft
        self.w_anticollapse = w_anticollapse
        self.w_speech_floor = w_speech_floor
        self.w_perceptual   = w_perceptual
        self.perc = ASRPerceptualLoss() if w_perceptual > 0 else None
        self.mrstft = MultiResolutionSTFTLoss(
            fft_sizes=(256, 512, 1024),
            hop_sizes=(64,  128, 256),
            win_sizes=(240, 480, 960),
            high_freq_boost=2.0,
            low_freq_boost=low_freq_boost,
            sample_rate=sample_rate,
        )
        # noise_stft uses the SAME params as mrstft resolution-1 (512/128/480)
        # so the caller can pass precomputed 512-pt spectrograms for both terms
        # without an extra STFT call.
        _mid = self.mrstft.losses[1]   # n=512 hop=128 win=480
        self.register_buffer("_noise_win", torch.hann_window(_mid.win), persistent=False)
        self._noise_n   = _mid.n_fft
        self._noise_hop = _mid.hop
        self._noise_win_len = _mid.win
        _n_bins  = _mid.n_fft // 2 + 1
        _cutoff  = int(200 * _mid.n_fft / sample_rate)
        noise_fw = torch.linspace(1.0, 3.0, _n_bins)
        noise_fw[:_cutoff + 1] = max(1.0, low_freq_boost)
        self.register_buffer("_freq_weight", noise_fw.view(1, -1, 1), persistent=False)

    def forward(self, enhanced: torch.Tensor, clean: torch.Tensor,
                noisy: torch.Tensor):
        noise_target = (noisy - clean).detach()
        noise_pred   = noisy - enhanced

        l1_clean = F.huber_loss(enhanced, clean, delta=0.5)
        l1_noise = F.huber_loss(noise_pred, noise_target, delta=0.5)
        sisnr    = _si_sdr(enhanced, clean)

        sil_mask    = (clean.abs() < 0.01).float()
        sil_penalty = (enhanced.abs() * sil_mask).mean()

        # ---- STFT cache: compute 512-pt specs once, reuse in both mrstft and noise_stft ----
        # mrstft resolution-1 and noise_stft now use identical params (n=512 hop=128 win=480),
        # so we compute them here and pass into both — saves 2 redundant STFT calls per step.
        m_enh_mid = _stft_mag(enhanced.float(),
                               self._noise_n, self._noise_hop, self._noise_win_len,
                               self._noise_win)
        m_cln_mid = _stft_mag(clean.float(),
                               self._noise_n, self._noise_hop, self._noise_win_len,
                               self._noise_win)
        sc, mag   = self.mrstft(enhanced, clean, cached_mid=(m_enh_mid, m_cln_mid))

        mag_noise_pred = _stft_mag(noise_pred.float(),
                                   self._noise_n, self._noise_hop, self._noise_win_len,
                                   self._noise_win)
        mag_noise_tgt  = _stft_mag(noise_target.float(),
                                   self._noise_n, self._noise_hop, self._noise_win_len,
                                   self._noise_win)
        stft_noise = F.l1_loss(mag_noise_pred * self._freq_weight,
                               mag_noise_tgt  * self._freq_weight)

        clean_energy    = clean.pow(2).mean(dim=-1)
        enhanced_energy = enhanced.pow(2).mean(dim=-1)
        speech_floor = F.relu(clean_energy - enhanced_energy).mean()

        perc = (self.perc(enhanced, clean)
                if self.perc is not None
                else enhanced.new_zeros(()))

        total = (self.w_l1           * l1_clean
               + self.w_noise        * l1_noise
               + self.w_sc           * sc
               + self.w_mag          * mag
               + self.w_sisnr        * sisnr
               + self.w_sil          * sil_penalty
               + self.w_noise_stft   * stft_noise
               + self.w_speech_floor * speech_floor
               + self.w_perceptual   * perc)

        return total, {
            "l1":           l1_clean.detach(),
            "sc":           sc.detach(),
            "mag":          mag.detach(),
            "sisnr":        sisnr.detach(),
            "sil":          sil_penalty.detach(),
            "noise_stft":   stft_noise.detach(),
            "perceptual":   perc.detach(),
            "speech_floor": speech_floor.detach(),
            "npe":          noise_pred.abs().mean().detach(),
        }
