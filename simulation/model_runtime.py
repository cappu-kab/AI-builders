"""
AI Model Runtime — Universal Dynamic Audio Pipeline
====================================================
Unified streaming inference wrapper.  Three layers of infrastructure ensure
any model — current or future — runs artifact-free on 256-sample real-time
chunks (CHUNK_SIZE configured in configs/config.py):

 Two processing paths, unified behind process_chunk():

 ┌─────────────────────────────────────────────────────────────────────┐
 │ Path A — Per-chunk  (UNet)                                          │
 │   UNet uses dilated convolutions with a fixed receptive field.      │
 │   It does not benefit from prepended history, so it runs directly   │
 │   on each raw 512-sample chunk with no context buffer.              │
 ├─────────────────────────────────────────────────────────────────────┤
 │ Path B1 — Long-context  (CRN)                                       │
 │   CRN's BiLSTM was trained on 3-second segments (~375 STFT frames). │
 │   It uses a dedicated 47616-sample (~3 s) context buffer so the     │
 │   BiLSTM sees ~372 frames per call — matching training conditions   │
 │   and enabling detection of periodic LF noise (50/60 Hz hum, fan). │
 │   Taper is flat (unity gain) with only a 32-sample warm-up edge.   │
 ├─────────────────────────────────────────────────────────────────────┤
 │ Path B2 — Long-context  (Resemble, MossFormer2)                     │
 │   Layer 1: Sliding Context Buffer — prepends HISTORY_SAMPLES=2048  │
 │     past samples (edge-only 32-sample fade-in during warm-up).      │
 │   Layer 2: Stationary Resampler — stateful polyphase up/down;       │
 │     auto-activated only when target_sr != SAMPLE_RATE (Resemble    │
 │     44.1 kHz).  Saves a short tail to eliminate FIR transients.     │
 │   Layer 3: Model inference → extract last CHUNK_SIZE output →      │
 │     noise estimate = raw - enhanced.                                │
 ├─────────────────────────────────────────────────────────────────────┤
 │ Uniform Interface                                                    │
 │       noise_est = runtime.process_chunk(raw_512)                    │
 │   predict() is an alias for backward compat with main_hardware.py. │
 └─────────────────────────────────────────────────────────────────────┘

Named models (pass model_name= to constructor):
    "mock"          — LP-filter mock (no checkpoint required)
    "crn"           — Convolutional Recurrent Network  (16 kHz)
    "unet"          — AdvancedUNetSE                   (16 kHz)
    "resemble"      — Resemble Enhance pre-trained     (44.1 kHz, auto-resampled)
    "resemble_ft"   — Resemble Enhance fine-tuned      (44.1 kHz, auto-resampled)
    "mossformer_ft" — MossFormer2 fine-tuned           (16 kHz)

Legacy path (model_name=None):
    Respects USE_MOCK_MODEL / MODEL_PATH → ONNX → TorchScript → mock.
"""
from __future__ import annotations

import importlib.util
import sys
from collections import deque
from math import gcd, ceil
from pathlib import Path
from typing import Optional

import numpy as np

from configs.config import (
    MODEL_PATH,
    USE_MOCK_MODEL,
    CHUNK_SIZE,
    SAMPLE_RATE,
)
from utils.logger_setup import get_logger

log = get_logger(__name__)

# ── Project layout ────────────────────────────────────────────────────────────
_ANC_DIR     = Path(__file__).resolve().parent      # …/ANC_Hardware_Test
_AI_DIR      = _ANC_DIR.parent                      # …/AI_builders
_CRN_ROOT    = _AI_DIR / "Run" / "CRN"
_UNET_ROOT   = _AI_DIR / "Run" / "U_net"
_RESEMBLE_SR = 44_100

# ── Streaming constants ───────────────────────────────────────────────────────
_HISTORY_SAMPLES     = 2048    # context for Resemble/MossFormer2 (~128 ms)
_CRN_CTX_N           = 47616  # CRN sliding context prefix: 47616 + 256 = 47872 smp.
                                # At hop_length=128 with center=True this is ~374 STFT
                                # frames per call — matches the model's 3-second
                                # training segments.  The BiLSTM bottleneck cannot
                                # persist hidden state across calls in a causal stream
                                # (backward LSTM is non-causal), so each call must see
                                # the full training-equivalent context window or the
                                # mask collapses toward zero.
                                # Inference cost: ~30–60 ms on RTX 5060 (well above the
                                # 16 ms chunk period; the simulator's deadline sleep
                                # absorbs the slack and falls behind real-time, which
                                # is fine for offline verification).
_RESAMP_TAIL         = 64      # resampler tail samples saved between calls
_CTX_EDGE_FADE       = 32      # soft fade-in at the very start of the history
                                # (prevents hard 0→signal step during warm-up only)

# UNet uses dilated convolutions (fixed receptive field) — no history buffer needed.
# CRN has a BiLSTM trained on 375 STFT frames; it gets its own 3-second buffer.
_PER_CHUNK_BACKENDS = frozenset({"unet"})


def _find_bench_ckpt(subdir: str, filename: str) -> Path:
    for base in (_AI_DIR / "Run" / "benchmark_outputs", _AI_DIR / "benchmark_outputs"):
        p = base / subdir / filename
        if p.exists():
            return p
    return _AI_DIR / "Run" / "benchmark_outputs" / subdir / filename


def _load_module_from_file(unique_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(unique_name, str(file_path))
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ══════════════════════════════════════════════════════════════════════════════
# Layer 1 — Sliding Context Buffer
# ══════════════════════════════════════════════════════════════════════════════

class _StreamingContextBuffer:
    """
    Ring buffer that prepends `history_samples` of past audio to every new
    chunk.  A half-Hann ramp fades the history edge to zero so the model
    never sees a hard discontinuity at the context boundary.

    Usage::
        buf   = _StreamingContextBuffer(CHUNK_SIZE, _HISTORY_SAMPLES)
        ctx   = buf.push(new_512_chunk)  # shape: (history + chunk_size,)
    """

    def __init__(self, chunk_size: int, history_samples: int) -> None:
        self._chunk   = chunk_size
        self._hist    = history_samples
        self._total   = history_samples + chunk_size
        self._buf     = np.zeros(self._total, dtype=np.float32)
        self._taper   = self._build_taper(history_samples, chunk_size)

    def push(self, chunk: np.ndarray) -> np.ndarray:
        """Advance ring buffer by one chunk; return tapered context window."""
        self._buf[: -self._chunk] = self._buf[self._chunk :]
        self._buf[-self._chunk :] = chunk
        return self._buf * self._taper          # returns a copy

    def reset(self) -> None:
        self._buf[:] = 0.0

    @staticmethod
    def _build_taper(hist: int, chunk: int) -> np.ndarray:
        """Flat (1.0) everywhere; only the first _CTX_EDGE_FADE samples get a
        soft half-Hann fade-in so the very start of the ring buffer (silence
        during warm-up) doesn't produce a hard 0→signal step.  The rest of the
        history window stays at unity gain so Resemble/MossFormer2 always see
        full-scale audio."""
        t    = np.ones(hist + chunk, dtype=np.float32)
        edge = min(_CTX_EDGE_FADE, hist)
        if edge > 0:
            t[:edge] = 0.5 * (1.0 - np.cos(np.pi * np.arange(edge) / edge))
        return t


# ══════════════════════════════════════════════════════════════════════════════
# Layer 2 — Stationary Resampler
# ══════════════════════════════════════════════════════════════════════════════

class _StationaryResampler:
    """
    Stateful polyphase resampler that eliminates inter-chunk FIR boundary
    artefacts by saving `_RESAMP_TAIL` samples of each call's input and
    prepending them to the next call.

    This ensures the filter always has real historical context at the start
    of every block instead of zero-padding, producing a continuous output
    stream even when applied to successive overlapping context windows.
    """

    def __init__(self, src_sr: int, tgt_sr: int) -> None:
        g           = gcd(src_sr, tgt_sr)
        self._up    = tgt_sr // g
        self._down  = src_sr // g
        self._tail  = np.zeros(_RESAMP_TAIL, dtype=np.float32)
        # Pre-compute expected number of output samples discarded per call
        self._tail_out = int(round(_RESAMP_TAIL * self._up / self._down))

    def process(self, block: np.ndarray) -> np.ndarray:
        from scipy.signal import resample_poly
        block  = np.asarray(block, dtype=np.float32)
        joined = np.concatenate([self._tail, block])
        # Save new tail before resampling (tail is the last _RESAMP_TAIL input samples)
        if len(block) >= _RESAMP_TAIL:
            self._tail = block[-_RESAMP_TAIL:].copy()
        else:
            self._tail = np.concatenate([self._tail[len(block):], block])
        y = resample_poly(joined, self._up, self._down).astype(np.float32)
        return y[self._tail_out:]   # discard samples corresponding to prepended tail

    def reset(self) -> None:
        self._tail[:] = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# ModelRuntime
# ══════════════════════════════════════════════════════════════════════════════

class ModelRuntime:
    """
    Load and stream audio through any ANC noise-estimation model.

    Primary interface::
        runtime = ModelRuntime("crn")
        noise   = runtime.process_chunk(raw_512)   # float32 1-D, len=CHUNK_SIZE

    predict() is a backward-compatible alias so main_hardware.py needs no edits.

    Parameters
    ----------
    model_name : str or None
        "mock", "crn", "unet", "resemble", "resemble_ft", "mossformer_ft", or
        None (legacy USE_MOCK_MODEL / ONNX / TorchScript path).
    """

    NAMED_MODELS = frozenset({"crn", "unet", "resemble", "resemble_ft", "mossformer_ft"})

    def __init__(self, model_name: Optional[str] = None,
                 bilstm_lookahead: int = 1,
                 ckpt_path: Optional[str] = None) -> None:
        self._name              = (model_name or "").lower().strip()
        self._backend           = "mock"
        self._device            = None
        self._torch_mdl         = None
        self._mock              = None
        # Optional override for the checkpoint file (CRN / UNet)
        self._ckpt_path         = Path(ckpt_path) if ckpt_path else None
        # Look-ahead depth for BiLSTM CRN — must be set before _init_streaming
        self._bilstm_lookahead  = max(1, int(bilstm_lookahead))
        # Legacy ONNX / TorchScript
        self._session      = None
        self._input_name   = None
        self._output_name  = None
        self._ts_mdl       = None
        # Streaming infrastructure — initialised after model load
        self._ctx_buf      : Optional[_StreamingContextBuffer] = None
        self._up_resampler : Optional[_StationaryResampler]    = None
        self._dn_resampler : Optional[_StationaryResampler]    = None
        # BiLSTM look-ahead queue (CRN only) — set by _init_streaming
        self._crn_lookahead_q  : Optional[deque]               = None
        # CRN stateful streaming state — persistent between process_chunk() calls
        self._crn_rnn_hidden                                    = None   # GRU: tensor | LSTM: (h,c)
        self._crn_stft_ctx     : Optional[np.ndarray]          = None   # n_fft-sample STFT look-back

        if self._name in self.NAMED_MODELS:
            self._load_named(self._name)
        else:
            self._load_legacy()

        self._init_streaming()

    # ── Streaming infrastructure ──────────────────────────────────────────────

    def _init_streaming(self) -> None:
        """Construct (or re-construct) the context buffer and stationary resamplers."""
        if self._backend in _PER_CHUNK_BACKENDS:
            # UNet: dilated-conv fixed receptive field — no benefit from history buffer.
            self._ctx_buf         = None
            self._up_resampler    = None
            self._dn_resampler    = None
            self._crn_lookahead_q = None
            self._crn_stft_ctx    = None
            self._crn_rnn_hidden  = None
            log.info("[%s] per-chunk mode — context buffer disabled.", self._backend)
            return

        if self._backend == "crn" and self._torch_mdl is not None:
            # Stateful streaming mode — large sliding context + persistent RNN hidden state.
            #
            # Root cause of the old zero-output bug was twofold:
            #   1. 47616-sample context → 374 STFT frames per call → 50–200 ms inference
            #      → hit the 128 ms output-queue timeout → zeros every call.
            #   2. When the context was cut to only n_fft=512 smp (6 STFT frames), the
            #      GRU had too few local frames to activate the noise mask, so
            #      noise_wave ≈ 0, enhanced ≈ x, and noise_estimate ≈ 0 (silent failure).
            #
            # Fix: use _CRN_CTX_N=3840 smp prefix so each call processes 4096 smp total
            # (~32 STFT frames at hop=128).  That gives the GRU enough local context to
            # produce a real mask while keeping inference time ~2–5 ms.  The RNN hidden
            # state is preserved across calls so long-range temporal memory accumulates
            # chunk-by-chunk without re-processing the entire history every time.
            self._crn_stft_ctx    = np.zeros(_CRN_CTX_N, dtype=np.float32)
            self._crn_rnn_hidden  = None    # populated on first call; preserved thereafter
            self._ctx_buf         = None    # not used for CRN stateful path
            self._up_resampler    = None
            self._dn_resampler    = None
            self._crn_lookahead_q = None
            log.info(
                "[crn] stateful streaming — ctx=%d smp (total=%d smp, ~%d STFT frames) "
                "+ 1-chunk look-ahead extraction (256 smp output latency).",
                _CRN_CTX_N, _CRN_CTX_N + CHUNK_SIZE,
                (_CRN_CTX_N + CHUNK_SIZE) // 128,
            )
            return

        self._ctx_buf = _StreamingContextBuffer(CHUNK_SIZE, _HISTORY_SAMPLES)
        tgt_sr = self.target_sample_rate
        if tgt_sr != SAMPLE_RATE:
            self._up_resampler = _StationaryResampler(SAMPLE_RATE, tgt_sr)
            self._dn_resampler = _StationaryResampler(tgt_sr, SAMPLE_RATE)
            log.info(
                "Streaming resamplers active: %d Hz <-> %d Hz (tail=%d samples)",
                SAMPLE_RATE, tgt_sr, _RESAMP_TAIL,
            )
        else:
            self._up_resampler = None
            self._dn_resampler = None
        self._crn_lookahead_q = None
        self._crn_stft_ctx    = None
        self._crn_rnn_hidden  = None

    def reset(self) -> None:
        """
        Reset all stateful streaming buffers to silence.
        Call this between simulation runs to avoid history bleed.
        """
        if self._ctx_buf is not None:
            self._ctx_buf.reset()
        if self._up_resampler is not None:
            self._up_resampler.reset()
        if self._dn_resampler is not None:
            self._dn_resampler.reset()
        if self._crn_lookahead_q is not None:
            self._crn_lookahead_q.clear()
        # CRN stateful path: clear STFT look-back and RNN hidden state.
        # _crn_rnn_hidden is set to None (not zeroed) so the next call starts
        # with PyTorch's default zero h0 — matching training initialisation.
        if self._crn_stft_ctx is not None:
            self._crn_stft_ctx[:] = 0.0
        self._crn_rnn_hidden = None
        log.debug("ModelRuntime streaming state reset.")

    def warm_up(self, audio: np.ndarray) -> int:
        """
        Pre-fill the context buffer with audio from BEFORE the stream starts.

        WARNING — do NOT pass the same audio you are about to stream.
        If audio[0:CHUNK_SIZE] appears in both the warm-up history AND as the
        first streamed chunk, the model's context window will contain that
        segment twice at a ~3 s delay, creating severe comb-filter echo.
        Only call this when you have distinct preceding audio (e.g. an earlier
        recording that immediately precedes the file being processed).

        Pushes floor(len(audio) / CHUNK_SIZE) chunks into _ctx_buf without
        running model inference.  Returns the number of samples consumed.
        """
        if self._ctx_buf is None:
            return 0
        audio = np.asarray(audio, dtype=np.float32)
        n     = len(audio) // CHUNK_SIZE
        for i in range(n):
            self._ctx_buf.push(audio[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE])
        if self._crn_lookahead_q is not None:
            self._crn_lookahead_q.clear()
        consumed = n * CHUNK_SIZE
        log.info(
            "[%s] warm_up: %d chunks (%d smp / %.2f s) pre-loaded into context buffer.",
            self._backend, n, consumed, consumed / SAMPLE_RATE,
        )
        return consumed

    # ── Model properties ──────────────────────────────────────────────────────

    @property
    def target_sample_rate(self) -> int:
        """Native sample rate the underlying model expects."""
        if self._backend in ("resemble", "resemble_ft"):
            return _RESEMBLE_SR
        return SAMPLE_RATE

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def model_name(self) -> str:
        return self._name or self._backend

    @property
    def look_ahead_delay(self) -> int:
        """Samples of output delay from the BiLSTM look-ahead queue (0 if inactive)."""
        if self._crn_lookahead_q is not None:
            return CHUNK_SIZE * self._bilstm_lookahead
        return 0

    # ── Named-model dispatch ──────────────────────────────────────────────────

    def _load_named(self, name: str) -> None:
        try:
            import torch
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            {
                "crn":           self._load_crn,
                "unet":          self._load_unet,
                "resemble":      lambda: self._load_resemble(ft_ckpt=None),
                "resemble_ft":   lambda: self._load_resemble(
                                     ft_ckpt=_find_bench_ckpt("resemble_ft", "denoiser_ft.pt")),
                "mossformer_ft": self._load_mossformer_ft,
            }[name]()
            self._backend = name
            log.info("Loaded model '%s' on device=%s", name, self._device)
        except Exception as exc:
            log.error(
                "Failed to load '%s': %s — falling back to mock.", name, exc, exc_info=True,
            )
            self._torch_mdl = None
            self._load_mock()

    # ── Individual loaders ────────────────────────────────────────────────────

    def _load_crn(self) -> None:
        ckpt = self._ckpt_path if self._ckpt_path else _CRN_ROOT / "checkpoints" / "best.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"CRN checkpoint not found: {ckpt}")
        if str(_CRN_ROOT) not in sys.path:
            sys.path.insert(0, str(_CRN_ROOT))
        mod = _load_module_from_file("_anc_crn_inference", _CRN_ROOT / "inference.py")
        self._torch_mdl, _ = mod.load_model(ckpt, self._device)
        log.info("[crn] %s", ckpt)

    def _load_unet(self) -> None:
        ckpt = self._ckpt_path if self._ckpt_path else _UNET_ROOT / "checkpoints" / "best.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"UNet checkpoint not found: {ckpt}")
        if str(_UNET_ROOT) not in sys.path:
            sys.path.insert(0, str(_UNET_ROOT))
        mod = _load_module_from_file("_anc_unet_inference", _UNET_ROOT / "inference.py")
        self._torch_mdl, _ = mod.load_model(ckpt, self._device)
        log.info("[unet] %s", ckpt)

    def _load_resemble(self, ft_ckpt: Optional[Path]) -> None:
        import torch
        self._patch_resemble_hparams()
        from resemble_enhance.denoiser.denoiser import Denoiser
        from resemble_enhance.denoiser.hparams import HParams as ResembleHParams

        if ft_ckpt is None:
            from resemble_enhance.enhancer.download import download as _re_download
            run_dir  = _re_download()
            hp       = ResembleHParams.load(run_dir)
            model    = Denoiser(hp)
            pt       = run_dir / "ds" / "G" / "default" / "mp_rank_00_model_states.pt"
            state    = torch.load(str(pt), map_location="cpu")["module"]
            model.load_state_dict(state)
            log.info("[resemble] pre-trained: %s", pt)
        else:
            if not ft_ckpt.exists():
                raise FileNotFoundError(f"Resemble FT checkpoint not found: {ft_ckpt}")
            hp    = ResembleHParams()
            model = Denoiser(hp)
            ckpt  = torch.load(str(ft_ckpt), map_location="cpu")
            model.load_state_dict(ckpt["state_dict"])
            log.info("[resemble_ft] %s", ft_ckpt)

        self._torch_mdl = model.eval().to(self._device)

    @staticmethod
    def _patch_resemble_hparams() -> None:
        try:
            import dataclasses
            from omegaconf import OmegaConf
            import resemble_enhance.hparams as _mod
            from resemble_enhance.hparams import HParams as _Base

            @classmethod
            def _safe_from_yaml(cls, path):
                raw      = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
                known    = {f.name for f in dataclasses.fields(cls)}
                filtered = {k: v for k, v in raw.items() if k in known}
                return cls(**dict(OmegaConf.merge(cls(), OmegaConf.create(filtered))))

            _Base.from_yaml      = _safe_from_yaml
            _mod.HParams.from_yaml = _safe_from_yaml
        except Exception as exc:
            log.debug("Resemble HParams patch skipped: %s", exc)

    def _load_mossformer_ft(self) -> None:
        import torch
        ft = _find_bench_ckpt("mossformer_ft", "mossformer_ft.pt")
        if not ft.exists():
            raise FileNotFoundError(f"MossFormer2 FT checkpoint not found: {ft}")
        from mossformer2.models.mossformer2 import MossFormer2
        from mossformer2.utils.checkpoint import load_pretrained
        model = MossFormer2(
            in_channels=256, out_channels=256, num_layers=24, d_model=256,
            attn_dropout=0.0, ffn_dropout=0.0,
        )
        try:
            load_pretrained(model, tag="MossFormer2_SE_16K")
            log.info("[mossformer] pre-trained base loaded.")
        except Exception as exc:
            log.warning("[mossformer] pre-trained base unavailable (%s).", exc)
        ckpt = torch.load(str(ft), map_location="cpu")
        model.load_state_dict(ckpt["state_dict"], strict=False)
        self._torch_mdl = model.eval().to(self._device)
        log.info("[mossformer_ft] %s", ft)

    # ── Legacy loaders ────────────────────────────────────────────────────────

    def _load_legacy(self) -> None:
        if USE_MOCK_MODEL:
            log.info("USE_MOCK_MODEL=True — loading mock estimator.")
            self._load_mock()
            return
        if not Path(MODEL_PATH).exists():
            log.warning("Checkpoint not found at '%s'. Falling back to mock.", MODEL_PATH)
            self._load_mock()
            return
        if not self._try_onnx():
            if not self._try_torchscript():
                log.warning("All real backends failed. Falling back to mock.")
                self._load_mock()

    def _try_onnx(self) -> bool:
        try:
            import onnxruntime as ort
            available = ort.get_available_providers()
            preferred = [
                ep for ep in (
                    "TensorrtExecutionProvider",
                    "CUDAExecutionProvider",
                    "CPUExecutionProvider",
                )
                if ep in available
            ]
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.intra_op_num_threads = 2
            self._session     = ort.InferenceSession(
                MODEL_PATH, sess_options=opts, providers=preferred,
            )
            self._input_name  = self._session.get_inputs()[0].name
            self._output_name = self._session.get_outputs()[0].name
            self._backend     = "onnx"
            log.info("ONNX: providers=%s | input='%s'", preferred, self._input_name)
            return True
        except ImportError:
            log.debug("onnxruntime not installed.")
        except Exception as exc:
            log.warning("ONNX load failed: %s", exc, exc_info=True)
        return False

    def _try_torchscript(self) -> bool:
        try:
            import torch
            self._ts_mdl = torch.jit.load(MODEL_PATH, map_location="cpu")
            self._ts_mdl.eval()
            self._backend = "torchscript"
            log.info("TorchScript model loaded on CPU.")
            return True
        except ImportError:
            log.debug("PyTorch not installed.")
        except Exception as exc:
            log.warning("TorchScript load failed: %s", exc, exc_info=True)
        return False

    def _load_mock(self) -> None:
        from mock.mock_model import MockNoiseEstimator
        self._mock    = MockNoiseEstimator()
        self._backend = "mock"

    # ══════════════════════════════════════════════════════════════════════════
    # Layer 3 — Uniform process_chunk() Interface
    # ══════════════════════════════════════════════════════════════════════════

    def process_chunk(self, raw_chunk: np.ndarray) -> np.ndarray:
        """
        Universal entry point for one CHUNK_SIZE audio block.

        Routing:
            UNet        — _infer_per_chunk(): raw chunk, no context buffer.
            CRN         — _infer_crn_stateful(): 3840-smp sliding context (→ 4096 smp
                          total, ~32 STFT frames) + persistent RNN hidden state.
            Resemble / MossFormer2 — three-layer pipeline:
                1. Sliding context buffer + edge taper
                2. Stationary resampler (only when target_sr != SAMPLE_RATE)
                3. Model inference → extract chunk-aligned output → noise estimate

        Args:
            raw_chunk: 1-D float32, length == CHUNK_SIZE (256), range [-1, 1].
        Returns:
            noise_estimate: 1-D float32, same length.
        """
        chunk = np.asarray(raw_chunk, dtype=np.float32)

        # ── Fast paths: legacy / mock backends skip streaming pipeline ────────
        if self._backend == "mock":
            return self._mock.predict(chunk)
        if self._backend == "onnx":
            return self._infer_onnx(chunk)
        if self._backend == "torchscript":
            return self._infer_torchscript(chunk)

        # ── Per-chunk backends (UNet): bypass context buffer entirely ────────────
        if self._backend in _PER_CHUNK_BACKENDS:
            return self._infer_per_chunk(chunk)

        # ── CRN stateful streaming path ───────────────────────────────────────
        # 3840-smp sliding context gives ~32 STFT frames per call; persistent
        # RNN hidden state carries long-range memory across calls without
        # re-processing the full session history each chunk.
        if self._backend == "crn" and self._crn_stft_ctx is not None:
            return self._infer_crn_stateful(chunk)

        # ── Named torch models: full three-layer pipeline ─────────────────────

        # Layer 1: sliding context window with edge taper
        ctx_16k = self._ctx_buf.push(chunk)         # shape: (HISTORY + CHUNK_SIZE,)

        # Layer 2: stationary resampling to model native SR
        if self._up_resampler is not None:
            ctx_model = self._up_resampler.process(ctx_16k)
        else:
            ctx_model = ctx_16k

        # Model inference
        import torch
        with torch.inference_mode():
            x   = torch.from_numpy(ctx_model).unsqueeze(0).to(self._device)
            out = self._torch_mdl(x)
            if isinstance(out, (tuple, list)):
                out = out[0]
            enh_model = out.squeeze(0).detach().cpu().numpy().astype(np.float32)

        # Layer 2 (inverse): resample model output back to pipeline SR
        if self._dn_resampler is not None:
            enhanced = self._dn_resampler.process(enh_model)
        else:
            enhanced = enh_model

        n = CHUNK_SIZE

        # BiLSTM look-ahead (CRN with bidirectional RNN):
        # The backward LSTM at the end of the context window has seen zero future
        # frames during training but is abruptly cut off here.  We delay output
        # by N chunks: the deque holds the last N raw chunks, the current chunk
        # serves as one of N future frames of look-ahead for the backward pass.
        # While the deque is filling (first N calls) we output zeros to avoid
        # emitting corrupted estimates based on an under-filled look-ahead.
        if self._crn_lookahead_q is not None:
            N = self._bilstm_lookahead
            if len(self._crn_lookahead_q) < N:
                # Still warming up: mute output until the queue is full
                self._crn_lookahead_q.append(chunk)
                return np.zeros(n, dtype=np.float32)

            # Queue full: oldest entry is the chunk we now have N frames of look-ahead for
            prev_raw = self._crn_lookahead_q[0].copy()
            self._crn_lookahead_q.append(chunk)     # auto-evicts the oldest entry
            if len(enhanced) >= (N + 1) * n:
                enh_at_prev = enhanced[-(N + 1) * n : -N * n]
            else:
                enh_at_prev = enhanced[-n:]          # fallback (shouldn't happen)
                prev_raw    = chunk
            return (prev_raw - enh_at_prev).astype(np.float32)

        # Standard path: GRU / unidirectional LSTM / Resemble / MossFormer2.
        # Extract the CHUNK_SIZE samples for the newly ingested chunk.
        if len(enhanced) >= n:
            enh_chunk = enhanced[-n:]
        else:
            # Rare: short output from a wide-kernel model — left-pad with zeros
            enh_chunk = np.pad(enhanced, (n - len(enhanced), 0))

        # Noise estimate: the component the enhancer subtracted
        return (chunk - enh_chunk).astype(np.float32)

    def predict(self, audio_chunk: np.ndarray) -> np.ndarray:
        """Backward-compatible alias for process_chunk().  Used by main_hardware.py."""
        return self.process_chunk(audio_chunk)

    # ── Per-chunk inference (UNet) ────────────────────────────────────────────

    def _infer_per_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """
        Feed exactly CHUNK_SIZE samples to UNet.

        UNet's dilated convolutions have a fixed receptive field and do not
        accumulate inter-chunk state, so no context buffer is needed.

        The inference module's enhance_waveform() handles batching internally.
        We call the model directly here to avoid a redundant module import on
        the hot path.
        """
        import torch
        with torch.inference_mode():
            x   = torch.from_numpy(chunk).unsqueeze(0).to(self._device)
            out = self._torch_mdl(x)
            if isinstance(out, (tuple, list)):
                out = out[0]
            enhanced = out.squeeze(0).detach().cpu().numpy().astype(np.float32)

        n = CHUNK_SIZE
        if len(enhanced) >= n:
            enh_chunk = enhanced[-n:]
        else:
            enh_chunk = np.pad(enhanced, (n - len(enhanced), 0))

        return (chunk - enh_chunk).astype(np.float32)

    # ── CRN stateful streaming inference ─────────────────────────────────────

    def _infer_crn_stateful(self, chunk: np.ndarray) -> np.ndarray:
        """
        Streaming CRN inference via direct model.forward() call, with
        1-chunk look-ahead extraction.

        Per-call input: [_crn_stft_ctx (_CRN_CTX_N smp) | chunk (CHUNK_SIZE smp)]
                      = 47616 + 256 = 47872 smp = ~374 STFT frames at hop=128.

        Two design choices:

        1. Large context window (~training segment length).
           BiLSTM was trained on 375 STFT frames; a smaller window collapses
           the mask toward zero because the backward LSTM has insufficient
           context to localise noise vs speech.

        2. Look-ahead extraction.
           torch.stft with center=True reflect-pads n_fft//2 samples on the
           right edge.  The iSTFT reconstruction of the LAST CHUNK_SIZE samples
           is therefore distorted (the model sees mirrored "future" content).
           Extracting from enhanced[-2*CHUNK:-CHUNK] instead of enhanced[-CHUNK:]
           moves the extraction into the model's interior, where reconstruction
           is undistorted because the corresponding STFT frames see real audio
           on both sides.  Cost: the returned noise_est corresponds to the
           chunk submitted on the PREVIOUS call (256-sample output latency),
           which the simulator's 257-sample reference delay already absorbs.

        Returns noise_est for the previous chunk (causal 1-chunk look-ahead).
        On the first call the previous-chunk slot is zero context, so the
        returned noise_est is ~0 — the simulator's WARMUP_CHUNKS mute makes
        this transparent.
        """
        import torch

        ctx_n = len(self._crn_stft_ctx)
        inp   = np.concatenate([self._crn_stft_ctx, chunk]).astype(np.float32)

        with torch.inference_mode():
            x       = torch.from_numpy(inp).unsqueeze(0).to(self._device)  # (1, L)
            out     = self._torch_mdl(x)                                    # (1, L) enhanced
            if isinstance(out, (tuple, list)):
                out = out[0]
            enhanced = out.squeeze(0).cpu().numpy().astype(np.float32)

        # Slide the context window forward by one chunk.
        self._crn_stft_ctx = inp[-ctx_n:]

        # Look-ahead extraction (see docstring): take the [-2*CHUNK : -CHUNK]
        # slice — this is one chunk earlier than the model's right boundary,
        # outside the reflect-pad-distorted region.  The same slice of the
        # input gives the matching raw segment for the residual.
        prev_chunk_in  = inp[-2 * CHUNK_SIZE:-CHUNK_SIZE]
        prev_chunk_enh = enhanced[-2 * CHUNK_SIZE:-CHUNK_SIZE]
        return (prev_chunk_in - prev_chunk_enh).astype(np.float32)

    # ── Legacy ONNX / TorchScript inference ──────────────────────────────────

    def _infer_onnx(self, chunk: np.ndarray) -> np.ndarray:
        x       = chunk.reshape(1, 1, -1)
        outputs = self._session.run([self._output_name], {self._input_name: x})
        return outputs[0].reshape(-1).astype(np.float32)

    def _infer_torchscript(self, chunk: np.ndarray) -> np.ndarray:
        import torch
        with torch.no_grad():
            x = torch.from_numpy(chunk).unsqueeze(0).unsqueeze(0)
            return self._ts_mdl(x).squeeze().numpy().astype(np.float32)

    # ── Diagnostics ────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        if self._crn_stft_ctx is not None:
            ctx = f"stateful STFT look-back={len(self._crn_stft_ctx)} smp"
        elif self._ctx_buf is None:
            ctx = "none (per-chunk)"
        else:
            ctx = f"{self._ctx_buf._hist} smp"
        if self._crn_lookahead_q is not None:
            la = f", BiLSTM_lookahead={self._bilstm_lookahead}×{CHUNK_SIZE}smp"
        else:
            la = ""
        return (
            f"ModelRuntime(backend='{self._backend}', device={self._device}, "
            f"target_sr={self.target_sample_rate}, ctx={ctx}{la})"
        )
