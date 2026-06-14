import io
import numpy as np
import gradio as gr
from app import denoise_core
from app import model_selector
from app import transcript_tab


def _load_mono_float(path):
    from scipy.io import wavfile as wv
    sr, raw = wv.read(path)
    pcm = raw.astype(np.float32)
    if raw.dtype == np.int16:
        pcm /= 32768.0
    elif raw.dtype == np.int32:
        pcm /= 2147483648.0
    if pcm.ndim > 1:
        pcm = pcm.mean(axis=1)
    return pcm, sr


def _estimate_snr(pcm):
    """Energy-based SNR estimate. Noise floor = bottom-15% frames; signal = mean energy."""
    frame_n = 320  # ~20 ms at 16 kHz
    n = (len(pcm) // frame_n) * frame_n
    if n < frame_n * 5:
        return 0.0
    frames = pcm[:n].reshape(-1, frame_n)
    energies = np.mean(frames ** 2, axis=1)
    noise_energy = float(np.percentile(energies, 15))
    signal_energy = float(np.mean(energies))
    if noise_energy < 1e-10:
        return 60.0
    return float(10.0 * np.log10(max(signal_energy / noise_energy, 1.0)))


def _spectrogram(wav_path, title, vmin, vmax):
    import librosa, librosa.display, matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    pcm, sr = _load_mono_float(wav_path)
    if sr != 16000:
        import scipy.signal as ss
        pcm = ss.resample(pcm, int(len(pcm) * 16000 / sr)).astype(np.float32)
        sr = 16000

    S_db = librosa.power_to_db(
        librosa.feature.melspectrogram(y=pcm, sr=sr, n_mels=128, fmax=8000),
        ref=np.max
    )
    fig, ax = plt.subplots(figsize=(7, 3), dpi=110)
    img = librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="mel",
                                   fmax=8000, ax=ax, vmin=vmin, vmax=vmax, cmap="magma")
    fig.colorbar(img, ax=ax, format="%+.0f dB")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def run(in_wav_path, model_choice):
    if in_wav_path is None:
        yield None, None, "", None, None, "", None
        return

    yield None, None, "", None, None, "⏳ Processing...", None

    if model_choice == "CRN (default)":
        denoised_path = denoise_core.process(in_wav_path, gate=True, agc=True, compress=True)
    else:
        denoised_path = model_selector.process(in_wav_path, model_choice)

    try:
        pcm_in,  _ = _load_mono_float(in_wav_path)
        pcm_out, _ = _load_mono_float(denoised_path)
        snr_in  = _estimate_snr(pcm_in)
        snr_out = _estimate_snr(pcm_out)
        improvement = snr_out - snr_in
        score_text = f"[{model_choice}] SNR: +{improvement:.1f} dB improvement ({snr_in:.1f} → {snr_out:.1f} dB)"
    except Exception as exc:
        score_text = f"SNR error: {exc}"

    VMIN, VMAX = -80, 0
    try:
        spec_in  = _spectrogram(in_wav_path,    "Input (noisy)",      VMIN, VMAX)
        spec_out = _spectrogram(denoised_path,  "Output (denoised)",  VMIN, VMAX)
    except Exception as exc:
        spec_in = spec_out = None
        score_text += f" | Spectrogram error: {exc}"

    yield in_wav_path, denoised_path, score_text, spec_in, spec_out, "✅ Done", denoised_path


with gr.Blocks(title="Speech Denoiser") as demo:
    denoised_state = gr.State(None)

    with gr.Tabs():
        with gr.Tab("Denoise"):
            gr.Markdown("## Speech Denoiser\nUpload or record noisy speech. Get clean speech back.")

            with gr.Row():
                audio_in  = gr.Audio(label="Noisy Input", type="filepath",
                                     sources=["upload", "microphone"])
                model_dropdown = gr.Dropdown(
                    choices=["CRN (default)", "UNet", "Resemble-FT"],
                    value="CRN (default)",
                    label="Model",
                )
            btn = gr.Button("Denoise", variant="primary")
            status_md = gr.Markdown("")

            with gr.Row():
                audio_orig = gr.Audio(label="Original",            type="filepath")
                audio_out  = gr.Audio(label="Denoised + Enhanced", type="filepath")

            with gr.Row():
                score_box = gr.Textbox(label="Quality Score", interactive=False)

            with gr.Row():
                spec_in_img  = gr.Image(label="Input Spectrogram",  type="pil")
                spec_out_img = gr.Image(label="Output Spectrogram", type="pil")

            gr.Examples(
                examples=[
                    ["examples/example_lf_severe.wav"],
                    ["examples/example_lf_moderate.wav"],
                    ["examples/example_general_noise.wav"],
                ],
                inputs=[audio_in],
                label="Example inputs (LF severe / LF moderate / general noise)",
            )

            btn.click(fn=run, inputs=[audio_in, model_dropdown],
                      outputs=[audio_orig, audio_out, score_box, spec_in_img, spec_out_img,
                                status_md, denoised_state])

        transcript_tab.build_transcript_tab(denoised_state)

if __name__ == "__main__":
    demo.launch()
