"""
transcript_tab.py — Thai Whisper transcription + Typhoon LLM cleanup.
Integrated into app.py as a second Gradio tab.
Call build_transcript_tab(denoised_state) inside a gr.Blocks context.
"""
from __future__ import annotations

import os
import re

import gradio as gr


MAX_DURATION_S = 60


def fix_spacing(text: str) -> str:
    text = re.sub(r'([A-Za-z0-9])([ก-๙])', r'\1 \2', text)
    text = re.sub(r'([ก-๙])([A-Za-z0-9])', r'\1 \2', text)
    return text


def _transcribe(audio_path: str) -> tuple[str, str]:
    """Transcribe via Whisper large-v3 on HF Inference API. Returns (text, warning)."""
    import soundfile as sf
    try:
        duration = sf.info(audio_path).duration
    except Exception:
        duration = 0

    if duration > MAX_DURATION_S:
        return "", (f"Audio is {duration:.0f}s — trim to under {MAX_DURATION_S}s "
                    "for reliable transcription on free-tier.")

    token = os.environ.get("HF_TOKEN") or None
    if not token:
        return "", "⚠️ Set HF_TOKEN for transcription (free HuggingFace account required — hf.co/settings/tokens)"

    try:
        from huggingface_hub import InferenceClient
        client = InferenceClient("openai/whisper-large-v3", token=token)
        result = client.automatic_speech_recognition(
            audio_path,
            extra_body={"language": "th", "task": "transcribe"},
        )
        return result.text.strip(), ""
    except Exception as exc:
        err = str(exc)
        if any(k in err for k in ("429", "rate", "Rate", "too many", "Too Many")):
            return "", "⚠️ Rate limit hit — try again in a moment or add your own HF_TOKEN"
        if any(k in err for k in ("401", "403", "nauthorized", "orbidden")):
            return "", "⚠️ Invalid HF_TOKEN — check your token at hf.co/settings/tokens"
        return "", f"⚠️ Transcription failed: {err[:150]}"


# ── Typhoon LLM (HF Inference API, no local load) ─────────────────────────────
_TYPHOON_MODEL  = "scb10x/llama3.1-typhoon2-8b-instruct"
_FALLBACK_MODEL = "Qwen/Qwen2.5-7B-Instruct"

_SYSTEM_PROMPT = """คุณคือผู้เชี่ยวชาญแก้ไขข้อความที่ได้จากการถอดเสียงภาษาไทยอัตโนมัติ (ASR)

ระบบ ASR ทำงานโดยฟังเสียงและเดาคำที่ใกล้เคียงที่สุด
ดังนั้นข้อผิดพลาดจะเป็นแบบ "เสียงคล้ายกันแต่คนละคำ" หรือ "ช่องว่างหาย"
ไม่ใช่การพิมพ์ผิดธรรมดา

หลักการแก้:
- ฟังเสียงในใจก่อน แล้วดูว่าคำที่ถูกต้องในบริบทนั้นควรเป็นอะไร
- ดูประโยคแวดล้อมเพื่อเข้าใจ context ก่อนตัดสิน
- ถ้าเป็นชื่อคนหรือชื่อเฉพาะที่ไม่แน่ใจ ให้คงไว้
- ตอบเฉพาะข้อความที่แก้แล้ว ห้ามอธิบาย

ตัวอย่างประเภทข้อผิดพลาดที่พบบ่อย:

ประเภท 1 — เสียงคล้าย สระ/วรรณยุกต์ผิด:
ก่อน: "และขอตอบรับเข้าสู่งานวันนี้"
หลัง: "และขอต้อนรับเข้าสู่งานวันนี้"
เหตุผล: "ตอบรับ" ออกเสียงคล้าย "ต้อนรับ" แต่ความหมายต่างกัน บริบทเป็นการกล่าวเปิดงาน

ประเภท 2 — คำทับศัพท์สะกดตามเสียงไทย:
ก่อน: "เขาทำงานอยู่ที่บริษัทฟาร์ฟฟัก ซึ่งเป็นบริษัทค้าปลีก"
หลัง: "เขาทำงานอยู่ที่บริษัท Farfetch ซึ่งเป็นบริษัทค้าปลีก"
เหตุผล: "ฟาร์ฟฟัก" ออกเสียงคล้าย Farfetch บริบทบอกว่าเป็นชื่อบริษัทค้าปลีก

ประเภท 3 — ช่องว่างหาย:
ก่อน: "ฉบับปี 2026ก่อนอื่นขอแนะนำตัว"
หลัง: "ฉบับปี 2026 ก่อนอื่นขอแนะนำตัว"
เหตุผล: Whisper ไม่ใส่ช่องว่างระหว่างตัวเลขกับคำไทย

ประเภท 4 — พยางค์หาย:
ก่อน: "ฉันชอบกินข้าวกับแกงเขียวหวานมาก"
หลัง: "ฉันชอบกินข้าวกับแกงเขียวหวานมาก"
เหตุผล: บางคำฟังดูถูกแต่พยางค์ขาด ให้ตรวจให้ครบ

ประเภท 5 — คำที่ออกเสียงเหมือนกันแต่ต่างความหมาย:
ก่อน: "เขาใช้วิธีการที่ถูกต้องในการแก้ปัญหา"
หลัง: "เขาใช้วิธีการที่ถูกต้องในการแก้ปัญหา"
เหตุผล: ถ้าคำถูกแล้วไม่ต้องแก้ แต่ต้องตรวจทุกคำ"""

_USER_TEMPLATE = """ข้อความต่อไปนี้ถอดมาจากเสียงพูดภาษาไทย มีข้อผิดพลาดแบบ ASR ซ่อนอยู่
ตรวจและแก้ทุกคำที่ผิด โดยใช้บริบทของประโยคช่วยตัดสิน:

{raw}

ตอบเฉพาะข้อความที่แก้แล้วเท่านั้น"""


def _llm_clean(raw_text: str) -> tuple[str | None, str | None]:
    """Pre-process → Typhoon → post-process. Returns (corrected, error)."""
    from huggingface_hub import InferenceClient

    pre = fix_spacing(raw_text)
    token = os.environ.get("HF_TOKEN") or None
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": _USER_TEMPLATE.format(raw=pre)},
    ]
    max_tokens = max(1024, len(pre) * 3)

    for model_id in (_TYPHOON_MODEL, _FALLBACK_MODEL):
        try:
            client = InferenceClient(model=model_id, token=token)
            resp = client.chat_completion(messages=messages, max_tokens=max_tokens, temperature=0.1)
            corrected = resp.choices[0].message.content.strip()
            return fix_spacing(corrected), None
        except Exception as exc:
            err = str(exc)
            if any(k in err for k in ("429", "rate", "Rate")):
                return None, "Rate-limited — try again in a moment."
            if any(k in err for k in ("401", "403", "nauthorized")):
                return None, "API key required. Set HF_TOKEN environment variable."
            if model_id == _FALLBACK_MODEL:
                return None, f"AI cleanup unavailable: {err[:120]}"

    return None, "AI cleanup unavailable."


# ── Tab builder ───────────────────────────────────────────────────────────────

def build_transcript_tab(denoised_state: gr.State) -> None:
    """
    Build the 'Transcribe (Thai)' tab inside the caller's gr.Blocks context.
    denoised_state: gr.State holding the denoised WAV path from the Denoise tab.
    """
    with gr.Tab("Transcribe (Thai)"):
        gr.Markdown(
            "**Step 1:** Denoise audio in the Denoise tab first.\n\n"
            "**Step 2:** Click **Transcribe** — uses denoised audio automatically.\n\n"
            "**Step 3 (optional):** Click **Clean with AI** to refine the raw transcript.\n\n"
            "_Transcription uses Whisper large-v3 via HuggingFace (free account required — set `HF_TOKEN` env var)._"
        )

        audio_src = gr.Audio(
            label="Audio source (auto-filled from Denoise tab, or upload separately)",
            type="filepath",
            sources=["upload"],
        )

        transcribe_btn    = gr.Button("Transcribe (Thai)", variant="primary")
        transcribe_status = gr.Markdown("")

        warning_box = gr.Textbox(
            label="Warning", visible=False, interactive=False, lines=2
        )

        raw_box = gr.Textbox(
            label="Raw transcript (Whisper large-v3, Thai)",
            lines=7,
            interactive=False,
            placeholder="Transcript appears here...",
        )

        clean_btn    = gr.Button("Clean with AI (Typhoon)", variant="secondary", interactive=False)
        clean_status = gr.Markdown("")

        cleaned_box = gr.Textbox(
            label="AI-edited draft (Typhoon) — verify against raw transcript above",
            lines=7,
            interactive=False,
            placeholder="AI-cleaned version appears here...",
            visible=False,
        )

        gr.Markdown(
            "_Disclaimer: AI output may contain errors. "
            "Do not use for medical, legal, or official purposes without human review._"
        )

        # ── Auto-fill audio_src when Denoise tab finishes ─────────────────────
        denoised_state.change(
            fn=lambda p: gr.update(value=p) if p else gr.update(),
            inputs=[denoised_state],
            outputs=[audio_src],
        )

        # ── Transcribe ────────────────────────────────────────────────────────
        def _do_transcribe(audio_path):
            if not audio_path:
                yield ("", gr.update(value="", visible=False),
                       "Upload audio or run Denoise first.",
                       gr.update(interactive=False),
                       gr.update(visible=False))
                return

            yield ("", gr.update(value="", visible=False),
                   "⏳ Transcribing via Whisper large-v3 API...",
                   gr.update(interactive=False),
                   gr.update(visible=False))

            text, warning = _transcribe(audio_path)

            if not text and warning:
                yield ("", gr.update(value=warning, visible=True),
                       "❌ " + warning,
                       gr.update(interactive=False),
                       gr.update(visible=False))
                return

            yield (text,
                   gr.update(value=warning, visible=bool(warning)),
                   "✓ Transcription done",
                   gr.update(interactive=bool(text)),
                   gr.update(visible=False))

        transcribe_btn.click(
            fn=_do_transcribe,
            inputs=[audio_src],
            outputs=[raw_box, warning_box, transcribe_status, clean_btn, cleaned_box],
        )

        # ── LLM clean ─────────────────────────────────────────────────────────
        def _do_clean(raw_text):
            if not raw_text.strip():
                yield gr.update(visible=False), "No transcript to clean."
                return

            yield gr.update(visible=False), "⏳ Calling Typhoon AI..."

            cleaned, err = _llm_clean(raw_text)

            if err:
                yield (gr.update(visible=False),
                       f"⚠️ {err} — raw transcript above is still valid.")
                return

            yield gr.update(value=cleaned, visible=True), "✓ AI cleanup done"

        clean_btn.click(
            fn=_do_clean,
            inputs=[raw_box],
            outputs=[cleaned_box, clean_status],
        )
