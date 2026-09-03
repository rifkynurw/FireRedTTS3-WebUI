
import os
import sys
import time
import math
import random
import traceback
import gc
import subprocess
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

import gradio as gr
import numpy as np
import soundfile as sf
import torch
import torchaudio

ROOT_DIR = Path(os.environ["FIRERED_REPO_DIR"]).resolve()
MODEL_DIR = Path(os.environ["FIRERED_MODEL_DIR"]).resolve()
OUTPUT_DIR = Path(os.environ["FIRERED_OUTPUT_DIR"]).resolve()
URL_FILE = Path(os.environ["FIRERED_URL_FILE"]).resolve()
RUBBERBAND_HQ_BIN = Path(os.environ["FIRERED_RUBBERBAND_HQ_BIN"]).resolve()
RUBBERBAND_LIB_DIR = Path(os.environ["FIRERED_RUBBERBAND_LIB_DIR"]).resolve()

MAX_TARGET_CHARS = int(os.environ.get("FIRERED_MAX_TARGET_CHARS", "500"))
MIN_PROMPT_SECONDS = float(os.environ.get("FIRERED_MIN_PROMPT_SECONDS", "2.0"))
MAX_PROMPT_SECONDS = float(os.environ.get("FIRERED_MAX_PROMPT_SECONDS", "20.0"))
PORT = int(os.environ.get("FIRERED_GRADIO_PORT", "7860"))
DEFAULT_SEED = int(os.environ.get("FIRERED_DEFAULT_SEED", "1986"))

SUPPORTED_LANGUAGES = [
    "Arabic", "Cantonese", "Chinese", "Czech", "Dutch", "English",
    "Finnish", "French", "German", "Greek", "Hindi", "Indonesian",
    "Italian", "Japanese", "Korean", "Polish", "Portuguese", "Romanian",
    "Russian", "Spanish", "Thai", "Turkish", "Ukrainian", "Vietnamese",
]

sys.path.insert(0, str(ROOT_DIR))
from fireredtts3.core import FireRedTTS3

tts = FireRedTTS3(
    str(MODEL_DIR),
    use_wetext=True,
    use_llm_tn=False,
)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable in WebUI process; CPU fallback is disabled.")

MODEL_SR = 24000
MIN_OUTPUT_SPEED = 0.85
MAX_OUTPUT_SPEED = 1.15
MIN_PITCH_SEMITONES = -2.0
MAX_PITCH_SEMITONES = 2.0

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def validate_reference_audio(path):
    if not path:
        raise gr.Error("Reference audio belum dipilih.")
    p = Path(path)
    if not p.is_file():
        raise gr.Error("Reference audio tidak ditemukan.")
    if p.stat().st_size <= 0:
        raise gr.Error("Reference audio kosong.")
    try:
        info = sf.info(str(p))
    except Exception as e:
        raise gr.Error(f"Reference audio tidak dapat dibaca: {e}")
    if info.frames <= 0 or info.samplerate <= 0:
        raise gr.Error("Metadata reference audio tidak valid.")
    duration = info.frames / float(info.samplerate)
    if duration < MIN_PROMPT_SECONDS:
        raise gr.Error(
            f"Reference audio terlalu pendek ({duration:.2f}s). "
            f"Gunakan minimal {MIN_PROMPT_SECONDS:.1f}s."
        )
    if duration > MAX_PROMPT_SECONDS:
        raise gr.Error(
            f"Reference audio terlalu panjang ({duration:.2f}s). "
            f"Gunakan <= {MAX_PROMPT_SECONDS:.0f}s."
        )
    return str(p), duration, int(info.samplerate)

def prepare_reference_audio(path):
    path, duration, sr = validate_reference_audio(path)
    try:
        waveform, loaded_sr = torchaudio.load(path)
    except Exception as e:
        raise gr.Error(f"Reference audio gagal dimuat: {e}")
    waveform = waveform.float()
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 2 or waveform.shape[-1] <= 0:
        raise gr.Error(f"Waveform reference tidak valid: {tuple(waveform.shape)}")
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if not torch.isfinite(waveform).all():
        raise gr.Error("Reference audio mengandung NaN/Inf.")
    return waveform.cpu(), int(loaded_sr), duration

def validate_generated_audio(audio):
    if not torch.is_tensor(audio):
        audio = torch.as_tensor(audio)
    audio = audio.detach().float().cpu()
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    if audio.ndim != 2 or audio.shape[-1] <= 0:
        raise RuntimeError(f"Unexpected generated audio shape: {tuple(audio.shape)}")
    if not torch.isfinite(audio).all():
        raise gr.Error("Generated audio mengandung NaN/Inf; tidak disimpan.")
    peak = float(audio.abs().max().item())
    if not math.isfinite(peak) or peak <= 0:
        raise gr.Error("Generated audio kosong/tidak valid.")
    if peak > 1.0:
        audio = audio / peak * 0.999
    return audio

def _run_rubberband(cmd):
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise gr.Error(
            "Rubber Band timeout setelah 120 detik. Coba kalimat yang lebih pendek."
        ) from exc

    if result.returncode != 0:
        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        raise gr.Error(
            f"Rubber Band gagal (returncode={result.returncode}).\n"
            f"COMMAND: {' '.join(cmd)}\n"
            f"STDOUT:\n{stdout[-2500:]}\n"
            f"STDERR:\n{stderr[-3500:]}"
        )
    return result


def _run_rubberband_hq(speed, pitch, input_path, output_path, sr):
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(RUBBERBAND_LIB_DIR) + ":" + env.get("LD_LIBRARY_PATH", "")
    cmd = [
        str(RUBBERBAND_HQ_BIN), str(int(sr)), f"{float(speed):.8f}",
        f"{float(pitch):.8f}", str(input_path), str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180, env=env)
    if result.returncode != 0:
        raise gr.Error(
            "Rubber Band R3 HQ gagal.\n"
            f"returncode={result.returncode}\n"
            f"STDOUT:\n{result.stdout[-6000:]}\n"
            f"STDERR:\n{result.stderr[-6000:]}"
        )
    return result


def apply_voice_controls(audio_tensor, sr, speed, pitch):
    speed = float(speed)
    pitch = float(pitch)
    if not MIN_OUTPUT_SPEED <= speed <= MAX_OUTPUT_SPEED:
        raise gr.Error(f"Speed harus {MIN_OUTPUT_SPEED:.2f}–{MAX_OUTPUT_SPEED:.2f}.")
    if not MIN_PITCH_SEMITONES <= pitch <= MAX_PITCH_SEMITONES:
        raise gr.Error(
            f"Pitch harus {MIN_PITCH_SEMITONES:+.0f} hingga {MAX_PITCH_SEMITONES:+.0f} semitone."
        )
    if not torch.is_tensor(audio_tensor):
        audio_tensor = torch.as_tensor(audio_tensor)
    x = audio_tensor.detach().float().cpu()
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if x.ndim != 2 or x.shape[0] != 1 or x.shape[-1] == 0:
        raise gr.Error(f"Output audio tidak valid untuk voice controls: {tuple(x.shape)}")
    if not torch.isfinite(x).all():
        raise gr.Error("Output audio mengandung NaN/Inf.")
    if abs(speed - 1.0) <= 1e-9 and abs(pitch) <= 1e-9:
        return x.squeeze(0).numpy().astype(np.float32, copy=True)

    waveform = x.squeeze(0).numpy().astype(np.float32, copy=False)
    with tempfile.TemporaryDirectory(prefix="fireredtts3_rb_hq_", dir="/content") as tmpdir:
        tmp = Path(tmpdir)
        input_raw = tmp / "input.f32"
        output_raw = tmp / "output.f32"
        output_wav = tmp / "output.wav"
        sf.write(str(tmp / "input.wav"), waveform, int(sr), subtype="FLOAT")
        input_raw.write_bytes(waveform.tobytes(order="C"))
        result = _run_rubberband_hq(speed, pitch, input_raw, output_raw, int(sr))
        if result.stdout:
            print("[Rubber Band R3 HQ]", result.stdout.strip(), flush=True)
        if not output_raw.is_file() or output_raw.stat().st_size <= 0:
            raise gr.Error("Rubber Band R3 HQ tidak menghasilkan output float32.")
        raw = np.fromfile(output_raw, dtype=np.float32)
        if raw.size == 0 or not np.isfinite(raw).all():
            raise gr.Error("Output Rubber Band R3 HQ kosong atau mengandung NaN/Inf.")
        peak = float(np.max(np.abs(raw)))
        if not math.isfinite(peak) or peak <= 0:
            raise gr.Error("Output Rubber Band R3 HQ tidak valid.")
        if peak > 1.0:
            raw = raw / peak * 0.999
        sf.write(str(output_wav), raw, int(sr), subtype="FLOAT")
        check, out_sr = sf.read(str(output_wav), dtype="float32", always_2d=True)
        if int(out_sr) != int(sr) or check.ndim != 2 or check.shape[1] != 1:
            raise gr.Error(f"Output Rubber Band R3 HQ WAV tidak valid/mono: shape={check.shape}, sr={out_sr}")
        check = check[:, 0]
        if check.size == 0 or not np.isfinite(check).all():
            raise gr.Error("Output Rubber Band R3 HQ WAV invalid setelah penulisan.")
        return np.asarray(check, dtype=np.float32)

def random_seed():
    return random.randint(1, 2147483647)

def reset_controls():
    return 1.0, 0.0

def generate_voice(
    target_text,
    reference_transcript,
    reference_audio,
    language,
    seed,
    output_speed,
    pitch_semitones,
):
    try:
        target_text = (target_text or "").strip()
        reference_transcript = (reference_transcript or "").strip()
        if not target_text:
            raise gr.Error("Target text kosong.")
        if not reference_transcript:
            raise gr.Error("Reference transcript kosong. Isi PERSIS ucapan pada reference audio.")
        if len(target_text) > MAX_TARGET_CHARS:
            raise gr.Error(
                f"Target text terlalu panjang ({len(target_text)} karakter); maksimum {MAX_TARGET_CHARS}."
            )
        if language not in SUPPORTED_LANGUAGES:
            raise gr.Error(f"Unsupported language: {language}")

        seed = int(seed)
        speed = float(output_speed)
        pitch = float(pitch_semitones)

        prompt_audio, prompt_audio_sr, duration = prepare_reference_audio(reference_audio)
        set_seed(seed)

        print(
            f"[INFO] language={language} prompt_duration={duration:.2f}s "
            f"prompt_sr={prompt_audio_sr} seed={seed} "
            f"speed={speed:.2f} pitch={pitch:+.1f}st",
            flush=True,
        )

        # Explicit known-good model-side baseline.
        with torch.inference_mode():
            gen_audio, gen_audio_sr = tts.generate(
                language=language,
                prompt_text=reference_transcript,
                prompt_audio=prompt_audio,
                prompt_audio_sr=prompt_audio_sr,
                text=target_text,
                n_timesteps=10,
                inference_cfg=2.0,
                seed=seed,
                do_tn=True,
            )

        audio = validate_generated_audio(gen_audio)
        processed = apply_voice_controls(audio, int(gen_audio_sr), speed, pitch)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        safe_speed = f"{speed:.2f}"
        safe_pitch = f"{pitch:+.1f}".replace("+", "p").replace("-", "m")
        out_path = OUTPUT_DIR / (
            f"fireredtts3_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            f"_spd{safe_speed}_pit{safe_pitch}st.wav"
        )
        master_path = out_path.with_name(out_path.stem + "_HQ_FLOAT32.wav")
        pcm16_path = out_path.with_name(out_path.stem + "_PCM16.wav")
        sf.write(str(master_path), processed, int(gen_audio_sr), subtype="FLOAT")
        sf.write(str(pcm16_path), processed, int(gen_audio_sr), subtype="PCM_16")
        return (int(gen_audio_sr), processed), str(master_path)

    except gr.Error:
        raise
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"FireRedTTS3 inference gagal: {type(e).__name__}: {e}")
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

APP_DIR = Path(__file__).resolve().parent
THEME_CSS_FILE = APP_DIR / "theme.css"
css = THEME_CSS_FILE.read_text(encoding="utf-8") if THEME_CSS_FILE.is_file() else ""

with gr.Blocks(title="FireRed Studio — Rifky Wijayanto", css=css, theme=gr.themes.Soft()) as demo:
    gr.HTML(
        """<div class="hero">
          <div class="brand">FIRERED STUDIO</div>
          <div class="title">Indonesian Voice Cloning</div>
          <div class="subtitle">Natural zero-shot voice cloning with FireRedTTS3-Base and Rubber Band R3 HQ post-processing.</div>
          <div class="credit">Created &amp; branded by <b>Rifky Wijayanto</b> &nbsp;·&nbsp; <span class="badge">T4 Ready</span> <span class="badge">Rubber Band R3 HQ</span></div>
        </div>"""
    )

    gr.Markdown("### 1 · Create the voice", elem_classes=["section"])
    with gr.Row():
        with gr.Column(scale=1):
            reference_audio = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Reference Audio",
            )
            gr.Markdown("Gunakan 5–15 detik, 1 speaker, suara jelas, noise dan reverb rendah.")
        with gr.Column(scale=1):
            reference_transcript = gr.Textbox(
                label="Reference Transcript",
                lines=5,
                placeholder="Masukkan PERSIS ucapan pada reference audio.",
            )

    gr.Markdown("### 2 · Write your script", elem_classes=["section"])
    target_text = gr.Textbox(
        label="Text to Speech",
        lines=5,
        max_lines=8,
        placeholder="Tulis teks Bahasa Indonesia yang ingin dibacakan.",
    )
    language = gr.Dropdown(
        choices=SUPPORTED_LANGUAGES,
        value="Indonesian",
        label="Target Language",
        info="Untuk target utama gunakan Indonesian.",
    )
    seed = gr.Number(
        value=DEFAULT_SEED,
        precision=0,
        label="Inference Seed",
        info="Pertahankan seed saat membandingkan controls.",
    )

    gr.Markdown("### 3 · Voice controls", elem_classes=["section"])
    with gr.Accordion("Speed / Pitch", open=True):
        with gr.Row():
            output_speed = gr.Slider(
                minimum=MIN_OUTPUT_SPEED,
                maximum=MAX_OUTPUT_SPEED,
                value=1.0,
                step=0.01,
                label="Speed",
                info="1.00× = original; 0.98× / 0.97× = sedikit lebih lambat. Rubber Band R3 HQ digunakan.",
            )
            pitch_semitones = gr.Slider(
                minimum=MIN_PITCH_SEMITONES,
                maximum=MAX_PITCH_SEMITONES,
                value=0.0,
                step=0.5,
                label="Pitch",
                info="0 = original; + lebih tinggi; − lebih rendah.",
            )
        gr.Markdown(
            "**Intonation:** dipertahankan dari model. Tidak ada contour post-processing otomatis."
        )
        reset_button = gr.Button("Reset Voice Controls")

    with gr.Row():
        random_button = gr.Button("🎲 Randomize Seed")
        generate_button = gr.Button("Generate Voice", variant="primary", elem_classes=["generate-btn"])

    gr.Markdown("### 4 · Result", elem_classes=["section"])
    generated_audio = gr.Audio(label="Generated Audio", autoplay=False)
    saved_path = gr.Textbox(label="Saved HQ WAV Path", interactive=False)

    reset_button.click(
        reset_controls,
        inputs=[],
        outputs=[output_speed, pitch_semitones],
    )
    random_button.click(random_seed, inputs=[], outputs=[seed])
    generate_button.click(
        generate_voice,
        inputs=[
            target_text,
            reference_transcript,
            reference_audio,
            language,
            seed,
            output_speed,
            pitch_semitones,
        ],
        outputs=[generated_audio, saved_path],
    )

demo.queue(max_size=4, default_concurrency_limit=1)

launch_result = demo.launch(
    server_name="0.0.0.0",
    server_port=PORT,
    share=True,
    show_error=True,
    prevent_thread_lock=True,
)

try:
    _, local_url, share_url = launch_result
except Exception:
    local_url, share_url = None, None

selected_url = share_url or local_url
if selected_url:
    URL_FILE.write_text(str(selected_url), encoding="utf-8")
    print("FIREREDTTS3_WEBUI_URL:", selected_url, flush=True)
else:
    print("FIREREDTTS3_WEBUI_URL: unavailable", flush=True)

while True:
    time.sleep(3600)
