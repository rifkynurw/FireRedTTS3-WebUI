import os
import sys
import time
import math
import random
import traceback
import gc
import subprocess
import tempfile
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

def set_voice_preset(preset_type):
    if preset_type == "Natural":
        return 1.0, 0.0
    elif preset_type == "Deep & Calm":
        return 0.96, -1.0
    elif preset_type == "Upbeat / Fast":
        return 1.06, +0.5
    elif preset_type == "News Broadcaster":
        return 0.98, -0.5
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
    # Hero Banner
    gr.HTML(
        """<div class="hero-card">
          <div class="hero-brand">FIRERED STUDIO PRO</div>
          <div class="hero-title">Zero-Shot Indonesian Voice Cloning</div>
          <div class="hero-subtitle">Teknologi sintesis suara presisi tinggi bertenaga FireRedTTS3-Base & Rubber Band R3 HQ DSP Engine.</div>
          <div class="badge-container">
            <span class="badge">🎙️ Rifky Wijayanto Studio</span>
            <span class="badge badge-cyan">⚡ T4 GPU Accelerated</span>
            <span class="badge">🎛️ Rubber Band R3 HQ</span>
          </div>
        </div>"""
    )

    with gr.Row(equal_height=False):
        # Left Panel (Controls & Inputs)
        with gr.Column(scale=3):
            # Step 1: Voice Source
            with gr.Group(elem_classes=["glass-panel"]):
                gr.HTML('<div class="section-header"><span class="step-badge">1</span> Sumber Suara Acuan (Voice Sample)</div>')
                with gr.Row():
                    reference_audio = gr.Audio(
                        sources=["upload", "microphone"],
                        type="filepath",
                        label="Audio Referensi (5–15 Detik)",
                    )
                    reference_transcript = gr.Textbox(
                        label="Transkrip Tepat Reference Audio",
                        lines=4,
                        placeholder="Ketik PERSIS kata-kata yang diucapkan dalam audio referensi...",
                    )

            # Step 2: Target Script
            with gr.Group(elem_classes=["glass-panel"]):
                gr.HTML('<div class="section-header"><span class="step-badge">2</span> Skrip Naskah Baru</div>')
                target_text = gr.Textbox(
                    label="Teks yang Akan Diucapkan (TTS)",
                    lines=4,
                    max_lines=6,
                    placeholder="Tuliskan kalimat Bahasa Indonesia yang ingin disintesiskan ke dalam suara baru...",
                )
                
                # Interactive Text Samples
                gr.Markdown("<small style='opacity:0.7;'> Contoh Cepat Naskah:</small>")
                with gr.Row():
                    sample_1 = gr.Button("📢 Sambutan", elem_classes=["preset-btn"])
                    sample_2 = gr.Button("🎙️ Berita Tech", elem_classes=["preset-btn"])
                    sample_3 = gr.Button("💬 Santai", elem_classes=["preset-btn"])

                with gr.Row():
                    language = gr.Dropdown(
                        choices=SUPPORTED_LANGUAGES,
                        value="Indonesian",
                        label="Target Bahasa",
                    )
                    seed = gr.Number(
                        value=DEFAULT_SEED,
                        precision=0,
                        label="Inference Seed",
                    )
                    random_button = gr.Button("🎲 Random Seed", elem_classes=["preset-btn"])

            # Step 3: Voice Tuning
            with gr.Group(elem_classes=["glass-panel"]):
                gr.HTML('<div class="section-header"><span class="step-badge">3</span> Tuning Suara (DSP HQ)</div>')
                
                # Quick Preset Buttons
                gr.Markdown("<small style='opacity:0.7;'> Preset Suara Cepat:</small>")
                with gr.Row():
                    preset_nat = gr.Button("✨ Natural (1.0x / 0st)", elem_classes=["preset-btn"])
                    preset_deep = gr.Button("🎙️ Deep (0.96x / -1.0st)", elem_classes=["preset-btn"])
                    preset_fast = gr.Button("⚡ Upbeat (1.06x / +0.5st)", elem_classes=["preset-btn"])
                    preset_news = gr.Button("📰 News (0.98x / -0.5st)", elem_classes=["preset-btn"])

                with gr.Row():
                    output_speed = gr.Slider(
                        minimum=MIN_OUTPUT_SPEED,
                        maximum=MAX_OUTPUT_SPEED,
                        value=1.0,
                        step=0.01,
                        label="Kecepatan Bicara (Speed)",
                    )
                    pitch_semitones = gr.Slider(
                        minimum=MIN_PITCH_SEMITONES,
                        maximum=MAX_PITCH_SEMITONES,
                        value=0.0,
                        step=0.5,
                        label="Niti / Tinggi Nada (Pitch)",
                    )
                reset_button = gr.Button("🔄 Reset Audio Controls", elem_classes=["preset-btn"])

        # Right Panel (Output & Preview)
        with gr.Column(scale=2):
            with gr.Group(elem_classes=["glass-panel"]):
                gr.HTML('<div class="section-header"><span class="step-badge">4</span> Hasil Sintesis Suara</div>')
                
                generate_button = gr.Button(
                    "🚀 Hasilkan Suara Klona", 
                    variant="primary", 
                    elem_classes=["generate-btn"]
                )
                
                gr.Markdown("---")
                generated_audio = gr.Audio(label="Audio Output (HQ Preview)", autoplay=False)
                saved_path = gr.Textbox(label="Lokasi Penyimpanan Master File (.wav)", interactive=False)

    # Event Handlers & Interactions
    sample_1.click(
        lambda: "Selamat datang di FireRed Studio. Suara ini dikloning secara presisi menggunakan teknologi AI terbaru.",
        outputs=[target_text]
    )
    sample_2.click(
        lambda: "Perkembangan kecerdasan buatan dalam pemrosesan audio kini memungkinkan pembacaan teks dengan artikulasi yang sangat alami.",
        outputs=[target_text]
    )
    sample_3.click(
        lambda: "Halo semuanya! Semoga hari kalian menyenangkan dan proyek audio kalian berjalan dengan lancar ya.",
        outputs=[target_text]
    )

    preset_nat.click(lambda: set_voice_preset("Natural"), outputs=[output_speed, pitch_semitones])
    preset_deep.click(lambda: set_voice_preset("Deep & Calm"), outputs=[output_speed, pitch_semitones])
    preset_fast.click(lambda: set_voice_preset("Upbeat / Fast"), outputs=[output_speed, pitch_semitones])
    preset_news.click(lambda: set_voice_preset("News Broadcaster"), outputs=[output_speed, pitch_semitones])

    reset_button.click(reset_controls, inputs=[], outputs=[output_speed, pitch_semitones])
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