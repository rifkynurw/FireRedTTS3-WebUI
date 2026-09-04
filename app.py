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

# Opsi Teks Sampel
TEXT_OPTION_1 = "Selamat datang di era baru teknologi penyuaraan digital. Di sini, setiap kata yang Anda tuliskan mampu diubah menjadi alunan suara yang jernih, alami, dan penuh emosi. Teknologi ini dirancang untuk membantu para kreator konten, penulis, serta profesional dalam menghidupkan narasi mereka secara presisi. Mulailah mengekspresikan ide-ide terbaik Anda dengan kualitas audio berstandar studio tinggi sekarang juga."
TEXT_OPTION_2 = "Perkembangan teknologi audio saat ini memungkinkan proses kloning suara dilakukan dengan sangat cepat dan akurat. Anda tidak perlu lagi melakukan rekaman ulang berulang kali untuk mendapatkan hasil penyampaian yang sempurna. Cukup masukkan teks serta contoh sampel suara referensi, lalu biarkan sistem bekerja menghasilkan artikulasi yang alami, intonasi yang pas, serta kualitas suara yang sangat jernih."
TEXT_OPTION_3 = "Halo semuanya! Selamat datang kembali di ruang kreatif kita. Hari ini saya mau berbagi cerita menarik tentang bagaimana ide-ide kecil bisa diubah menjadi karya besar dengan bantuan alat audio yang tepat. Jangan lupa untuk terus mengeksplorasi potensi diri Anda, mencoba hal-hal baru, dan menciptakan konten yang menginspirasi banyak orang. Terima kasih sudah mendengarkan dan semoga hari Anda selalu menyenangkan!"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def validate_reference_audio(path):
    if not path:
        raise gr.Error("Audio referensi belum dipilih.")
    p = Path(path)
    if not p.is_file():
        raise gr.Error("Audio referensi tidak ditemukan.")
    if p.stat().st_size <= 0:
        raise gr.Error("Audio referensi kosong.")
    try:
        info = sf.info(str(p))
    except Exception as e:
        raise gr.Error(f"Audio referensi tidak dapat dibaca: {e}")
    if info.frames <= 0 or info.samplerate <= 0:
        raise gr.Error("Metadata audio referensi tidak valid.")
    duration = info.frames / float(info.samplerate)
    if duration < MIN_PROMPT_SECONDS:
        raise gr.Error(f"Audio referensi terlalu pendek ({duration:.2f}s). Gunakan minimal {MIN_PROMPT_SECONDS:.1f}s.")
    if duration > MAX_PROMPT_SECONDS:
        raise gr.Error(f"Audio referensi terlalu panjang ({duration:.2f}s). Gunakan <= {MAX_PROMPT_SECONDS:.0f}s.")
    return str(p), duration, int(info.samplerate)

def prepare_reference_audio(path):
    path, duration, sr = validate_reference_audio(path)
    try:
        waveform, loaded_sr = torchaudio.load(path)
    except Exception as e:
        raise gr.Error(f"Audio referensi gagal dimuat: {e}")
    waveform = waveform.float()
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 2 or waveform.shape[-1] <= 0:
        raise gr.Error(f"Waveform referensi tidak valid: {tuple(waveform.shape)}")
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if not torch.isfinite(waveform).all():
        raise gr.Error("Audio referensi mengandung nilai tidak valid.")
    return waveform.cpu(), int(loaded_sr), duration

def validate_generated_audio(audio):
    if not torch.is_tensor(audio):
        audio = torch.as_tensor(audio)
    audio = audio.detach().float().cpu()
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    if audio.ndim != 2 or audio.shape[-1] <= 0:
        raise RuntimeError(f"Ukuran audio tidak valid: {tuple(audio.shape)}")
    if not torch.isfinite(audio).all():
        raise gr.Error("Audio sintesis mengandung nilai tidak valid.")
    peak = float(audio.abs().max().item())
    if not math.isfinite(peak) or peak <= 0:
        raise gr.Error("Audio sintesis kosong.")
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
        raise gr.Error("Proses penyelarasan nada & kecepatan audio gagal.")
    return result

def apply_voice_controls(audio_tensor, sr, speed, pitch):
    speed = float(speed)
    pitch = float(pitch)
    if not MIN_OUTPUT_SPEED <= speed <= MAX_OUTPUT_SPEED:
        raise gr.Error(f"Kecepatan harus {MIN_OUTPUT_SPEED:.2f}–{MAX_OUTPUT_SPEED:.2f}.")
    if not MIN_PITCH_SEMITONES <= pitch <= MAX_PITCH_SEMITONES:
        raise gr.Error(f"Nada harus {MIN_PITCH_SEMITONES:+.0f} hingga {MAX_PITCH_SEMITONES:+.0f} semitone.")
    if not torch.is_tensor(audio_tensor):
        audio_tensor = torch.as_tensor(audio_tensor)
    x = audio_tensor.detach().float().cpu()
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if x.ndim != 2 or x.shape[0] != 1 or x.shape[-1] == 0:
        raise gr.Error("Output audio tidak valid.")
    if not torch.isfinite(x).all():
        raise gr.Error("Output audio mengandung nilai tidak valid.")
    if abs(speed - 1.0) <= 1e-9 and abs(pitch) <= 1e-9:
        return x.squeeze(0).numpy().astype(np.float32, copy=True)

    waveform = x.squeeze(0).numpy().astype(np.float32, copy=False)
    with tempfile.TemporaryDirectory(prefix="cangkemanmu_rb_", dir="/content") as tmpdir:
        tmp = Path(tmpdir)
        input_raw = tmp / "input.f32"
        output_raw = tmp / "output.f32"
        output_wav = tmp / "output.wav"
        sf.write(str(tmp / "input.wav"), waveform, int(sr), subtype="FLOAT")
        input_raw.write_bytes(waveform.tobytes(order="C"))
        result = _run_rubberband_hq(speed, pitch, input_raw, output_raw, int(sr))
        if not output_raw.is_file() or output_raw.stat().st_size <= 0:
            raise gr.Error("Proses pemrosesan nada audio gagal.")
        raw = np.fromfile(output_raw, dtype=np.float32)
        if raw.size == 0 or not np.isfinite(raw).all():
            raise gr.Error("Hasil pemrosesan audio tidak valid.")
        peak = float(np.max(np.abs(raw)))
        if not math.isfinite(peak) or peak <= 0:
            raise gr.Error("Output audio tidak valid.")
        if peak > 1.0:
            raw = raw / peak * 0.999
        sf.write(str(output_wav), raw, int(sr), subtype="FLOAT")
        check, out_sr = sf.read(str(output_wav), dtype="float32", always_2d=True)
        if int(out_sr) != int(sr) or check.ndim != 2 or check.shape[1] != 1:
            raise gr.Error("Output audio akhir tidak valid.")
        check = check[:, 0]
        if check.size == 0 or not np.isfinite(check).all():
            raise gr.Error("Hasil akhir audio tidak valid.")
        return np.asarray(check, dtype=np.float32)

def generate_voice(target_text, reference_transcript, reference_audio, language, seed, output_speed, pitch_semitones):
    try:
        target_text = (target_text or "").strip()
        reference_transcript = (reference_transcript or "").strip()
        if not target_text:
            raise gr.Error("Naskah target masih kosong.")
        if not reference_transcript:
            raise gr.Error("Transkrip referensi kosong. Isi sesuai ucapan sampel audio.")
        if len(target_text) > MAX_TARGET_CHARS:
            raise gr.Error(f"Naskah terlalu panjang ({len(target_text)} karakter); maksimum {MAX_TARGET_CHARS}.")
        if language not in SUPPORTED_LANGUAGES:
            raise gr.Error(f"Bahasa tidak didukung: {language}")

        seed = int(seed)
        speed = float(output_speed)
        pitch = float(pitch_semitones)

        prompt_audio, prompt_audio_sr, duration = prepare_reference_audio(reference_audio)
        set_seed(seed)

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
        out_path = OUTPUT_DIR / (f"cangkemanmu_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_spd{safe_speed}_pit{safe_pitch}st.wav")
        master_path = out_path.with_name(out_path.stem + "_MASTER.wav")
        pcm16_path = out_path.with_name(out_path.stem + "_PCM16.wav")
        sf.write(str(master_path), processed, int(gen_audio_sr), subtype="FLOAT")
        sf.write(str(pcm16_path), processed, int(gen_audio_sr), subtype="PCM_16")
        return (int(gen_audio_sr), processed), str(master_path)

    except gr.Error:
        raise
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"Kloning suara gagal: {type(e).__name__}: {e}")
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

APP_DIR = Path(__file__).resolve().parent
THEME_CSS_FILE = APP_DIR / "theme.css"
css = THEME_CSS_FILE.read_text(encoding="utf-8") if THEME_CSS_FILE.is_file() else ""

def handle_preset_change(preset_name):
    if "Alami" in preset_name: return 1.0, 0.0
    if "Dalam" in preset_name: return 0.96, -1.0
    if "Enerjik" in preset_name: return 1.06, +0.5
    if "Formal" in preset_name: return 0.98, -0.5
    return 1.0, 0.0

with gr.Blocks(title="CANGKEMANMU", css=css, theme=gr.themes.Default()) as demo:

    # State tersembunyi untuk input seed yang diwajibkan oleh backend
    hidden_seed = gr.Number(value=DEFAULT_SEED, visible=False)

    with gr.Row(elem_classes="app-container"):
        # Sidebar: hanya elemen navigasi yang sudah ada di project.
        with gr.Column(elem_classes="sidebar"):
            gr.HTML("""
            <div class="brand">
                <span class="brand-mark" aria-hidden="true">II</span>
                <span class="brand-name">Cangkemanmu</span>
            </div>

            <div class="nav-section">
                <a href="#" class="nav-item">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><path d="M3 11.5 12 4l9 7.5"/><path d="M5.5 10.5V20h13v-9.5"/><path d="M9.5 20v-5h5v5"/></svg>
                    <span>Home</span>
                </a>
                <a href="#" class="nav-item">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><circle cx="8" cy="8" r="2.5"/><circle cx="16" cy="8" r="2.5"/><circle cx="8" cy="16" r="2.5"/><circle cx="16" cy="16" r="2.5"/></svg>
                    <span>Voices</span><span class="plus-icon">+</span>
                </a>
                <a href="#" class="nav-item">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><path d="M5 5.5h10l4 4V19a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V6.5a1 1 0 0 1 1-1Z"/><path d="M15 5.5V10h4"/><path d="M8 14h8M8 17h5"/></svg>
                    <span>Studio Suara</span>
                </a>
                <a href="#" class="nav-item">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><path d="m6 6 12 6-12 6V6Z"/><path d="M12 3v18" opacity="0"/></svg>
                    <span>Templat</span>
                </a>
                <a href="#" class="nav-item">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><path d="M4.5 7.5h15v11h-15z"/><path d="M7 7.5V5h10v2.5"/></svg>
                    <span>Aset</span>
                </a>
            </div>

            <div class="nav-section-title">Pintasan</div>
            <div class="nav-section">
                <a href="#" class="nav-item active">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><path d="M4 12h3l2-5 3 10 2-5h6"/></svg>
                    <span>Kloning Suara</span>
                </a>
                <a href="#" class="nav-item">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.9l.1.1-2 2-.1-.1a1.7 1.7 0 0 0-1.9-.3 1.7 1.7 0 0 0-1 1.5v.1h-2.8v-.1a1.7 1.7 0 0 0-1-1.5 1.7 1.7 0 0 0-1.9.3l-.1.1-2-2 .1-.1a1.7 1.7 0 0 0 .3-1.9 1.7 1.7 0 0 0-1.5-1H5v-2.8h.1a1.7 1.7 0 0 0 1.5-1 1.7 1.7 0 0 0-.3-1.9l-.1-.1 2-2 .1.1a1.7 1.7 0 0 0 1.9.3 1.7 1.7 0 0 0 1-1.5V5h2.8v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.9-.3l.1-.1 2 2-.1.1a1.7 1.7 0 0 0-.3 1.9 1.7 1.7 0 0 0 1.5 1h.1V14h-.1a1.7 1.7 0 0 0-1.5 1Z"/></svg>
                    <span>Pengaturan Lanjutan</span>
                </a>
            </div>

            <div class="sidebar-bottom">
                <a href="#" class="nav-item">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><rect x="4" y="4" width="16" height="16" rx="2"/><circle cx="9" cy="9" r="1.5"/><path d="m20 15-4.5-4.5L7 19"/></svg>
                    <span>Cangkemanmu</span>
                </a>
                <div class="bottom-switch-wrap">
                    <a href="#" class="nav-item">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><path d="M4 12h3l2-5 3 10 2-5h6"/></svg>
                        <span>Creative Suite</span>
                    </a>
                    <button class="switch-btn" type="button">Switch</button>
                </div>
            </div>
            """)

        with gr.Column(elem_classes="main-wrapper"):
            gr.HTML("""
            <div class="top-header">
                <div class="header-title">
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="4" y="4" width="16" height="16" rx="2"/><path d="M8 9h8M8 13h6"/></svg>
                    <span>Text to Speech</span>
                </div>
                <div class="header-spacer"></div>
                <div class="header-actions">
                    <div class="header-search">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="11" cy="11" r="7"/><path d="m20 20-4-4"/></svg>
                        <input type="text" placeholder="Pencarian..." aria-label="Pencarian" />
                        <span class="search-shortcut">⌘&nbsp; K</span>
                    </div>
                    <button type="button">Feedback</button>
                    <button type="button">Docs</button>
                    <button type="button">Ask</button>
                    <button class="icon-btn" type="button" aria-label="Folder">
                        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M3.5 6.5h6l2 2h9v10a1 1 0 0 1-1 1h-16z"/><path d="M3.5 9h17"/></svg>
                    </button>
                    <button class="icon-btn has-dot" type="button" aria-label="Notifikasi">
                        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M18 9a6 6 0 0 0-12 0c0 5-2 5.5-2 7h16c0-1.5-2-2-2-7Z"/><path d="M10 19a2 2 0 0 0 4 0"/></svg>
                    </button>
                    <div class="avatar">h</div>
                </div>
            </div>
            """)

            with gr.Row(elem_classes="workspace-split"):
                with gr.Column(elem_classes="pane-left"):
                    target_text = gr.Textbox(
                        lines=12,
                        show_label=False,
                        elem_id="target-input",
                        placeholder="Mulai mengetik di sini atau tempel teks apa pun yang ingin diubah menjadi suara alami..."
                    )

                    reference_transcript = gr.Textbox(
                        lines=2,
                        show_label=False,
                        elem_id="ref-transcript-input",
                        placeholder="Transkrip referensi — ketik persis ucapan pada sampel audio..."
                    )

                    gr.HTML("<div class='prompt-label'>Coba naskah contoh</div>")
                    with gr.Row(elem_classes="prompt-grid"):
                        btn_story = gr.Button("📖  Narasi sebuah cerita", elem_classes="prompt-btn")
                        btn_joke = gr.Button("☺  Ceritakan lelucon lucu", elem_classes="prompt-btn")
                        btn_ad = gr.Button("◉  Rekam iklan", elem_classes="prompt-btn")
                        btn_lang = gr.Button("文  Berbicara dalam berbagai bahasa", elem_classes="prompt-btn")
                        btn_movie = gr.Button("▣  Arahkan adegan film dramatis", elem_classes="prompt-btn")
                        btn_game = gr.Button("◉  Dengarkan dari karakter video game", elem_classes="prompt-btn")
                        btn_pod = gr.Button("◉  Perkenalkan podcast Anda", elem_classes="prompt-btn")
                        btn_med = gr.Button("◌  Pandu kelas meditasi", elem_classes="prompt-btn")

                with gr.Column(elem_classes="pane-right"):
                    with gr.Tabs(elem_classes="settings-tabs"):
                        with gr.Tab("Pengaturan", id=1):
                            gr.HTML("""
                            <div class="promo-banner">
                                <div class="promo-icon">
                                    <div class="promo-glow"></div>
                                    <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="1.6"><path d="M6 5v10m0 0c0 2 1.6 3.5 3.5 3.5S13 17 13 15V9c0-1.7 1.3-3 3-3s3 1.3 3 3v6"/><circle cx="6" cy="5" r="1.3" fill="white" stroke="none"/><circle cx="19" cy="15" r="1.3" fill="white" stroke="none"/></svg>
                                </div>
                                <div class="promo-text">
                                    <div class="promo-title">Try Flows</div>
                                    <div class="promo-desc">Node-based canvas for creating image, video, speech, music, all in one place.</div>
                                </div>
                                <div class="promo-close">×</div>
                            </div>
                            """)

                            gr.HTML("<div class='setting-title'>Voice</div>")
                            voice_preset = gr.Dropdown(
                                choices=["Preset Alami - Tenang, Jernih", "Dalam & Tenang", "Cepat & Enerjik", "Formal / Narasi"],
                                value="Preset Alami - Tenang, Jernih",
                                show_label=False,
                                elem_classes="custom-dropdown"
                            )

                            gr.HTML("<div class='setting-title section-gap'>Bahasa</div>")
                            language = gr.Dropdown(
                                choices=SUPPORTED_LANGUAGES,
                                value="Indonesian",
                                show_label=False,
                                elem_classes="custom-dropdown"
                            )

                            with gr.Row(elem_classes="sliders-row"):
                                output_speed = gr.Slider(minimum=MIN_OUTPUT_SPEED, maximum=MAX_OUTPUT_SPEED, value=1.0, step=0.01, label="Kecepatan", elem_classes="slim-slider")
                                pitch_semitones = gr.Slider(minimum=MIN_PITCH_SEMITONES, maximum=MAX_PITCH_SEMITONES, value=0.0, step=0.5, label="Nada", elem_classes="slim-slider")

                            gr.HTML("<div class='setting-title section-gap'>Sampel Suara Referensi</div>")
                            reference_audio = gr.Audio(
                                sources=["upload", "microphone"],
                                type="filepath",
                                label=None,
                                elem_classes="custom-audio-uploader"
                            )

                            gr.HTML("<div class='setting-title section-gap result-title'>Hasil Sintesis</div>")
                            generated_audio = gr.Audio(show_label=False, autoplay=False, elem_classes="custom-audio-player")
                            saved_path = gr.Textbox(visible=False)

                            generate_btn = gr.Button("⚡  Kloning Suara Sekarang", elem_classes="generate-btn")

                        with gr.Tab("Riwayat", id=2):
                            gr.HTML("<div class='history-empty'>Belum ada riwayat kloning suara.</div>")

    # -------------------------------------------------------------
    # EVENT HANDLERS
    # -------------------------------------------------------------
    # Mengikat Preset Dropdown ke Slider
    voice_preset.change(handle_preset_change, inputs=[voice_preset], outputs=[output_speed, pitch_semitones])

    # Mengikat Naskah Sampel ke Target Text
    btn_story.click(lambda: TEXT_OPTION_1, outputs=[target_text])
    btn_joke.click(lambda: TEXT_OPTION_2, outputs=[target_text])
    btn_ad.click(lambda: TEXT_OPTION_3, outputs=[target_text])
    btn_lang.click(lambda: TEXT_OPTION_1, outputs=[target_text])
    btn_movie.click(lambda: TEXT_OPTION_2, outputs=[target_text])
    btn_game.click(lambda: TEXT_OPTION_3, outputs=[target_text])
    btn_pod.click(lambda: TEXT_OPTION_1, outputs=[target_text])
    btn_med.click(lambda: TEXT_OPTION_2, outputs=[target_text])

    # Eksekusi Generate
    generate_btn.click(
        generate_voice,
        inputs=[target_text, reference_transcript, reference_audio, language, hidden_seed, output_speed, pitch_semitones],
        outputs=[generated_audio, saved_path]
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
    print("CANGKEMANMU_WEBUI_URL:", selected_url, flush=True)
else:
    print("CANGKEMANMU_WEBUI_URL: unavailable", flush=True)

while True:
    time.sleep(3600)