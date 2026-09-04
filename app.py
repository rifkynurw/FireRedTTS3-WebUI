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
        raise gr.Error(
            f"Audio referensi terlalu pendek ({duration:.2f}s). "
            f"Gunakan minimal {MIN_PROMPT_SECONDS:.1f}s."
        )
    if duration > MAX_PROMPT_SECONDS:
        raise gr.Error(
            f"Audio referensi terlalu panjang ({duration:.2f}s). "
            f"Gunakan <= {MAX_PROMPT_SECONDS:.0f}s."
        )
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
        raise RuntimeError(f"Ukuran audio hasil sintesis tidak valid: {tuple(audio.shape)}")
    if not torch.isfinite(audio).all():
        raise gr.Error("Audio hasil sintesis mengandung nilai tidak valid.")
    peak = float(audio.abs().max().item())
    if not math.isfinite(peak) or peak <= 0:
        raise gr.Error("Audio hasil sintesis kosong.")
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
        raise gr.Error("Pemrosesan nada dan kecepatan audio gagal.")
    return result

def apply_voice_controls(audio_tensor, sr, speed, pitch):
    speed = float(speed)
    pitch = float(pitch)
    if not MIN_OUTPUT_SPEED <= speed <= MAX_OUTPUT_SPEED:
        raise gr.Error(f"Kecepatan harus {MIN_OUTPUT_SPEED:.2f}–{MAX_OUTPUT_SPEED:.2f}.")
    if not MIN_PITCH_SEMITONES <= pitch <= MAX_PITCH_SEMITONES:
        raise gr.Error(
            f"Nada harus {MIN_PITCH_SEMITONES:+.0f} hingga {MAX_PITCH_SEMITONES:+.0f} semitone."
        )
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
            raise gr.Error("Pemrosesan audio gagal menghasilkan berkas.")
        raw = np.fromfile(output_raw, dtype=np.float32)
        if raw.size == 0 or not np.isfinite(raw).all():
            raise gr.Error("Pemrosesan audio tidak valid.")
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
            raise gr.Error("Naskah target masih kosong.")
        if not reference_transcript:
            raise gr.Error("Transkrip referensi kosong. Isi persis ucapan pada sampel suara.")
        if len(target_text) > MAX_TARGET_CHARS:
            raise gr.Error(
                f"Naskah terlalu panjang ({len(target_text)} karakter); maksimum {MAX_TARGET_CHARS}."
            )
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
        out_path = OUTPUT_DIR / (
            f"cangkemanmu_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            f"_spd{safe_speed}_pit{safe_pitch}st.wav"
        )
        master_path = out_path.with_name(out_path.stem + "_MASTER.wav")
        pcm16_path = out_path.with_name(out_path.stem + "_PCM16.wav")
        sf.write(str(master_path), processed, int(gen_audio_sr), subtype="FLOAT")
        sf.write(str(pcm16_path), processed, int(gen_audio_sr), subtype="PCM_16")
        return (int(gen_audio_sr), processed), str(master_path)

    except gr.Error:
        raise
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"Proses kloning suara gagal: {type(e).__name__}: {e}")
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

APP_DIR = Path(__file__).resolve().parent
THEME_CSS_FILE = APP_DIR / "theme.css"
css = THEME_CSS_FILE.read_text(encoding="utf-8") if THEME_CSS_FILE.is_file() else ""

with gr.Blocks(title="CANGKEMANMU — Kloning Suara Studio", css=css, theme=gr.themes.Soft()) as demo:
    gr.HTML("""
    <div class="topbar">
      <div class="brand-lockup">
        <div class="brand-mark">
          <span></span><span></span><span></span>
        </div>
        <div>
          <div class="brand-name">CANGKEMANMU</div>
          <div class="brand-product">BY RIFKY WIJAYANTO</div>
        </div>
      </div>
      <div class="topbar-status">
        <span class="status-dot"></span> STUDIO ONLINE <b>•</b> SIAP
      </div>
    </div>
    """)
    
    gr.HTML("""
    <section class="hero-v2">
      <div class="hero-grid"></div>
      <div class="hero-copy">
        <div class="eyebrow"><span class="eyebrow-line"></span> KLONING SUARA PRESISI</div>
        <h1>Sintesis Suara <em>Sempurna & Alami.</em></h1>
        <p>Kloning karakter suara Anda dan ubah naskah teks menjadi rekaman audio yang realistis, jernih, dan penuh ekspresi secara instan.</p>
        <div class="hero-metrics">
          <span><b>24+</b> Bahasa</span><i></i>
          <span><b>24 kHz</b> Kualitas Studio</span><i></i>
          <span><b>Pro</b> Voice Engine</span>
        </div>
      </div>
      <div class="hero-visual">
        <div class="voice-orb">
          <div class="orb-core"></div>
        </div>
        <div class="wave-stack">
          <span></span><span></span><span></span><span></span><span></span>
          <span></span><span></span><span></span><span></span><span></span>
          <span></span><span></span><span></span><span></span><span></span>
        </div>
        <div class="orb-caption">KLONING SUARA<br><b>READY</b></div>
      </div>
    </section>
    """)
    
    with gr.Row(elem_classes=["workspace"]):
      with gr.Column(scale=7):
        with gr.Group(elem_classes=["glass-panel","step-panel"]):
          gr.HTML("""
          <div class="panel-head">
            <div class="step-icon">01</div>
            <div>
              <div class="panel-kicker">SUARA REFERENSI</div>
              <div class="panel-title">Unggah sampel suara</div>
            </div>
            <div class="panel-check">● READY</div>
          </div>
          """)
          with gr.Row():
            reference_audio = gr.Audio(sources=["upload","microphone"], type="filepath", label="Audio Referensi (2–20 detik)")
            reference_transcript = gr.Textbox(label="Transkrip Tepat Suara Referensi", lines=5, placeholder="Tuliskan kata-kata yang diucapkan pada sampel suara secara tepat...")
            
        with gr.Group(elem_classes=["glass-panel","step-panel"]):
          gr.HTML("""
          <div class="panel-head">
            <div class="step-icon">02</div>
            <div>
              <div class="panel-kicker">NASKAH SUARA</div>
              <div class="panel-title">Tulis kalimat yang ingin diucapkan</div>
            </div>
            <div class="char-pill">MAKS 500 KARAKTER</div>
          </div>
          """)
          target_text = gr.Textbox(label="", lines=5, max_lines=8, placeholder="Mulai ketik naskah suara Anda di sini...")
          gr.Markdown("<div class='micro-label'>NASKAH CEPAT</div>")
          with gr.Row():
            sample_1 = gr.Button("✦  Sambutan", elem_classes=["chip-btn"])
            sample_2 = gr.Button("◈  Berita", elem_classes=["chip-btn"])
            sample_3 = gr.Button("○  Santai", elem_classes=["chip-btn"])
          with gr.Row(elem_classes=["compact-row"]):
            language = gr.Dropdown(choices=SUPPORTED_LANGUAGES, value="Indonesian", label="Bahasa Suara")
            seed = gr.Number(value=DEFAULT_SEED, precision=0, label="Seed Variasi")
            random_button = gr.Button("↻", elem_classes=["icon-btn"], scale=0)
            
        with gr.Group(elem_classes=["glass-panel","step-panel"]):
          gr.HTML("""
          <div class="panel-head">
            <div class="step-icon">03</div>
            <div>
              <div class="panel-kicker">PENGATURAN SUARA</div>
              <div class="panel-title">Atur ekspresi & karakter</div>
            </div>
            <div class="panel-check">PRO AUDIO</div>
          </div>
          """)
          gr.Markdown("<div class='micro-label'>PRESET KARAKTER</div>")
          with gr.Row():
            preset_nat = gr.Button("Alami", elem_classes=["preset-btn"])
            preset_deep = gr.Button("Berat & Tenang", elem_classes=["preset-btn"])
            preset_fast = gr.Button("Cepat", elem_classes=["preset-btn"])
            preset_news = gr.Button("Formal", elem_classes=["preset-btn"])
          with gr.Row():
            output_speed = gr.Slider(minimum=MIN_OUTPUT_SPEED, maximum=MAX_OUTPUT_SPEED, value=1.0, step=0.01, label="Kecepatan")
            pitch_semitones = gr.Slider(minimum=MIN_PITCH_SEMITONES, maximum=MAX_PITCH_SEMITONES, value=0.0, step=0.5, label="Nada (Pitch)")
          reset_button = gr.Button("Riset Pengaturan", elem_classes=["ghost-btn"])
          
      with gr.Column(scale=5, elem_classes=["output-column"]):
        with gr.Group(elem_classes=["output-card"]):
          gr.HTML("""
          <div class="output-top">
            <div>
              <div class="panel-kicker">04 · HASIL SUARA</div>
              <div class="output-title">Hasil Kloning Suara</div>
            </div>
            <div class="render-badge"><span></span> HQ AUDIO</div>
          </div>
          <div class="render-visual">
            <div class="render-glow"></div>
            <div class="render-bars">
              <i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i>
              <i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i>
              <i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i>
            </div>
            <div class="render-center">24<span>kHz</span></div>
          </div>
          """)
          generate_button = gr.Button("Kloning Suara Sekarang  →", variant="primary", elem_classes=["generate-btn"])
          generated_audio = gr.Audio(label="Pratinjau Suara", autoplay=False)
          saved_path = gr.Textbox(label="Lokasi Berkas Master", interactive=False, elem_classes=["file-output"])
          gr.HTML("""
          <div class="output-note">
            <span>⚡</span> CANGKEMANMU Voice Engine <span class="note-right">24kHz AUDIO HQ</span>
          </div>
          """)
          
    gr.HTML("""
    <div class="footer-line">
      <span>CANGKEMANMU</span><span>·</span><span>BY RIFKY WIJAYANTO</span>
      <span class="footer-right">STUDIO KLONING SUARA</span>
    </div>
    """)
    
    sample_1.click(lambda: "Selamat datang di CANGKEMANMU. Tempat terbaik untuk kloning suara dengan kualitas jernih dan presisi tinggi.", outputs=[target_text])
    sample_2.click(lambda: "Teknologi kloning suara modern memungkinkan pembacaan teks dengan artikulasi yang sangat alami dan ekspresif.", outputs=[target_text])
    sample_3.click(lambda: "Halo semuanya! Semoga hari Anda menyenangkan dan proyek audio Anda berjalan dengan lancar ya.", outputs=[target_text])
    
    preset_nat.click(lambda: set_voice_preset("Natural"), outputs=[output_speed, pitch_semitones])
    preset_deep.click(lambda: set_voice_preset("Deep & Calm"), outputs=[output_speed, pitch_semitones])
    preset_fast.click(lambda: set_voice_preset("Upbeat / Fast"), outputs=[output_speed, pitch_semitones])
    preset_news.click(lambda: set_voice_preset("News Broadcaster"), outputs=[output_speed, pitch_semitones])
    
    reset_button.click(reset_controls, inputs=[], outputs=[output_speed, pitch_semitones])
    random_button.click(random_seed, inputs=[], outputs=[seed])
    
    generate_button.click(
        generate_voice,
        inputs=[target_text, reference_transcript, reference_audio, language, seed, output_speed, pitch_semitones],
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