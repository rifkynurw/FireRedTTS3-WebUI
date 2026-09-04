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

LANGUAGE_LABELS = {
    "Arabic": "Arab", "Cantonese": "Kanton", "Chinese": "Mandarin", "Czech": "Ceko",
    "Dutch": "Belanda", "English": "Inggris", "Finnish": "Finlandia", "French": "Prancis",
    "German": "Jerman", "Greek": "Yunani", "Hindi": "Hindi", "Indonesian": "Indonesia",
    "Italian": "Italia", "Japanese": "Jepang", "Korean": "Korea", "Polish": "Polandia",
    "Portuguese": "Portugis", "Romanian": "Rumania", "Russian": "Rusia", "Spanish": "Spanyol",
    "Thai": "Thailand", "Turkish": "Turki", "Ukrainian": "Ukraina", "Vietnamese": "Vietnam",
}
SUPPORTED_LANGUAGES = list(LANGUAGE_LABELS.keys())
LANGUAGE_CHOICES = [(LANGUAGE_LABELS[code], code) for code in SUPPORTED_LANGUAGES]

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
            f"Audio referensi terlalu pendek ({duration:.2f} detik). "
            f"Gunakan minimal {MIN_PROMPT_SECONDS:.1f} detik."
        )
    if duration > MAX_PROMPT_SECONDS:
        raise gr.Error(
            f"Audio referensi terlalu panjang ({duration:.2f} detik). "
            f"Gunakan maksimal {MAX_PROMPT_SECONDS:.0f} detik."
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
        raise gr.Error(f"Bentuk gelombang referensi tidak valid: {tuple(waveform.shape)}")
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if not torch.isfinite(waveform).all():
        raise gr.Error("Audio referensi mengandung nilai tidak valid (NaN/Inf).")
    return waveform.cpu(), int(loaded_sr), duration

def validate_generated_audio(audio):
    if not torch.is_tensor(audio):
        audio = torch.as_tensor(audio)
    audio = audio.detach().float().cpu()
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    if audio.ndim != 2 or audio.shape[-1] <= 0:
        raise RuntimeError(f"Bentuk audio hasil tidak terduga: {tuple(audio.shape)}")
    if not torch.isfinite(audio).all():
        raise gr.Error("Audio hasil mengandung nilai tidak valid (NaN/Inf); tidak disimpan.")
    peak = float(audio.abs().max().item())
    if not math.isfinite(peak) or peak <= 0:
        raise gr.Error("Audio hasil kosong atau tidak valid.")
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
        raise gr.Error(f"Kecepatan harus antara {MIN_OUTPUT_SPEED:.2f}–{MAX_OUTPUT_SPEED:.2f}.")
    if not MIN_PITCH_SEMITONES <= pitch <= MAX_PITCH_SEMITONES:
        raise gr.Error(
            f"Nada harus antara {MIN_PITCH_SEMITONES:+.0f} hingga {MAX_PITCH_SEMITONES:+.0f} semitone."
        )
    if not torch.is_tensor(audio_tensor):
        audio_tensor = torch.as_tensor(audio_tensor)
    x = audio_tensor.detach().float().cpu()
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if x.ndim != 2 or x.shape[0] != 1 or x.shape[-1] == 0:
        raise gr.Error(f"Audio keluaran tidak valid untuk pengaturan suara: {tuple(x.shape)}")
    if not torch.isfinite(x).all():
        raise gr.Error("Audio keluaran mengandung nilai tidak valid (NaN/Inf).")
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
            raise gr.Error("Rubber Band R3 HQ tidak menghasilkan keluaran float32.")
        raw = np.fromfile(output_raw, dtype=np.float32)
        if raw.size == 0 or not np.isfinite(raw).all():
            raise gr.Error("Keluaran Rubber Band R3 HQ kosong atau mengandung nilai tidak valid.")
        peak = float(np.max(np.abs(raw)))
        if not math.isfinite(peak) or peak <= 0:
            raise gr.Error("Keluaran Rubber Band R3 HQ tidak valid.")
        if peak > 1.0:
            raw = raw / peak * 0.999
        sf.write(str(output_wav), raw, int(sr), subtype="FLOAT")
        check, out_sr = sf.read(str(output_wav), dtype="float32", always_2d=True)
        if int(out_sr) != int(sr) or check.ndim != 2 or check.shape[1] != 1:
            raise gr.Error(f"Berkas WAV keluaran Rubber Band R3 HQ tidak valid/tidak mono: shape={check.shape}, sr={out_sr}")
        check = check[:, 0]
        if check.size == 0 or not np.isfinite(check).all():
            raise gr.Error("Berkas WAV keluaran Rubber Band R3 HQ tidak valid setelah ditulis.")
        return np.asarray(check, dtype=np.float32)

def random_seed():
    return random.randint(1, 2147483647)

def reset_controls():
    return 1.0, 0.0

def set_voice_preset(preset_type):
    if preset_type == "Alami":
        return 1.0, 0.0
    elif preset_type == "Dalam & Tenang":
        return 0.96, -1.0
    elif preset_type == "Ceria":
        return 1.06, +0.5
    elif preset_type == "Berita":
        return 0.98, -0.5
    return 1.0, 0.0

def describe_reference_audio(path):
    if not path:
        return "Unggah klip referensi untuk memulai."
    try:
        _, duration, sr = validate_reference_audio(path)
    except gr.Error as e:
        return str(e)
    except Exception as e:
        return f"Berkas tidak dapat dibaca: {e}"
    return f"Terekam {duration:.1f} detik pada {sr:,} Hz — siap dikloning."

def describe_char_count(text):
    n = len(text or "")
    if n == 0:
        return f"Maksimal {MAX_TARGET_CHARS} karakter."
    if n > MAX_TARGET_CHARS:
        return f"{n} karakter — kelebihan {n - MAX_TARGET_CHARS} dari batas."
    return f"{n} dari {MAX_TARGET_CHARS} karakter."

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
        t0 = time.time()
        target_text = (target_text or "").strip()
        reference_transcript = (reference_transcript or "").strip()
        if not target_text:
            raise gr.Error("Naskah masih kosong.")
        if not reference_transcript:
            raise gr.Error("Transkrip referensi masih kosong. Isi PERSIS ucapan pada audio referensi.")
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
            f"cangkeman_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            f"_spd{safe_speed}_pit{safe_pitch}st.wav"
        )
        master_path = out_path.with_name(out_path.stem + "_HQ_FLOAT32.wav")
        pcm16_path = out_path.with_name(out_path.stem + "_PCM16.wav")
        sf.write(str(master_path), processed, int(gen_audio_sr), subtype="FLOAT")
        sf.write(str(pcm16_path), processed, int(gen_audio_sr), subtype="PCM_16")

        elapsed = time.time() - t0
        duration_out = processed.shape[-1] / float(gen_audio_sr)
        bahasa_label = LANGUAGE_LABELS.get(language, language)
        summary = (
            f"**{duration_out:.1f} detik** audio pada **{int(gen_audio_sr):,} Hz**, "
            f"selesai dirender dalam {elapsed:.1f} detik.\n\n"
            f"Bahasa {bahasa_label} · seed {seed} · kecepatan {speed:.2f}x · nada {pitch:+.1f}st"
        )
        return (int(gen_audio_sr), processed), summary, [str(master_path), str(pcm16_path)]

    except gr.Error:
        raise
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"Proses inferensi FireRedTTS3 gagal: {type(e).__name__}: {e}")
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

APP_DIR = Path(__file__).resolve().parent
THEME_CSS_FILE = APP_DIR / "theme.css"
css = THEME_CSS_FILE.read_text(encoding="utf-8") if THEME_CSS_FILE.is_file() else ""

with gr.Blocks(title="Cangkeman — Ruang Kerja Suara", css=css, theme=gr.themes.Soft()) as demo:
    with gr.Sidebar(label="Pengaturan", position="left", width=300):
        gr.HTML("""<div class="side-head"><div class="side-title">Pengaturan</div><div class="side-sub">Atur bahasa dan karakter suara</div></div>""")
        language = gr.Dropdown(choices=LANGUAGE_CHOICES, value="Indonesian", label="Bahasa")
        gr.HTML("""<div class="side-label">Preset karakter</div>""")
        with gr.Row(elem_classes=["preset-row"]):
            preset_nat = gr.Button("Alami", elem_classes=["preset"])
            preset_deep = gr.Button("Dalam & Tenang", elem_classes=["preset"])
        with gr.Row(elem_classes=["preset-row"]):
            preset_fast = gr.Button("Ceria", elem_classes=["preset"])
            preset_news = gr.Button("Berita", elem_classes=["preset"])
        output_speed = gr.Slider(minimum=MIN_OUTPUT_SPEED, maximum=MAX_OUTPUT_SPEED, value=1.0, step=0.01, label="Kecepatan")
        pitch_semitones = gr.Slider(minimum=MIN_PITCH_SEMITONES, maximum=MAX_PITCH_SEMITONES, value=0.0, step=0.5, label="Nada (semitone)")
        with gr.Row(elem_classes=["compact-row"]):
            seed = gr.Number(value=DEFAULT_SEED, precision=0, label="Seed")
            random_button = gr.Button("↻", elem_classes=["icon-btn"], scale=0)
        reset_button = gr.Button("Atur ulang kontrol suara", elem_classes=["ghost-link"])
        gr.HTML("""<div class="side-foot">Cangkeman oleh Rifky Wijayanto</div>""")

    gr.HTML("""<div class="topbar"><div class="brand-lockup"><div class="brand-mark">C</div><div><div class="brand-name">Cangkeman</div><div class="brand-product">oleh Rifky Wijayanto</div></div></div><div class="topbar-status"><span class="status-dot"></span>Model siap digunakan</div></div>""")
    gr.HTML("""<section class="hero"><div class="eyebrow">Ruang kerja suara</div><h1>Kasih suara pada <span class="accent">kata-katamu.</span></h1><p>Klon suara dari rekaman singkat, atur karakternya, lalu ubah naskah jadi suara yang terdengar natural — dalam lebih dari dua puluh bahasa, dengan kualitas 24kHz.</p></section>""")

    with gr.Column(elem_classes=["sheet"]):
        gr.HTML("""<div class="section-head"><span class="section-num">1</span><span class="section-label">Sumber suara</span></div>""")
        with gr.Row():
            reference_audio = gr.Audio(sources=["upload", "microphone"], type="filepath", label=f"Audio referensi · {MIN_PROMPT_SECONDS:.0f}–{MAX_PROMPT_SECONDS:.0f} detik")
            reference_transcript = gr.Textbox(label="Transkrip persis", lines=5, placeholder="Ketik persis apa yang diucapkan pada audio referensi…")
        ref_status = gr.Markdown("Unggah klip referensi untuk memulai.", elem_classes=["field-hint"])

        gr.HTML("""<hr class="divider"><div class="section-head"><span class="section-num">2</span><span class="section-label">Naskah</span></div>""")
        target_text = gr.Textbox(label="", lines=6, max_lines=10, placeholder="Mulai tulis naskahmu… bisa santai, dramatis, atau gaya kamu sendiri.")
        char_counter = gr.Markdown(f"Maksimal {MAX_TARGET_CHARS} karakter.", elem_classes=["field-hint"])
        gr.HTML("""<div class="side-label" style="margin-top:14px">Naskah cepat</div>""")
        with gr.Row():
            sample_1 = gr.Button("Sapaan", elem_classes=["chip"])
            sample_2 = gr.Button("Berita teknologi", elem_classes=["chip"])
            sample_3 = gr.Button("Santai", elem_classes=["chip"])
        generate_button = gr.Button("Hasilkan suara", variant="primary", elem_classes=["generate-btn"])

        gr.HTML("""<hr class="divider"><div class="section-head"><span class="section-num">✓</span><span class="section-label">Hasil</span></div>""")
        generated_audio = gr.Audio(label="Pratinjau", autoplay=False)
        generation_summary = gr.Markdown("Belum ada hasil — lengkapi langkah di atas, lalu klik Hasilkan suara.", elem_classes=["summary-box"])
        output_files = gr.File(label="Unduh berkas", file_count="multiple", elem_classes=["file-output"])
        gr.HTML("""<div class="output-note">Dirender dengan FireRedTTS3 dan dimaster dengan Rubber Band R3 HQ. Tersedia dua berkas: master kualitas tinggi (float32) dan salinan standar (PCM16).</div>""")

    gr.HTML("""<div class="footer">Cangkeman oleh Rifky Wijayanto — ruang kerja suara pribadi.</div>""")

    sample_1.click(lambda: "Selamat datang di Cangkeman. Suara ini dikloning secara presisi menggunakan teknologi AI terbaru.", outputs=[target_text])
    sample_2.click(lambda: "Perkembangan kecerdasan buatan dalam pemrosesan audio kini memungkinkan pembacaan teks dengan artikulasi yang sangat alami.", outputs=[target_text])
    sample_3.click(lambda: "Halo semuanya! Semoga hari kalian menyenangkan dan proyek audio kalian berjalan dengan lancar ya.", outputs=[target_text])
    preset_nat.click(lambda: set_voice_preset("Alami"), outputs=[output_speed, pitch_semitones])
    preset_deep.click(lambda: set_voice_preset("Dalam & Tenang"), outputs=[output_speed, pitch_semitones])
    preset_fast.click(lambda: set_voice_preset("Ceria"), outputs=[output_speed, pitch_semitones])
    preset_news.click(lambda: set_voice_preset("Berita"), outputs=[output_speed, pitch_semitones])
    reset_button.click(reset_controls, inputs=[], outputs=[output_speed, pitch_semitones])
    random_button.click(random_seed, inputs=[], outputs=[seed])
    reference_audio.change(describe_reference_audio, inputs=[reference_audio], outputs=[ref_status])
    target_text.change(describe_char_count, inputs=[target_text], outputs=[char_counter])
    generate_button.click(generate_voice, inputs=[target_text, reference_transcript, reference_audio, language, seed, output_speed, pitch_semitones], outputs=[generated_audio, generation_summary, output_files])

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