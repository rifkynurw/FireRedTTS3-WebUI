import os
import time
import math
import random
import time
from pathlib import Path

import gradio as gr

ROOT_DIR = Path(os.environ["FIRERED_REPO_DIR"]).resolve()
OUTPUT_DIR = Path(os.environ["FIRERED_OUTPUT_DIR"]).resolve()
URL_FILE = Path(os.environ["FIRERED_URL_FILE"]).resolve()
LOCAL_URL_FILE = Path(os.environ.get("FIRERED_LOCAL_URL_FILE", "/content/fireredtts3_webui.local.url")).resolve()
PORT = int(os.environ.get("FIRERED_GRADIO_PORT", "7860"))
MAX_TARGET_CHARS = int(os.environ.get("FIRERED_MAX_TARGET_CHARS", "500"))
MIN_PROMPT_SECONDS = float(os.environ.get("FIRERED_MIN_PROMPT_SECONDS", "2.0"))
MAX_PROMPT_SECONDS = float(os.environ.get("FIRERED_MAX_PROMPT_SECONDS", "20.0"))
DEFAULT_SEED = int(os.environ.get("FIRERED_DEFAULT_SEED", "1986"))

# Fix14 contract: the repository is UI-only. The notebook owns the stable backend.
import sys
sys.path.insert(0, str(ROOT_DIR))
from fireredtts3_runtime.backend import (
    SUPPORTED_LANGUAGES,
    DEFAULT_SEED as BACKEND_DEFAULT_SEED,
    MIN_OUTPUT_SPEED,
    MAX_OUTPUT_SPEED,
    MIN_PITCH_SEMITONES,
    MAX_PITCH_SEMITONES,
    generate_voice as backend_generate_voice,
    random_seed,
    reset_controls,
)

LANGUAGE_LABELS = {
    "Arabic": "Arab", "Cantonese": "Kanton", "Chinese": "Mandarin", "Czech": "Ceko",
    "Dutch": "Belanda", "English": "Inggris", "Finnish": "Finlandia", "French": "Prancis",
    "German": "Jerman", "Greek": "Yunani", "Hindi": "Hindi", "Indonesian": "Indonesia",
    "Italian": "Italia", "Japanese": "Jepang", "Korean": "Korea", "Polish": "Polandia",
    "Portuguese": "Portugis", "Romanian": "Rumania", "Russian": "Rusia", "Spanish": "Spanyol",
    "Thai": "Thailand", "Turkish": "Turki", "Ukrainian": "Ukraina", "Vietnamese": "Vietnam",
}
LANGUAGE_CHOICES = [(LANGUAGE_LABELS.get(code, code), code) for code in SUPPORTED_LANGUAGES]
DEFAULT_SEED = BACKEND_DEFAULT_SEED if BACKEND_DEFAULT_SEED is not None else DEFAULT_SEED

def _validate_reference_audio(path):
    if not path:
        raise gr.Error("Audio referensi belum dipilih.")
    p = Path(path)
    if not p.is_file() or p.stat().st_size <= 0:
        raise gr.Error("Audio referensi tidak ditemukan atau kosong.")
    try:
        import soundfile as sf
        info = sf.info(str(p))
    except Exception as exc:
        raise gr.Error(f"Audio referensi tidak dapat dibaca: {exc}")
    duration = info.frames / float(info.samplerate) if info.samplerate else 0.0
    if duration < MIN_PROMPT_SECONDS:
        raise gr.Error(f"Audio referensi terlalu pendek ({duration:.2f} detik). Minimal {MIN_PROMPT_SECONDS:.1f} detik.")
    if duration > MAX_PROMPT_SECONDS:
        raise gr.Error(f"Audio referensi terlalu panjang ({duration:.2f} detik). Maksimal {MAX_PROMPT_SECONDS:.0f} detik.")
    return duration, int(info.samplerate)

def describe_reference_audio(path):
    if not path:
        return "Unggah klip referensi untuk memulai."
    try:
        duration, sr = _validate_reference_audio(path)
    except gr.Error as exc:
        return str(exc)
    return f"Terekam {duration:.1f} detik pada {sr:,} Hz — siap dikloning."

def describe_char_count(text):
    n = len(text or "")
    if n == 0:
        return f"Maksimal {MAX_TARGET_CHARS} karakter."
    if n > MAX_TARGET_CHARS:
        return f"{n} karakter — kelebihan {n - MAX_TARGET_CHARS} dari batas."
    return f"{n} dari {MAX_TARGET_CHARS} karakter."

def set_voice_preset(preset_type):
    return {
        "Alami": (1.0, 0.0),
        "Dalam & Tenang": (0.96, -1.0),
        "Ceria": (1.06, +0.5),
        "Berita": (0.98, -0.5),
    }.get(preset_type, (1.0, 0.0))

def generate_voice_ui(target_text, reference_transcript, reference_audio, language, seed, output_speed, pitch_semitones):
    started = time.time()
    audio_result, master_path = backend_generate_voice(
        target_text, reference_transcript, reference_audio, language, seed, output_speed, pitch_semitones
    )
    sample_rate, audio = audio_result
    duration = audio.shape[-1] / float(sample_rate)
    master = Path(master_path)
    pcm = master.with_name(master.name.replace("_HQ_FLOAT32.wav", "_PCM16.wav"))
    files = [str(p) for p in (master, pcm) if p.is_file()]
    language_label = LANGUAGE_LABELS.get(language, language)
    elapsed = time.time() - started
    summary = (
        f"**{duration:.1f} detik** audio pada **{int(sample_rate):,} Hz**, "
        f"selesai dirender dalam {elapsed:.1f} detik.\n\n"
        f"Bahasa {language_label} · seed {int(seed)} · kecepatan {float(output_speed):.2f}x · nada {float(pitch_semitones):+.1f}st"
    )
    return audio_result, summary, files

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
    generate_button.click(generate_voice_ui, inputs=[target_text, reference_transcript, reference_audio, language, seed, output_speed, pitch_semitones], outputs=[generated_audio, generation_summary, output_files])

demo.queue(max_size=4, default_concurrency_limit=1)

# Fix18: public Gradio share is intentional so browser upload/file endpoints stay on the same public origin; model loading remains lazy in notebook backend.
launch_result = demo.launch(
    server_name="0.0.0.0",
    server_port=PORT,
    share=True,
    show_error=True,
    prevent_thread_lock=True,
    allowed_paths=[str(OUTPUT_DIR)],
)

try:
    _, local_url, share_url = launch_result
except Exception:
    local_url, share_url = None, None

if local_url:
    LOCAL_URL_FILE.write_text(str(local_url), encoding="utf-8")
    print("FIREREDTTS3_WEBUI_LOCAL_URL:", local_url, flush=True)
if share_url:
    URL_FILE.write_text(str(share_url), encoding="utf-8")
    print("FIREREDTTS3_WEBUI_PUBLIC_URL:", share_url, flush=True)
else:
    print("FIREREDTTS3_WEBUI_PUBLIC_URL: unavailable", flush=True)

while True:
    time.sleep(3600)
