import os
import time
import math
import random
import inspect
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

# Repository remains UI-only; the notebook owns the stable backend/model runtime.
import sys
sys.path.insert(0, str(ROOT_DIR))
from fireredtts3_runtime import backend as _backend

SUPPORTED_LANGUAGES = _backend.SUPPORTED_LANGUAGES
DEFAULT_SEED = _backend.DEFAULT_SEED if _backend.DEFAULT_SEED is not None else DEFAULT_SEED
MIN_OUTPUT_SPEED = _backend.MIN_OUTPUT_SPEED
MAX_OUTPUT_SPEED = _backend.MAX_OUTPUT_SPEED
MIN_PITCH_SEMITONES = _backend.MIN_PITCH_SEMITONES
MAX_PITCH_SEMITONES = _backend.MAX_PITCH_SEMITONES
backend_generate_voice = _backend.generate_voice
random_seed = _backend.random_seed
reset_controls = _backend.reset_controls

# The prosody-enabled backend is intentionally optional at import time so the
# current stable UI still starts. Once backend.py is patched, this function
# will be used automatically.
_backend_generate_voice_with_prosody = getattr(
    _backend, "generate_voice_with_prosody", None
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

PROSODY_PRESETS = {
    "Natural": "",
    "Serius": (
        "Speak in a serious, authoritative manner. "
        "Use deliberate pacing, firm emphasis, restrained intonation, "
        "clear phrase boundaries, and confident sentence endings."
    ),
    "Santai": (
        "Speak casually and naturally, like talking to a close friend. "
        "Use a relaxed rhythm, warm conversational intonation, natural pauses, "
        "and avoid sounding formal or announcer-like."
    ),
    "Presenter": (
        "Speak like a professional television presenter. "
        "Use clear articulation, confident projection, polished pacing, "
        "controlled pauses, deliberate emphasis, and an engaging but composed delivery."
    ),
    "Komedian": (
        "Speak like an expressive comedian. "
        "Use playful intonation, varied rhythm, dynamic emphasis, natural pauses, "
        "and stronger vocal emphasis around punchlines while keeping the voice believable."
    ),
    "Dramatis": (
        "Speak in a dramatic, cinematic manner. "
        "Use expressive pitch movement, deliberate pauses, strong emphasis on key words, "
        "and a controlled build of intensity without shouting."
    ),
    "Enerjik": (
        "Speak with upbeat, energetic delivery. "
        "Use lively rhythm, brighter expressive intonation, active emphasis, "
        "and confident forward momentum while remaining intelligible."
    ),
}

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
        raise gr.Error(
            f"Audio referensi terlalu pendek ({duration:.2f} detik). "
            f"Minimal {MIN_PROMPT_SECONDS:.1f} detik."
        )
    if duration > MAX_PROMPT_SECONDS:
        raise gr.Error(
            f"Audio referensi terlalu panjang ({duration:.2f} detik). "
            f"Maksimal {MAX_PROMPT_SECONDS:.0f} detik."
        )
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
    # These presets remain as convenience presets for the existing pitch/speed
    # controls. Prosody is handled separately by PROSODY_PRESETS.
    return {
        "Alami": (1.0, 0.0),
        "Dalam & Tenang": (0.96, -1.0),
        "Ceria": (1.06, +0.5),
        "Berita": (0.98, -0.5),
    }.get(preset_type, (1.0, 0.0))

def resolve_prosody_instruction(prosody_mode, custom_instruction):
    mode = (prosody_mode or "Natural").strip()
    if mode == "Custom":
        instruction = (custom_instruction or "").strip()
        if not instruction:
            raise gr.Error("Isi instruksi prosodi custom terlebih dahulu.")
        return instruction
    return PROSODY_PRESETS.get(mode, "")

def _call_backend(
    target_text,
    reference_transcript,
    reference_audio,
    language,
    seed,
    output_speed,
    pitch_value,
    prosody_instruction,
):
    # Natural path preserves the current stable pipeline exactly.
    if not prosody_instruction:
        return backend_generate_voice(
            target_text,
            reference_transcript,
            reference_audio,
            language,
            seed,
            output_speed,
            pitch_value,
        )

    if _backend_generate_voice_with_prosody is None:
        raise gr.Error(
            "Backend prosodi belum terpasang. Patch backend.py terlebih dahulu "
            "agar menyediakan generate_voice_with_prosody(...)."
        )

    return _backend_generate_voice_with_prosody(
        target_text=target_text,
        reference_transcript=reference_transcript,
        reference_audio=reference_audio,
        language=language,
        seed=seed,
        output_speed=output_speed,
        pitch_semitones=pitch_value,
        prosody_instruction=prosody_instruction,
    )

def generate_voice_ui(
    target_text,
    reference_transcript,
    reference_audio,
    language,
    seed,
    output_speed,
    pitch_semitones,
    prosody_mode,
    custom_instruction,
):
    started = time.time()
    pitch_value = float(pitch_semitones)
    prosody_instruction = resolve_prosody_instruction(prosody_mode, custom_instruction)

    audio_result, master_path = _call_backend(
        target_text=target_text,
        reference_transcript=reference_transcript,
        reference_audio=reference_audio,
        language=language,
        seed=seed,
        output_speed=output_speed,
        pitch_value=pitch_value,
        prosody_instruction=prosody_instruction,
    )

    sample_rate, audio = audio_result
    duration = audio.shape[-1] / float(sample_rate)
    master = Path(master_path)
    pcm = master.with_name(master.name.replace("_HQ_FLOAT32.wav", "_PCM16.wav"))
    files = [str(p) for p in (master, pcm) if p.is_file()]
    language_label = LANGUAGE_LABELS.get(language, language)
    elapsed = time.time() - started

    style_label = prosody_mode if prosody_mode != "Custom" else "Custom"
    summary = (
        f"**{duration:.1f} detik** audio pada **{int(sample_rate):,} Hz**, "
        f"selesai dirender dalam {elapsed:.1f} detik.\n\n"
        f"Bahasa {language_label} · seed {int(seed)} · "
        f"kecepatan {float(output_speed):.2f}x · nada {pitch_value:+.1f}st · "
        f"prosodi **{style_label}**"
    )
    return audio_result, summary, files

def toggle_custom_instruction(mode):
    return gr.update(visible=(mode == "Custom"))

def preset_from_sidebar(preset_type):
    speed, pitch = set_voice_preset(preset_type)
    return speed, pitch

APP_DIR = Path(__file__).resolve().parent
THEME_CSS_FILE = APP_DIR / "theme.css"
css = THEME_CSS_FILE.read_text(encoding="utf-8") if THEME_CSS_FILE.is_file() else ""

with gr.Blocks(
    title="Cangkeman — Ruang Kerja Suara",
    css=css,
    theme=gr.themes.Soft(),
) as demo:
    with gr.Sidebar(label="Pengaturan", position="left", width=320):
        gr.HTML(
            """<div class="side-head">
            <div class="side-title">Pengaturan</div>
            <div class="side-sub">Atur bahasa, prosodi, dan karakter suara</div>
            </div>"""
        )

        language = gr.Dropdown(
            choices=LANGUAGE_CHOICES,
            value="Indonesian",
            label="Bahasa",
        )

        gr.HTML("""<div class="side-label">Presets speed & pitch</div>""")
        with gr.Row(elem_classes=["preset-row"]):
            preset_nat = gr.Button("Alami", elem_classes=["preset"])
            preset_deep = gr.Button("Dalam & Tenang", elem_classes=["preset"])
        with gr.Row(elem_classes=["preset-row"]):
            preset_fast = gr.Button("Ceria", elem_classes=["preset"])
            preset_news = gr.Button("Berita", elem_classes=["preset"])

        output_speed = gr.Slider(
            minimum=MIN_OUTPUT_SPEED,
            maximum=MAX_OUTPUT_SPEED,
            value=1.0,
            step=0.01,
            label="Kecepatan",
        )
        pitch_semitones = gr.Slider(
            minimum=MIN_PITCH_SEMITONES,
            maximum=MAX_PITCH_SEMITONES,
            value=0.0,
            step=0.5,
            label="Nada (semitone)",
        )

        gr.HTML("""<div class="side-label">Prosodi / gaya bicara</div>""")
        prosody_mode = gr.Dropdown(
            choices=list(PROSODY_PRESETS.keys()) + ["Custom"],
            value="Natural",
            label="Gaya",
            info="Mengubah cara penyampaian, bukan sekadar pitch/speed.",
        )
        custom_instruction = gr.Textbox(
            label="Instruksi prosodi custom",
            lines=4,
            max_lines=8,
            visible=False,
            placeholder=(
                "Contoh: bicara seperti host podcast yang santai, hangat, "
                "percaya diri, beri jeda sebelum punchline..."
            ),
        )

        with gr.Row(elem_classes=["compact-row"]):
            seed = gr.Number(value=DEFAULT_SEED, precision=0, label="Seed")
            random_button = gr.Button("↻", elem_classes=["icon-btn"], scale=0)

        reset_button = gr.Button(
            "Atur ulang kontrol suara",
            elem_classes=["ghost-link"],
        )
        gr.HTML("""<div class="side-foot">Cangkeman oleh Rifky Wijayanto</div>""")

    gr.HTML(
        """<div class="topbar"><div class="brand-lockup">
        <div class="brand-mark">C</div><div>
        <div class="brand-name">Cangkeman</div>
        <div class="brand-product">oleh Rifky Wijayanto</div>
        </div></div><div class="topbar-status">
        <span class="status-dot"></span>Model siap digunakan</div></div>"""
    )
    gr.HTML(
        """<section class="hero"><div class="eyebrow">Ruang kerja suara</div>
        <h1>Kasih suara pada <span class="accent">kata-katamu.</span></h1>
        <p>Klon suara dari rekaman singkat, atur prosodi dan karakter penyampaiannya,
        lalu ubah naskah jadi suara yang terdengar natural — dalam lebih dari dua puluh
        bahasa, dengan kualitas 24kHz.</p></section>"""
    )

    with gr.Column(elem_classes=["sheet"]):
        gr.HTML(
            """<div class="section-head"><span class="section-num">1</span>
            <span class="section-label">Sumber suara</span></div>"""
        )
        with gr.Row():
            reference_audio = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label=f"Audio referensi · {MIN_PROMPT_SECONDS:.0f}–{MAX_PROMPT_SECONDS:.0f} detik",
            )
            reference_transcript = gr.Textbox(
                label="Transkrip persis",
                lines=5,
                placeholder="Ketik persis apa yang diucapkan pada audio referensi…",
            )
        ref_status = gr.Markdown(
            "Unggah klip referensi untuk memulai.",
            elem_classes=["field-hint"],
        )

        gr.HTML(
            """<hr class="divider"><div class="section-head">
            <span class="section-num">2</span><span class="section-label">Naskah</span></div>"""
        )
        target_text = gr.Textbox(
            label="",
            lines=6,
            max_lines=10,
            placeholder="Mulai tulis naskahmu… bisa santai, dramatis, atau gaya kamu sendiri.",
        )
        char_counter = gr.Markdown(
            f"Maksimal {MAX_TARGET_CHARS} karakter.",
            elem_classes=["field-hint"],
        )

        gr.HTML("""<div class="side-label" style="margin-top:14px">Naskah cepat</div>""")
        with gr.Row():
            sample_1 = gr.Button("Sapaan", elem_classes=["chip"])
            sample_2 = gr.Button("Berita teknologi", elem_classes=["chip"])
            sample_3 = gr.Button("Santai", elem_classes=["chip"])

        generate_button = gr.Button(
            "Hasilkan suara",
            variant="primary",
            elem_classes=["generate-btn"],
        )

        gr.HTML(
            """<hr class="divider"><div class="section-head">
            <span class="section-num">✓</span><span class="section-label">Hasil</span></div>"""
        )
        generated_audio = gr.Audio(label="Pratinjau", autoplay=False)
        generation_summary = gr.Markdown(
            "Belum ada hasil — lengkapi langkah di atas, lalu klik Hasilkan suara.",
            elem_classes=["summary-box"],
        )
        output_files = gr.File(
            label="Unduh berkas",
            file_count="multiple",
            elem_classes=["file-output"],
        )
        gr.HTML(
            """<div class="output-note">Natural mempertahankan pipeline cloning lama.
            Preset prosodi menggunakan FireRedTTS3-Instruct Acoustic Edit pada audio hasil clone,
            lalu Pitch/Speed tetap menjadi fine tuning tambahan.</div>"""
        )

    gr.HTML("""<div class="footer">Cangkeman oleh Rifky Wijayanto — ruang kerja suara pribadi.</div>""")

    sample_1.click(
        lambda: "Selamat datang di Cangkeman. Suara ini dikloning secara presisi menggunakan teknologi AI terbaru.",
        outputs=[target_text],
    )
    sample_2.click(
        lambda: "Perkembangan kecerdasan buatan dalam pemrosesan audio kini memungkinkan pembacaan teks dengan artikulasi yang sangat alami.",
        outputs=[target_text],
    )
    sample_3.click(
        lambda: "Halo semuanya! Semoga hari kalian menyenangkan dan proyek audio kalian berjalan dengan lancar ya.",
        outputs=[target_text],
    )

    preset_nat.click(lambda: preset_from_sidebar("Alami"), outputs=[output_speed, pitch_semitones])
    preset_deep.click(lambda: preset_from_sidebar("Dalam & Tenang"), outputs=[output_speed, pitch_semitones])
    preset_fast.click(lambda: preset_from_sidebar("Ceria"), outputs=[output_speed, pitch_semitones])
    preset_news.click(lambda: preset_from_sidebar("Berita"), outputs=[output_speed, pitch_semitones])

    reset_button.click(reset_controls, inputs=[], outputs=[output_speed, pitch_semitones])
    random_button.click(random_seed, inputs=[], outputs=[seed])
    prosody_mode.change(
        toggle_custom_instruction,
        inputs=[prosody_mode],
        outputs=[custom_instruction],
    )
    reference_audio.change(
        describe_reference_audio,
        inputs=[reference_audio],
        outputs=[ref_status],
    )
    target_text.change(
        describe_char_count,
        inputs=[target_text],
        outputs=[char_counter],
    )

    generate_button.click(
        generate_voice_ui,
        inputs=[
            target_text,
            reference_transcript,
            reference_audio,
            language,
            seed,
            output_speed,
            pitch_semitones,
            prosody_mode,
            custom_instruction,
        ],
        outputs=[generated_audio, generation_summary, output_files],
    )

demo.queue(max_size=4, default_concurrency_limit=1)

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
