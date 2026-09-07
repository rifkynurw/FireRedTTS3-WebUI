import os
import time
from datetime import datetime
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
HISTORY_LIMIT = 6

import sys
sys.path.insert(0, str(ROOT_DIR))
from fireredtts3_runtime import backend
from fireredtts3_runtime.backend import (
    SUPPORTED_LANGUAGES,
    DEFAULT_SEED,
    MIN_OUTPUT_SPEED,
    MAX_OUTPUT_SPEED,
    MIN_PITCH_SEMITONES,
    MAX_PITCH_SEMITONES,
    MIN_DESIGN_CFG,
    MAX_DESIGN_CFG,
    DEFAULT_DESIGN_CFG,
    MIN_DESIGN_STEPS,
    MAX_DESIGN_STEPS,
    DEFAULT_DESIGN_STEPS,
    generate_voice,
    generate_voice_design,
    random_seed,
    reset_clone_controls,
)

LANGUAGE_LABELS = {
    "Arabic":"Arab","Cantonese":"Kanton","Chinese":"Mandarin","Czech":"Ceko","Dutch":"Belanda",
    "English":"Inggris","Finnish":"Finlandia","French":"Prancis","German":"Jerman","Greek":"Yunani",
    "Hindi":"Hindi","Indonesian":"Indonesia","Italian":"Italia","Japanese":"Jepang","Korean":"Korea",
    "Polish":"Polandia","Portuguese":"Portugis","Romanian":"Rumania","Russian":"Rusia","Spanish":"Spanyol",
    "Thai":"Thailand","Turkish":"Turki","Ukrainian":"Ukraina","Vietnamese":"Vietnam",
}
LANGUAGE_CHOICES = [(LANGUAGE_LABELS.get(code, code), code) for code in SUPPORTED_LANGUAGES]
STYLE_CHOICES = list(backend.STYLE_INSTRUCTIONS.keys())

NARRATION_PRESETS = {
    "🎙 Narasi Natural":"Hari ini kita akan membahas sebuah cerita sederhana, dengan alur yang mengalir dan cara penyampaian yang terdengar alami.",
    "📖 Storytelling":"Malam itu, hujan turun perlahan ketika ia berjalan sendirian menyusuri jalan yang sudah lama tidak ia lewati.",
    "📰 Berita":"Pemerintah hari ini mengumumkan langkah baru yang diharapkan dapat meningkatkan pelayanan publik dan memberikan manfaat bagi masyarakat.",
    "🎧 Podcast":"Halo semuanya, di episode kali ini kita akan ngobrol santai tentang pengalaman, pelajaran, dan hal-hal menarik yang sering kita temui sehari-hari.",
    "🎬 Dokumenter":"Di balik hiruk-pikuk kota, terdapat kisah tentang orang-orang yang bekerja dalam diam untuk menjaga kehidupan tetap berjalan.",
    "📢 Promosi":"Temukan pengalaman baru yang lebih praktis, lebih nyaman, dan dirancang untuk membantu aktivitas Anda setiap hari.",
    "🌙 Misterius":"Tidak ada yang tahu siapa yang meninggalkan pesan itu, tetapi sejak malam tersebut, sesuatu terasa berbeda.",
    "💡 Inspiratif":"Setiap langkah kecil tetap berarti. Terus bergerak, belajar dari proses, dan jangan berhenti percaya bahwa perubahan bisa dimulai dari hari ini.",
}

PERSON_PRESETS = {
    "👩 Indonesia Muda":("Female","Young Adult","Natural","Indonesian","Perempuan Indonesia dewasa muda, penutur asli Bahasa Indonesia dari Indonesia. Identitas penutur harus terasa seperti orang yang tumbuh di Indonesia dan berbicara Bahasa Indonesia sebagai bahasa utama. Suara natural, conversational, hangat, dengan pelafalan, ritme, intonasi, dan prosodi khas Bahasa Indonesia."),
    "👨 Indonesia Muda":("Male","Young Adult","Natural","Indonesian","Laki-laki Indonesia dewasa muda, penutur asli Bahasa Indonesia dari Indonesia. Identitas penutur harus terasa seperti orang yang tumbuh di Indonesia dan berbicara Bahasa Indonesia sebagai bahasa utama. Suara natural, conversational, santai, dengan pelafalan, ritme, intonasi, dan prosodi khas Bahasa Indonesia."),
    "👩 Indonesia Dewasa":("Female","Adult","Warm","Indonesian","Perempuan Indonesia dewasa, penutur asli Bahasa Indonesia dari Indonesia. Suara matang, hangat, tenang, natural, seperti pembicara Indonesia sehari-hari. Pertahankan pelafalan, ritme, intonasi, dan prosodi Bahasa Indonesia."),
    "👨 Indonesia Dewasa":("Male","Adult","Warm","Indonesian","Laki-laki Indonesia dewasa, penutur asli Bahasa Indonesia dari Indonesia. Suara matang, hangat, tenang, natural, seperti pembicara Indonesia sehari-hari. Pertahankan pelafalan, ritme, intonasi, dan prosodi Bahasa Indonesia."),
    "🎙 Presenter Indonesia":("Female","Adult","Crisp","Indonesian","Presenter profesional Indonesia, perempuan dewasa, penutur asli Bahasa Indonesia. Suara jelas, percaya diri, komunikatif, natural seperti presenter Indonesia. Bukan penutur asing yang membaca Bahasa Indonesia."),
    "🎧 Podcaster Indonesia":("Male","Adult","Soft","Indonesian","Podcaster Indonesia, laki-laki dewasa, penutur asli Bahasa Indonesia. Suara dekat, santai, nyaman, conversational, dengan jeda alami seperti podcast Indonesia."),
    "📰 Penyiar Indonesia":("Male","Mature","Crisp","Indonesian","Penyiar berita Indonesia, laki-laki dewasa matang, penutur asli Bahasa Indonesia. Suara tegas, jernih, profesional, natural seperti penyiar radio Indonesia."),
    "📖 Pendongeng Indonesia":("Female","Adult","Rich","Indonesian","Pendongeng Indonesia, perempuan dewasa, penutur asli Bahasa Indonesia. Suara ekspresif, hangat, hidup, natural seperti pendongeng Indonesia."),
}


def apply_narration_preset(name):
    return NARRATION_PRESETS.get(name, "")


def apply_person_preset(name):
    return PERSON_PRESETS.get(name, ("Auto", "Auto", "Natural", "Indonesian", ""))


def random_clone_seed():
    return random_seed()


def reset_design_controls():
    return "Auto", "Auto", "Natural", "Indonesian", "Natural", "", DEFAULT_DESIGN_CFG, DEFAULT_DESIGN_STEPS, DEFAULT_SEED


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
        return "Upload audio referensi untuk mulai cloning."
    try:
        duration, sr = _validate_reference_audio(path)
    except gr.Error as exc:
        return str(exc)
    return f"Siap dikloning · {duration:.1f} detik · {sr:,} Hz"


def describe_char_count(text):
    n = len(text or "")
    if n == 0:
        return f"Maksimal {MAX_TARGET_CHARS} karakter."
    if n > MAX_TARGET_CHARS:
        return f"{n} karakter · kelebihan {n - MAX_TARGET_CHARS}."
    return f"{n} / {MAX_TARGET_CHARS} karakter"


def set_clone_preset(name):
    return {
        "Alami": (1.0, 0.0),
        "Dalam & Tenang": (0.96, -1.0),
        "Ceria": (1.06, 0.5),
        "Berita": (0.98, -0.5),
    }.get(name, (1.0, 0.0))


def _history_updates(history):
    history = list(history or [])[:HISTORY_LIMIT]
    updates = []
    for index in range(HISTORY_LIMIT):
        if index < len(history):
            item = history[index]
            stamp = item.get("time", "")
            mode = item.get("mode", "Voice")
            duration = item.get("duration", "")
            label = f"{index + 1:02d} · {mode} · {duration} · {stamp}"
            path = item.get("audio_path")
            updates.extend([
                gr.update(visible=True, label=label),
                gr.update(value=item.get("text", "")),
                gr.update(value=path, visible=True),
                gr.update(value=path, interactive=bool(path)),
            ])
        else:
            updates.extend([
                gr.update(visible=False),
                gr.update(value=""),
                gr.update(value=None, visible=False),
                gr.update(value=None, interactive=False),
            ])
    return updates


def _history_push(history, *, mode, text, download_path, duration):
    updated = [{
        "mode": mode,
        "text": text,
        "audio_path": download_path,
        "duration": f"{duration:.1f}s",
        "time": datetime.now().strftime("%H:%M"),
    }]
    updated.extend(list(history or []))
    return updated[:HISTORY_LIMIT]


def _begin_generation():
    return (
        gr.update(value="⏳ Preparing…", interactive=False),
        gr.update(visible=False, value=None),
        gr.update(visible=True, value="""<div class='output-loading-card'><div class='loading-spinner'></div><div class='loading-title'>Menyiapkan generate…</div><div class='loading-subtitle'>Output sebelumnya digantikan loading.</div></div>"""),
        gr.update(interactive=False),
        gr.update(value="Generating…", visible=True),
    )


def _generation_phase():
    return (
        gr.update(value="🔊 Generating…", interactive=False),
        gr.update(value="""<div class='output-loading-card output-loading-active'><div class='loading-spinner'></div><div class='loading-title'>Generating speech…</div><div class='loading-subtitle'>Model sedang menghasilkan audio.</div></div>"""),
        gr.update(value="Output sedang diproses…"),
    )


def generate_dispatch(
    mode, target_text, reference_audio, reference_transcript, language, seed,
    design_seed, design_language, output_speed, pitch_semitones, gender, age,
    timbre, accent, style, custom_instruction, design_cfg, design_steps
):
    started = time.time()
    if mode == "cloning":
        audio_result, master_path = generate_voice(
            target_text, reference_transcript, reference_audio, language,
            seed, output_speed, pitch_semitones
        )
        sr, audio = audio_result
        duration = audio.shape[-1] / float(sr)
        pcm = Path(master_path).with_name(Path(master_path).name.replace("_HQ_FLOAT32.wav", "_PCM16.wav"))
        elapsed = time.time() - started
        summary = (
            f"**{duration:.1f} detik** · {int(sr):,} Hz · **Voice Cloning** · {elapsed:.1f}s\n\n"
            f"seed `{int(seed)}` · speed `{float(output_speed):.2f}x` · pitch `{float(pitch_semitones):+.1f}st`"
        )
        return audio_result, summary, target_text.strip(), str(pcm) if pcm.is_file() else str(master_path), duration

    audio_result, master_path, voice_plan, instruction = generate_voice_design(
        target_text, design_language, gender, age, timbre, accent, style,
        custom_instruction, design_seed, design_cfg, design_steps
    )
    sr, audio = audio_result
    duration = audio.shape[-1] / float(sr)
    pcm = Path(master_path).with_name(Path(master_path).name.replace("_HQ_FLOAT32.wav", "_PCM16.wav"))
    elapsed = time.time() - started
    summary = (
        f"**{duration:.1f} detik** · {int(sr):,} Hz · **Voice Design** · {elapsed:.1f}s\n\n"
        f"gaya **{style}** · language **{design_language}** · gender **{gender}** · age **{age}** · "
        f"timbre **{timbre}** · accent **{accent}** · seed `{int(design_seed)}` · "
        f"CFG `{float(design_cfg):.2f}` · steps `{int(design_steps)}`\n\n"
        f"**Voice plan:** {voice_plan}"
    )
    return audio_result, summary, target_text.strip(), str(pcm) if pcm.is_file() else str(master_path), duration


def run_generation_ui(
    history, mode, target_text, reference_audio, reference_transcript, language, seed,
    design_seed, design_language, output_speed, pitch_semitones, gender, age, timbre,
    accent, style, custom_instruction, design_cfg, design_steps
):
    try:
        audio_result, summary, prompt_text, audio_path, duration = generate_dispatch(
            mode, target_text, reference_audio, reference_transcript, language, seed,
            design_seed, design_language, output_speed, pitch_semitones, gender, age,
            timbre, accent, style, custom_instruction, design_cfg, design_steps
        )
        mode_label = "Cloning" if mode == "cloning" else "Design"
        new_history = _history_push(
            history, mode=mode_label, text=prompt_text,
            download_path=audio_path, duration=duration,
        )
        return (
            gr.update(visible=True, value=audio_result),
            gr.update(value=summary),
            gr.update(value=prompt_text),
            gr.update(visible=False, value=""),
            gr.update(value="Generate Speech", interactive=True),
            gr.update(interactive=True),
            gr.update(value="Ready · output generated"),
            new_history,
            *_history_updates(new_history),
        )
    except Exception as exc:
        message = f"❌ Generate gagal · {type(exc).__name__}: {exc}"
        return (
            gr.update(visible=False, value=None),
            gr.update(value=message),
            gr.update(value=(target_text or "").strip()),
            gr.update(visible=True, value="""<div class='output-error-card'>Generation gagal. Periksa parameter dan coba lagi.</div>"""),
            gr.update(value="Generate Speech", interactive=True),
            gr.update(interactive=True),
            gr.update(value="Generation failed"),
            list(history or []),
            *_history_updates(history),
        )


OUTPUT_SCROLL_JS = """() => { const el = document.querySelector('#output-stage'); if (el) el.scrollIntoView({behavior:'smooth', block:'start'}); }"""

APP_DIR = Path(__file__).resolve().parent
THEME_CSS_FILE = APP_DIR / "theme.css"
css = THEME_CSS_FILE.read_text(encoding="utf-8") if THEME_CSS_FILE.is_file() else ""

with gr.Blocks(title="Cangkeman — AI Voice Studio") as demo:
    mode_state = gr.State("cloning")
    history_state = gr.State([])

    with gr.Row(elem_classes=["app-shell"]):
        with gr.Column(scale=0, min_width=255, elem_classes=["nav-col"]):
            gr.HTML("""<div class='brand'><div class='brand-mark'>C</div><div><div class='brand-name'>Cangkeman</div><div class='brand-tag'>AI Voice Studio</div></div></div>""")
            gr.Markdown("TOOLS", elem_classes=["nav-caption"])
            mode_nav = gr.Radio(
                choices=["🎙  Voice Cloning", "🎨  Voice Design"],
                value="🎙  Voice Cloning",
                label=None,
                show_label=False,
                container=False,
                elem_classes=["mode-nav"],
            )
            active_nav_status = gr.Markdown("● Voice Cloning aktif", elem_classes=["mode-status"])
            gr.Markdown("HISTORY", elem_classes=["nav-caption", "history-heading"])
            history_hint = gr.Markdown("Hasil terbaru akan muncul di sini.", elem_classes=["history-hint"])
            history_accordions = []
            history_outputs = []
            for i in range(HISTORY_LIMIT):
                with gr.Accordion(
                    f"{i + 1:02d} · Belum ada hasil", open=False, visible=False,
                    elem_classes=["history-slot"]
                ) as history_accordion:
                    history_text = gr.Textbox(
                        label="Generated text", lines=3, interactive=False,
                        buttons=["copy"], elem_classes=["history-text"]
                    )
                    history_audio = gr.Audio(
                        value=None, label="", show_label=False,
                        interactive=False, visible=False,
                        autoplay=False, elem_classes=["history-audio"]
                    )
                    history_download = gr.DownloadButton(
                        "⬇ Unduh WAV", value=None, interactive=False,
                        elem_classes=["history-download"]
                    )
                history_accordions.append(history_accordion)
                history_outputs.extend([history_accordion, history_text, history_audio, history_download])
            gr.Markdown("FireRedTTS3 · T4", elem_classes=["nav-foot"])

        with gr.Column(scale=1, elem_classes=["workspace-col"]):
            gr.Markdown("Workspace", elem_classes=["eyebrow"])
            target_text = gr.Textbox(
                label="", lines=9, max_lines=14,
                placeholder="Tulis teks yang ingin diucapkan…",
                elem_classes=["main-text"]
            )
            gr.Markdown("**Preset Teks**", elem_classes=["preset-heading"])
            with gr.Row(elem_classes=["preset-grid"]):
                narration_buttons = [gr.Button(name, elem_classes=["narration-preset-btn"]) for name in NARRATION_PRESETS]
            char_counter = gr.Markdown(f"Maksimal {MAX_TARGET_CHARS} karakter.", elem_classes=["field-hint", "char-counter"])
            generate_btn = gr.Button("Generate Speech", variant="primary", elem_classes=["generate-btn"])

            gr.Markdown("Output", elem_classes=["eyebrow", "output-heading"])
            with gr.Column(elem_id="output-stage", elem_classes=["output-stage"]):
                output_status = gr.Markdown("Ready · belum ada output", elem_classes=["output-status"])
                with gr.Column(elem_classes=["output-media-card"]):
                    generated_audio = gr.Audio(
                        value=None, label="", show_label=False, autoplay=False,
                        visible=True, elem_classes=["output-audio"]
                    )
                    output_loading = gr.HTML(
                        """<div class='output-empty-card'><div class='output-empty-icon'>◉</div><div class='loading-title'>Belum ada audio</div><div class='loading-subtitle'>Hasil generate akan tampil di sini.</div></div>""",
                        visible=True, elem_classes=["output-loading"]
                    )
            generation_summary = gr.Markdown(
                "Parameter, durasi, waktu proses, dan voice plan akan tampil di sini.",
                elem_classes=["summary-box"]
            )
            generation_prompt = gr.Textbox(
                label="Generated Text", lines=3, interactive=False,
                buttons=["copy"], elem_classes=["generation-prompt"]
            )

        with gr.Column(scale=0, min_width=330, elem_classes=["settings-col"]):
            active_title = gr.Markdown("🎙 **Voice Cloning**", elem_classes=["settings-mode-title"])
            settings_sync = gr.Markdown("● Voice Cloning aktif", elem_classes=["settings-mode-status"])

            with gr.Column(visible=True, elem_classes=["settings-panel"]) as clone_panel:
                gr.Markdown("### Voice Cloning")
                reference_audio = gr.Audio(
                    sources=["upload", "microphone"], type="filepath",
                    label=f"Reference Audio · {MIN_PROMPT_SECONDS:.0f}–{MAX_PROMPT_SECONDS:.0f} detik"
                )
                reference_transcript = gr.Textbox(
                    label="Reference Transcript", lines=4,
                    placeholder="Tulis persis ucapan pada audio referensi…"
                )
                ref_status = gr.Markdown("Upload audio referensi untuk mulai cloning.", elem_classes=["field-hint"])
                language = gr.Dropdown(choices=LANGUAGE_CHOICES, value="Indonesian", label="Language")
                gr.Markdown("### Fine Control")
                with gr.Row():
                    output_speed = gr.Slider(minimum=MIN_OUTPUT_SPEED, maximum=MAX_OUTPUT_SPEED, value=1.0, step=0.01, label="Speed")
                    pitch_semitones = gr.Slider(minimum=MIN_PITCH_SEMITONES, maximum=MAX_PITCH_SEMITONES, value=0.0, step=0.5, label="Pitch")
                gr.Markdown("### Presets")
                with gr.Row():
                    preset_nat = gr.Button("Alami", elem_classes=["small-btn"])
                    preset_deep = gr.Button("Tenang", elem_classes=["small-btn"])
                with gr.Row():
                    preset_fast = gr.Button("Ceria", elem_classes=["small-btn"])
                    preset_news = gr.Button("Berita", elem_classes=["small-btn"])
                with gr.Row():
                    random_seed_btn = gr.Button("Random Seed", elem_classes=["small-btn"])
                    reset_btn = gr.Button("Reset controls", elem_classes=["text-btn"])
                seed = gr.Number(value=DEFAULT_SEED, precision=0, label="Seed")

            with gr.Column(visible=False, elem_classes=["settings-panel"]) as design_panel:
                gr.Markdown("### Voice Design")
                gr.Markdown("**Preset Person**", elem_classes=["preset-heading"])
                with gr.Row(elem_classes=["person-grid"]):
                    person_buttons = [gr.Button(name, elem_classes=["person-preset-btn"]) for name in PERSON_PRESETS]
                gender = gr.Dropdown(choices=["Auto","Male","Female"], value="Auto", label="Gender")
                age = gr.Dropdown(choices=["Auto","Young Adult","Adult","Mature","Senior"], value="Auto", label="Age")
                timbre = gr.Dropdown(choices=["Natural","Warm","Bright","Deep","Soft","Crisp","Breathy","Rich"], value="Natural", label="Timbre")
                design_language = gr.Dropdown(choices=LANGUAGE_CHOICES, value="Indonesian", label="Language")
                accent = gr.Dropdown(choices=["Auto","Indonesian","American English","British English","International English"], value="Indonesian", label="Accent")
                style = gr.Dropdown(choices=STYLE_CHOICES, value="Natural", label="Gaya Suara", info="Gaya mencakup cara bicara, ekspresi, energi, dan ritme.")
                custom_instruction = gr.Textbox(label="Custom Style", lines=4, max_lines=7, placeholder="Contoh: warm, relaxed, like a late-night podcast host with natural pauses…")
                gr.Markdown("### Advanced")
                design_cfg = gr.Slider(minimum=MIN_DESIGN_CFG, maximum=MAX_DESIGN_CFG, value=DEFAULT_DESIGN_CFG, step=0.1, label="CFG")
                design_steps = gr.Slider(minimum=MIN_DESIGN_STEPS, maximum=MAX_DESIGN_STEPS, value=DEFAULT_DESIGN_STEPS, step=1, label="Steps")
                with gr.Row():
                    design_random_seed_btn = gr.Button("Random Seed", elem_classes=["small-btn"])
                    design_reset_btn = gr.Button("Reset Design", elem_classes=["text-btn"])
                design_seed = gr.Number(value=DEFAULT_SEED, precision=0, label="Seed")
                gr.Markdown("FireRedTTS3-Instruct membuat voice baru dari deskripsi, tanpa reference audio.", elem_classes=["field-hint"])

    def _nav_changed(label):
        mode = "cloning" if (label or "").startswith("🎙") else "design"
        is_clone = mode == "cloning"
        title = "🎙 **Voice Cloning**" if is_clone else "🎨 **Voice Design**"
        status = "● Voice Cloning aktif" if is_clone else "● Voice Design aktif"
        return (
            mode,
            gr.update(visible=is_clone),
            gr.update(visible=not is_clone),
            gr.update(value=title),
            gr.update(value=status),
            gr.update(value=status),
        )

    mode_nav.change(
        _nav_changed,
        inputs=[mode_nav],
        outputs=[mode_state, clone_panel, design_panel, active_title, active_nav_status, settings_sync],
    )

    preset_nat.click(lambda: set_clone_preset("Alami"), outputs=[output_speed, pitch_semitones])
    preset_deep.click(lambda: set_clone_preset("Dalam & Tenang"), outputs=[output_speed, pitch_semitones])
    preset_fast.click(lambda: set_clone_preset("Ceria"), outputs=[output_speed, pitch_semitones])
    preset_news.click(lambda: set_clone_preset("Berita"), outputs=[output_speed, pitch_semitones])
    random_seed_btn.click(random_clone_seed, outputs=[seed])
    reset_btn.click(reset_clone_controls, inputs=[], outputs=[output_speed, pitch_semitones])
    design_random_seed_btn.click(random_seed, outputs=[design_seed])
    design_reset_btn.click(reset_design_controls, inputs=[], outputs=[gender, age, timbre, accent, style, custom_instruction, design_cfg, design_steps, design_seed])
    reference_audio.change(describe_reference_audio, inputs=[reference_audio], outputs=[ref_status])
    target_text.change(describe_char_count, inputs=[target_text], outputs=[char_counter])

    for button, preset_name in zip(narration_buttons, NARRATION_PRESETS):
        button.click(lambda name=preset_name: apply_narration_preset(name), inputs=[], outputs=[target_text])
    for button, person_name in zip(person_buttons, PERSON_PRESETS):
        button.click(lambda name=person_name: apply_person_preset(name), inputs=[], outputs=[gender, age, timbre, accent, custom_instruction])

    generation_event = generate_btn.click(
        _begin_generation,
        inputs=[],
        outputs=[generate_btn, generated_audio, output_loading, mode_nav, output_status],
        js=OUTPUT_SCROLL_JS,
    ).then(
        _generation_phase,
        inputs=[],
        outputs=[generate_btn, output_loading, output_status],
    ).then(
        run_generation_ui,
        inputs=[
            history_state, mode_state, target_text, reference_audio, reference_transcript,
            language, seed, design_seed, design_language, output_speed, pitch_semitones,
            gender, age, timbre, accent, style, custom_instruction, design_cfg, design_steps,
        ],
        outputs=[
            generated_audio, generation_summary, generation_prompt, output_loading,
            generate_btn, mode_nav, output_status, history_state,
            *history_outputs,
        ],
    )

import threading
threading.Thread(target=backend.preload_model, name="fireredtts3-base-preload", daemon=True).start()

demo.queue(max_size=2, default_concurrency_limit=1)
launch_result = demo.launch(
    server_name="0.0.0.0", server_port=PORT, share=True, show_error=True,
    prevent_thread_lock=True, allowed_paths=[str(OUTPUT_DIR)], css=css,
    theme=gr.themes.Soft()
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

print("[MODEL] Base preload scheduled in background; local model cache is active.", flush=True)

while True:
    time.sleep(3600)
