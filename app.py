import time
from pathlib import Path

import gradio as gr

# ==========================================
# CSS — dimuat dari theme.css (satu sumber desain)
# ==========================================
CSS_PATH = Path(__file__).parent / "theme.css"
custom_css = CSS_PATH.read_text(encoding="utf-8") if CSS_PATH.exists() else ""

CUSTOM_VOICE_LABEL = "Kustom (unggah referensi baru)"

PRESET_VOICES = [
    "Indonesia — Perempuan (Clara)",
    "Indonesia — Laki-laki (Budi)",
    "Inggris — Perempuan (Aria)",
    "Inggris — Laki-laki (Guy)",
]

QUICK_PROMPTS = [
    ("Bacakan cerita pendek", "Di sebuah desa kecil di lereng gunung, hiduplah seorang perajin yang membuat lonceng dari tanah liat."),
    ("Buat naskah iklan", "Rasakan kesegaran alami dalam setiap tegukan — hanya dari bahan pilihan, tanpa pengawet."),
    ("Sampaikan lelucon ringan", "Kenapa komputer tidak pernah kedinginan? Karena dia selalu punya banyak Windows."),
    ("Perkenalkan podcast", "Selamat datang kembali di podcast kita. Hari ini kita akan membahas topik yang sudah lama ditunggu."),
    ("Narasikan adegan film", "Hujan turun perlahan saat dua sosok berdiri berhadapan di ujung dermaga, tak ada yang berani bicara lebih dulu."),
    ("Panduan meditasi singkat", "Tarik napas perlahan. Rasakan udara mengisi dadamu, lalu hembuskan dengan tenang."),
]


# ==========================================
# STATE HELPERS
# ==========================================
def render_history(history):
    if not history:
        return "<div class='history-empty'>Belum ada riwayat pembuatan suara pada sesi ini.</div>"
    rows = []
    for item in history:
        rows.append(
            f"""<div class="history-item">
                    <div class="history-item-text">{item['text']}</div>
                    <div class="history-item-meta">{item['voice']} · {item['time']}</div>
                </div>"""
        )
    return "<div class='history-list'>" + "".join(rows) + "</div>"


def render_voice_cards(custom_map):
    cards = []
    for name in PRESET_VOICES:
        cards.append(
            f"""<div class="voice-card">
                    <div class="voice-card-avatar preset">{name[0]}</div>
                    <div>
                        <div class="voice-card-name">{name}</div>
                        <div class="voice-card-tag">Preset bawaan</div>
                    </div>
                </div>"""
        )
    for name in custom_map:
        cards.append(
            f"""<div class="voice-card">
                    <div class="voice-card-avatar custom">{name[0].upper()}</div>
                    <div>
                        <div class="voice-card-name">{name}</div>
                        <div class="voice-card-tag">Hasil kloning kamu</div>
                    </div>
                </div>"""
        )
    return "<div class='voice-grid'>" + "".join(cards) + "</div>"


# ==========================================
# FUNGSI INFERENSI
# (Gantikan isi generate_speech dengan pemanggilan backend FireRedTTS3 asli)
# ==========================================
def generate_speech(text, voice, ref_audio, ref_text, speed, stability, style, seed, custom_map, history):
    if not text or not text.strip():
        msg = "Teks masih kosong — tulis dulu kalimat yang ingin disintesis."
        return None, msg, history, render_history(history)

    if voice == CUSTOM_VOICE_LABEL:
        if not ref_audio:
            msg = "Unggah audio referensi terlebih dahulu untuk suara kustom."
            return None, msg, history, render_history(history)
        audio_path = ref_audio
    elif voice in custom_map:
        audio_path = custom_map[voice]
    else:
        audio_path = None  # suara preset — belum ada file audio asli, ini placeholder

    time.sleep(1.2)  # simulasi proses inferensi

    status = f"Selesai · suara \u201c{voice}\u201d · speed {speed}x · stabilitas {stability}"
    entry = {
        "text": (text.strip()[:70] + "…") if len(text.strip()) > 70 else text.strip(),
        "voice": voice,
        "time": time.strftime("%H:%M:%S"),
    }
    new_history = ([entry] + history)[:20]
    return audio_path, status, new_history, render_history(new_history)


def toggle_custom_reference(voice):
    return gr.Group(visible=(voice == CUSTOM_VOICE_LABEL))


def save_custom_voice(name, audio_path, custom_map, current_choices):
    name = (name or "").strip()
    if not name:
        return custom_map, gr.Dropdown(choices=current_choices), None, "", render_voice_cards(custom_map), \
            "Nama suara wajib diisi."
    if not audio_path:
        return custom_map, gr.Dropdown(choices=current_choices), None, name, render_voice_cards(custom_map), \
            "Unggah contoh audio suara terlebih dahulu."

    new_map = dict(custom_map)
    new_map[name] = audio_path
    new_choices = current_choices if name in current_choices else current_choices[:-1] + [name, CUSTOM_VOICE_LABEL]
    status = f"Suara \u201c{name}\u201d tersimpan dan bisa dipilih di halaman Text to Speech."
    return new_map, gr.Dropdown(choices=new_choices), None, "", render_voice_cards(new_map), status


# ==========================================
# NAVIGASI SIDEBAR <-> HALAMAN
# ==========================================
PAGE_IDS = ["tts", "voices", "system"]
PAGE_TITLES = {
    "tts": "### Text to Speech",
    "voices": "### Voices",
    "system": "### System",
}


def switch_page(page_id):
    def _handler():
        classes = [
            ["nav-item", "active"] if pid == page_id else ["nav-item"]
            for pid in PAGE_IDS
        ]
        return (
            gr.Tabs(selected=page_id),
            PAGE_TITLES[page_id],
            gr.Button(elem_classes=classes[0]),
            gr.Button(elem_classes=classes[1]),
            gr.Button(elem_classes=classes[2]),
        )
    return _handler


# ==========================================
# TEMA GRADIO
# ==========================================
theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.emerald,
    secondary_hue=gr.themes.colors.stone,
    neutral_hue=gr.themes.colors.stone,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
)

with gr.Blocks(theme=theme, css=custom_css, title="Cangkeman AI Voice Studio") as demo:

    custom_voices_state = gr.State({})       # {nama_suara: filepath}
    history_state = gr.State([])             # [{text, voice, time}, ...]

    with gr.Row(elem_classes=["app-shell"]):

        # ---------------- SIDEBAR ----------------
        with gr.Column(scale=2, min_width=220, elem_classes=["app-sidebar"]):
            gr.HTML("""
                <div class="brand-row sidebar-brand">
                    <div class="brand-mark">C</div>
                    <div class="brand-name">Cangkeman<span>AI Voice Studio</span></div>
                </div>
            """)
            nav_tts = gr.Button("Text to Speech", elem_classes=["nav-item", "active"])
            nav_voices = gr.Button("Voices", elem_classes=["nav-item"])
            nav_system = gr.Button("System", elem_classes=["nav-item"])
            gr.HTML('<div class="sidebar-footnote">FireRedTTS3 · runtime lokal</div>')

        # ---------------- KONTEN UTAMA ----------------
        with gr.Column(scale=10, elem_classes=["app-main-col"]):

            with gr.Row(elem_classes=["topbar"]):
                page_title = gr.Markdown(PAGE_TITLES["tts"], elem_classes=["topbar-title"])
                gr.HTML('<span class="status-badge">Siap · CUDA</span>')

            with gr.Tabs(elem_id="page-tabs") as page_tabs:

                # ======= HALAMAN 1 : TEXT TO SPEECH =======
                with gr.TabItem("Text to Speech", id="tts"):
                    with gr.Row():

                        # --- Kolom tengah: teks & hasil ---
                        with gr.Column(scale=7, elem_classes=["app-main"]):
                            main_text = gr.Textbox(
                                placeholder="Ketik atau tempel teks yang ingin diubah menjadi suara…",
                                lines=9,
                                show_label=False,
                                elem_classes=["main-textbox"],
                            )

                            gr.Markdown("Mulai dengan contoh", elem_classes=["quickstart-label"])
                            with gr.Row(elem_classes=["chip-row"]):
                                for label, sample in QUICK_PROMPTS:
                                    btn = gr.Button(label, elem_classes=["chip-btn"])
                                    btn.click(fn=(lambda s=sample: s), inputs=None, outputs=main_text)

                            with gr.Group(visible=False, elem_classes=["custom-card", "custom-ref-group"]) as custom_ref_group:
                                gr.Markdown("**Referensi suara kustom**")
                                ref_audio = gr.Audio(
                                    label="Sampel suara sumber (3–10 detik)",
                                    type="filepath",
                                    sources=["upload", "microphone"],
                                )
                                ref_text = gr.Textbox(
                                    label="Transkrip audio referensi (opsional)",
                                    placeholder="Isi teks yang diucapkan pada audio di atas untuk hasil lebih akurat…",
                                    lines=2,
                                )

                            with gr.Row(elem_classes=["generate-bar"]):
                                btn_generate = gr.Button("Buat suara", elem_classes=["btn-primary", "btn-generate"])

                            with gr.Group(elem_classes=["custom-card"]):
                                gr.Markdown("**Hasil**")
                                output_audio = gr.Audio(
                                    label="Audio hasil", type="filepath",
                                    interactive=False, elem_classes=["output-audio"],
                                )
                                output_status = gr.Markdown("Belum ada proses berjalan.", elem_id="status-output")

                        # --- Kolom kanan: pengaturan / riwayat ---
                        with gr.Column(scale=4, elem_classes=["app-rightpanel"]):
                            with gr.Tabs(elem_id="settings-tabs"):
                                with gr.TabItem("Pengaturan"):
                                    with gr.Group(elem_classes=["custom-card"]):
                                        gr.Markdown("Voice", elem_classes=["field-label"])
                                        voice_select = gr.Dropdown(
                                            choices=PRESET_VOICES + [CUSTOM_VOICE_LABEL],
                                            value=PRESET_VOICES[0],
                                            show_label=False,
                                        )

                                        gr.Markdown("Model", elem_classes=["field-label"])
                                        gr.Textbox(
                                            value="FireRedTTS3 — Multilingual",
                                            interactive=False, show_label=False,
                                            elem_classes=["spec-field"],
                                        )

                                        gr.Markdown("Speed", elem_classes=["field-label"])
                                        speed = gr.Slider(0.5, 2.0, value=1.0, step=0.1, show_label=False)

                                        gr.Markdown("Stabilitas", elem_classes=["field-label"])
                                        stability = gr.Slider(0.0, 1.0, value=0.6, step=0.05, show_label=False,
                                                               elem_classes=["range-slider"])
                                        gr.HTML('<div class="range-caption"><span>Lebih variatif</span>'
                                                '<span>Lebih stabil</span></div>')

                                        gr.Markdown("Ekspresi", elem_classes=["field-label"])
                                        style = gr.Slider(0.0, 1.0, value=0.3, step=0.05, show_label=False)

                                        seed = gr.Number(value=-1, label="Seed (-1 = acak)")

                                with gr.TabItem("Riwayat"):
                                    history_display = gr.HTML(render_history([]))

                    btn_generate.click(
                        fn=generate_speech,
                        inputs=[main_text, voice_select, ref_audio, ref_text, speed, stability, style, seed,
                                custom_voices_state, history_state],
                        outputs=[output_audio, output_status, history_state, history_display],
                    )
                    voice_select.change(fn=toggle_custom_reference, inputs=voice_select, outputs=custom_ref_group)

                # ======= HALAMAN 2 : VOICES =======
                with gr.TabItem("Voices", id="voices"):
                    with gr.Row():
                        with gr.Column(scale=6):
                            gr.Markdown("**Pustaka suara**")
                            voice_gallery = gr.HTML(render_voice_cards({}))

                        with gr.Column(scale=4):
                            with gr.Group(elem_classes=["custom-card"]):
                                gr.Markdown("**Kloning suara baru**")
                                new_voice_name = gr.Textbox(label="Nama suara", placeholder="mis. Suara Narator 1")
                                new_voice_audio = gr.Audio(
                                    label="Contoh suara (3–10 detik)",
                                    type="filepath",
                                    sources=["upload", "microphone"],
                                )
                                btn_save_voice = gr.Button("Simpan suara", elem_classes=["btn-primary"])
                                save_voice_status = gr.Markdown("", elem_id="status-output")

                    btn_save_voice.click(
                        fn=save_custom_voice,
                        inputs=[new_voice_name, new_voice_audio, custom_voices_state, voice_select],
                        outputs=[custom_voices_state, voice_select, new_voice_audio, new_voice_name,
                                 voice_gallery, save_voice_status],
                    )

                # ======= HALAMAN 3 : SYSTEM =======
                with gr.TabItem("System", id="system"):
                    with gr.Group(elem_classes=["custom-card"]):
                        gr.Markdown("**Status model dan runtime**")
                        with gr.Row():
                            gr.Textbox(label="Perangkat", value="CUDA — NVIDIA GeForce RTX / Colab GPU",
                                       interactive=False, elem_classes=["spec-field"])
                            gr.Textbox(label="Presisi", value="FP16 (half precision)",
                                       interactive=False, elem_classes=["spec-field"])
                        with gr.Row():
                            gr.Textbox(label="Direktori bobot model", value="./models/FireRedTTS3",
                                       interactive=False, elem_classes=["spec-field"])
                            gr.Textbox(label="Sample rate keluaran", value="24000 Hz",
                                       interactive=False, elem_classes=["spec-field"])

            gr.HTML('<div class="footer-text">Cangkeman AI Voice Studio · antarmuka FireRedTTS3</div>')

    nav_tts.click(switch_page("tts"), outputs=[page_tabs, page_title, nav_tts, nav_voices, nav_system])
    nav_voices.click(switch_page("voices"), outputs=[page_tabs, page_title, nav_tts, nav_voices, nav_system])
    nav_system.click(switch_page("system"), outputs=[page_tabs, page_title, nav_tts, nav_voices, nav_system])

if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
