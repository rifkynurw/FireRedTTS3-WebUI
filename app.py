import time
from pathlib import Path

import gradio as gr

# ==========================================
# CSS — dimuat dari theme.css (satu sumber desain, tidak diduplikasi inline)
# ==========================================
CSS_PATH = Path(__file__).parent / "theme.css"
custom_css = CSS_PATH.read_text(encoding="utf-8") if CSS_PATH.exists() else ""


# ==========================================
# FUNGSI INFERENSI
# (Gantikan isi fungsi ini dengan pemanggilan backend FireRedTTS3 asli)
# ==========================================
def generate_zero_shot_tts(ref_audio, ref_text, gen_text, temperature, top_p, speed):
    if not ref_audio:
        return None, "Audio referensi belum diunggah."
    if not gen_text or not gen_text.strip():
        return None, "Teks target masih kosong — isi dulu teks yang ingin disintesis."

    start_time = time.time()
    time.sleep(1.5)  # simulasi proses inferensi
    elapsed = round(time.time() - start_time, 2)

    status = f"Selesai dalam {elapsed} detik · speed {speed}x · temp {temperature}"
    return ref_audio, status  # placeholder: mengembalikan audio referensi sebagai demo


def generate_preset_tts(text_input, voice_preset, speed):
    if not text_input or not text_input.strip():
        return None, "Teks input masih kosong."
    return None, f"Selesai — suara '{voice_preset}' · speed {speed}x"


def clear_clone_form():
    return None, "", "", 0.7, 0.9, 1.0, None, "Form dibersihkan. Siap memproses input baru."


# ==========================================
# TEMA GRADIO — palet & tipografi selaras dengan theme.css
# ==========================================
theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.emerald,
    secondary_hue=gr.themes.colors.stone,
    neutral_hue=gr.themes.colors.stone,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
)

with gr.Blocks(theme=theme, css=custom_css, title="Cangkeman AI Voice Studio") as demo:

    # 1. HEADER
    with gr.Row(elem_classes=["main-header"]):
        with gr.Column(scale=4):
            gr.HTML("""
                <div class="brand-row">
                    <div class="brand-mark">C</div>
                    <div style="flex:1;">
                        <div style="display:flex; align-items:center; gap:10px; flex-wrap:wrap;">
                            <h1>Cangkeman AI Voice Studio</h1>
                            <span class="status-badge">Siap · CUDA</span>
                        </div>
                        <p>Kloning suara dan text-to-speech berbahasa Indonesia, dari referensi audio hingga hasil siap pakai.</p>
                    </div>
                </div>
            """)

    # 2. RUANG KERJA UTAMA
    with gr.Tabs():

        # TAB 1 — ZERO-SHOT VOICE CLONING
        with gr.TabItem("Kloning Suara"):
            with gr.Row():
                with gr.Column(scale=5):
                    with gr.Group(elem_classes=["custom-card"]):
                        gr.Markdown("**1 · Audio referensi**")
                        ref_audio = gr.Audio(
                            label="Sampel suara sumber (3–10 detik)",
                            type="filepath",
                            sources=["upload", "microphone"],
                        )
                        ref_text = gr.Textbox(
                            label="Transkrip audio referensi (opsional)",
                            placeholder="Isi teks yang diucapkan pada audio di atas untuk hasil yang lebih akurat…",
                            lines=2,
                        )

                    with gr.Group(elem_classes=["custom-card"]):
                        gr.Markdown("**2 · Teks yang disintesis**")
                        gen_text = gr.Textbox(
                            label="Teks target",
                            placeholder="Tulis kalimat yang ingin dibacakan dengan suara kloningan…",
                            lines=4,
                        )

                    with gr.Accordion("Parameter lanjutan", open=False):
                        with gr.Row():
                            temperature = gr.Slider(0.1, 1.2, value=0.7, step=0.05, label="Temperature")
                            top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-P")
                        with gr.Row():
                            speed = gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="Kecepatan bicara")
                            seed = gr.Number(value=-1, label="Seed (-1 = acak)", precision=0)

                    with gr.Row():
                        btn_clear = gr.Button("Bersihkan", elem_classes=["btn-secondary"], scale=1)
                        btn_generate = gr.Button("Buat suara", elem_classes=["btn-primary"], scale=2)

                with gr.Column(scale=5):
                    with gr.Group(elem_classes=["custom-card"]):
                        gr.Markdown("**3 · Hasil**")
                        output_audio = gr.Audio(
                            label="Audio hasil",
                            type="filepath",
                            interactive=False,
                            elem_classes=["output-audio"],
                        )
                        output_status = gr.Markdown(
                            value="Belum ada proses berjalan.",
                            elem_id="status-output",
                        )

                    with gr.Group(elem_classes=["custom-card", "tips-card"]):
                        gr.Markdown("**Agar hasil kloning maksimal**")
                        gr.Markdown(
                            "- Gunakan audio bersih tanpa suara latar.\n"
                            "- Durasi ideal referensi: 3–8 detik dengan intonasi jelas.\n"
                            "- Mengisi transkrip referensi membantu kejernihan hasil."
                        )

            btn_generate.click(
                fn=generate_zero_shot_tts,
                inputs=[ref_audio, ref_text, gen_text, temperature, top_p, speed],
                outputs=[output_audio, output_status],
            )
            btn_clear.click(
                fn=clear_clone_form,
                inputs=None,
                outputs=[ref_audio, ref_text, gen_text, temperature, top_p, speed, output_audio, output_status],
            )

        # TAB 2 — PRESET TEXT-TO-SPEECH
        with gr.TabItem("Suara Preset"):
            with gr.Row():
                with gr.Column(scale=5):
                    with gr.Group(elem_classes=["custom-card"]):
                        gr.Markdown("**Pilih suara dan tulis teks**")
                        voice_preset = gr.Dropdown(
                            choices=[
                                "Indonesia - Perempuan (Clara)",
                                "Indonesia - Laki-laki (Budi)",
                                "Inggris - Perempuan (Aria)",
                                "Inggris - Laki-laki (Guy)",
                            ],
                            value="Indonesia - Perempuan (Clara)",
                            label="Preset suara",
                        )
                        preset_text = gr.Textbox(
                            label="Teks input",
                            placeholder="Tulis kalimat yang ingin diubah menjadi suara…",
                            lines=5,
                        )
                        preset_speed = gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="Kecepatan bicara")
                        btn_preset_gen = gr.Button("Buat audio", elem_classes=["btn-primary"])

                with gr.Column(scale=5):
                    with gr.Group(elem_classes=["custom-card"]):
                        gr.Markdown("**Hasil**")
                        preset_audio_out = gr.Audio(label="Audio hasil", interactive=False, elem_classes=["output-audio"])
                        preset_status_out = gr.Markdown(value="Menunggu input.", elem_id="status-output")

            btn_preset_gen.click(
                fn=generate_preset_tts,
                inputs=[preset_text, voice_preset, preset_speed],
                outputs=[preset_audio_out, preset_status_out],
            )

        # TAB 3 — PENGATURAN SISTEM
        with gr.TabItem("Sistem"):
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

    # 3. FOOTER
    gr.HTML('<div class="footer-text">Cangkeman AI Voice Studio · antarmuka FireRedTTS3</div>')

if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
