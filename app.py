import gradio as gr
import time

# ==========================================
# CUSTOM CSS UNTUK TEMA TERANG PROFESIONAL
# ==========================================
custom_css = """
/* Latar Belakang & Font Utama */
body, .gradio-container {
    background-color: #f8fafc !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
}

/* Header Banner */
.main-header {
    background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%);
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}

.main-header h1 {
    color: #0f172a !important;
    font-weight: 700 !important;
    font-size: 1.875rem !important;
    margin-bottom: 6px !important;
}

.main-header p {
    color: #475569 !important;
    font-size: 0.95rem !important;
}

/* Badge Status */
.status-badge {
    display: inline-block;
    background-color: #dcfce7;
    color: #15803d;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 9999px;
    border: 1px solid #bbf7d0;
}

/* Kustomisasi Group / Card Panel */
.custom-card {
    background-color: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    padding: 18px !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.02) !important;
    margin-bottom: 12px !important;
}

/* Tombol Utas / Action Button */
.btn-primary {
    background: linear-gradient(135deg, #4f46e5 0%, #2563eb 100%) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    border: none !important;
    box-shadow: 0 2px 4px rgba(37, 99, 235, 0.2) !important;
    transition: all 0.2s ease !important;
}

.btn-primary:hover {
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.35) !important;
    transform: translateY(-1px);
}

.btn-secondary {
    background-color: #f1f5f9 !important;
    color: #334155 !important;
    border: 1px solid #cbd5e1 !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
}

.btn-secondary:hover {
    background-color: #e2e8f0 !important;
}

/* Label & Input Box Styling */
label span {
    font-weight: 600 !important;
    color: #1e293b !important;
    font-size: 0.875rem !important;
}

textarea, input[type="text"] {
    border-radius: 8px !important;
    border: 1px solid #cbd5e1 !important;
    background-color: #ffffff !important;
}

textarea:focus, input[type="text"]:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.15) !important;
}

/* Tab Styling */
.tabs button.selected {
    border-bottom-color: #2563eb !important;
    color: #2563eb !important;
    font-weight: 600 !important;
}

/* Footer Info */
.footer-text {
    text-align: center;
    color: #64748b;
    font-size: 0.8rem;
    margin-top: 24px;
}
"""

# ==========================================
# DUMMY / PLACEHOLDER INFERENCE FUNCTION
# (Gantikan dengan fungsi inferensi asli anda)
# ==========================================
def generate_zero_shot_tts(ref_audio, ref_text, gen_text, temperature, top_p, speed):
    if not gen_text:
        return None, "⚠️ Harap masukkan teks target untuk disintesis."
    
    # Simulasi proses inferensi
    start_time = time.time()
    time.sleep(1.5)  # Simulasi proses
    elapsed = round(time.time() - start_time, 2)
    
    info_msg = f"✅ Audio berhasil dibuat dalam {elapsed} detik. (Speed: {speed}x)"
    # Kembalikan path audio hasil dan pesan status
    return ref_audio, info_msg  # Output dummy mengembalikan ref_audio sebagai demo

def generate_preset_tts(text_input, voice_preset, speed):
    if not text_input:
        return None, "⚠️ Teks input tidak boleh kosong."
    
    return None, f"✅ Sintesis dengan sampel '{voice_preset}' selesai."

# ==========================================
# DESAIN TAMPILAN (GRADIO BLOCKS)
# ==========================================
theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"]
)

with gr.Blocks(theme=theme, css=custom_css, title="FireRedTTS v3 Studio") as demo:
    
    # 1. HEADER SECTION
    with gr.Row(elem_classes=["main-header"]):
        with gr.Column(scale=4):
            gr.HTML("""
                <div style="display: flex; align-items: center; gap: 10px;">
                    <h1>FireRedTTS v3 Studio</h1>
                    <span class="status-badge">● Ready (CUDA)</span>
                </div>
                <p>Platform Sintesis Suara AI & Zero-Shot Voice Cloning Berkualitas Tinggi</p>
            """)
    
    # 2. MAIN WORKSPACE (TABS)
    with gr.Tabs(elem_classes=["tabs"]):
        
        # TAB 1: ZERO-SHOT VOICE CLONING
        with gr.TabItem("🎙️ Zero-Shot Voice Cloning"):
            with gr.Row():
                # KOLOM KIRI: INPUT & KONTROL
                with gr.Column(scale=5):
                    with gr.Group(elem_classes=["custom-card"]):
                        gr.Markdown("### 1. Audio Referensi Suara")
                        ref_audio = gr.Audio(
                            label="Upload / Rekam sampel suara (3-10 detik)",
                            type="filepath",
                            sources=["upload", "microphone"]
                        )
                        ref_text = gr.Textbox(
                            label="Transkrip Audio Referensi (Opsional)",
                            placeholder="Ketik teks yang diucapkan pada audio di atas untuk akurasi lebih tinggi...",
                            lines=2
                        )
                    
                    with gr.Group(elem_classes=["custom-card"]):
                        gr.Markdown("### 2. Teks Target (Output)")
                        gen_text = gr.Textbox(
                            label="Teks yang Ingin Disintesis",
                            placeholder="Ketik teks yang ingin dibacakan oleh suara kloningan di sini...",
                            lines=4
                        )
                    
                    with gr.Accordion("⚙️ Parameter Generasi Lanjutan", open=False):
                        with gr.Row():
                            temperature = gr.Slider(0.1, 1.2, value=0.7, step=0.05, label="Temperature")
                            top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-P")
                        with gr.Row():
                            speed = gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="Kecepatan Bicara (Speed)")
                            seed = gr.Number(value=-1, label="Seed (-1 untuk Acak)", precision=0)

                    with gr.Row():
                        btn_clear = gr.Button("🗑️ Bersihkan", elem_classes=["btn-secondary"], scale=1)
                        btn_generate = gr.Button("⚡ Generasi Suara", elem_classes=["btn-primary"], scale=2)

                # KOLOM KANAN: OUTPUT & RESULT
                with gr.Column(scale=5):
                    with gr.Group(elem_classes=["custom-card"]):
                        gr.Markdown("### 3. Hasil Sintesis Audio")
                        output_audio = gr.Audio(
                            label="Pemutar Audio Hasil",
                            type="filepath",
                            interactive=False
                        )
                        output_status = gr.Markdown(
                            value="*Status: Siap memproses input.*",
                            elem_id="status-output"
                        )
                    
                    with gr.Group(elem_classes=["custom-card"]):
                        gr.Markdown("💡 **Tips untuk hasil kloning maksimal:**")
                        gr.Markdown("""
                        - Gunakan sampel audio yang bersih tanpa musik latar (*background noise*).
                        - Durasi ideal sampel rujukan adalah 3–8 detik dengan intonasi jelas.
                        - Mengisi transkrip audio referensi secara tepat dapat meningkatkan kejernihan output secara signifikan.
                        """)

            # Event Handlers
            btn_generate.click(
                fn=generate_zero_shot_tts,
                inputs=[ref_audio, ref_text, gen_text, temperature, top_p, speed],
                outputs=[output_audio, output_status]
            )
            
            btn_clear.click(
                fn=lambda: (None, "", "", 0.7, 0.9, 1.0, None, "*Status: Form dibersihkan.*"),
                inputs=None,
                outputs=[ref_audio, ref_text, gen_text, temperature, top_p, speed, output_audio, output_status]
            )

        # TAB 2: PRESET TEXT-TO-SPEECH
        with gr.TabItem("🔊 Standard TTS (Preset)"):
            with gr.Row():
                with gr.Column(scale=5):
                    with gr.Group(elem_classes=["custom-card"]):
                        gr.Markdown("### Pilih Suara & Input Teks")
                        voice_preset = gr.Dropdown(
                            choices=["Indonesian - Female (Clara)", "Indonesian - Male (Budi)", "English - Female (Aria)", "English - Male (Guy)"],
                            value="Indonesian - Female (Clara)",
                            label="Preset Suara"
                        )
                        preset_text = gr.Textbox(
                            label="Teks Input",
                            placeholder="Ketik kalimat yang ingin diubah menjadi suara...",
                            lines=5
                        )
                        preset_speed = gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="Kecepatan Bicara")
                        btn_preset_gen = gr.Button("🎵 Generasi Audio Preset", elem_classes=["btn-primary"])

                with gr.Column(scale=5):
                    with gr.Group(elem_classes=["custom-card"]):
                        gr.Markdown("### Hasil Audio")
                        preset_audio_out = gr.Audio(label="Audio Output", interactive=False)
                        preset_status_out = gr.Markdown(value="*Status: Menunggu input.*")

            btn_preset_gen.click(
                fn=generate_preset_tts,
                inputs=[preset_text, voice_preset, preset_speed],
                outputs=[preset_audio_out, preset_status_out]
            )

        # TAB 3: PENGATURAN MODEL & SISTEM
        with gr.TabItem("⚙️ Pengaturan Sistem"):
            with gr.Group(elem_classes=["custom-card"]):
                gr.Markdown("### Status & Konfigurasi Model")
                with gr.Row():
                    gr.Textbox(label="Device Target", value="CUDA (NVIDIA GeForce RTX / Colab GPU)", interactive=False)
                    gr.Textbox(label="Presisi Precision", value="FP16 (Half Precision)", interactive=False)
                with gr.Row():
                    gr.Textbox(label="Direktori Bobot Model", value="./models/FireRedTTS3", interactive=False)
                    gr.Textbox(label="Sample Rate Output", value="24000 Hz (24kHz)", interactive=False)

    # 3. FOOTER
    gr.HTML("""
        <div class="footer-text">
            FireRedTTS v3 WebUI Interface • Didesain secara profesional dengan tema terang seragam
        </div>
    """)

if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )