import os
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
LANGUAGE_CHOICES=[(LANGUAGE_LABELS.get(code,code),code) for code in SUPPORTED_LANGUAGES]

STYLE_CHOICES=list(backend.STYLE_INSTRUCTIONS.keys())


def _validate_reference_audio(path):
    if not path: raise gr.Error("Audio referensi belum dipilih.")
    p=Path(path)
    if not p.is_file() or p.stat().st_size<=0: raise gr.Error("Audio referensi tidak ditemukan atau kosong.")
    try:
        import soundfile as sf
        info=sf.info(str(p))
    except Exception as exc:
        raise gr.Error(f"Audio referensi tidak dapat dibaca: {exc}")
    duration=info.frames/float(info.samplerate) if info.samplerate else 0.0
    if duration<MIN_PROMPT_SECONDS: raise gr.Error(f"Audio referensi terlalu pendek ({duration:.2f} detik). Minimal {MIN_PROMPT_SECONDS:.1f} detik.")
    if duration>MAX_PROMPT_SECONDS: raise gr.Error(f"Audio referensi terlalu panjang ({duration:.2f} detik). Maksimal {MAX_PROMPT_SECONDS:.0f} detik.")
    return duration,int(info.samplerate)

def describe_reference_audio(path):
    if not path: return "Upload audio referensi untuk mulai cloning."
    try: duration,sr=_validate_reference_audio(path)
    except gr.Error as exc: return str(exc)
    return f"Siap dikloning · {duration:.1f} detik · {sr:,} Hz"

def describe_char_count(text):
    n=len(text or "")
    if n==0: return f"Maksimal {MAX_TARGET_CHARS} karakter."
    if n>MAX_TARGET_CHARS: return f"{n} karakter · kelebihan {n-MAX_TARGET_CHARS}."
    return f"{n} / {MAX_TARGET_CHARS} karakter"

def set_clone_preset(name):
    return {"Alami":(1.0,0.0),"Dalam & Tenang":(0.96,-1.0),"Ceria":(1.06,0.5),"Berita":(0.98,-0.5)}.get(name,(1.0,0.0))

def select_mode(mode):
    is_clone = mode == "cloning"
    return (
        gr.update(visible=is_clone),
        gr.update(visible=not is_clone),
        gr.update(value="🎙 Voice Cloning" if is_clone else "🎨 Voice Design"),
        gr.update(value="Voice cloning" if is_clone else "Voice design"),
    )
def generate_dispatch(mode,target_text,reference_audio,reference_transcript,language,seed,design_seed,output_speed,pitch_semitones,gender,age,timbre,accent,style,custom_instruction,design_cfg,design_steps):
    started=time.time()
    if mode=="cloning":
        audio_result,master_path=generate_voice(target_text,reference_transcript,reference_audio,language,seed,output_speed,pitch_semitones)
        sr,audio=audio_result
        duration=audio.shape[-1]/float(sr)
        files=[str(p) for p in [Path(master_path),Path(master_path).with_name(Path(master_path).name.replace("_HQ_FLOAT32.wav","_PCM16.wav"))] if p.is_file()]
        summary=(f"**{duration:.1f} detik** · {int(sr):,} Hz · "
                 f"Voice Cloning · seed {int(seed)} · speed {float(output_speed):.2f}x · pitch {float(pitch_semitones):+.1f}st")
        return audio_result,summary,files
    audio_result,master_path,voice_plan,instruction=generate_voice_design(target_text,gender,age,timbre,accent,style,custom_instruction,design_seed,design_cfg,design_steps)
    sr,audio=audio_result
    duration=audio.shape[-1]/float(sr)
    files=[str(p) for p in [Path(master_path),Path(master_path).with_name(Path(master_path).name.replace("_HQ_FLOAT32.wav","_PCM16.wav"))] if p.is_file()]
    summary=(f"**{duration:.1f} detik** · {int(sr):,} Hz · Voice Design · "
             f"gaya **{style}** · seed {int(seed)} · CFG {float(design_cfg):.2f} · steps {int(design_steps)}\n\n"
             f"**Voice plan:** {voice_plan}")
    return audio_result,summary,files

APP_DIR=Path(__file__).resolve().parent
THEME_CSS_FILE=APP_DIR/"theme.css"
css=THEME_CSS_FILE.read_text(encoding="utf-8") if THEME_CSS_FILE.is_file() else ""

with gr.Blocks(title="Cangkeman — AI Voice Studio") as demo:
    mode_state=gr.State("cloning")
    active_title=gr.Markdown("🎙 **Voice Cloning**",elem_classes=["mode-chip"])
    active_subtitle=gr.Markdown("",visible=False)

    with gr.Row(elem_classes=["app-shell"]):
        # LEFT — tool navigation only
        with gr.Column(scale=0, min_width=190, elem_classes=["nav-col"]):
            gr.HTML("""<div class='brand'><div class='brand-mark'>C</div><div><div class='brand-name'>Cangkeman</div><div class='brand-tag'>AI Voice Studio</div></div></div>""")
            gr.Markdown("TOOLS",elem_classes=["nav-caption"])
            mode_nav=gr.Radio(
                choices=["🎙  Voice Cloning","🎨  Voice Design"],
                value="🎙  Voice Cloning",
                label=None,
                show_label=False,
                container=False,
                elem_classes=["mode-nav"],
            )
            gr.Markdown("",elem_classes=["nav-spacer"])
            gr.Markdown("FireRedTTS3 · T4",elem_classes=["nav-foot"])

        # CENTER — workspace
        with gr.Column(scale=1,elem_classes=["workspace-col"]):
            gr.Markdown("Workspace",elem_classes=["eyebrow"])
            target_text=gr.Textbox(label="",lines=9,max_lines=14,placeholder="Tulis teks yang ingin diucapkan…",elem_classes=["main-text"])
            char_counter=gr.Markdown(f"Maksimal {MAX_TARGET_CHARS} karakter.",elem_classes=["field-hint","char-counter"])
            generate_btn=gr.Button("Generate Speech",variant="primary",elem_classes=["generate-btn"])
            gr.Markdown("Output",elem_classes=["eyebrow","output-heading"])
            generated_audio=gr.Audio(label="",autoplay=False,elem_classes=["output-audio"])
            generation_summary=gr.Markdown("Belum ada hasil — isi teks lalu Generate Speech.",elem_classes=["summary-box"])
            output_files=gr.File(label="Export",file_count="multiple",elem_classes=["file-output"])

        # RIGHT — settings follow active model capability
        with gr.Column(scale=0,min_width=315,elem_classes=["settings-col"]):
            gr.Markdown("Settings",elem_classes=["eyebrow"])

            with gr.Column(visible=True,elem_classes=["settings-panel"]) as clone_panel:
                gr.Markdown("### Voice Cloning")
                reference_audio=gr.Audio(sources=["upload","microphone"],type="filepath",label=f"Reference Audio · {MIN_PROMPT_SECONDS:.0f}–{MAX_PROMPT_SECONDS:.0f} detik")
                reference_transcript=gr.Textbox(label="Reference Transcript",lines=4,placeholder="Tulis persis ucapan pada audio referensi…")
                ref_status=gr.Markdown("Upload audio referensi untuk mulai cloning.",elem_classes=["field-hint"])
                language=gr.Dropdown(choices=LANGUAGE_CHOICES,value="Indonesian",label="Language")
                gr.Markdown("### Fine Control")
                with gr.Row():
                    output_speed=gr.Slider(minimum=MIN_OUTPUT_SPEED,maximum=MAX_OUTPUT_SPEED,value=1.0,step=0.01,label="Speed")
                    pitch_semitones=gr.Slider(minimum=MIN_PITCH_SEMITONES,maximum=MAX_PITCH_SEMITONES,value=0.0,step=0.5,label="Pitch")
                gr.Markdown("### Presets")
                with gr.Row():
                    preset_nat=gr.Button("Alami",elem_classes=["small-btn"])
                    preset_deep=gr.Button("Tenang",elem_classes=["small-btn"])
                with gr.Row():
                    preset_fast=gr.Button("Ceria",elem_classes=["small-btn"])
                    preset_news=gr.Button("Berita",elem_classes=["small-btn"])
                gr.Markdown("### Advanced")
                seed=gr.Number(value=DEFAULT_SEED,precision=0,label="Seed")
                reset_btn=gr.Button("Reset controls",elem_classes=["text-btn"])

            with gr.Column(visible=False,elem_classes=["settings-panel"]) as design_panel:
                gr.Markdown("### Voice Design")
                gender=gr.Dropdown(choices=["Auto","Male","Female"],value="Auto",label="Gender")
                age=gr.Dropdown(choices=["Auto","Young Adult","Adult","Mature","Senior"],value="Auto",label="Age")
                timbre=gr.Dropdown(choices=["Natural","Warm","Bright","Deep","Soft","Crisp","Breathy","Rich"],value="Natural",label="Timbre")
                accent=gr.Dropdown(choices=["Auto","Indonesian","American English","British English","International English"],value="Auto",label="Accent")
                style=gr.Dropdown(choices=STYLE_CHOICES,value="Natural",label="Gaya Suara",info="Gaya mencakup cara bicara, ekspresi, energi, dan ritme.")
                custom_instruction=gr.Textbox(label="Custom Style",lines=4,max_lines=7,placeholder="Contoh: warm, relaxed, like a late-night podcast host with natural pauses…")
                gr.Markdown("### Advanced")
                design_cfg=gr.Slider(minimum=MIN_DESIGN_CFG,maximum=MAX_DESIGN_CFG,value=DEFAULT_DESIGN_CFG,step=0.1,label="CFG")
                design_steps=gr.Slider(minimum=MIN_DESIGN_STEPS,maximum=MAX_DESIGN_STEPS,value=DEFAULT_DESIGN_STEPS,step=1,label="Steps")
                design_seed=gr.Number(value=DEFAULT_SEED,precision=0,label="Seed")
                gr.Markdown("FireRedTTS3-Instruct membuat voice baru dari deskripsi, tanpa reference audio.",elem_classes=["field-hint"])

    def _nav_changed(label):
        mode = "cloning" if (label or "").startswith("🎙") else "design"
        return (mode, *select_mode(mode))

    mode_nav.change(
        _nav_changed,
        inputs=[mode_nav],
        outputs=[mode_state,clone_panel,design_panel,active_title,active_subtitle],
    )
    preset_nat.click(lambda:set_clone_preset("Alami"),outputs=[output_speed,pitch_semitones])
    preset_deep.click(lambda:set_clone_preset("Dalam & Tenang"),outputs=[output_speed,pitch_semitones])
    preset_fast.click(lambda:set_clone_preset("Ceria"),outputs=[output_speed,pitch_semitones])
    preset_news.click(lambda:set_clone_preset("Berita"),outputs=[output_speed,pitch_semitones])
    reset_btn.click(reset_clone_controls,inputs=[],outputs=[output_speed,pitch_semitones])
    reference_audio.change(describe_reference_audio,inputs=[reference_audio],outputs=[ref_status])
    target_text.change(describe_char_count,inputs=[target_text],outputs=[char_counter])
    generate_btn.click(
        generate_dispatch,
        inputs=[mode_state,target_text,reference_audio,reference_transcript,language,seed,design_seed,output_speed,pitch_semitones,gender,age,timbre,accent,style,custom_instruction,design_cfg,design_steps],
        outputs=[generated_audio,generation_summary,output_files],
    )

demo.queue(max_size=4,default_concurrency_limit=1)
launch_result=demo.launch(server_name="0.0.0.0",server_port=PORT,share=True,show_error=True,prevent_thread_lock=True,allowed_paths=[str(OUTPUT_DIR)],css=css,theme=gr.themes.Soft())
try:
    _,local_url,share_url=launch_result
except Exception:
    local_url,share_url=None,None
if local_url:
    LOCAL_URL_FILE.write_text(str(local_url),encoding="utf-8")
    print("FIREREDTTS3_WEBUI_LOCAL_URL:",local_url,flush=True)
if share_url:
    URL_FILE.write_text(str(share_url),encoding="utf-8")
    print("FIREREDTTS3_WEBUI_PUBLIC_URL:",share_url,flush=True)
else:
    print("FIREREDTTS3_WEBUI_PUBLIC_URL: unavailable",flush=True)

# Fix26.3: keep the public interface alive and lazy-load models on demand.
# This preserves the known-good Fix25 behavior and avoids startup GPU/thread contention.
print("[MODEL] Lazy-load mode enabled; model loads on first Generate.", flush=True)

while True:
    time.sleep(3600)
