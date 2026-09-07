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

NARRATION_PRESETS={
 "🎙 Narasi Natural":"Hari ini kita akan membahas sebuah cerita sederhana, dengan alur yang mengalir dan cara penyampaian yang terdengar alami.",
 "📖 Storytelling":"Malam itu, hujan turun perlahan ketika ia berjalan sendirian menyusuri jalan yang sudah lama tidak ia lewati.",
 "📰 Berita":"Pemerintah hari ini mengumumkan langkah baru yang diharapkan dapat meningkatkan pelayanan publik dan memberikan manfaat bagi masyarakat.",
 "🎧 Podcast":"Halo semuanya, di episode kali ini kita akan ngobrol santai tentang pengalaman, pelajaran, dan hal-hal menarik yang sering kita temui sehari-hari.",
 "🎬 Dokumenter":"Di balik hiruk-pikuk kota, terdapat kisah tentang orang-orang yang bekerja dalam diam untuk menjaga kehidupan tetap berjalan.",
 "📢 Promosi":"Temukan pengalaman baru yang lebih praktis, lebih nyaman, dan dirancang untuk membantu aktivitas Anda setiap hari.",
 "🌙 Misterius":"Tidak ada yang tahu siapa yang meninggalkan pesan itu, tetapi sejak malam tersebut, sesuatu terasa berbeda.",
 "💡 Inspiratif":"Setiap langkah kecil tetap berarti. Terus bergerak, belajar dari proses, dan jangan berhenti percaya bahwa perubahan bisa dimulai dari hari ini.",
}

PERSON_PRESETS={
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
    return PERSON_PRESETS.get(name, ("Auto","Auto","Natural","Indonesian",""))

def random_clone_seed():
    return random_seed()

def reset_design_controls():
    return "Auto","Auto","Natural","Indonesian","Natural", "", DEFAULT_DESIGN_CFG, DEFAULT_DESIGN_STEPS, DEFAULT_SEED

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

def generate_dispatch(mode,target_text,reference_audio,reference_transcript,language,seed,design_seed,design_language,output_speed,pitch_semitones,gender,age,timbre,accent,style,custom_instruction,design_cfg,design_steps):
    started=time.time()
    if mode=="cloning":
        audio_result,master_path=generate_voice(target_text,reference_transcript,reference_audio,language,seed,output_speed,pitch_semitones)
        sr,audio=audio_result
        duration=audio.shape[-1]/float(sr)
        files=[str(p) for p in [Path(master_path),Path(master_path).with_name(Path(master_path).name.replace("_HQ_FLOAT32.wav","_PCM16.wav"))] if p.is_file()]
        elapsed=time.time()-started
        summary=(f"**{duration:.1f} detik** · {int(sr):,} Hz · Voice Cloning · "
                 f"{elapsed:.1f}s · seed {int(seed)} · speed {float(output_speed):.2f}x · pitch {float(pitch_semitones):+.1f}st")
        return audio_result,summary,files
    audio_result,master_path,voice_plan,instruction=generate_voice_design(target_text,design_language,gender,age,timbre,accent,style,custom_instruction,design_seed,design_cfg,design_steps)
    sr,audio=audio_result
    duration=audio.shape[-1]/float(sr)
    files=[str(p) for p in [Path(master_path),Path(master_path).with_name(Path(master_path).name.replace("_HQ_FLOAT32.wav","_PCM16.wav"))] if p.is_file()]
    elapsed=time.time()-started
    summary=(f"**{duration:.1f} detik** · {int(sr):,} Hz · Voice Design · "
             f"{elapsed:.1f}s · gaya **{style}** · language **{design_language}** · gender **{gender}** · age **{age}** · "
             f"timbre **{timbre}** · accent **{accent}** · seed {int(design_seed)} · CFG {float(design_cfg):.2f} · steps {int(design_steps)}\n\n"
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
        with gr.Column(scale=0, min_width=190, elem_classes=["nav-col"]):
            gr.HTML("""<div class='brand'><div class='brand-mark'>C</div><div><div class='brand-name'>Cangkeman</div><div class='brand-tag'>AI Voice Studio</div></div></div>""")
            gr.Markdown("TOOLS",elem_classes=["nav-caption"])
            mode_nav=gr.Radio(
                choices=["🎙  Voice Cloning","🎨  Voice Design"],
                value="🎙  Voice Cloning",
                label=None,show_label=False,container=False,elem_classes=["mode-nav"],
            )
            gr.Markdown("",elem_classes=["nav-spacer"])
            gr.Markdown("FireRedTTS3 · T4",elem_classes=["nav-foot"])

        with gr.Column(scale=1,elem_classes=["workspace-col"]):
            gr.Markdown("Workspace",elem_classes=["eyebrow"])
            target_text=gr.Textbox(label="",lines=9,max_lines=14,placeholder="Tulis teks yang ingin diucapkan…",elem_classes=["main-text"])
            gr.Markdown("**Preset Teks**",elem_classes=["preset-heading"])
            with gr.Row(elem_classes=["preset-grid"]):
                narration_buttons=[]
                for _preset_name in NARRATION_PRESETS:
                    narration_buttons.append(gr.Button(_preset_name,elem_classes=["narration-preset-btn"]))
            char_counter=gr.Markdown(f"Maksimal {MAX_TARGET_CHARS} karakter.",elem_classes=["field-hint","char-counter"])
            generate_btn=gr.Button("Generate Speech",variant="primary",elem_classes=["generate-btn"])
            gr.Markdown("Output",elem_classes=["eyebrow","output-heading"])
            generated_audio=gr.Audio(label="",autoplay=False,elem_classes=["output-audio"])
            generation_summary=gr.Markdown("Belum ada hasil — isi teks lalu Generate Speech.",elem_classes=["summary-box"])
            output_files=gr.File(label="Export",file_count="multiple",elem_classes=["file-output"])

        with gr.Column(scale=0,min_width=315,elem_classes=["settings-col"]):
            gr.Markdown("Settings",elem_classes=["eyebrow"])

            with gr.Column(visible=True,elem_classes=["settings-panel"]) as clone_panel:
                gr.Markdown("### Voice Cloning")
                reference_audio=gr.Audio(sources=["upload","microphone"],type="filepath",label=f"Reference Audio · {MIN_PROMPT_SECONDS:.0f}–{MAX_PROMPT_SECONDS:.0f} detik")
                reference_transcript=gr.Textbox(label="Reference Transcript",lines=4,placeholder="Tulis persis ucapan pada audio referensi…")
                ref_status=gr.Markdown("Upload audio referensi untuk mulai cloning.",elem_classes=["field-hint"])
                language=gr.Dropdown(choices=LANGUAGE_CHOICES,value="Indonesian",label="Language")
                with gr.Accordion("Fine Control",open=True):
                    with gr.Row():
                        output_speed=gr.Slider(minimum=MIN_OUTPUT_SPEED,maximum=MAX_OUTPUT_SPEED,value=1.0,step=0.01,label="Speed")
                        pitch_semitones=gr.Slider(minimum=MIN_PITCH_SEMITONES,maximum=MAX_PITCH_SEMITONES,value=0.0,step=0.5,label="Pitch")
                with gr.Accordion("Presets",open=False):
                    with gr.Row():
                        preset_nat=gr.Button("Alami",elem_classes=["small-btn"])
                        preset_deep=gr.Button("Tenang",elem_classes=["small-btn"])
                    with gr.Row():
                        preset_fast=gr.Button("Ceria",elem_classes=["small-btn"])
                        preset_news=gr.Button("Berita",elem_classes=["small-btn"])
                with gr.Accordion("Advanced",open=False):
                    with gr.Row():
                        seed=gr.Number(value=DEFAULT_SEED,precision=0,label="Seed")
                        random_seed_btn=gr.Button("🎲",elem_classes=["icon-btn"],variant="secondary")
                    reset_btn=gr.Button("Reset controls",elem_classes=["text-btn"])

            with gr.Column(visible=False,elem_classes=["settings-panel"]) as design_panel:
                gr.Markdown("### Voice Design")
                with gr.Accordion("Person presets",open=True):
                    with gr.Row(elem_classes=["preset-grid"]):
                        person_buttons=[]
                        for _person_name in PERSON_PRESETS:
                            person_buttons.append(gr.Button(_person_name,elem_classes=["person-preset-btn"]))
                gender=gr.Dropdown(choices=["Auto","Male","Female"],value="Auto",label="Gender")
                age=gr.Dropdown(choices=["Auto","Young Adult","Adult","Mature","Senior"],value="Auto",label="Age")
                timbre=gr.Dropdown(choices=["Natural","Warm","Bright","Deep","Soft","Crisp","Breathy","Rich"],value="Natural",label="Timbre")
                design_language=gr.Dropdown(choices=LANGUAGE_CHOICES,value="Indonesian",label="Language")
                accent=gr.Dropdown(choices=["Auto","Indonesian","American English","British English","International English"],value="Indonesian",label="Accent")
                style=gr.Dropdown(choices=STYLE_CHOICES,value="Natural",label="Gaya Suara")
                custom_instruction=gr.Textbox(label="Custom Style",lines=3,max_lines=6,placeholder="Contoh: warm, relaxed, seperti host podcast dengan jeda alami…")
                with gr.Accordion("Advanced",open=False):
                    design_cfg=gr.Slider(minimum=MIN_DESIGN_CFG,maximum=MAX_DESIGN_CFG,value=DEFAULT_DESIGN_CFG,step=0.1,label="CFG")
                    design_steps=gr.Slider(minimum=MIN_DESIGN_STEPS,maximum=MAX_DESIGN_STEPS,value=DEFAULT_DESIGN_STEPS,step=1,label="Steps")
                    with gr.Row():
                        design_seed=gr.Number(value=DEFAULT_SEED,precision=0,label="Seed")
                        design_random_seed_btn=gr.Button("🎲",elem_classes=["icon-btn"],variant="secondary")
                    design_reset_btn=gr.Button("Reset Voice Design",elem_classes=["text-btn"])
                gr.Markdown("Voice Design membuat suara baru tanpa reference audio. Accent adalah preferensi model, bukan jaminan akustik.",elem_classes=["field-hint"])

    def _nav_changed(label):
        mode = "cloning" if (label or "").startswith("🎙") else "design"
        return (mode, *select_mode(mode))

    mode_nav.change(_nav_changed,inputs=[mode_nav],outputs=[mode_state,clone_panel,design_panel,active_title,active_subtitle])
    preset_nat.click(lambda:set_clone_preset("Alami"),outputs=[output_speed,pitch_semitones])
    preset_deep.click(lambda:set_clone_preset("Dalam & Tenang"),outputs=[output_speed,pitch_semitones])
    preset_fast.click(lambda:set_clone_preset("Ceria"),outputs=[output_speed,pitch_semitones])
    preset_news.click(lambda:set_clone_preset("Berita"),outputs=[output_speed,pitch_semitones])
    reset_btn.click(reset_clone_controls,inputs=[],outputs=[output_speed,pitch_semitones])
    random_seed_btn.click(random_clone_seed,inputs=[],outputs=[seed])
    design_random_seed_btn.click(random_clone_seed,inputs=[],outputs=[design_seed])
    design_reset_btn.click(reset_design_controls,inputs=[],outputs=[gender,age,timbre,accent,style,custom_instruction,design_cfg,design_steps,design_seed])
    reference_audio.change(describe_reference_audio,inputs=[reference_audio],outputs=[ref_status])
    target_text.change(describe_char_count,inputs=[target_text],outputs=[char_counter])
    for _btn, _preset_name in zip(narration_buttons, NARRATION_PRESETS):
        _btn.click(lambda name=_preset_name: apply_narration_preset(name), inputs=[], outputs=[target_text])
    for _btn, _person_name in zip(person_buttons, PERSON_PRESETS):
        _btn.click(lambda name=_person_name: apply_person_preset(name), inputs=[], outputs=[gender,age,timbre,accent,custom_instruction])
    generate_btn.click(
        generate_dispatch,
        inputs=[mode_state,target_text,reference_audio,reference_transcript,language,seed,design_seed,design_language,output_speed,pitch_semitones,gender,age,timbre,accent,style,custom_instruction,design_cfg,design_steps],
        outputs=[generated_audio,generation_summary,output_files],
        show_progress="minimal",
    )

import threading
threading.Thread(target=backend.preload_model, name="fireredtts3-base-preload", daemon=True).start()

demo.queue(max_size=2,default_concurrency_limit=1)
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

print("[MODEL] Base preload scheduled in background; Instruct remains lazy until Voice Design.", flush=True)

while True:
    time.sleep(3600)
