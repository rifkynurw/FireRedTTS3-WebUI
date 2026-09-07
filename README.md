# FireRedTTS3 AI Voice Studio — Dokumentasi Proyek

## 1. Ringkasan Proyek

Proyek ini adalah WebUI berbasis Gradio untuk FireRedTTS3 yang ditujukan sebagai **AI Voice Studio** di Google Colab dengan GPU NVIDIA T4.

Aplikasi mempunyai dua mode utama:

- **Voice Cloning** — menghasilkan speech dari audio referensi dan transcript.
- **Voice Design** — membuat karakter suara baru dari instruksi/atribut seperti gender, usia, timbre, accent, style, dan teks target.

Target utama penggunaan proyek saat ini adalah **Bahasa Indonesia**, dengan fokus pada suara natural untuk narasi, podcast, berita, dokumenter, promosi, storytelling, dan kebutuhan voice-over lainnya.

---

## 2. Struktur Arsitektur

Arsitektur aplikasi terbagi menjadi beberapa bagian:

```text
Google Colab / Runtime
        │
        ├── Environment & model cache
        │
        ├── FireRedTTS3 Base
        │
        ├── FireRedTTS3-Instruct
        │
        ├── RedAE
        │
        ├── Runtime backend yang dibentuk dari notebook
        │
        └── Gradio WebUI
                  │
                  ├── Voice Cloning
                  │       ├── Reference Audio
                  │       ├── Reference Transcript
                  │       ├── Language
                  │       ├── Speed
                  │       └── Pitch
                  │
                  └── Voice Design
                          ├── Gender
                          ├── Age
                          ├── Timbre
                          ├── Language
                          ├── Accent
                          ├── Style
                          ├── Custom Style
                          ├── CFG
                          └── Steps
```

`app.py` bertindak sebagai layer WebUI dan dispatcher. Fungsi backend tidak disimpan sebagai `backend.py` terpisah di repository asli, tetapi dibangun/materialisasi dari source yang terdapat di notebook.

---

## 3. Komponen WebUI

WebUI menggunakan Gradio.

Di `app.py`, parameter Voice Design diteruskan melalui fungsi:

```python
generate_dispatch(
    mode,
    target_text,
    reference_audio,
    reference_transcript,
    language,
    seed,
    design_seed,
    design_language,
    output_speed,
    pitch_semitones,
    gender,
    age,
    timbre,
    accent,
    style,
    custom_instruction,
    design_cfg,
    design_steps
)
```

Untuk mode Voice Design, parameter tersebut diteruskan ke:

```python
generate_voice_design(
    target_text,
    design_language,
    gender,
    age,
    timbre,
    accent,
    style,
    custom_instruction,
    design_seed,
    design_cfg,
    design_steps
)
```

Dengan demikian, dropdown `Gender`, `Age`, `Timbre`, `Language`, dan `Accent` memang masuk ke jalur Voice Design.

---

## 4. Language dan Accent

Language Voice Design menggunakan pilihan bahasa yang berasal dari `SUPPORTED_LANGUAGES`.

Bahasa Indonesia tersedia sebagai:

```text
Indonesian
```

Dropdown Accent menyediakan:

```text
Auto
Indonesian
American English
British English
International English
```

Pada awal proyek, default Accent adalah `Auto`. Kemudian di beberapa patch default diarahkan ke `Indonesian` untuk mengurangi kemungkinan fallback ke aksen asing.

Penting:

**Language dan Accent bukan hal yang sama.**

`language = Indonesian` berarti bahasa target adalah Indonesia.

`accent = Indonesian` berarti instruksi Voice Design meminta karakteristik aksen Indonesia.

Keduanya sudah berhasil dikirim dari UI.

---

## 5. Masalah Utama: Voice Design Bahasa Indonesia Tetap Terdengar Asing

Masalah terpenting proyek adalah:

> Voice Design menghasilkan audio berbahasa Indonesia tetapi aksennya terdengar seperti penutur asing, terutama Inggris/Amerika, dan pada beberapa percobaan terdengar seperti bahasa/aksen Asia Timur.

Ini terjadi meskipun:

```text
Language = Indonesian
Accent = Indonesian
```

dan instruksi juga sudah diperkuat agar meminta:

```text
native Indonesian speaker
natural Indonesian pronunciation
Indonesian rhythm
Indonesian intonation
no English accent
```

---

## 6. Temuan Penting dari Voice Plan

Pada salah satu hasil awal, Voice Plan menjadi:

```text
[gender] male
[age] middle_age
[pitch] medium_low
[texture] smooth
[volume] normal
[accent] american
[emotion] neutral
[fluency] fluent
[speed] moderate
[clarity] clear
[tone] questioning
[personality] thoughtful
```

Hal tersebut awalnya mengindikasikan kemungkinan bahwa nilai accent dari UI tertimpa oleh backend/planner.

Setelah backend dan prompt diperiksa lebih jauh, ditemukan bahwa `app.py` memang meneruskan nilai dropdown ke fungsi Voice Design.

Tidak ditemukan mapping sederhana seperti:

```text
Indonesian → American
```

di layer WebUI.

Dengan kata lain, `american` pada Voice Plan dapat berasal dari proses planning model, bukan dari dropdown Gradio.

---

## 7. Kasus Voice Plan Berbahasa Mandarin

Pada percobaan selanjutnya muncul Voice Plan:

```text
[性别] 男
[年龄] 中年
[音高] 中高
[音色质感] 清亮
[音量] 正常
[口音] 非中文
[情绪] 客观
[流畅度] 流畅
[语速] 快
[清晰度] 清晰
[语调] 陈述
[性格] 审慎严谨
```

Ini menunjukkan bahwa planner tidak selalu mengeluarkan structured plan dalam bahasa Inggris.

Hal ini penting karena validator sebelumnya hanya mencari bentuk:

```text
[gender]
[age]
[accent]
```

sehingga Voice Plan Mandarin tidak terdeteksi sebagai mismatch.

Validator kemudian dikembangkan agar dapat menangani variasi output planner.

---

## 8. Kasus Gender Tidak Sesuai

Pengguna memilih:

```text
Gender = Female
```

tetapi audio yang dihasilkan terdengar seperti male.

Pada saat yang sama, Voice Plan dapat menunjukkan:

```text
[gender] male
```

atau setelah perbaikan:

```text
[gender] female
```

Temuan ini menghasilkan dua kesimpulan:

1. Parameter UI dapat diterima oleh planner.
2. Voice Plan yang benar belum tentu menjamin karakter akustik yang benar-benar sesuai.

Hal ini menjadi bukti bahwa structured voice plan bukan kontrol deterministik penuh terhadap audio.

---

## 9. Kasus Voice Plan Sudah Benar Tetapi Audio Masih Salah

Ini adalah temuan paling penting pada tahap akhir.

Contoh hasil terbaru:

```text
language = Indonesian
gender = Female
age = Young Adult
timbre = Natural
accent = Indonesian
```

Voice Plan:

```text
[gender] female
[age] young_adult
[pitch] high
[texture] bright
[volume] normal
[accent] indonesian
[emotion] neutral
[fluency] fluent
[speed] moderate
[clarity] clear
[tone] declarative
[personality] methodical
```

Namun hasil audio **masih terdengar beraksen asing/Inggris**.

Ini sangat penting karena menunjukkan bahwa masalah tidak lagi dapat dijelaskan sebagai:

```text
dropdown salah
```

atau:

```text
accent tidak masuk ke planner
```

Planner sudah memahami parameter yang diminta.

Masalahnya berada pada tahap:

```text
Voice plan → acoustic synthesis
```

yaitu bagaimana model menerjemahkan structured voice plan menjadi karakter suara yang terdengar.

---

## 10. Kesimpulan tentang Accent Control

Kesimpulan proyek saat ini:

> `accent = Indonesian` dapat berhasil muncul pada Voice Plan, tetapi belum dapat dianggap sebagai kontrol deterministik yang menjamin audio akan terdengar seperti penutur Indonesia.

Karena itu penambahan kata-kata seperti:

```text
native Indonesian
Indonesian accent
Indonesian pronunciation
no English accent
```

dapat membantu planner, tetapi tidak menjamin hasil akustik.

Ini juga berarti tidak efektif terus melakukan iterasi berupa:

```text
Fix28.16
Fix28.17
Fix28.18
...
```

hanya dengan menambahkan kata "Indonesian" ke prompt.

---

## 11. Pendekatan Person / Character

Salah satu pendekatan yang sempat diuji adalah mengubah Voice Design menjadi konsep **Person / Character**.

Contoh konsep:

```text
Perempuan Indonesia Muda
Laki-laki Indonesia Muda
Pria Indonesia Dewasa
Wanita Indonesia Dewasa
Presenter Indonesia
Podcaster Indonesia
Penyiar Indonesia
Pendongeng Indonesia
```

Tujuannya adalah memberi model konteks speaker yang jauh lebih konkret daripada hanya:

```text
accent = Indonesian
```

Contoh deskripsi:

```text
Perempuan Indonesia dewasa muda, penutur asli Bahasa Indonesia.
Tinggal dan berbicara sehari-hari di Indonesia.
Suara percakapan natural seperti presenter atau podcaster Indonesia.
Pelafalan Bahasa Indonesia sangat natural, dengan ritme, intonasi,
tekanan kata, dan jeda khas penutur Indonesia.
Bukan penutur bahasa Inggris yang sedang membaca Bahasa Indonesia.
Hindari aksen English, Chinese, Mandarin, Cantonese, Japanese,
atau aksen asing lainnya.
```

Pendekatan ini secara konseptual lebih masuk akal untuk Voice Design, tetapi pada pengujian terakhir aksen asing masih muncul.

Dengan demikian, Person preset dianggap sebagai **eksperimen yang layak**, bukan solusi yang sudah terbukti.

---

## 12. Preset Teks vs Preset Person

Terjadi miskonsepsi pada implementasi preset sebelumnya.

### Preset Person

Preset Person dimaksudkan untuk mengubah karakter Voice Design:

```text
Person
 ├── Gender
 ├── Age
 ├── Timbre
 ├── Accent
 └── Character description
```

Contoh:

```text
👩 Indonesia Muda
👩 Indonesia Dewasa
🎙 Presenter Indonesia
🎧 Podcaster Indonesia
📻 Penyiar Indonesia
📖 Pendongeng Indonesia
```

### Preset Teks

Preset Teks dimaksudkan untuk mengisi **textbox yang akan dibacakan**.

Contoh:

```text
Narasi Natural
Storytelling
Berita
Podcast
Dokumenter
Promosi
Misterius
Inspiratif
```

Preset Teks tidak seharusnya mengubah karakter suara.

---

## 13. Masalah Syntax Notebook

Beberapa iterasi mengalami kerusakan notebook karena source backend dimasukkan ke variabel Python seperti:

```python
BACKEND_SOURCE = '...kode backend...'
```

sementara source backend tersebut mengandung banyak quote seperti:

```python
"Indonesian"
"fireredtts3_instruct"
"float32"
```

Akibatnya terjadi:

```text
SyntaxError: unterminated string literal
```

Ini bukan error FireRedTTS3 dan bukan error GPU.

Penyelesaiannya adalah menyimpan backend source sebagai Python string literal yang aman, bukan melakukan quote manual dengan single quote biasa.

---

## 14. Masalah Validator

Ada juga validator yang pernah mencari string:

```text
direct Instruct synthesis
```

sebagai invariant.

String tersebut tidak merupakan bagian wajib dari backend aktual sehingga validator menghasilkan:

```text
RuntimeError:
STOP: required backend invariant missing:
direct Instruct synthesis
```

Validator kemudian diperbaiki agar memeriksa invariant yang benar, misalnya:

```text
def get_tts():
def get_instruct():
def build_design_instruction
def generate_voice_design
model.generate_voice_design(
RedAE.from_pretrained=_shared_redae
```

Validator seharusnya memeriksa **struktur dan API yang benar-benar digunakan**, bukan komentar/prose marker yang kebetulan pernah dipakai dalam patch.

---

## 15. Arsitektur Model di T4

Notebook menggunakan pendekatan shared RedAE dan hanya menjaga salah satu core model aktif di GPU pada satu waktu.

Konsepnya:

```text
Base aktif
    ↓
Instruct dilepas jika perlu

atau

Instruct aktif
    ↓
Base dilepas jika perlu
```

Tujuannya menghemat VRAM pada NVIDIA T4 16 GB.

Notebook juga memiliki lazy staging untuk model Instruct sehingga asset Instruct dapat disalin dari persistent model cache bila diperlukan.

---

## 16. Precision

Beberapa eksperimen sebelumnya mencoba optimasi FP16/autocast untuk Instruct.

Eksperimen tersebut kemudian dianggap tidak stabil dan tidak dijadikan jalur utama.

Kontrak yang dipertahankan pada patch stabil:

```text
FireRedTTS3-Instruct
↓
official FP32 load path
↓
tidak menggunakan selective .half()
↓
tidak menggunakan experimental autocast boundary hooks
```

Validasi notebook juga pernah ditambahkan untuk memastikan patch precision eksperimental lama tidak kembali masuk.

---

## 17. Parameter Generasi yang Digunakan

Baseline yang digunakan dalam beberapa pengujian:

```text
Output sample rate : 24,000 Hz
CFG                 : 1.20
Steps               : 10
Seed default        : 1986
```

Salah satu log terakhir:

```text
8.2 detik
24,000 Hz
Voice Design
gaya Natural
language Indonesian
gender Female
age Young Adult
timbre Natural
accent Indonesian
seed 1986
CFG 1.20
steps 10
```

Voice Plan yang dihasilkan:

```text
[gender] female
[age] young_adult
[pitch] high
[texture] bright
[volume] normal
[accent] indonesian
[emotion] neutral
[fluency] fluent
[speed] moderate
[clarity] clear
[tone] declarative
[personality] methodical
```

Masalah aksen masih ada meskipun parameter di atas terlihat benar.

---

## 18. Status Terakhir Proyek

Status terakhir:

### Berhasil

- WebUI Gradio berjalan.
- Voice Cloning tersedia.
- Voice Design tersedia.
- Language Indonesian tersedia.
- Accent Indonesian tersedia.
- Gender/Age/Timbre tersedia.
- Voice Plan dapat ditampilkan.
- Model Base dan Instruct dapat dikelola pada T4.
- Lazy loading/cache Instruct tersedia.
- Preset Person dan Preset Teks sudah dibedakan secara konsep.
- Beberapa masalah syntax notebook sudah diperbaiki.

### Belum terselesaikan

**Voice Design tanpa reference audio belum konsisten menghasilkan aksen Indonesia yang natural.**

Kasus paling kuat adalah:

```text
requested:
accent = Indonesian

planner:
accent = Indonesian

result:
audio masih terdengar asing/Inggris
```

Jadi bug utama bukan lagi UI.

---

## 19. Kesimpulan Teknis

Masalah utama proyek dapat digambarkan sebagai:

```text
UI
 ↓
Parameter benar
 ↓
Instruction benar
 ↓
Voice plan benar
 ↓
Audio masih tidak sesuai aksen
```

Dengan kata lain:

```text
parameter control ≠ guaranteed acoustic result
```

Untuk kebutuhan aksen Indonesia yang benar-benar konsisten, **Voice Cloning dengan reference speaker Indonesia** tetap menjadi jalur yang lebih dapat diprediksi.

Voice Design tetap berguna untuk:

- membuat karakter suara baru;
- mengatur gender;
- usia;
- timbre;
- emosi;
- kecepatan;
- personality;
- style.

Tetapi proyek belum membuktikan bahwa Voice Design dapat memberikan **aksen Indonesia akustik yang konsisten tanpa reference audio**.

---

## 20. Arah Pengembangan Berikutnya

Arah pengembangan sebaiknya tidak lagi fokus pada menambah variasi prompt "Indonesian".

Prioritas yang lebih masuk akal:

### A. Pertahankan Voice Design sebagai Character Generator

Gunakan Person preset untuk membangun identitas speaker:

```text
Person
+
Style
+
Language
+
Text
```

Tetapi dokumentasikan bahwa accent adalah preference, bukan guarantee.

### B. Perkuat Voice Cloning untuk Indonesian

Sediakan UX yang jelas untuk:

```text
Reference Audio Indonesia
Reference Transcript
Language = Indonesian
```

Gunakan reference speaker yang benar-benar native Indonesian.

### C. Pisahkan UX dengan jelas

```text
Voice Design
    Person
    Voice attributes
    Style

Voice Cloning
    Reference speaker
    Target language
    Speed
    Pitch
```

### D. Jangan menyembunyikan mismatch

Jika:

```text
requested accent = Indonesian
```

tetapi planner atau hasil internal mengindikasikan mismatch, tampilkan status seperti:

```text
Requested: Indonesian accent
Planner: Indonesian
Acoustic accent: not guaranteed
```

Jangan menyatakan "aksen Indonesia berhasil" hanya berdasarkan `[accent] indonesian`.

---

## 21. File dan Versi Penting

File proyek yang telah digunakan dalam pengembangan:

```text
app.py

FireRedTTS3_Base_Instruct_Colab_T4_GDrive_AI_Voice_Studio_Fix28_10.ipynb
FireRedTTS3_Base_Instruct_AI_Voice_Studio_Fix28_11.ipynb
FireRedTTS3_Base_Instruct_AI_Voice_Studio_Fix28_12.ipynb
FireRedTTS3_Base_Instruct_AI_Voice_Studio_Fix28_13.ipynb
FireRedTTS3_Base_Instruct_AI_Voice_Studio_Fix28_14.ipynb
FireRedTTS3_Base_Instruct_AI_Voice_Studio_Fix28_15.ipynb
FireRedTTS3_Base_Instruct_AI_Voice_Studio_Fix28_16.ipynb
```

Versi `Fix28.16` adalah iterasi yang menggabungkan konsep:

```text
Person presets
+
Text presets
```

tetapi masalah aksen Indonesia tetap belum terselesaikan secara deterministik.

---

## 22. Catatan untuk Pengembangan Selanjutnya

Jangan menganggap:

```text
Voice Plan = acoustic truth
```

Voice Plan hanya menunjukkan apa yang planner model tulis sebagai atribut suara.

Pengujian kualitas harus tetap dilakukan pada audio.

Untuk setiap eksperimen sebaiknya dicatat:

```text
Seed
CFG
Steps
Gender
Age
Timbre
Language
Accent
Person preset
Text preset
Voice Plan
Persepsi aksen audio
```

Contoh record pengujian:

```text
Seed: 1986
Gender: Female
Age: Young Adult
Timbre: Natural
Language: Indonesian
Accent: Indonesian
CFG: 1.20
Steps: 10

Planner:
accent = indonesian

Audio:
masih terdengar asing
```

Catatan seperti ini lebih berguna daripada hanya mencatat apakah Generate berhasil atau gagal.

---

## 23. Kesimpulan Akhir

Proyek **FireRedTTS3 AI Voice Studio** secara teknis sudah memiliki pipeline Voice Cloning dan Voice Design yang berfungsi.

Masalah yang tersisa bukan lagi masalah UI sederhana.

Masalah inti adalah:

> **FireRedTTS3-Instruct Voice Design dapat memahami dan menuliskan atribut `accent=Indonesian` pada Voice Plan, tetapi hasil audio belum konsisten mempertahankan aksen Indonesia yang diharapkan.**

Karena itu:

- jangan lagi menganggap dropdown Accent rusak;
- jangan lagi menganggap `american` selalu berasal dari hard-code UI;
- jangan menganggap Voice Plan yang benar berarti audio sudah benar;
- jangan terus menambah prompt Indonesia tanpa evaluasi audio;
- gunakan Voice Design terutama sebagai character/voice generator;
- gunakan Voice Cloning dengan reference Indonesia ketika aksen Indonesia merupakan requirement utama.

Dokumen ini dimaksudkan sebagai **baseline teknis/handoff** agar pengembangan berikutnya tidak mengulang eksperimen dan diagnosis yang sama.


## Patch28.16a — Performance, WebUI, dan Reliability

### Perubahan utama

1. **Startup lebih cepat**
   - Base tetap dipreload di background.
   - Instruct tidak lagi di-download/stage pada startup normal.
   - Instruct baru disiapkan saat Voice Design pertama kali dipakai.

2. **Voice generation lebih cepat**
   - Voice Design retry dibatasi ke `FIRERED_MAX_DESIGN_ATTEMPTS=2`.
   - Retry hanya dipakai bila voice plan melanggar hard setting.
   - `torch.cuda.empty_cache()` tidak lagi dipanggil setelah setiap request sukses. Cache GPU dibersihkan ketika model benar-benar berpindah.

3. **WebUI lebih rapi**
   - Preset teks dan preset person dipisahkan dengan jelas.
   - Advanced settings memakai accordion.
   - Random seed button tersedia untuk Cloning dan Design.
   - Generate summary menampilkan elapsed time.
   - Queue diturunkan menjadi `max_size=2` dengan concurrency `1` agar T4 tidak menerima beban paralel.

4. **Reliability**
   - Notebook sekarang memmaterialisasi source dengan raw triple-quoted string agar quote di source backend tidak mudah merusak notebook.
   - Validator memeriksa kontrak lazy Instruct dan batas retry.
   - Official Instruct FP32 tetap dipertahankan; tidak ada selective `.half()` atau autocast eksperimental.

### Konfigurasi baru

```text
FIRERED_MAX_DESIGN_ATTEMPTS=2
```

Nilai ini dapat dinaikkan untuk eksperimen terkontrol, tetapi nilai `2` adalah baseline untuk menjaga waktu generate tetap masuk akal.

### Dampak arsitektur

```text
Notebook start
   ↓
GPU / environment / Base assets
   ↓
Base preload (background)
   ↓
WebUI ready

Voice Cloning
   ↓
re-use Base

Voice Design (first use)
   ↓
release Base
   ↓
stage/load Instruct
   ↓
reuse Instruct for subsequent Design requests
```

Tidak ada perubahan terhadap prinsip bahwa Voice Plan bukan jaminan acoustic accent. Untuk aksen Indonesia yang konsisten, Voice Cloning dengan reference speaker Indonesia tetap menjadi jalur yang lebih dapat diprediksi.

## Patch28.16b — WebUI materialization regression fix

### Root cause
The Fix28.16a notebook used `STANDARD_WEBUI_FILE.write_text(STANDARD_WEBUI_SOURCE, ...)` while `STANDARD_WEBUI_SOURCE` had not been explicitly assigned. This caused:

```text
NameError: name 'STANDARD_WEBUI_SOURCE' is not defined
```

### Fix
The canonical WebUI source is now defined explicitly before theme/materialization:

```python
STANDARD_WEBUI_SOURCE = APP_PATCH_SOURCE
```

The materialization step also contains a defensive guard:

```python
if not STANDARD_WEBUI_SOURCE:
    raise RuntimeError("STOP: STANDARD_WEBUI_SOURCE is empty; WebUI source materialization is unsafe.")
```

The validation cell now fails early with a clear diagnostic if the canonical WebUI source is missing from notebook state.

### Validation
- Notebook JSON validation: PASS
- All code cells AST parse: PASS
- Materialization cell executed with isolated temporary paths: PASS
- Generated `backend.py` syntax: PASS
- Generated `standard_ui.py` syntax: PASS


## Patch28.16c — Generate UX, Output Loading, History, dan Mode Sync

Patch ini menyempurnakan WebUI tanpa mengubah kontrak model FireRedTTS3 yang sudah stabil.

### 1. Generate button dikunci selama proses

Saat user menekan Generate Speech:

```text
Generate Speech
      ↓
⏳ Preparing…
      ↓
🔊 Generating…
      ↓
Generate Speech
```

Button menjadi `interactive=False` selama request berjalan. Navigasi Voice Cloning / Voice Design di sidebar kiri juga dinonaktifkan selama request agar `mode_state` tidak berubah di tengah generation.

### 2. Output audio berubah menjadi loading

Audio hasil sebelumnya disembunyikan segera setelah generation dimulai dan digantikan loading card dengan spinner. Setelah request selesai, audio baru kembali ditampilkan.

Struktur output:

```text
Output
 ├── Loading / Audio
 ├── Output metadata + Voice Plan
 ├── Generated Text / Prompt (copyable)
 └── Export
```

Metadata dan generated text tetap berada di bawah stage audio/loading sehingga konteks request sebelumnya tidak hilang selama proses berjalan.

### 3. History di sidebar kiri

Sidebar sekarang menyimpan maksimal `HISTORY_LIMIT = 6` hasil terbaru dalam sesi UI. Setiap entry berisi:

- mode dan waktu generation;
- text yang digenerate;
- textbox dengan `buttons=["copy"]`;
- tombol `Unduh WAV`.

Entry terbaru berada paling atas. History menggunakan `gr.State`, sehingga tidak menambah I/O Drive hanya untuk menyimpan metadata sesi.

### 4. Voice Cloning / Voice Design selalu sinkron

`mode_state` menjadi single source of truth. Radio di sidebar kiri mengubah:

```text
mode_state
clone_panel visibility
design_panel visibility
active mode title
left mode status
right settings status
```

secara bersamaan. Navigasi dikunci selama generation untuk mencegah mismatch antara mode yang diproses dan settings yang terlihat.

### 5. Fix materialisasi notebook

Canonical WebUI source sekarang didefinisikan secara eksplisit:

```python
APP_PATCH_SOURCE = ...
STANDARD_WEBUI_SOURCE = APP_PATCH_SOURCE
```

sebelum `STANDARD_WEBUI_FILE.write_text(...)`. Source backend/UI/CSS juga dibentuk menggunakan `repr(...)` agar quote internal tidak dapat menyebabkan `SyntaxError` pada notebook.

### 6. Validasi Fix28.16c

Validation tambahan memastikan source aktif memiliki:

```text
HISTORY_LIMIT
_begin_generation
_generation_phase
run_generation_ui
output_loading
history_state
gr.DownloadButton
buttons=["copy"]
mode_state
```

serta tetap mempertahankan validasi backend, lazy Base/Instruct switching, official Instruct FP32 path, dan Rubber Band R3 HQ dari patch sebelumnya.


### Fix28.16c compatibility note

Textbox copy actions use the Gradio 6-compatible component API `buttons=["copy"]` rather than the removed/unsupported `show_copy_button` argument. This was validated with a real Gradio 6.x component constructor.
