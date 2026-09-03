# FireRedTTS3-WebUI

This repository is the **source of truth for the FireRedTTS3 Gradio WebUI** used by the Colab notebook.

## Repository layout

```text
FireRedTTS3-WebUI/
├── app.py          # runnable Gradio entry point
├── theme.css       # visual design
├── assets/         # optional visual assets
└── README.md       # this tutorial
```

## How the Colab notebook uses this repository

The notebook fetches the selected Git ref at startup:

```text
Git repository
     ↓
/content/FireRedTTS3-WebUI/
     ↓
app.py
     ↓
Gradio
```

With `WEBUI_REPO_REF="main"`, the newest committed UI on `main` is used.

For a reproducible UI version, set the notebook to a tag or stable branch.

## Design tutorial

### Change the appearance

Edit `theme.css`.

Use it for:

- colors and backgrounds
- typography
- cards and sections
- buttons
- sliders
- badges
- spacing
- responsive rules

`app.py` loads `theme.css` automatically when the file exists beside it.

### Change the layout

Edit `app.py`.

Typical components:

```python
gr.Row()
gr.Column()
gr.Accordion()
gr.Textbox()
gr.Dropdown()
gr.Number()
gr.Slider()
gr.Button()
gr.Audio()
gr.Markdown()
```

You can redesign the layout without changing the FireRedTTS3 model.

### Add assets

Place files under:

```text
assets/
```

and reference them from `app.py` or `theme.css`.

## Backend contract

The notebook supplies these environment variables:

```text
FIRERED_REPO_DIR
FIRERED_MODEL_DIR
FIRERED_OUTPUT_DIR
FIRERED_URL_FILE
FIRERED_MAX_TARGET_CHARS
FIRERED_MIN_PROMPT_SECONDS
FIRERED_MAX_PROMPT_SECONDS
FIRERED_DEFAULT_SEED
FIRERED_GRADIO_PORT
FIRERED_RUBBERBAND_HQ_BIN
FIRERED_RUBBERBAND_LIB_DIR
```

Keep those names and the existing backend behavior intact when doing UI-only work.

Do not reintroduce:

```text
librosa.effects.time_stretch
librosa.effects.pitch_shift
```

Speed/Pitch must continue through the notebook's Rubber Band R3 HQ helper.

## Publish a design update

```bash
git add app.py theme.css assets/ README.md
git commit -m "Update WebUI design"
git push
```

Then run the same notebook again. The notebook will fetch the newest commit on the configured ref.

## Recommended design workflow

1. Make changes in `theme.css` for visual work.
2. Make layout/label changes in `app.py`.
3. Test locally when possible.
4. Commit and push.
5. Run the Colab notebook.
6. Confirm the printed WebUI commit matches the expected repository revision.

## Fallback behavior

When the repository cannot be reached, the notebook uses its bundled **standard Gradio fallback UI**. The TTS model and Rubber Band R3 HQ backend are not replaced.
