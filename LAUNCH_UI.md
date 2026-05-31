# How to actually launch the LTX-2 UI

## ❌ Why "127.0.0.1 refused to connect"

Nothing is listening yet. The UI isn't running because:

1. The Python env (`.venv/`) hasn't been created.
2. **You're on macOS Apple Silicon (Darwin arm64).** LTX-2 is a 22 B-parameter
   CUDA-only model. The repo's `pyproject.toml` pins
   `https://download.pytorch.org/whl/cu129` — there are no wheels for macOS,
   and even with PyTorch-MPS the model is far too big to run on a Mac. So
   `uv sync --frozen` will fail on your machine.

You have three realistic options:

---

## Option A — Run on a CUDA Linux machine (recommended)

Any RunPod / Vast.ai / Lambda / your own NVIDIA box.

```bash
git clone https://github.com/debzody/LTX-2.git
cd LTX-2
uv sync --frozen
source .venv/bin/activate
uv pip install "gradio>=4.44"

# Download the 4 model artifacts (see README.md) into /models/...

python app.py \
    --checkpoint-path /models/ltx-2.3-22b-distilled-1.1.safetensors \
    --distilled-lora /models/ltx-2.3-22b-distilled-lora-384-1.1.safetensors \
    --spatial-upsampler-path /models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
    --gemma-root /models/gemma-3-12b-it-qat-q4_0-unquantized \
    --host 0.0.0.0 \
    --port 7860 \
    --share          # optional: get a public gradio.live URL
```

If you have <40 GB VRAM, add:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python app.py --quantization fp8-cast ...
```

---

## Option B — Use the hosted LTX-2 playground (no local install)

Lightricks runs an official cloud playground at
<https://console.ltx.video/playground>. Same model, free, no setup.

---

## Option C — Use ComfyUI on a CUDA machine

Lightricks ships ComfyUI nodes:
<https://github.com/Lightricks/ComfyUI-LTXVideo/>. ComfyUI gives you a richer
UI than my minimal Gradio one, and supports the same checkpoints. The
`app.py` here is intended for people who want a one-screen prompt → mp4 flow
without the ComfyUI graph editor.

---

## What this repo's UI _can_ do on your Mac right now

You can still:

- Open `app.py` and `app_pipeline.py` and review/edit them.
- Sanity-check that the Python parses: `python3 -m py_compile app.py app_pipeline.py`.
- Push the code to a CUDA box and run it there.

You **cannot** actually generate a video on Apple Silicon — that part needs an
NVIDIA GPU. The UI itself is plain Gradio + Python and is GPU-agnostic; it's
the underlying LTX-2 pipeline that requires CUDA.
