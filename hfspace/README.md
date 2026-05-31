---
title: LTX-2 Video Generator
emoji: 🎬
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
license: other
hardware: t4-small
suggested_hardware: t4-small
---

# LTX-2 Video Generator (Hugging Face Space)

This is the Gradio UI from <https://github.com/debzody/LTX-2> packaged for HF Spaces.

> ⚠️ **Requires a paid GPU Space** (T4 small or larger). LTX-2 is a 22B-parameter
> CUDA-only model — it will not run on the free CPU tier.

## Setup

1. **Duplicate this Space** to your account: click the ⋮ menu → "Duplicate this Space".
2. In the duplicated Space, go to **Settings**:
   - **Hardware** → pick **T4 small** ($0.40/h, billed by the second).
   - **Variables and secrets** → **New secret** → name `HF_TOKEN`, value = your HF read token (needed to download the gated Gemma text encoder).
   - **Persistent storage** → ≥ 50 GB recommended (otherwise the 30 GB of weights get re-downloaded on every restart, which is slow and counts as compute time).
3. Accept the Gemma license once at
   <https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized>
   while logged into the same HF account whose token you stored.
4. Save settings — the Space rebuilds automatically.

## How it works

`app.py` is the entry point. On boot it:

1. Downloads the LTX-2 distilled checkpoint and Gemma text encoder
   from Hugging Face into `/data` (persistent storage).
2. Starts the Gradio UI on port 7860 (Spaces default).
3. The "one-stage" pipeline is the default — fits in 16 GB VRAM with FP8.

## Cost

A T4 small Space costs **$0.40/hour**, billed only when the Space is awake.
Spaces auto-sleep after 48h of inactivity (configurable). Generating a 4-second
video at 512×768 takes ~2 minutes of GPU time → ~$0.013 / video.

## Files

This directory contains:

- `app.py` — the Gradio UI (auto-downloads weights on first start).
- `requirements.txt` — pip dependencies for the Space.
- `packages/` — vendored copies of `ltx-core` and `ltx-pipelines` (or installed via pip from the upstream repo).