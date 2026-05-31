# 🎬 LTX-2 Video Generation Web UI

A simple Gradio-based web UI for generating videos with the LTX-2 model. You type a prompt, click a button, and get an MP4.

> The UI lives in two files at the repo root:
>
> - `app.py` — Gradio interface
> - `app_pipeline.py` — model loading + generation logic

## 1. Install

From the repo root:

```bash
# Create the env and install LTX-2 + UI deps
uv sync --frozen
source .venv/bin/activate

# Install Gradio
uv pip install "gradio>=4.44"
```

> CUDA / FlashAttention / xFormers extras are documented in the main `README.md` — install whichever match your GPU.

## 2. Download model weights

You need 4 files (see the main `README.md` for direct download links):

| Purpose | File |
| ------- | ---- |
| Main checkpoint | `ltx-2.3-22b-distilled-1.1.safetensors` (recommended) **or** `ltx-2.3-22b-dev.safetensors` |
| Distilled LoRA | `ltx-2.3-22b-distilled-lora-384-1.1.safetensors` (only needed for the **two-stage** pipeline) |
| Spatial upsampler | `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` |
| Gemma text encoder | All files from [`google/gemma-3-12b-it-qat-q4_0-unquantized`](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized) (download the entire folder) |

Put them anywhere — you'll pass paths on the command line.

## 3. Launch the UI

```bash
python app.py \
    --checkpoint-path /models/ltx-2.3-22b-distilled-1.1.safetensors \
    --distilled-lora /models/ltx-2.3-22b-distilled-lora-384-1.1.safetensors \
    --spatial-upsampler-path /models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
    --gemma-root /models/gemma-3-12b-it-qat-q4_0-unquantized
```

Then open <http://127.0.0.1:7860>.

### Optional flags

| Flag | Description |
| ---- | ----------- |
| `--quantization fp8-cast` | Lower VRAM via FP8 weight casting. Run with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. |
| `--output-dir /path/to/dir` | Where to write generated `.mp4` files (default: `<tmp>/ltx2_outputs`). |
| `--host 0.0.0.0` | Listen on all interfaces (e.g. for remote access). |
| `--port 7861` | Custom port. |
| `--share` | Create a public Gradio share link (tunnel). |

Example with FP8:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python app.py \
    --quantization fp8-cast \
    --checkpoint-path /models/ltx-2.3-22b-distilled-1.1.safetensors \
    --distilled-lora /models/ltx-2.3-22b-distilled-lora-384-1.1.safetensors \
    --spatial-upsampler-path /models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
    --gemma-root /models/gemma-3-12b-it-qat-q4_0-unquantized
```

## 4. Using the UI

1. **Pipeline** — pick `two-stage` (best quality) or `distilled` (fastest).
2. **Prompt** — describe your shot. See the prompting tips in the main `README.md`. Example:
   > *A cinematic shot of a fox running through a snowy forest at dawn, soft golden light filtering through the trees, breath visible in the cold air, camera tracking smoothly alongside, shallow depth of field.*
3. **Negative Prompt** *(optional)* — discouraged content; defaults to `worst quality, low quality, blurry, distorted`.
4. **Conditioning image** *(optional)* — drop in an image to do **image-to-video**; the image is used at frame 0.
5. **Advanced settings** — width/height (multiples of 32), frames (snapped to `8k+1`), FPS, inference steps, seed, CFG scales.
6. Hit **🎬 Generate Video**.

The output MP4 is shown in the player and saved to disk; the path is printed below the player.

## 5. Hardware requirements

LTX-2 is a **22B** parameter model. You'll want:

- **GPU**: an NVIDIA datacenter GPU (H100/H200/B200) for full bf16. On consumer GPUs (e.g. RTX 4090 24 GB), use `--quantization fp8-cast`.
- **Disk**: ~50–80 GB for the model weights.
- **RAM**: 64 GB+ recommended.

The first generation will take a few minutes to load weights; subsequent generations only run inference.

## 6. Troubleshooting

- **`ModuleNotFoundError: ltx_pipelines`** — run from the repo root with the env active (`source .venv/bin/activate`).
- **OOM on a 24 GB card** — add `--quantization fp8-cast` and the `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` env var; try the `distilled` pipeline; reduce resolution and frames.
- **`assert_resolution`-style errors** — the UI snaps width/height to multiples of 32 and frames to `8k+1`, but extreme values can still fail; try the defaults first.
- **Blank video / silence** — try a higher `Video CFG` (e.g. 4.0–5.0) and a more detailed prompt.

## 7. Extending

`app_pipeline.py` only wires up `TI2VidTwoStagesPipeline` and `DistilledPipeline` for simplicity. Other pipelines (`ic_lora`, `keyframe_interpolation`, `a2vid_two_stage`, `retake`, `lipdub`, `hdr_ic_lora`) live in [`packages/ltx-pipelines/src/ltx_pipelines/`](packages/ltx-pipelines/src/ltx_pipelines/) and follow the same shape — add a new branch in `load_pipeline()` and a matching call in `generate()` to expose them in the UI.