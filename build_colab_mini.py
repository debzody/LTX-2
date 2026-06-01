"""Build colab_mini_ui.ipynb cleanly via the Python API (avoids shell heredoc apostrophes)."""
import json

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text})


def code(src):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src,
    })


md(
    "# LTX-Video Mini - Free Google Colab UI\n\n"
    "A prompt-to-video Gradio UI using **LTX-Video v0.9.7-distilled** - the "
    "predecessor to LTX-2 from the same team (Lightricks). It is a much smaller "
    "model (~6 GB instead of 22 GB) so it fits comfortably on free Colab T4.\n\n"
    "Trade-offs vs LTX-2: lower visual quality, no audio, but **runs free on Colab** "
    "with no /tmp tricks, no gated-model logins, no quantization juggling. Just works.\n\n"
    "## How to use\n\n"
    "1. Open this notebook in Colab.\n"
    "2. Runtime > Change runtime type > T4 GPU (free).\n"
    "3. Runtime > Run all.\n"
    "4. The last cell prints a public https://*.gradio.live URL - open it.\n\n"
    "First boot takes ~5 min (downloads ~6 GB). Each video then takes 30-60 seconds."
)

md("## 1. GPU sanity check")
code(
    "!nvidia-smi || echo 'No GPU. Runtime > Change runtime type > T4 GPU.'\n"
    "!df -h /content"
)

md("## 2. Install dependencies (~2 min)")
code(
    "# diffusers >= 0.32 ships LTXPipeline. Use Colab preinstalled torch.\n"
    "!pip install -q --upgrade 'diffusers>=0.32' 'transformers>=4.45' 'accelerate>=1.0' 'gradio>=4.44' sentencepiece imageio imageio-ffmpeg\n"
    "import torch\n"
    "print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
)

md(
    "## 3. Write the Gradio app file\n\n"
    "We write to `/content/ltx_mini_app.py` so Gradio runs in its own process "
    "(avoids cell-output buffering hangs)."
)

# Build the app source as a list of lines (no f-strings, no apostrophes inside strings).
app_lines = [
    '"""LTX-Video distilled prompt-to-video UI for free Colab T4."""',
    "import os",
    "import time",
    "from pathlib import Path",
    "",
    "import gradio as gr",
    "import torch",
    "from diffusers import LTXPipeline",
    "from diffusers.utils import export_to_video",
    "",
    "os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')",
    "",
    "MODEL_ID = 'Lightricks/LTX-Video-0.9.7-distilled'",
    "_PIPE = None",
    "OUT_DIR = Path('/content/outputs')",
    "OUT_DIR.mkdir(parents=True, exist_ok=True)",
    "",
    "",
    "def get_pipeline():",
    "    global _PIPE",
    "    if _PIPE is not None:",
    "        return _PIPE",
    "    print('Loading LTX-Video pipeline (first time downloads ~6 GB)...')",
    "    pipe = LTXPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)",
    "    # Sequential offload: move each layer to GPU only when needed.",
    "    # Slower than enable_sequential_cpu_offload + VAE tiling() but uses much less peak VRAM.",
    "    pipe.enable_sequential_cpu_offload()",
    "    # VAE tiling: decode the video in tiles instead of all at once.",
    "    # Critical on T4 — without this, decoding 49 frames OOMs.",
    "    if hasattr(pipe, 'vae') and hasattr(pipe.vae, 'enable_tiling'):",
    "        pipe.vae.enable_tiling()",
    "    _PIPE = pipe",
    "    return pipe",
    "",
    "",
    "def generate(prompt, negative_prompt, width, height, num_frames, num_inference_steps, fps, seed,",
    "             progress=gr.Progress(track_tqdm=True)):",
    "    if not prompt or not prompt.strip():",
    "        raise gr.Error('Please enter a prompt.')",
    "    progress(0.0, desc='Loading pipeline...')",
    "    pipe = get_pipeline()",
    "    width = max(256, (int(width) // 32) * 32)",
    "    height = max(256, (int(height) // 32) * 32)",
    "    nf = int(num_frames)",
    "    if (nf - 1) % 8 != 0:",
    "        nf = ((nf - 1) // 8) * 8 + 1",
    "    nf = max(nf, 9)",
    "    progress(0.1, desc='Running diffusion...')",
    "    gen = torch.Generator(device='cuda').manual_seed(int(seed))",
    "    out = pipe(",
    "        prompt=prompt,",
    "        negative_prompt=negative_prompt or 'worst quality, blurry, distorted, low resolution',",
    "        width=width,",
    "        height=height,",
    "        num_frames=nf,",
    "        num_inference_steps=int(num_inference_steps),",
    "        generator=gen,",
    "    )",
    "    video_frames = out.frames[0]",
    "    out_path = str(OUT_DIR / ('ltxv_' + str(int(time.time())) + '_' + str(int(seed)) + '.mp4'))",
    "    progress(0.95, desc='Encoding video...')",
    "    export_to_video(video_frames, out_path, fps=int(fps))",
    "    progress(1.0, desc='Done')",
    "    return out_path, 'Saved to ' + out_path",
    "",
    "",
    "PROMPT_HINT = (",
    "    'A cinematic shot of a fox running through a snowy forest at dawn, '",
    "    'soft golden light filtering through the trees, smooth tracking camera, '",
    "    'shallow depth of field.'",
    ")",
    "",
    "",
    "with gr.Blocks(title='LTX-Video Mini') as demo:",
    "    gr.Markdown('# LTX-Video Mini\\nRunning **' + MODEL_ID + '** on Colab T4 with sequential_cpu_offload + VAE tiling. ~30-60 sec/video at 768x512x49 frames.')",
    "    with gr.Row():",
    "        with gr.Column(scale=3):",
    "            prompt = gr.Textbox(label='Prompt', lines=4, placeholder=PROMPT_HINT)",
    "            neg = gr.Textbox(label='Negative Prompt', value='worst quality, blurry, distorted, low resolution', lines=2)",
    "            with gr.Accordion('Advanced settings', open=False):",
    "                with gr.Row():",
    "                    width = gr.Slider(256, 1024, value=512, step=32, label='Width')",
    "                    height = gr.Slider(256, 1024, value=320, step=32, label='Height')",
    "                with gr.Row():",
    "                    num_frames = gr.Slider(9, 161, value=25, step=8, label='Frames (8k+1)')",
    "                    fps = gr.Slider(8, 30, value=24, step=1, label='FPS')",
    "                with gr.Row():",
    "                    steps = gr.Slider(4, 20, value=8, step=1, label='Inference steps (distilled = ~8)')",
    "                    seed = gr.Number(value=42, precision=0, label='Seed')",
    "            btn = gr.Button('Generate Video', variant='primary')",
    "        with gr.Column(scale=2):",
    "            out_video = gr.Video(label='Result', autoplay=True)",
    "            status = gr.Markdown('')",
    "    gr.Examples(",
    "        examples=[",
    "            ['A serene aerial shot drifting over a turquoise lagoon at sunset, palm trees swaying in the warm golden light.'],",
    "            ['A close-up of a hummingbird hovering near a bright red flower, slow motion, soft natural backlight.'],",
    "            ['A neon-lit cyberpunk alley at night, rain on the asphalt reflecting the signs, a lone figure walking away from camera.'],",
    "        ],",
    "        inputs=[prompt],",
    "    )",
    "    btn.click(generate, [prompt, neg, width, height, num_frames, steps, fps, seed], [out_video, status])",
    "",
    "demo.queue().launch(share=True, server_name='0.0.0.0', server_port=7860)",
]

# Wrap in a writefile magic so Colab persists it to disk.
app_src = "%%writefile /content/ltx_mini_app.py\n" + "\n".join(app_lines) + "\n"
code(app_src)

md("## 4. Run the app and get the public URL")
code("!python /content/ltx_mini_app.py")

md(
    "## Tips\n\n"
    "- Distilled variant uses 8 inference steps by default. Bumping it does not help much.\n"
    "- Default 768x512x49 frames is ~2 sec at 24 fps. Push to 768x512x97 for ~4 sec; higher only if VRAM allows.\n"
    "- First generation: ~90 sec (model load + first run). Subsequent: ~30-60 sec.\n"
    "- Generated MP4s land in /content/outputs/ - visible in Colab left-side Files panel.\n"
    "- Free Colab disconnects after ~12h or 90min idle. Re-run the last cell to relaunch (HF cache survives).\n"
    "- This is NOT LTX-2 - it is the older v0.9.7. Quality is good but not LTX-2 level."
)

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open("colab_mini_ui.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print(f"Wrote colab_mini_ui.ipynb with {len(cells)} cells")