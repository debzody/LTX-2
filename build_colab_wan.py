"""Build colab_wan_ui.ipynb cleanly via the Python API."""
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
    "# Wan-2.1 1.3B - Free Google Colab UI\n\n"
    "A prompt-to-video Gradio UI using **Wan-2.1 T2V 1.3B** by Alibaba's Wan team.\n\n"
    "- Only ~3 GB on disk (3x smaller than LTX-Video distilled).\n"
    "- Surprisingly good quality for its size.\n"
    "- Apache-2.0 license.\n"
    "- Fits comfortably on free Colab T4 with `enable_model_cpu_offload()`.\n\n"
    "## How to use\n\n"
    "1. Open this notebook in Colab.\n"
    "2. Runtime > Change runtime type > T4 GPU (free).\n"
    "3. Runtime > Run all.\n"
    "4. The last cell prints a public https://*.gradio.live URL - open it.\n\n"
    "Cold start: ~3-5 min (downloads ~3 GB). Each video then takes ~1-2 min."
)

md("## 1. GPU sanity check")
code(
    "!nvidia-smi || echo 'No GPU. Runtime > Change runtime type > T4 GPU.'\n"
    "!df -h /content"
)

md("## 2. Install dependencies (~2 min)")
code(
    "# WanPipeline lives in diffusers >= 0.32. ftfy is needed by Wan's text encoder.\n"
    "!pip install -q --upgrade 'diffusers>=0.32' 'transformers>=4.45' 'accelerate>=1.0' 'gradio>=4.44' ftfy imageio imageio-ffmpeg sentencepiece\n"
    "import torch\n"
    "print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
)

md(
    "## 3. Write the Gradio app file\n\n"
    "We write to `/content/wan_app.py` so Gradio runs in its own process."
)

app_lines = [
    '"""Wan-2.1 T2V 1.3B prompt-to-video UI for free Colab T4."""',
    "import os",
    "import time",
    "from pathlib import Path",
    "",
    "import gradio as gr",
    "import torch",
    "from diffusers import AutoencoderKLWan, WanPipeline",
    "from diffusers.utils import export_to_video",
    "",
    "os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')",
    "",
    "MODEL_ID = 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers'",
    "_PIPE = None",
    "OUT_DIR = Path('/content/outputs')",
    "OUT_DIR.mkdir(parents=True, exist_ok=True)",
    "",
    "",
    "def get_pipeline():",
    "    global _PIPE",
    "    if _PIPE is not None:",
    "        return _PIPE",
    "    print('Loading Wan-2.1 1.3B pipeline (first time downloads ~3 GB)...')",
    "    # Wan ships its VAE separately; load it in fp32 for stability then cast the rest in bf16.",
    "    vae = AutoencoderKLWan.from_pretrained(MODEL_ID, subfolder='vae', torch_dtype=torch.float32)",
    "    pipe = WanPipeline.from_pretrained(MODEL_ID, vae=vae, torch_dtype=torch.bfloat16)",
    "    pipe.enable_model_cpu_offload()",
    "    if hasattr(pipe, 'vae') and hasattr(pipe.vae, 'enable_tiling'):",
    "        pipe.vae.enable_tiling()",
    "    _PIPE = pipe",
    "    return pipe",
    "",
    "",
    "def generate(prompt, negative_prompt, width, height, num_frames, num_inference_steps, fps, guidance_scale, seed,",
    "             progress=gr.Progress(track_tqdm=True)):",
    "    if not prompt or not prompt.strip():",
    "        raise gr.Error('Please enter a prompt.')",
    "    progress(0.0, desc='Loading pipeline...')",
    "    pipe = get_pipeline()",
    "    width = max(256, (int(width) // 16) * 16)",
    "    height = max(256, (int(height) // 16) * 16)",
    "    nf = max(int(num_frames), 9)",
    "    progress(0.1, desc='Running diffusion...')",
    "    gen = torch.Generator(device='cuda').manual_seed(int(seed))",
    "    out = pipe(",
    "        prompt=prompt,",
    "        negative_prompt=negative_prompt or 'low quality, blurry, distorted, watermark',",
    "        width=width,",
    "        height=height,",
    "        num_frames=nf,",
    "        num_inference_steps=int(num_inference_steps),",
    "        guidance_scale=float(guidance_scale),",
    "        generator=gen,",
    "    )",
    "    video_frames = out.frames[0]",
    "    out_path = str(OUT_DIR / ('wan_' + str(int(time.time())) + '_' + str(int(seed)) + '.mp4'))",
    "    progress(0.95, desc='Encoding video...')",
    "    export_to_video(video_frames, out_path, fps=int(fps))",
    "    progress(1.0, desc='Done')",
    "    return out_path, 'Saved to ' + out_path",
    "",
    "",
    "PROMPT_HINT = (",
    "    'A cinematic shot of a fox running through a snowy forest at dawn, '",
    "    'soft golden light filtering through the trees, smooth tracking camera.'",
    ")",
    "",
    "",
    "with gr.Blocks(title='Wan-2.1 1.3B') as demo:",
    "    gr.Markdown('# Wan-2.1 T2V 1.3B\\nRunning **' + MODEL_ID + '** on Colab T4 with cpu_offload + VAE tiling. ~1-2 min/video at 480x320x33 frames.')",
    "    with gr.Row():",
    "        with gr.Column(scale=3):",
    "            prompt = gr.Textbox(label='Prompt', lines=4, placeholder=PROMPT_HINT)",
    "            neg = gr.Textbox(label='Negative Prompt', value='low quality, blurry, distorted, watermark, text', lines=2)",
    "            with gr.Accordion('Advanced settings', open=False):",
    "                with gr.Row():",
    "                    width = gr.Slider(256, 1024, value=480, step=16, label='Width')",
    "                    height = gr.Slider(256, 1024, value=320, step=16, label='Height')",
    "                with gr.Row():",
    "                    num_frames = gr.Slider(9, 121, value=33, step=4, label='Frames')",
    "                    fps = gr.Slider(8, 30, value=16, step=1, label='FPS')",
    "                with gr.Row():",
    "                    steps = gr.Slider(10, 50, value=25, step=1, label='Inference steps')",
    "                    cfg = gr.Slider(1.0, 10.0, value=5.0, step=0.5, label='Guidance scale')",
    "                seed = gr.Number(value=42, precision=0, label='Seed')",
    "            btn = gr.Button('Generate Video', variant='primary')",
    "        with gr.Column(scale=2):",
    "            out_video = gr.Video(label='Result', autoplay=True)",
    "            status = gr.Markdown('')",
    "    gr.Examples(",
    "        examples=[",
    "            ['A serene aerial shot drifting over a turquoise lagoon at sunset, palm trees swaying in the warm golden light.'],",
    "            ['A close-up of a hummingbird hovering near a bright red flower, slow motion, soft natural backlight.'],",
    "            ['A neon-lit cyberpunk alley at night, rain on the asphalt reflecting the signs, a lone figure walking away from camera.'],",
    "            ['Cute orange cat playing with a ball of yarn on a sunlit wooden floor, soft focus, slow motion.'],",
    "        ],",
    "        inputs=[prompt],",
    "    )",
    "    btn.click(generate, [prompt, neg, width, height, num_frames, steps, fps, cfg, seed], [out_video, status])",
    "",
    "demo.queue().launch(share=True, server_name='0.0.0.0', server_port=7860)",
]

app_src = "%%writefile /content/wan_app.py\n" + "\n".join(app_lines) + "\n"
code(app_src)

md("## 4. Run the app and get the public URL")
code("!python /content/wan_app.py")

md(
    "## Tips\n\n"
    "- Default 480x320x33 frames is ~2 sec at 16 fps. Push to 480x320x65 for ~4 sec.\n"
    "- Wan-2.1 likes more inference steps than distilled models. 25-40 is the sweet spot.\n"
    "- Higher `guidance_scale` (5-7) = stricter prompt adherence; lower (3-4) = more diverse motion.\n"
    "- First generation: ~3 min (model load + first run). Subsequent: ~1-2 min.\n"
    "- Generated MP4s land in /content/outputs/ - visible in Colab left-side Files panel.\n"
    "- Free Colab disconnects after ~12h or 90min idle. Re-run the last cell to relaunch.\n"
    "- The model is Apache-2.0 licensed - completely open. No HF token needed."
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

with open("colab_wan_ui.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print(f"Wrote colab_wan_ui.ipynb with {len(cells)} cells")