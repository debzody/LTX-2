"""Wan-2.1 T2V 1.3B prompt-to-video Gradio UI for Apple Silicon (M1/M2/M3/M4).

Uses PyTorch's MPS backend. Tested on M4 Pro 24 GB. Smaller M-series chips
(8 GB / 16 GB) may need to drop the resolution further.

Usage:
    python mac_wan_app.py

Then open http://127.0.0.1:7860 in your browser.

Cold start downloads ~3 GB of Wan-2.1 weights into ~/.cache/huggingface/hub
the first time. Each generation is ~3-6 minutes on M4 Pro at 480x320x33 frames
(MPS is roughly 2-3x slower than a CUDA T4 for diffusion models).
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

# Required env vars BEFORE torch import.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # for ops not yet on MPS
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import gradio as gr
import torch
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video


MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
_PIPE = None


def pick_device_dtype():
    if torch.backends.mps.is_available():
        # MPS in 2.x supports float16 well; bfloat16 is hit-or-miss.
        return torch.device("mps"), torch.float16, "Apple Silicon (MPS)"
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.bfloat16, f"CUDA ({torch.cuda.get_device_name(0)})"
    return torch.device("cpu"), torch.float32, "CPU (very slow!)"


DEVICE, DTYPE, DEVICE_DESC = pick_device_dtype()
print(f"[boot] Device: {DEVICE_DESC}  dtype: {DTYPE}")


def get_pipeline():
    global _PIPE
    if _PIPE is not None:
        return _PIPE
    print("[boot] Loading Wan-2.1 1.3B pipeline (first time downloads ~3 GB)...")
    # Wan's VAE prefers fp32 for stability.
    vae = AutoencoderKLWan.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(MODEL_ID, vae=vae, torch_dtype=DTYPE)
    pipe = pipe.to(DEVICE)
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
    _PIPE = pipe
    return pipe


def generate(
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    num_frames: int,
    num_inference_steps: int,
    fps: int,
    guidance_scale: float,
    seed: int,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
):
    if not prompt or not prompt.strip():
        raise gr.Error("Please enter a prompt.")
    progress(0.0, desc=f"Loading pipeline on {DEVICE_DESC}...")
    pipe = get_pipeline()

    width = max(256, (int(width) // 16) * 16)
    height = max(256, (int(height) // 16) * 16)
    nf = max(int(num_frames), 9)

    progress(0.1, desc="Running diffusion...")
    # MPS doesn't support torch.Generator directly — use CPU generator.
    gen = torch.Generator(device="cpu").manual_seed(int(seed))

    out = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt or "low quality, blurry, distorted, watermark",
        width=width,
        height=height,
        num_frames=nf,
        num_inference_steps=int(num_inference_steps),
        guidance_scale=float(guidance_scale),
        generator=gen,
    )
    video_frames = out.frames[0]

    out_dir = Path.home() / "Desktop" / "wan_videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(out_dir / f"wan_{int(time.time())}_{int(seed)}.mp4")

    progress(0.95, desc="Encoding video...")
    export_to_video(video_frames, out_path, fps=int(fps))
    progress(1.0, desc="Done")
    return out_path, f"✅ Saved to {out_path}"


PROMPT_HINT = (
    "A cinematic shot of a fox running through a snowy forest at dawn, "
    "soft golden light filtering through the trees, smooth tracking camera."
)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Wan-2.1 1.3B (Local Mac)") as demo:
        gr.Markdown(
            f"# 🎬 Wan-2.1 T2V 1.3B (Local)\n"
            f"Running **{MODEL_ID}** on **{DEVICE_DESC}**.\n"
            f"Generated MP4s save to `~/Desktop/wan_videos/`."
        )
        with gr.Row():
            with gr.Column(scale=3):
                prompt = gr.Textbox(label="Prompt", lines=4, placeholder=PROMPT_HINT)
                neg = gr.Textbox(
                    label="Negative Prompt",
                    value="low quality, blurry, distorted, watermark, text",
                    lines=2,
                )
                with gr.Accordion("Advanced settings", open=False):
                    with gr.Row():
                        width = gr.Slider(256, 1024, value=480, step=16, label="Width")
                        height = gr.Slider(256, 1024, value=320, step=16, label="Height")
                    with gr.Row():
                        num_frames = gr.Slider(9, 121, value=33, step=4, label="Frames")
                        fps = gr.Slider(8, 30, value=16, step=1, label="FPS")
                    with gr.Row():
                        steps = gr.Slider(10, 50, value=25, step=1, label="Inference steps")
                        cfg = gr.Slider(1.0, 10.0, value=5.0, step=0.5, label="Guidance scale")
                    seed = gr.Number(value=42, precision=0, label="Seed")
                btn = gr.Button("🎬 Generate Video", variant="primary")
            with gr.Column(scale=2):
                out_video = gr.Video(label="Result", autoplay=True)
                status = gr.Markdown("")

        gr.Examples(
            examples=[
                [
                    "A serene aerial shot drifting over a turquoise lagoon at sunset, palm trees swaying in the warm golden light."
                ],
                [
                    "A close-up of a hummingbird hovering near a bright red flower, slow motion, soft natural backlight."
                ],
                [
                    "Cute orange cat playing with a ball of yarn on a sunlit wooden floor, soft focus, slow motion."
                ],
            ],
            inputs=[prompt],
        )
        btn.click(
            generate,
            [prompt, neg, width, height, num_frames, steps, fps, cfg, seed],
            [out_video, status],
        )
    return demo


def main():
    p = argparse.ArgumentParser(description="Wan-2.1 1.3B local Mac UI")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="Public *.gradio.live URL")
    args = p.parse_args()

    demo = build_ui()
    demo.queue().launch(
        server_name=args.host, server_port=args.port, share=args.share, inbrowser=True
    )


if __name__ == "__main__":
    main()