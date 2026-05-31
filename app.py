"""LTX-2 Video Generation Web UI (Gradio).

Usage:
    python app.py \\
        --checkpoint-path /path/to/ltx-2.3-22b-distilled-1.1.safetensors \\
        --distilled-lora /path/to/ltx-2.3-22b-distilled-lora-384-1.1.safetensors \\
        --spatial-upsampler-path /path/to/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \\
        --gemma-root /path/to/gemma-3-12b-it-qat-q4_0-unquantized
"""
from __future__ import annotations

import argparse
import logging
import traceback

import gradio as gr

# CFG / generate are imported lazily inside main() so the UI can also run in
# `--demo` mode (no torch / no LTX dependency installed).
CFG: dict = {}
PROMPT_PLACEHOLDER = (
    "A cinematic shot of a fox running through a snowy forest at dawn, "
    "soft golden light filtering through the trees, breath visible in the cold air, "
    "camera tracking smoothly alongside, shallow depth of field."
)
_GENERATE = None  # set in main()
_DEMO_MODE = False


def safe_generate(*args, **kwargs):
    if _DEMO_MODE or _GENERATE is None:
        raise gr.Error(
            "Demo mode: this build has no model backend wired up. "
            "Run on a CUDA machine without --demo and pass --checkpoint-path / "
            "--distilled-lora / --spatial-upsampler-path / --gemma-root."
        )
    try:
        return _GENERATE(*args, **kwargs)
    except gr.Error:
        raise
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        raise gr.Error(f"Generation failed: {exc}") from exc


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="LTX-2 Video Generator", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 🎬 LTX-2 Video Generator\n"
            "Enter a prompt and generate a video. **two-stage** = best quality, "
            "**distilled** = fastest."
        )

        with gr.Row():
            with gr.Column(scale=3):
                prompt = gr.Textbox(label="Prompt", lines=4, placeholder=PROMPT_PLACEHOLDER)
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    lines=2,
                    value="worst quality, low quality, blurry, distorted",
                )
                image_input = gr.Image(
                    label="Optional: conditioning image (image-to-video)",
                    type="filepath",
                )

                with gr.Row():
                    pipeline_kind = gr.Radio(
                        choices=["two-stage", "distilled"],
                        value="two-stage",
                        label="Pipeline",
                    )
                    enhance_prompt = gr.Checkbox(value=False, label="Enhance prompt")

                with gr.Accordion("Advanced settings", open=False):
                    with gr.Row():
                        width = gr.Slider(256, 1280, value=768, step=32, label="Width")
                        height = gr.Slider(256, 1280, value=512, step=32, label="Height")
                    with gr.Row():
                        num_frames = gr.Slider(9, 257, value=121, step=8, label="Frames (8k+1)")
                        frame_rate = gr.Slider(8, 30, value=25, step=1, label="FPS")
                    with gr.Row():
                        num_inference_steps = gr.Slider(
                            8, 60, value=30, step=1, label="Inference steps (two-stage)"
                        )
                        seed = gr.Number(value=42, precision=0, label="Seed")
                    with gr.Row():
                        video_cfg = gr.Slider(1.0, 8.0, value=3.0, step=0.1, label="Video CFG")
                        audio_cfg = gr.Slider(1.0, 12.0, value=7.0, step=0.1, label="Audio CFG")

                generate_btn = gr.Button("🎬 Generate Video", variant="primary")

            with gr.Column(scale=2):
                video_output = gr.Video(label="Result", autoplay=True)
                status = gr.Markdown("")

        gr.Examples(
            examples=[
                ["A cinematic shot of a fox running through a snowy forest at dawn, "
                 "soft golden light, breath visible in cold air, smooth tracking camera."],
                ["Aerial drone footage flying over a tropical island at sunset, "
                 "turquoise water, palm trees, golden hour lighting."],
                ["A close-up of a hummingbird hovering near a bright red flower, "
                 "wings beating rapidly, slow motion, soft natural light."],
            ],
            inputs=[prompt],
        )

        generate_btn.click(
            fn=safe_generate,
            inputs=[
                prompt, negative_prompt, pipeline_kind,
                width, height, num_frames, frame_rate,
                num_inference_steps, seed,
                video_cfg, audio_cfg,
                enhance_prompt, image_input,
            ],
            outputs=[video_output, status],
        )

    return demo


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LTX-2 Video Generator UI")
    p.add_argument(
        "--demo", action="store_true",
        help="Launch the UI without the LTX-2 backend (lets you preview the layout "
             "on a machine that doesn't have torch / CUDA / model weights).",
    )
    p.add_argument("--checkpoint-path", required=False, help="Path to LTX-2 .safetensors")
    p.add_argument("--distilled-lora", required=False, help="Path to distilled LoRA")
    p.add_argument(
        "--distilled-lora-strength", type=float, default=0.6, help="Distilled LoRA strength"
    )
    p.add_argument(
        "--spatial-upsampler-path", required=False, help="Path to spatial upsampler"
    )
    p.add_argument("--gemma-root", required=False, help="Path to Gemma text encoder dir")
    p.add_argument(
        "--quantization", choices=["none", "fp8-cast"], default="none",
        help="Optional weight quantization",
    )
    p.add_argument("--output-dir", default=None, help="Where to write generated mp4s")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    return p.parse_args()


def main() -> None:
    global _GENERATE, _DEMO_MODE, CFG, PROMPT_PLACEHOLDER

    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    if args.demo:
        _DEMO_MODE = True
        print("[INFO] Running in --demo mode: UI only, no model backend.")
    else:
        # Try to import the heavy backend. If it fails (no torch / not CUDA),
        # fall back to demo mode so the UI still starts.
        try:
            from app_pipeline import (
                CFG as _BACKEND_CFG,
                generate as _BACKEND_GENERATE,
                PROMPT_PLACEHOLDER as _BACKEND_PROMPT,
            )
            CFG = _BACKEND_CFG
            _GENERATE = _BACKEND_GENERATE
            PROMPT_PLACEHOLDER = _BACKEND_PROMPT
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Could not import LTX-2 backend ({exc!r}); starting UI in demo mode.")
            _DEMO_MODE = True

    CFG.update({
        "checkpoint_path": args.checkpoint_path,
        "distilled_lora": args.distilled_lora,
        "distilled_lora_strength": args.distilled_lora_strength,
        "spatial_upsampler_path": args.spatial_upsampler_path,
        "gemma_root": args.gemma_root,
        "quantization": None if args.quantization == "none" else args.quantization,
        "output_dir": args.output_dir,
    })

    if not _DEMO_MODE:
        missing = [k for k in (
            "checkpoint_path", "distilled_lora", "spatial_upsampler_path", "gemma_root"
        ) if not CFG.get(k)]
        if missing:
            print(
                "[WARN] Missing required model paths: "
                + ", ".join(missing)
                + "\nThe UI will start but generation will fail until you restart "
                "with --checkpoint-path / --distilled-lora / --spatial-upsampler-path / --gemma-root.",
            )

    demo = build_ui()
    demo.queue().launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()