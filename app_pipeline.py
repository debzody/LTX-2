"""Pipeline backend for the LTX-2 Gradio UI.

Kept separate from app.py so the UI module stays small and we can lazy-import
the heavy ML stack only when a generation actually runs.
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Optional

import gradio as gr

# Lazy-imported below.
torch = None
TI2VidTwoStagesPipeline = None
DistilledPipeline = None
MultiModalGuiderParams = None
LoraPathStrengthAndSDOps = None
LTXV_LORA_COMFY_RENAMING_MAP = None
TilingConfig = None
get_video_chunks_number = None
ImageConditioningInput = None
encode_video = None
build_fp8_cast_policy = None


def _import_ltx() -> None:
    global torch, TI2VidTwoStagesPipeline, DistilledPipeline
    global MultiModalGuiderParams, LoraPathStrengthAndSDOps, LTXV_LORA_COMFY_RENAMING_MAP
    global TilingConfig, get_video_chunks_number, ImageConditioningInput
    global encode_video, build_fp8_cast_policy

    if torch is not None:
        return

    import torch as _torch
    from ltx_core.components.guiders import MultiModalGuiderParams as _GP
    from ltx_core.loader import (
        LTXV_LORA_COMFY_RENAMING_MAP as _MAP,
        LoraPathStrengthAndSDOps as _LP,
    )
    from ltx_core.model.video_vae import (
        TilingConfig as _TC,
        get_video_chunks_number as _gvc,
    )
    from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline as _TP
    from ltx_pipelines.distilled import DistilledPipeline as _DP
    from ltx_pipelines.utils.args import ImageConditioningInput as _ICI
    from ltx_pipelines.utils.media_io import encode_video as _ev

    torch = _torch
    TI2VidTwoStagesPipeline = _TP
    DistilledPipeline = _DP
    MultiModalGuiderParams = _GP
    LoraPathStrengthAndSDOps = _LP
    LTXV_LORA_COMFY_RENAMING_MAP = _MAP
    TilingConfig = _TC
    get_video_chunks_number = _gvc
    ImageConditioningInput = _ICI
    encode_video = _ev

    try:
        from ltx_core.quantization.fp8_cast import build_policy as _bp
        build_fp8_cast_policy = _bp
    except Exception:  # noqa: BLE001
        build_fp8_cast_policy = None


CFG: dict = {}
_PIPELINE = None
_PIPELINE_KIND: Optional[str] = None


PROMPT_PLACEHOLDER = (
    "A cinematic shot of a fox running through a snowy forest at dawn, "
    "soft golden light filtering through the trees, breath visible in the cold air, "
    "camera tracking smoothly alongside, shallow depth of field."
)


def load_pipeline(kind: str):
    global _PIPELINE, _PIPELINE_KIND
    _import_ltx()

    if _PIPELINE is not None and _PIPELINE_KIND == kind:
        return _PIPELINE

    _PIPELINE = None
    _PIPELINE_KIND = None
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()

    quantization = None
    if CFG.get("quantization") == "fp8-cast" and build_fp8_cast_policy is not None:
        quantization = build_fp8_cast_policy(CFG["checkpoint_path"])

    if kind == "distilled":
        # DistilledPipeline does not take a separate distilled_lora argument; the
        # LoRA (if any) is passed via `loras`. The checkpoint should already be
        # a distilled one.
        _PIPELINE = DistilledPipeline(
            distilled_checkpoint_path=CFG["checkpoint_path"],
            spatial_upsampler_path=CFG["spatial_upsampler_path"],
            gemma_root=CFG["gemma_root"],
            loras=[],
            quantization=quantization,
        )
    elif kind == "two-stage":
        distilled_lora = [
            LoraPathStrengthAndSDOps(
                CFG["distilled_lora"],
                float(CFG.get("distilled_lora_strength", 0.6)),
                LTXV_LORA_COMFY_RENAMING_MAP,
            ),
        ]
        _PIPELINE = TI2VidTwoStagesPipeline(
            checkpoint_path=CFG["checkpoint_path"],
            distilled_lora=distilled_lora,
            spatial_upsampler_path=CFG["spatial_upsampler_path"],
            gemma_root=CFG["gemma_root"],
            loras=[],
            quantization=quantization,
        )
    else:
        raise ValueError("Unknown pipeline kind: " + kind)

    _PIPELINE_KIND = kind
    return _PIPELINE


def generate(
    prompt: str,
    negative_prompt: str,
    pipeline_kind: str,
    width: int,
    height: int,
    num_frames: int,
    frame_rate: float,
    num_inference_steps: int,
    seed: int,
    video_cfg_scale: float,
    audio_cfg_scale: float,
    enhance_prompt: bool,
    image_path: Optional[str],
    progress: gr.Progress = gr.Progress(track_tqdm=True),
):
    if not prompt or not prompt.strip():
        raise gr.Error("Please enter a prompt.")
    required = ["checkpoint_path", "spatial_upsampler_path", "gemma_root"]
    if pipeline_kind == "two-stage":
        required.append("distilled_lora")
    if not all(CFG.get(k) for k in required):
        raise gr.Error(
            "Pipeline is not configured. Start the app with --checkpoint-path, "
            "--spatial-upsampler-path, --gemma-root "
            "(and --distilled-lora for the two-stage pipeline)."
        )

    progress(0.0, desc="Loading pipeline (first run can take minutes)...")
    pipeline = load_pipeline(pipeline_kind)

    # Snap to legal dimensions: divisible by 32; frames as 8k+1.
    width = max(256, (int(width) // 32) * 32)
    height = max(256, (int(height) // 32) * 32)
    num_frames = int(num_frames)
    if (num_frames - 1) % 8 != 0:
        num_frames = ((num_frames - 1) // 8) * 8 + 1
    num_frames = max(num_frames, 9)

    images = []
    if image_path:
        images = [ImageConditioningInput(image_path, 0, 1.0, 33)]

    video_guider_params = MultiModalGuiderParams(
        cfg_scale=float(video_cfg_scale),
        stg_scale=1.0,
        rescale_scale=0.7,
        modality_scale=3.0,
        skip_step=0,
        stg_blocks=[29],
    )
    audio_guider_params = MultiModalGuiderParams(
        cfg_scale=float(audio_cfg_scale),
        stg_scale=1.0,
        rescale_scale=0.7,
        modality_scale=3.0,
        skip_step=0,
        stg_blocks=[29],
    )

    tiling_config = TilingConfig.default()
    chunks = get_video_chunks_number(num_frames, tiling_config)

    out_dir = Path(CFG.get("output_dir") or tempfile.gettempdir()) / "ltx2_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(out_dir / f"ltx2_{int(time.time())}_{int(seed)}.mp4")

    neg = negative_prompt or "worst quality, low quality, blurry, distorted"

    progress(0.1, desc="Running diffusion...")
    with torch.inference_mode():
        if pipeline_kind == "distilled":
            # Distilled pipeline doesn't use a negative prompt or guider params.
            video, audio = pipeline(
                prompt=prompt,
                seed=int(seed),
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=float(frame_rate),
                images=images,
                tiling_config=tiling_config,
                enhance_prompt=bool(enhance_prompt),
            )
        else:
            video, audio = pipeline(
                prompt=prompt,
                negative_prompt=neg,
                seed=int(seed),
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=float(frame_rate),
                num_inference_steps=int(num_inference_steps),
                video_guider_params=video_guider_params,
                audio_guider_params=audio_guider_params,
                images=images,
                tiling_config=tiling_config,
                enhance_prompt=bool(enhance_prompt),
            )

        progress(0.9, desc="Encoding video...")
        encode_video(
            video=video,
            fps=float(frame_rate),
            audio=audio,
            output_path=out_path,
            video_chunks_number=chunks,
        )

    progress(1.0, desc="Done")
    return out_path, f"✅ Saved to `{out_path}`"