"""HF Spaces entry point for the LTX-2 Gradio UI.

This file is what HF Spaces runs as `app.py`. It:
  1. Downloads the LTX-2 distilled checkpoint + Gemma text encoder on first boot.
  2. Imports our existing UI (vendored from the parent repo as `_ui.py` and
     `_pipeline.py`) and launches it.

Required Space Secrets:
  - HF_TOKEN  (read scope; must come from an account that has accepted the
               Gemma-3 license at
               https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized)

Required Space hardware:
  - T4 small or larger (CPU-only Spaces will refuse to launch).
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

# --- 1. Where to put weights -------------------------------------------------
# /data is HF Spaces persistent storage. Falls back to /tmp on Spaces without
# persistent storage (weights re-download every restart).
PERSIST = Path("/data") if Path("/data").exists() else Path("/tmp")
CACHE = PERSIST / "hf_cache"
CACHE.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(CACHE)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(CACHE)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# --- 2. HF login -------------------------------------------------------------
from huggingface_hub import login, hf_hub_download, snapshot_download
from huggingface_hub.utils import GatedRepoError

token = os.environ.get("HF_TOKEN")
if not token:
    raise SystemExit(
        "ERROR: HF_TOKEN secret not set in this Space.\n"
        "Settings -> Variables and secrets -> New secret:\n"
        "  Name = HF_TOKEN\n"
        "  Value = <your HF read token from https://huggingface.co/settings/tokens>\n"
        "Also accept the Gemma license at\n"
        "  https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized"
    )
login(token=token, add_to_git_credential=False)

# --- 3. Download weights -----------------------------------------------------
print(f"[boot] Cache dir: {CACHE}")
free = shutil.disk_usage(str(CACHE)).free / (1024 ** 3)
print(f"[boot] Free disk: {free:.1f} GB")

print("[boot] Downloading distilled LTX-2 checkpoint (~22 GB)...")
ckpt = hf_hub_download(
    repo_id="Lightricks/LTX-2.3",
    filename="ltx-2.3-22b-distilled-1.1.safetensors",
    cache_dir=str(CACHE),
)
print(f"[boot] checkpoint: {ckpt}")

print("[boot] Downloading Gemma-3 text encoder (~7 GB)...")
try:
    gemma = snapshot_download(
        repo_id="google/gemma-3-12b-it-qat-q4_0-unquantized",
        cache_dir=str(CACHE),
        allow_patterns=["*.json", "*.safetensors", "*.model", "tokenizer*", "special_tokens*"],
    )
except GatedRepoError as e:
    raise SystemExit(
        "Gemma access denied. Either:\n"
        "  - The HF_TOKEN secret is for an account that hasn't accepted the\n"
        "    Gemma license, or the token doesn't have Read scope.\n"
        f"  - Original error: {e}"
    )
print(f"[boot] gemma: {gemma}")

# --- 4. Hand off to our shared CLI ------------------------------------------
sys.argv = [
    "app.py",
    "--checkpoint-path", ckpt,
    "--gemma-root", gemma,
    "--quantization", "fp8-cast",
    "--output-dir", str(PERSIST / "outputs"),
    "--host", "0.0.0.0",
    "--port", "7860",
]

# Import and run the Gradio app entrypoint that lives in `_ui.py` next to us.
# (We rename app.py -> _ui.py inside the Space so this file can be the entry
# point. See the Makefile / README for how to assemble.)
from _ui import main as _main  # noqa: E402

_main()