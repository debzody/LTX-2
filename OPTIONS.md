# Where can I actually run this LTX-2 UI?

LTX-2 is a **22 B-parameter** model. Its weights are ~22 GB by themselves; the Gemma-3 text encoder it depends on is another ~7 GB; PyTorch + CUDA libraries are another ~10 GB on top. **You need ~40 GB free disk and an NVIDIA GPU with at least 16 GB VRAM** to run this locally.

This rules out free Google Colab (~85 GB usable disk) once you account for system files. Below are the paths that *do* work, ranked by friction.

| # | Option | Cost | DIY effort | Notes |
| - | ------ | ---- | ---------- | ----- |
| 1 | **Lightricks's hosted playground** at <https://console.ltx.video/playground> | Free tier | None | Same model. Sign in, type prompt, get video. **Easiest by far.** |
| 2 | **Hugging Face Spaces** community demos in <https://huggingface.co/Lightricks> | Free | None | Browser UI, runs on a GPU someone else pays for. Look for an LTX-2 Space. |
| 3 | **Kaggle Notebooks** — see [`kaggle_ltx2_ui.ipynb`](kaggle_ltx2_ui.ipynb) | Free (30 hrs/week T4) | Open & Run All | Same UI as our Colab notebook, but enough disk to actually fit. |
| 4 | **Colab Pro** (₹1k/month) | Paid | Run [`colab_ltx2_ui.ipynb`](colab_ltx2_ui.ipynb) | ~166 GB disk, A100/L4 — fits comfortably. |
| 5 | **GPU rental** (RunPod / Vast.ai / Lambda) | ~₹15–₹40/hour | SSH + run [`app.py`](app.py) | Pay only while generating. ~₹50 = a couple of hours. |
| 6 | **Your own NVIDIA box** (≥16 GB VRAM, ≥40 GB free disk) | One-time HW cost | Run `app.py` per [`UI_README.md`](UI_README.md) | The "ideal" path; needs hardware you may not have. |

## Why free Colab specifically doesn't work

Disk math after a fresh Colab T4 runtime:

- System / Python / preinstalled libs: **~25 GB used**
- LTX-2 22B checkpoint: **~22 GB**
- Gemma-3 text encoder: **~7 GB**
- LTX-2 Python deps (torch+cu, etc.): **~5 GB**
- HF cache + working room: **~10 GB**
- **Total: ~70 GB** of a ~85 GB usable disk → it just barely doesn't fit because:
  - HF Hub's xet downloader needs to hold the **incomplete download** plus the **final file** simultaneously (effectively 2× the file size for a moment).
  - That's where the "No space left on device" error comes from at 93% downloaded.

## Why Kaggle works

Free Kaggle Notebooks give you:

- **20 GB on `/kaggle/working`** (persisted between cells).
- **70 GB on `/tmp`** (volatile but huge).
- **GPU T4 ×2** or **GPU P100** with 16 GB VRAM.
- ~30 hours/week of GPU time.

It's the only mainstream **free** notebook host with enough disk + GPU to run LTX-2 end-to-end.

## TL;DR

**For zero setup and zero cost:** use the official playground (#1).

**For "I want my own UI to mess with, free":** open `kaggle_ltx2_ui.ipynb` on Kaggle (#3).

**Everything else** costs money.