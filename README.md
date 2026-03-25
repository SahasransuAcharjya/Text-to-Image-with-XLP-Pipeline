# 🎨 RealVis Image Studio


## 📌 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [How It Works](#how-it-works)
- [Tech Stack](#tech-stack)
- [Installation & Setup](#installation--setup)
- [Usage Guide](#usage-guide)
- [Style Presets](#style-presets)
- [Supported Resolutions](#supported-resolutions)
- [Advanced Settings](#advanced-settings)
- [VRAM Optimization Notes](#vram-optimization-notes)
- [Known Issues & Limitations](#known-issues--limitations)
- [Project Structure](#project-structure)

---

## Overview

**RealVis Image Studio** is a Jupyter notebook designed to run on **Google Colab (T4 GPU)**. It loads the `SG161222/RealVisXL_V4.0` model — a fine-tuned SDXL model known for its exceptional photorealism — and wraps it in an intuitive web UI using **Gradio**.

The project is self-contained: it installs all required dependencies, initializes the pipeline, and launches a shareable public link so you can generate images directly from your browser without writing any code.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🖼️ **Text-to-Image** | Generate high-quality images from any text prompt |
| 🎭 **Style Presets** | One-click Cinematic, Anime, Photographic, Digital Art & more |
| 📐 **Aspect Ratio Control** | Square, Portrait (3:4 / 9:16), Landscape (4:3 / 16:9) |
| 🎛️ **CFG Scale Slider** | Fine-tune how strictly the model follows your prompt (1.0–15.0) |
| 🔁 **Inference Steps Slider** | Trade speed for quality (10–60 steps) |
| ⚡ **VRAM Optimized** | VAE tiling + slicing enabled to fit in 16GB VRAM |
| 🌐 **Shareable Public URL** | Gradio generates a public link valid for 1 week |
| 🔒 **Negative Prompt Support** | Explicitly exclude unwanted elements from generations |

---

## How It Works

The notebook follows a **3-stage pipeline**:

```
User Prompt
    │
    ▼
┌─────────────────────────────┐
│   Style Template Applied    │  ← Prompt engineering & style injection
│   (e.g., "Cinematic still   │
│    {prompt}. highly detail" │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  RealVisXL V4.0 (SDXL)      │  ← Diffusion model running on CUDA
│  - CLIP Text Encoder x2     │
│  - UNet Denoiser             │
│  - VAE Decoder               │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│   Gradio Web Interface      │  ← Displays output, manages inputs
└─────────────────────────────┘
```

**Internally**, the `CustomImageEngine` class orchestrates the following on every generation:

1. The selected **style preset** wraps the user's prompt with carefully crafted descriptors.
2. A **negative prompt** (user-defined + style default) tells the model what to avoid.
3. The chosen **aspect ratio** maps to exact pixel dimensions (e.g., `1344×768` for 16:9).
4. The SDXL pipeline runs inference with `float16` precision on the GPU.
5. After generation, `torch.cuda.empty_cache()` and `gc.collect()` are called to free memory.

---

## 🛠️ Tech Stack

| Library | Version | Role |
|---|---|---|
| `diffusers` | 0.31.0 | SDXL pipeline & model loading |
| `transformers` | 4.46.1 | CLIP text encoders |
| `accelerate` | 0.34.2 | Efficient GPU model loading |
| `peft` | latest | Parameter-efficient model adapters |
| `safetensors` | latest | Safe model weight format |
| `gradio` | 4.44.1 | Web UI & public URL sharing |
| `torch` | 2.x (CUDA 12.x) | GPU tensor operations |
| `onnxruntime-gpu` | latest | GPU-accelerated ONNX inference |
| `insightface` | latest | Face detection & processing |
| `opencv-python` | 4.11.x | Image processing utilities |
| `numpy` | 1.26.4 | Numerical operations |

> ⚠️ **Note:** `numpy==1.26.4` is intentionally pinned to maintain compatibility with the specific `diffusers`/`transformers` versions used. Newer numpy versions break some of the pipeline internals.

---

## 📦 Installation & Setup

### Prerequisites

- A **Google Colab** account (free tier works; T4 GPU recommended)
- No local GPU or Python installation required

### Steps

**1. Open in Google Colab**

Upload `text2image.ipynb` to Google Colab or open it directly.

**2. Select a GPU Runtime**

```
Runtime → Change runtime type → Hardware accelerator → T4 GPU
```

**3. Run Cell 1 — Install Compatible Libraries**

```bash
pip install numpy==1.26.4 "huggingface_hub==0.25.2" "transformers==4.46.1" \
    "diffusers==0.31.0" "accelerate==0.34.2" "gradio==4.44.1" \
    onnxruntime-gpu insightface opencv-python peft safetensors
```

**4. Run Cell 2 — Launch the Engine**

This cell loads the `RealVisXL_V4.0` model (~6.94 GB download on first run), initializes the Gradio UI, and prints a public URL:

```
* Running on public URL: https://xxxxxxxxxxxxxxxx.gradio.live
```

Open the link in any browser to start generating images.

---

## 🖥️ Usage Guide

Once the Gradio interface is live:

1. **Enter a Prompt** — Describe what you want to generate in detail.
   - _Example: "A highly detailed 3D medical illustration of a human heart, glowing blue veins, dark background"_

2. **Choose a Style** — Select from the dropdown (e.g., `Digital Art`, `Cinematic`).

3. **Select an Aspect Ratio** — Pick the format that suits your use case.

4. **(Optional) Enter a Negative Prompt** — Things to exclude from the image.
   - _Default: `blurry, ugly, distorted, text, watermark`_

5. **(Optional) Expand Advanced Settings** — Adjust CFG Scale and inference steps.

6. **Click "Generate Image"** — The output appears on the right panel in ~15–45 seconds depending on GPU load.

---

## 🎭 Style Presets

Styles are implemented as **prompt wrappers** — they automatically append descriptors to your prompt for consistent aesthetic results.

| Style | Effect | Added Keywords |
|---|---|---|
| **None** | Raw prompt, no modification | — |
| **Cinematic** | Film-like quality, moody, bokeh | `emotional, harmonious, vignette, highly detailed, film grain` |
| **Digital Art** | Painterly, concept art feel | `digital artwork, illustrative, painterly, matte painting` |
| **Photographic** | Sharp, real-world photography | `35mm photograph, film, professional, 4k` |
| **Anime** | Japanese animation aesthetic | `anime style, key visual, vibrant, studio anime` |

Each style also injects a corresponding **negative prompt** to steer the model away from undesirable aesthetics (e.g., the Cinematic style discourages `anime, cartoon, graphic`).

---

## 📐 Supported Resolutions

| Name | Dimensions | Use Case |
|---|---|---|
| Square (1:1) | 1024 × 1024 | Social media, avatars |
| Portrait (3:4) | 896 × 1152 | Portraits, book covers |
| Portrait (9:16) | 768 × 1344 | Mobile wallpapers, stories |
| Landscape (4:3) | 1152 × 896 | Desktop wallpapers, presentations |
| Landscape (16:9) | 1344 × 768 | Cinematic widescreen |

All resolutions are optimized for SDXL's native resolution range to maximize quality.

---

## ⚙️ Advanced Settings

| Parameter | Range | Default | Description |
|---|---|---|---|
| **CFG Scale** | 1.0 – 15.0 | 7.5 | How strictly the model follows the prompt. Higher = more literal, lower = more creative. Values 6–9 work best for photorealism. |
| **Inference Steps** | 10 – 60 | 30 | Number of denoising steps. More steps = higher quality + slower generation. 25–35 is the sweet spot. |

---

## 💾 VRAM Optimization Notes

The engine employs two key VRAM optimizations to run on a **16GB T4 GPU**:

```python
self.pipe_t2i.vae.enable_tiling()   # Processes image in tiles to reduce peak VRAM
self.pipe_t2i.vae.enable_slicing()  # Processes batch items one at a time
```

Additionally, the memory allocator is configured at startup:

```python
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
```

This prevents PyTorch memory fragmentation, which is crucial for long Colab sessions where repeated generation can cause OOM errors.

After each generation, VRAM is explicitly cleared:

```python
torch.cuda.empty_cache()
gc.collect()
```

---

## ⚠️ Known Issues & Limitations

- **First run is slow** — The model (~6.94 GB) downloads from Hugging Face on the first execution. Subsequent runs in the same Colab session reuse the cached weights.
- **Gradio link expires in 1 week** — The public URL is temporary. Re-run the notebook cell to get a new one.
- **numpy version conflicts** — The pinned `numpy==1.26.4` may cause warnings with other Colab-preinstalled packages (e.g., JAX, rasterio). These are non-critical for this use case.
- **No HF_TOKEN set** — The notebook works without authentication for public models but benefits from setting a Hugging Face token for higher rate limits during download. Add it via _Colab Secrets_.
- **Single image per generation** — The current implementation generates one image at a time to conserve VRAM.

---

## 📁 Project Structure

```
text2image.ipynb
│
├── Cell 1: Dependency Installation
│   └── Pins compatible versions of all libraries
│
├── Cell 2: Dependency Installation (alternative)
│   └── Secondary install block for cross-checking
│
├── Cell 3: Core libraries update
│   └── Updates diffusers, transformers, accelerate
│
└── Cell 4: Main Engine + Gradio UI
    ├── STYLE_PRESETS dict       → Style presets
    ├── RESOLUTIONS dict         → Aspect ratio mappings
    ├── CustomImageEngine class  → Model loading & generation logic
    └── Gradio Blocks UI         → Web interface definition & launch
```

---

## 🙏 Credits & Acknowledgements

- **[SG161222/RealVisXL_V4.0](https://huggingface.co/SG161222/RealVisXL_V4.0)** — The underlying photorealistic SDXL model
- **[Hugging Face Diffusers](https://github.com/huggingface/diffusers)** — Pipeline framework
- **[Gradio](https://gradio.app/)** — Web UI framework

---

*Built for Google Colab · Requires T4 GPU or better · Python 3.12*
