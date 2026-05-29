# uncensored-ai-image-video-generator

A small PyQt desktop application for local image-generation experiments.

This repository currently contains two desktop apps:

- `unsensored_image_and_video_generator_ai_advanced.py` - real text-to-image generation through Hugging Face Diffusers and Stable Diffusion.
- `unsensored_image_and_video_generator_app.py` - a local PIL-based placeholder/demo that draws deterministic abstract images from a prompt. It does not run an AI model.

The Docker setup runs the advanced Diffusers app by default and exposes the desktop UI in your browser through noVNC.

## Current Maintenance Notes

Recent project updates:

- The desktop UI text was translated and normalized to English.
- The advanced app now launches generation through `generate_image_cli.py`, so the PyQt UI stays separate from the memory-heavy Diffusers process.
- Local Windows generation was verified with Python 3.11, PyTorch `2.9.0+cu126`, Diffusers `0.35.2`, Transformers `4.57.6`, and Hugging Face Hub `0.36.2`.
- A low-VRAM CUDA GPU path is handled explicitly. GPUs under 6 GB VRAM use CPU by default for stable output unless `SD_ALLOW_LOW_VRAM_CUDA=1` is set.
- Cached Hugging Face model files are reused from `./models/huggingface/hub`, and generated files are written to `./outputs`.

The repository intentionally ignores local model cache and generated images:

- `models/`
- `outputs/`

## Verified Project Claims

Supported by the current code:

- Text-to-image generation in the advanced app.
- Local image file output under `./outputs`.
- Prompt keyword safety checkboxes in the UI.
- Docker-based local workflow.

Not implemented in the current code:

- Text-to-video generation.
- Image-to-video generation.
- Reference-image workflows.
- Prompt-based image editing or transformation of an existing image.
- A verified one-command Windows installer.

Important limitations:

- The `hd` and `4k` quality labels do not create true 4K images. In the advanced app, `standard` uses 512x512 and both `hd` and `4k` use 768x768.
- The advanced app disables the built-in Diffusers safety checker and uses only a simple keyword filter before generation. This is not robust moderation.
- CPU generation can be very slow. A single image may take several minutes or longer depending on your machine.

## Is It Free?

The project source code is MIT licensed.

The advanced app does not call a paid hosted image API. It downloads model weights from Hugging Face and runs inference locally, so there is no per-image API charge from this app. You still pay for your own hardware, electricity, disk space, and internet bandwidth. Hugging Face-hosted APIs, Spaces, and Inference Endpoints have their own pricing and limits if you choose to use those separately.

The Stable Diffusion model weights are governed by their model licenses, which include use restrictions. Review the license for whichever model you run.

## How It Works

The advanced app flow is:

1. The PyQt UI collects a text prompt and a quality setting.
2. The selected safety checkboxes run a simple keyword match against the prompt.
3. On Generate, the app starts `generate_image_cli.py` as an isolated subprocess.
4. The CLI imports `diffusers` and `torch`, then selects a Stable Diffusion model:
   - `standard` and `hd`: `stable-diffusion-v1-5/stable-diffusion-v1-5`
   - `4k`: `stabilityai/stable-diffusion-2-1`
5. Diffusers downloads missing model files into `./models/huggingface/hub` on first use.
6. If the model files are already cached, the app loads them locally and skips the Hugging Face network probe.
7. PyTorch runs the denoising pipeline on CUDA when available and allowed, otherwise on CPU.
8. The generated PNG is saved to `./outputs`.

You can override the models with environment variables:

- `SD_MODEL_STANDARD`
- `SD_MODEL_HD`
- `SD_MODEL_4K`

Useful runtime environment variables:

- `SD_DEVICE=cpu` - force CPU generation.
- `SD_ALLOW_LOW_VRAM_CUDA=1` - try CUDA even on GPUs under 6 GB VRAM.
- `SD_MODEL_VARIANT=none` - use the default model variant instead of `fp16`.
- `SD_STEPS=15` - override the number of inference steps.
- `SD_GUIDANCE_SCALE=7.5` - override guidance scale.

## Run With Docker

Prerequisites:

- Docker Desktop
- At least 8 GB RAM available to Docker; more is better
- Internet access on first generation so the model can be downloaded

Build and start:

```powershell
docker compose up --build
```

Open the app:

```text
http://localhost:6080/vnc.html
```

Click Connect in noVNC. No password is configured.

Generated images are written to:

```text
./outputs
```

Stop the app:

```powershell
docker compose down
```

Reset the downloaded model cache:

```powershell
docker compose down -v
```

## Run Directly on Windows

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python .\unsensored_image_and_video_generator_ai_advanced.py
```

The first AI generation run downloads the selected Stable Diffusion model, so keep the terminal open and expect it to take time. The current `requirements.txt` uses the PyTorch CUDA 12.6 wheel index, but CUDA can still be disabled automatically when the detected GPU has too little VRAM for stable generation.

## Future Updates

Planned or useful next improvements:

- Add a clearer first-run model download progress screen.
- Add a model manager for choosing and validating local Stable Diffusion checkpoints.
- Add a true CPU-only requirements file for machines without CUDA.
- Improve generation cancellation and cleanup from the UI.
- Add tests around model selection, cache detection, and safety-filter behavior.
- Replace the placeholder/demo app or clearly separate it from the Diffusers app.
