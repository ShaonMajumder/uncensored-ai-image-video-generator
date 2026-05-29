# uncensored-ai-image-video-generator

A small PyQt desktop application for local image-generation experiments.

This repository currently contains two desktop apps:

- `unsensored_image_and_video_generator_ai_advanced.py` - real text-to-image generation through Hugging Face Diffusers and Stable Diffusion.
- `unsensored_image_and_video_generator_app.py` - a local PIL-based placeholder/demo that draws deterministic abstract images from a prompt. It does not run an AI model.

The Docker setup runs the advanced Diffusers app by default and exposes the desktop UI in your browser through noVNC.

## Verified Project Claims

Supported by the current code:

- Text-to-image generation in the advanced app.
- Local image file output under the user's `Pictures/ImageGenerator` directory, mapped to `./outputs` when using Docker Compose.
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
3. On Generate, the app imports `diffusers` and `torch`.
4. It selects a Stable Diffusion model:
   - `standard` and `hd`: `stable-diffusion-v1-5/stable-diffusion-v1-5`
   - `4k`: `stabilityai/stable-diffusion-2-1`
5. Diffusers downloads the model into the Hugging Face cache on first use.
6. PyTorch runs the denoising pipeline on CPU in the provided Docker image.
7. The generated PNG is saved to `Pictures/ImageGenerator`.

You can override the models with environment variables:

- `SD_MODEL_STANDARD`
- `SD_MODEL_HD`
- `SD_MODEL_4K`

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

The first AI generation run downloads the selected Stable Diffusion model, so keep the terminal open and expect it to take time. The included `requirements.txt` installs a CPU-only PyTorch build. If you want GPU acceleration, install the PyTorch build that matches your CUDA setup before running the app.
