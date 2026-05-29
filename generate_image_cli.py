#!/usr/bin/env python3
import argparse
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_CACHE_DIR = PROJECT_DIR / "models" / "huggingface"
MODEL_HUB_CACHE_DIR = MODEL_CACHE_DIR / "hub"
OUTPUT_DIR = PROJECT_DIR / "outputs"

MODEL_HUB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(MODEL_CACHE_DIR)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(MODEL_HUB_CACHE_DIR)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def log(message):
    print(message, flush=True)


def model_for_quality(quality):
    return {
        "standard": os.getenv("SD_MODEL_STANDARD", "stable-diffusion-v1-5/stable-diffusion-v1-5"),
        "hd": os.getenv("SD_MODEL_HD", "stable-diffusion-v1-5/stable-diffusion-v1-5"),
        "4k": os.getenv("SD_MODEL_4K", "stabilityai/stable-diffusion-2-1"),
    }.get(quality, os.getenv("SD_MODEL_STANDARD", "stable-diffusion-v1-5/stable-diffusion-v1-5"))


def env_flag(name):
    return os.getenv(name, "").lower() in {"1", "true", "yes", "on"}


def cached_snapshot_dir(model_name):
    repo_dir = MODEL_HUB_CACHE_DIR / f"models--{model_name.replace('/', '--')}"
    ref_file = repo_dir / "refs" / "main"
    if not ref_file.exists():
        return None

    revision = ref_file.read_text(encoding="utf-8").strip()
    snapshot_dir = repo_dir / "snapshots" / revision
    if (snapshot_dir / "model_index.json").exists():
        return snapshot_dir

    return None


def has_cached_pipeline(model_name, model_variant):
    snapshot_dir = cached_snapshot_dir(model_name)
    if snapshot_dir is None:
        return False

    suffix = f".{model_variant}.safetensors" if model_variant else ".safetensors"
    required_files = [
        "model_index.json",
        "scheduler/scheduler_config.json",
        "text_encoder/config.json",
        f"text_encoder/model{suffix}",
        "tokenizer/merges.txt",
        "tokenizer/vocab.json",
        "unet/config.json",
        f"unet/diffusion_pytorch_model{suffix}",
        "vae/config.json",
        f"vae/diffusion_pytorch_model{suffix}",
    ]

    return all((snapshot_dir / file_name).exists() for file_name in required_files)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--quality", default="standard", choices=["standard", "hd", "4k"])
    args = parser.parse_args()

    try:
        from transformers.utils import import_utils as transformers_import_utils

        transformers_import_utils._sklearn_available = False

        import torch
        from diffusers import AutoencoderKL, StableDiffusionPipeline

        model_name = model_for_quality(args.quality)
        force_cpu = os.getenv("SD_DEVICE", "").lower() == "cpu"
        allow_low_vram_cuda = env_flag("SD_ALLOW_LOW_VRAM_CUDA")
        device = "cuda" if torch.cuda.is_available() and not force_cpu else "cpu"

        vram_gb = 0
        if device == "cuda":
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / 1024**3
            if vram_gb < 6 and not allow_low_vram_cuda:
                log(f"Low-VRAM CUDA GPU detected: {props.name} ({vram_gb:.1f} GB VRAM)")
                log("Using CPU for stable output. Set SD_ALLOW_LOW_VRAM_CUDA=1 to try CUDA.")
                device = "cpu"
            else:
                log(f"Device: CUDA - {props.name} ({vram_gb:.1f} GB VRAM)")

        if device == "cpu":
            log("Device: CPU")

        log(f"Model cache: {MODEL_HUB_CACHE_DIR}")
        log(f"Loading model: {model_name}")

        model_variant = os.getenv("SD_MODEL_VARIANT", "fp16")
        if model_variant and model_variant.lower() in {"none", "default", "false", "0"}:
            model_variant = None
        log(f"Model variant: {model_variant or 'default'}")
        local_files_only = env_flag("HF_HUB_OFFLINE") or has_cached_pipeline(
            model_name, model_variant
        )
        if local_files_only:
            log("Using cached model files.")
        else:
            log("Downloading missing model files if needed. First run can take a long time.")

        pipe = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            use_safetensors=True,
            variant=model_variant,
            cache_dir=str(MODEL_HUB_CACHE_DIR),
            safety_checker=None,
            local_files_only=local_files_only,
        )
        pipe.enable_attention_slicing()

        if device == "cuda" and vram_gb and vram_gb < 6:
            log("Low-VRAM GPU detected; using CPU offload to avoid CUDA memory errors.")
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(device)

        size = {"standard": 512, "hd": 768, "4k": 768}.get(args.quality, 512)
        if device == "cuda" and vram_gb and vram_gb < 6 and size > 512:
            log("Low-VRAM GPU detected; using 512x512 output for this run.")
            size = 512

        default_steps = "15" if device == "cpu" else "25"
        steps = int(os.getenv("SD_STEPS", default_steps))
        guidance_scale = float(os.getenv("SD_GUIDANCE_SCALE", "7.5"))
        log(f"Generating image at {size}x{size} for {steps} steps...")

        reported_steps = set()

        def on_step_end(_pipeline, step_index, _timestep, callback_kwargs):
            step_number = step_index + 1
            if 1 <= step_number <= steps and step_number not in reported_steps:
                reported_steps.add(step_number)
                log(f"Step {step_number}/{steps}")
            return callback_kwargs

        with torch.no_grad():
            if device == "cuda":
                latents = pipe(
                    args.prompt,
                    height=size,
                    width=size,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    output_type="latent",
                    callback_on_step_end=on_step_end,
                ).images[0]
            else:
                image = pipe(
                    args.prompt,
                    height=size,
                    width=size,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    callback_on_step_end=on_step_end,
                ).images[0]

        if device == "cuda":
            log("Decoding image on CPU to avoid CUDA VAE crashes.")
            vae = AutoencoderKL.from_pretrained(
                model_name,
                subfolder="vae",
                torch_dtype=torch.float32,
                use_safetensors=True,
                cache_dir=str(MODEL_HUB_CACHE_DIR),
            ).to("cpu")
            vae.enable_slicing()

            latents = latents.detach().to("cpu", dtype=torch.float32).unsqueeze(0)
            scaling_factor = getattr(vae.config, "scaling_factor", 0.18215)
            with torch.no_grad():
                decoded = vae.decode(latents / scaling_factor, return_dict=False)[0]
            image = pipe.image_processor.postprocess(decoded, output_type="pil")[0]

        extrema = image.convert("RGB").getextrema()
        if all(low == high for low, high in extrema):
            raise RuntimeError(
                "The model returned a blank image. If you forced CUDA, clear "
                "SD_ALLOW_LOW_VRAM_CUDA and run again so the CPU path is used."
            )

        filename = OUTPUT_DIR / f"image_ai_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        image.save(filename)

        log(f"Image saved: {filename}")
        log(f"RESULT_PATH={filename}")
        return 0
    except Exception as exc:
        traceback.print_exc()
        log(f"ERROR={type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
