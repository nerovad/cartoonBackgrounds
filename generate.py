import argparse
import torch
from pathlib import Path
from safetensors.torch import load_file

from diffusers import StableDiffusionPipeline
from peft import LoraConfig, get_peft_model


def load_pipeline(model_id, lora_path, device):
    """Load base SD model and apply trained LoRA weights."""
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to(device)

    # Apply LoRA config then load the trained weights
    lora_config = LoraConfig(
        r=4,
        lora_alpha=16,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.0,
        bias="none",
    )
    pipe.unet = get_peft_model(pipe.unet, lora_config)
    state_dict = load_file(Path(lora_path) / "adapter_model.safetensors")
    pipe.unet.load_state_dict(state_dict, strict=False)
    pipe.unet.eval()

    return pipe


def generate(pipe, prompt, output_path, num_images=1, steps=50, guidance=7.5, seed=None):
    """Generate cartoon background images from a text prompt."""
    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)

    images = pipe(
        prompt,
        num_images_per_prompt=num_images,
        num_inference_steps=steps,
        guidance_scale=guidance,
        width=512,
        height=512,
        generator=generator,
    ).images

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find the next available index
    existing = sorted(output_path.glob("output_*.png"))
    if existing:
        last = int(existing[-1].stem.split("_")[1])
    else:
        last = 0

    saved = []
    for i, img in enumerate(images):
        name = output_path / f"output_{last + i + 1}.png"
        img.save(name)
        saved.append(name)
        print(f"Saved: {name}")

    return saved


def main():
    parser = argparse.ArgumentParser(description="Generate cartoon backgrounds using a trained LoRA model")
    parser.add_argument("prompt", type=str, help="Text prompt describing the background to generate")
    parser.add_argument("--model", type=str, default="runwayml/stable-diffusion-v1-5",
                        help="Base Stable Diffusion model ID (default: runwayml/stable-diffusion-v1-5)")
    parser.add_argument("--lora", type=str, default="./output/lora-model/epoch-10",
                        help="Path to trained LoRA adapter directory (default: ./output/lora-model/epoch-10)")
    parser.add_argument("--output", type=str, default="./output/generated",
                        help="Output directory for generated images (default: ./output/generated)")
    parser.add_argument("--num-images", type=int, default=1,
                        help="Number of images to generate (default: 1)")
    parser.add_argument("--steps", type=int, default=50,
                        help="Number of diffusion steps (default: 50)")
    parser.add_argument("--guidance", type=float, default=7.5,
                        help="Classifier-free guidance scale (default: 7.5)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    lora_path = Path(args.lora)
    if not lora_path.exists():
        # Fall back to the latest available epoch
        lora_base = Path("./output/lora-model")
        epochs = sorted(lora_base.glob("epoch-*"), key=lambda p: int(p.name.split("-")[1]))
        if not epochs:
            print(f"Error: No LoRA checkpoints found at {args.lora} or in {lora_base}")
            return
        lora_path = epochs[-1]
        print(f"Requested checkpoint not found, using latest: {lora_path}")

    print(f"Loading model: {args.model}")
    print(f"Loading LoRA adapter: {lora_path}")
    pipe = load_pipeline(args.model, str(lora_path), device)

    print(f"Generating {args.num_images} image(s) for prompt: \"{args.prompt}\"")
    generate(pipe, args.prompt, args.output, args.num_images, args.steps, args.guidance, args.seed)
    print("Done.")


if __name__ == "__main__":
    main()
