import math
import torch
import logging
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
from PIL import Image
from pathlib import Path

from diffusers import StableDiffusionPipeline
from peft import get_peft_model, LoraConfig

# --- LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# --- CONFIG ---
model_id = "runwayml/stable-diffusion-v1-5"
dataset_dir = Path("./dataset")
output_dir = Path("./output/lora-model")
sample_dir = Path("./output/samples")

image_size = 512
batch_size = 1
gradient_accumulation_steps = 8  # effective batch size = 8
num_epochs = 10
lr = 1e-4
warmup_steps = 50
device = "cuda" if torch.cuda.is_available() else "cpu"

# Prompts used to generate validation samples after each epoch
validation_prompts = [
    "interior, living room, 90s cartoon style, medium shot",
    "exterior, park, 90s cartoon style, wide shot",
    "interior, kitchen, 90s cartoon style, close up",
    "exterior, street, 90s cartoon style, high angle",
]

# --- DATASET ---
class CartoonDataset(Dataset):
    def __init__(self, root):
        self.root = Path(root)
        self.image_paths = sorted(
            list(self.root.glob("*.png")) + list(self.root.glob("*.jpg"))
        )
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        # Filter out images that can't be loaded
        valid = []
        for p in self.image_paths:
            try:
                Image.open(p).verify()
                valid.append(p)
            except Exception as e:
                logger.warning(f"Skipping invalid image {p.name}: {e}")
        self.image_paths = valid
        logger.info(f"Loaded {len(self.image_paths)} valid images from {root}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        caption_path = img_path.with_suffix(".txt")

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        if caption_path.exists():
            with open(caption_path, "r") as f:
                caption = f.read().strip()
                # Strip the BG-XXXX prefix — the model doesn't need the ID
                parts = caption.split(", ", 1)
                if len(parts) > 1 and parts[0].startswith("BG-"):
                    caption = parts[1]
        else:
            caption = "a cartoon background, 90s cartoon style"

        return {"pixel_values": image, "caption": caption}


# --- LOAD MODELS ---
logger.info(f"Loading base model: {model_id}")
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
).to(device)

unet = pipe.unet
tokenizer = pipe.tokenizer
text_encoder = pipe.text_encoder
vae = pipe.vae
scheduler = pipe.scheduler

# Freeze text encoder and VAE — we only train LoRA on the UNet
text_encoder.requires_grad_(False)
vae.requires_grad_(False)

# Move VAE to CPU in float32 for encoding (saves GPU memory)
vae.to("cpu", dtype=torch.float32)

# --- APPLY LORA ---
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    lora_dropout=0.05,
    bias="none",
    task_type=None,
)

unet = get_peft_model(unet, lora_config)
unet.train()

# Freeze non-LoRA layers
for name, param in unet.named_parameters():
    if "lora" not in name:
        param.requires_grad = False

trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
total = sum(p.numel() for p in unet.parameters())
logger.info(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

# --- PREP DATA ---
dataset = CartoonDataset(dataset_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, unet.parameters()), lr=lr, weight_decay=1e-2)

# Cosine LR schedule with warmup
total_steps = (len(dataloader) // gradient_accumulation_steps) * num_epochs

def lr_lambda(step):
    if step < warmup_steps:
        return step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))

lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

scaler = GradScaler()

# --- VALIDATION ---
@torch.no_grad()
def generate_samples(epoch_num):
    """Generate validation samples to check training progress."""
    sample_dir.mkdir(parents=True, exist_ok=True)
    unet.eval()

    # Temporarily move VAE back to GPU for generation
    pipe.vae = vae.to(device, dtype=torch.float16)
    pipe.unet = unet

    for i, prompt in enumerate(validation_prompts):
        image = pipe(
            prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            width=512,
            height=512,
        ).images[0]
        out_path = sample_dir / f"epoch-{epoch_num}_sample-{i+1}.png"
        image.save(out_path)
        logger.info(f"  Saved sample: {out_path.name} — \"{prompt}\"")

    # Move VAE back to CPU
    vae.to("cpu", dtype=torch.float32)
    pipe.vae = vae
    unet.train()


# --- TRAIN LOOP ---
logger.info(f"Starting training: {num_epochs} epochs, {len(dataloader)} steps/epoch, "
            f"grad accum {gradient_accumulation_steps}, effective batch {batch_size * gradient_accumulation_steps}")

global_step = 0
optimizer.zero_grad()

for epoch in range(num_epochs):
    epoch_loss = 0.0
    num_batches = 0

    for i, batch in enumerate(dataloader):
        pixel_values = batch["pixel_values"].to("cpu", dtype=torch.float32)
        prompts = batch["caption"]

        # Encode image to latent space (VAE is on CPU)
        with torch.no_grad():
            latent_dist = vae.encode(pixel_values).latent_dist
            latents = latent_dist.sample() * vae.config.scaling_factor

        if torch.isnan(latents).any() or torch.isinf(latents).any():
            logger.warning(f"Invalid latents at epoch {epoch+1} step {i+1} — skipping.")
            continue

        latents = latents.to(device, dtype=torch.float32)

        # Tokenize and encode text
        inputs = tokenizer(
            prompts, padding="max_length", truncation=True,
            max_length=77, return_tensors="pt"
        )
        input_ids = inputs.input_ids.to(device)
        with torch.no_grad():
            text_embeddings = text_encoder(input_ids=input_ids)[0]

        # Add noise
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
        noisy_latents = scheduler.add_noise(latents, noise, timesteps)

        # Predict noise with mixed precision
        with autocast():
            noise_pred = unet(
                sample=noisy_latents.half(),
                timestep=timesteps,
                encoder_hidden_states=text_embeddings,
            ).sample

            loss = torch.nn.functional.mse_loss(noise_pred.float(), noise)
            loss = loss / gradient_accumulation_steps

        scaler.scale(loss).backward()

        if (i + 1) % gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            lr_scheduler.step()
            global_step += 1

        step_loss = loss.item() * gradient_accumulation_steps
        epoch_loss += step_loss
        num_batches += 1

        if (i + 1) % 10 == 0 or (i + 1) == len(dataloader):
            current_lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} | Step {i+1}/{len(dataloader)} | "
                f"Loss: {step_loss:.4f} | LR: {current_lr:.2e}"
            )

    avg_loss = epoch_loss / max(num_batches, 1)
    logger.info(f"Epoch {epoch+1} complete — Avg Loss: {avg_loss:.4f}")

    # Save checkpoint
    save_path = output_dir / f"epoch-{epoch+1}"
    save_path.mkdir(parents=True, exist_ok=True)
    unet.save_pretrained(save_path)
    logger.info(f"Saved checkpoint: {save_path}")

    # Generate validation samples
    logger.info(f"Generating validation samples for epoch {epoch+1}...")
    generate_samples(epoch + 1)

logger.info("Training complete.")
