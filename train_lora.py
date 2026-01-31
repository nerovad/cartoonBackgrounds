import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import os

from diffusers import StableDiffusionPipeline, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from peft import get_peft_model, LoraConfig, TaskType

# --- CONFIG ---
model_id = "runwayml/stable-diffusion-v1-5"
dataset_dir = Path("./dataset")
output_dir = Path("./output/lora-model")
logging_dir = Path("./output/logs")

image_size = 512
batch_size = 1
num_epochs = 10
lr = 1e-5
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- DATASET ---
class CartoonDataset(Dataset):
    def __init__(self, root):
        self.root = Path(root)
        self.image_paths = list(self.root.glob("*.png")) + list(self.root.glob("*.jpg"))
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

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
        else:
            caption = "a cartoon background"

        return {"pixel_values": image, "caption": caption}

# --- LOAD MODELS ---
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
).to(device)

unet = pipe.unet
tokenizer = pipe.tokenizer
text_encoder = pipe.text_encoder
vae = pipe.vae
scheduler = pipe.scheduler

pipe.vae.to("cpu", dtype=torch.float32)

# --- APPLY LORA ---
lora_config = LoraConfig(
    r=4,
    lora_alpha=16,
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    lora_dropout=0.1,
    bias="none",
    task_type="UNET",
)

unet = get_peft_model(unet, lora_config)
unet.train()

# Freeze non-LoRA layers
for name, param in unet.named_parameters():
    if "lora" not in name:
        param.requires_grad = False

# --- PREP DATA ---
dataset = CartoonDataset(dataset_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, unet.parameters()), lr=lr)


# --- TRAIN LOOP ---
for epoch in range(num_epochs):
    for i, batch in enumerate(dataloader):
        pixel_values = batch["pixel_values"].to(device, dtype=torch.float16)

        print(f"Pixel values stats - min: {pixel_values.min()}, max: {pixel_values.max()}, mean: {pixel_values.mean()}")
        prompts = batch["caption"]

        # Tokenize text
        inputs = tokenizer(prompts, padding="max_length", truncation=True, max_length=77, return_tensors="pt")
        input_ids = inputs.input_ids.to(device)
        text_embeddings = text_encoder(input_ids=input_ids)[0]

        # Encode image to latent
        with torch.no_grad():
            latents = pipe.vae.encode(pixel_values.cpu().float()).latent_dist.sample()

        if torch.isnan(latents).any() or torch.isinf(latents).any():
            print("⚠️ Invalid latents detected — skipping.")
            continue

        latents = latents.to(device).clamp(-0.9, 0.9) * 0.18215
        latents = latents.float()

        noise = torch.randn_like(latents, dtype=torch.float32)
        timesteps = torch.randint(0, 1000, (latents.shape[0],), device=device).long()

        noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
        # Predict noise

        noise_pred = unet(
            sample=noisy_latents.half(),  # UNet expects float16
            timestep=timesteps,
            encoder_hidden_states=text_embeddings
        ).sample.float() 

        # Loss
        loss = torch.nn.functional.mse_loss(noise_pred, noise)


        diff = noise_pred - noise
        print(f"Diff stats — min: {diff.min()}, max: {diff.max()}, mean: {diff.mean()}")

        if torch.isnan(loss):
            print("NaN loss detected — skipping this step.")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        print(f"Epoch {epoch+1} | Step {i+1}/{len(dataloader)} | Loss: {loss.item():.4f}")

    # Save checkpoint
    unet.save_pretrained(output_dir / f"epoch-{epoch+1}")

print("Training complete.")
