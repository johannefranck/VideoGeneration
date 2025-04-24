import os
import argparse
from pathlib import Path
from PIL import Image
import torch
from torchvision.transforms.functional import to_tensor
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config

def load_model_from_config(config, ckpt_path):
    print(f"Loading model from {ckpt_path}")
    pl_sd = torch.load(ckpt_path, map_location="cpu")
    model = instantiate_from_config(config.model)
    model.load_state_dict(pl_sd["state_dict"], strict=False)
    model.cuda().eval()
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to source image (e.g., pred_128x128.png)")
    parser.add_argument("--output_dir", required=True, help="Where to save latents")
    parser.add_argument("--config", default="configs/stable-diffusion/v1-inference.yaml")
    parser.add_argument("--ckpt", default="chkpts/sd-v1-4.ckpt")
    parser.add_argument("--ddim_steps", type=int, default=250)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    config = OmegaConf.load(args.config)
    model = load_model_from_config(config, args.ckpt)

    # Load image and encode to latent
    img = Image.open(args.input).convert("RGB")
    img_tensor = to_tensor(img).unsqueeze(0).cuda() * 2 - 1  # Scale to [-1, 1]
    with torch.no_grad():
        z = model.encode_first_stage(img_tensor).mode() * 0.18215  # Latent

    # Simulate diffusion process (stay on CUDA, save to CPU)
    zt = z
    alphas_cumprod = model.alphas_cumprod.to(zt.device)  # Move all at once
    for t in range(args.ddim_steps):
        noise = torch.randn_like(zt)
        alpha = alphas_cumprod[t]
        zt = (alpha.sqrt() * z + (1 - alpha).sqrt() * noise)
        torch.save(zt.cpu(), output_dir / f"zt.{t:05}.pth")

    print(f"Saved {args.ddim_steps} latent steps to {output_dir}")

if __name__ == "__main__":
    main()
