import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Set which GPU to use

import torch
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from torchvision import utils
from torchvision.transforms.functional import to_tensor
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim_with_grad import DDIMSamplerWithGrad
from losses import FlowLoss
import shutil
from torch.amp import autocast 

# Suppress all warnings from HuggingFace transformers
from transformers import logging
logging.set_verbosity_error()   

def load_cli_config():
    # Load YAML config passed as CLI argument
    if len(sys.argv) < 2:
        print("Usage: python generate.py <path_to_yaml>")
        sys.exit(1)
    config_path = sys.argv[1]
    if not Path(config_path).is_file():
        print(f"Error: configuration file '{config_path}' not found.")
        sys.exit(1)
    return OmegaConf.load(config_path)

def load_model_from_config(cfg, ckpt, verbose=False):
    # Load model from checkpoint and config
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(cfg.model)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if verbose and missing:
        print("Missing keys:\n", missing)
    if verbose and unexpected:
        print("Unexpected keys:\n", unexpected)
    model.cuda().eval()
    return model

def prepare_environment(cfg):
    # Create necessary folders and save config
    input_dir = Path(cfg.input_dir)
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(cfg, save_dir / 'config.pth')
    print(cfg)
    return input_dir, save_dir

def initialize_model_and_sampler(cfg):
    # Initialize the diffusion model and DDIM sampler
    sd_cfg = OmegaConf.load(cfg.sd_model_config)
    model = load_model_from_config(sd_cfg, cfg.ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.DataParallel(model.to(device), device_ids=list(range(torch.cuda.device_count())))
    model.eval()
    sampler = DDIMSamplerWithGrad(model)
    torch.set_grad_enabled(False)
    return model, sampler

def load_guidance_data(cfg, input_dir):
    # Load input image, mask, cached latents, guidance flow, and compute guidance energy
    img_path = input_dir / 'pred.png'
    src_img = to_tensor(Image.open(img_path))[None] * 2 - 1
    src_img = src_img.cuda()
    start_zt = None if cfg.no_init_startzt else torch.load(input_dir / 'start_zt.pth')

    if cfg.edit_mask_path:
        mask_path = input_dir / 'masks' / cfg.edit_mask_path
        edit_mask = torch.load(mask_path)
    else:
        print("No edit mask provided, using all pixels.")
        edit_mask = torch.zeros(1,4,64,64).bool()

    guidance_schedule = np.load(cfg.guidance_schedule_path) if cfg.guidance_schedule_path else None
    cached_latents = None
    if cfg.use_cached_latents:
        latents = [torch.load(input_dir / 'latents' / f'zt.{i:05}.pth') for i in range(cfg.ddim_steps)]
        cached_latents = torch.stack(latents)

    flow = torch.load(input_dir / 'flows' / cfg.target_flow_name) if cfg.target_flow_name else None
    if flow is not None:
        flow = flow.float().cuda()

    guidance_energy = FlowLoss(cfg.color_weight, cfg.flow_weight,
                               oracle=cfg.oracle_flow, target_flow=flow,
                               occlusion_masking=not cfg.no_occlusion_masking).cuda()

    return src_img, start_zt, edit_mask, guidance_schedule, cached_latents, guidance_energy

def run_sampling(cfg, model, sampler, data, save_dir) -> torch.Tensor:
    # Perform diffusion sampling and save images, logs, and latent zt
    src_img, start_zt, edit_mask, guidance_schedule, cached_latents, guidance_energy = data
    uncond_embed = model.module.get_learned_conditioning([""])
    cond_embed = model.module.get_learned_conditioning([cfg.prompt])
    prefix = Path(cfg.target_flow_name).stem

    for idx in range(cfg.num_samples):
        print(f"Sampling {idx + 1}/{cfg.num_samples} for {prefix}")
        with autocast(device_type="cuda", enabled=getattr(cfg, "mixed_precision", False)):
            sample, start_zt, info = sampler.sample(
                num_ddim_steps=cfg.ddim_steps,
                cond_embed=cond_embed,
                uncond_embed=uncond_embed,
                batch_size=1,
                shape=[4, 64, 64],
                CFG_scale=cfg.scale,
                eta=cfg.ddim_eta,
                src_img=src_img,
                start_zt=start_zt,
                guidance_schedule=guidance_schedule,
                cached_latents=cached_latents,
                edit_mask=edit_mask,
                num_recursive_steps=cfg.num_recursive_steps,
                clip_grad=cfg.clip_grad,
                guidance_weight=cfg.guidance_weight,
                log_freq=cfg.log_freq,
                results_folder=None,
                guidance_energy=guidance_energy
            )

        # Decode image from latent
        img = model.module.decode_first_stage(sample)
        img = torch.clamp((img + 1.0) / 2.0, 0.0, 1.0)

        # Save result
        png_name = f"pred_{prefix}_{idx:03}.png" if cfg.num_samples > 1 else f"pred_{prefix}.png"
        save_dir.mkdir(parents=True, exist_ok=True)
        utils.save_image(img, save_dir / png_name)

        # Save guidance logs
        for key in ['losses','losses_flow','losses_color','noise_norms','guidance_norms']:
            np_name = f"{key}_{prefix}_{idx:03}.npy" if cfg.num_samples > 1 else f"{key}_{prefix}.npy"
            np.save(save_dir / np_name, info[key])

        # Save updated latent
        zt_name = f"start_zt_{prefix}_{idx:03}.pth" if cfg.num_samples > 1 else f"start_zt_{prefix}.pth"
        torch.save(start_zt, save_dir / zt_name)

    return start_zt

def make_results_gif(results_dir: str, gif_name: str = "flows.gif", duration: int = 500):
    # Create a looping GIF from initial_pred and pred images
    results_path = Path(results_dir)
    init_files = list(results_path.glob("initial_pred_*.png"))
    pred_files = list(results_path.glob("pred_*.png"))

    def idx_of(p: Path):
        stem = p.stem.split("initial_pred_")[-1].split("pred_")[-1]
        return int(stem.split("_")[-1])

    frames = []
    for idx in sorted({idx_of(p) for p in init_files + pred_files}):
        init_p = results_path / f"initial_pred_flow_{idx}.png"
        pred_p = results_path / f"pred_flow_{idx}.png"
        if init_p.exists():
            frames.append(Image.open(init_p))
        if pred_p.exists():
            frames.append(Image.open(pred_p))

    if not frames:
        raise RuntimeError(f"No suitable PNGs found in {results_dir}")

    save_path = results_path / gif_name
    frames[0].save(save_path, format="GIF", save_all=True, append_images=frames[1:], duration=duration, loop=0)
    print(f"GIF saved to {save_path}")

def main():
    # Entry point of the generation pipeline
    cfg = load_cli_config()
    input_dir, output_root = prepare_environment(cfg)
    model, sampler = initialize_model_and_sampler(cfg)

    # Copy original prediction image to output for reference
    orig = input_dir / "pred.png"
    if orig.exists():
        output_root.mkdir(parents=True, exist_ok=True)
        shutil.copy(orig, output_root / "initial_pred.png")

    flows_dir = input_dir / "flows"
    masks_dir = input_dir / "masks"

    flows = sorted(flows_dir.glob("flow_*.pth"))
    masks = {p.stem.split("_",1)[1]: p for p in masks_dir.glob("mask_*.pth")}

    if not flows:
        print(f"No flow_*.pth files found in {flows_dir}")
        return

    # Ensure initial latent exists or generate it
    start_zt_path = input_dir / 'start_zt.pth'
    if start_zt_path.exists():
        print("Found start_zt.pth, loading...")
        orig_start_zt = torch.load(start_zt_path).cuda()
    else:
        print("start_zt.pth not found, generating from pred.png...")
        src_img = to_tensor(Image.open(input_dir / 'pred.png'))[None] * 2 - 1
        src_img = src_img.cuda()
        with torch.no_grad():
            z = model.module.get_first_stage_encoding(model.module.encode_first_stage(src_img))
        torch.save(z, start_zt_path)
        orig_start_zt = z

    # Process each flow-mask pair
    for flow_path in flows:
        idx = flow_path.stem.split("_", 1)[1]
        mask_path = masks.get(idx, None)
        if mask_path is None:
            print(f"Warning: no matching mask_{idx}.pth for {flow_path.name}, skipping.")
            continue

        print(f"\n=== Chaining flow_{idx} → mask_{idx} ===")
        cfg.target_flow_name = flow_path.name
        cfg.edit_mask_path = mask_path.name

        src_img, _, edit_mask, guidance_schedule, cached_latents, guidance_energy = \
            load_guidance_data(cfg, input_dir)

        start_zt = orig_start_zt
        start_zt = run_sampling(cfg, model, sampler,
                                 (src_img, start_zt, edit_mask,
                                  guidance_schedule, cached_latents,
                                  guidance_energy),
                                 output_root)

    print(f"\nAll done! Results are in {output_root}")
    make_results_gif(output_root, gif_name="flows.gif", duration=300)

if __name__ == '__main__':
    main()
