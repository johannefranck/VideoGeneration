import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision import utils
from torchvision.transforms.functional import to_tensor
from omegaconf import OmegaConf
from pathlib import Path
from PIL import Image
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim_with_grad import DDIMSamplerWithGrad
from losses import FlowLoss


def load_cli_config(config_path: str = "cli_config.yaml"):
    """
    Load configuration from a YAML file.
    """
    if not Path(config_path).is_file():
        print(f"Error: configuration file '{config_path}' not found.")
        sys.exit(1)
    return OmegaConf.load(config_path)


def load_model_from_config(cfg, ckpt, verbose=False):
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
    input_dir = Path(cfg.input_dir)
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(cfg, save_dir / 'config.pth')
    print(cfg)
    return input_dir, save_dir


def initialize_model_and_sampler(cfg):
    sd_cfg = OmegaConf.load(cfg.sd_model_config)
    model = load_model_from_config(sd_cfg, cfg.ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.DataParallel(model.to(device), device_ids=list(range(torch.cuda.device_count())))
    model.eval()
    sampler = DDIMSamplerWithGrad(model)
    torch.set_grad_enabled(False)
    return model, sampler


def load_guidance_data(cfg, input_dir):
    # Source image
    img_path = input_dir / 'pred.png'
    src_img = to_tensor(Image.open(img_path))[None] * 2 - 1
    src_img = src_img.cuda()

    # Initial latent
    start_zt = None if cfg.no_init_startzt else torch.load(input_dir / 'start_zt.pth')

    # Edit mask
    if cfg.edit_mask_path:
        mask_path = input_dir / 'flows' / cfg.edit_mask_path
        edit_mask = torch.load(mask_path)
    else:
        edit_mask = torch.zeros(1,4,64,64).bool()

    # Guidance schedule
    guidance_schedule = np.load(cfg.guidance_schedule_path) if cfg.guidance_schedule_path else None

    # Cached latents
    cached_latents = None
    if cfg.use_cached_latents:
        latents = [torch.load(input_dir / 'latents' / f'zt.{i:05}.pth') for i in range(cfg.ddim_steps)]
        cached_latents = torch.stack(latents)

    # Target flow
    flow = torch.load(input_dir / 'flows' / cfg.target_flow_name) if cfg.target_flow_name else None

    if flow is not None:
    # ensure it’s float32 so requires_grad works
        flow = flow.float().cuda()

    guidance_energy = FlowLoss(
        cfg.color_weight,
        cfg.flow_weight,
        oracle=cfg.oracle_flow,
        target_flow=flow,
        occlusion_masking=not cfg.no_occlusion_masking
    ).cuda()

    return src_img, start_zt, edit_mask, guidance_schedule, cached_latents, guidance_energy


def run_sampling(cfg, model, sampler, data, save_dir):
    src_img, start_zt, edit_mask, guidance_schedule, cached_latents, guidance_energy = data
    uncond_embed = model.module.get_learned_conditioning([""])
    cond_embed = model.module.get_learned_conditioning([cfg.prompt])

    for idx in range(cfg.num_samples):
        print(f"Sampling {idx + 1}/{cfg.num_samples}")
        out_dir = save_dir / f'sample_{idx:03}'
        out_dir.mkdir(parents=True, exist_ok=True)

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
            results_folder=out_dir,
            guidance_energy=guidance_energy
        )

        # Decode and save
        img = model.module.decode_first_stage(sample)
        img = torch.clamp((img + 1.0) / 2.0, 0.0, 1.0)
        utils.save_image(img, out_dir / 'pred.png')
        for key in ['losses', 'losses_flow', 'losses_color', 'noise_norms', 'guidance_norms']:
            np.save(out_dir / f'{key}.npy', info[key])
        torch.save(start_zt, out_dir / 'start_zt.pth')


def make_results_gif(results_dir: str,
                     gif_name: str = "flows.gif",
                     duration: int = 500):
    """
    Scans all sub‑directories of `results_dir` for pred.png and
    writes them in sorted order into a looping GIF.

    Args:
        results_dir: path to your output root (where sample_000, sample_001, … live)
        gif_name:   filename of the resulting GIF saved into results_dir
        duration:   frame duration in milliseconds
    """
    results_path = Path(results_dir)
    # find all pred.png under sample_* dirs
    png_paths = sorted(results_path.glob("sample_*/pred.png"))
    if not png_paths:
        raise RuntimeError(f"No pred.png files found in {results_dir}")

    frames = [Image.open(p) for p in png_paths]

    # save as animated GIF
    save_path = results_path / gif_name
    frames[0].save(
        save_path,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0
    )
    print(f"GIF saved to {save_path}")


def main():
    cfg = load_cli_config()
    input_dir, output_root = prepare_environment(cfg)
    model, sampler = initialize_model_and_sampler(cfg)

    flow_dir = input_dir / 'flows'
    flow_files = sorted(flow_dir.glob('*.pth'))
    if not flow_files:
        print(f"No .pth flow files found in {flow_dir}")
        return
    
    for flow_path in flow_files:
        flow_name = flow_path.stem
        print(f"\nProcessing flow: {flow_path.name}")
        cfg.target_flow_name = flow_path.name
        data = load_guidance_data(cfg, input_dir)
        run_sampling(cfg, model, sampler, data, output_root, flow_name)

    print(f"All flows processed. Outputs saved in {output_root}")
    make_results_gif(output_root, gif_name="flows.gif", duration=300)


if __name__ == '__main__':
    main()
