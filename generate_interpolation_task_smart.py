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
from losses import FlowLoss, FlowInterpolate
import re

def load_cli_config(config_path: str = "cli_config_interpolation_task.yaml"):
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

def run_sampling(cfg, model, sampler, data, save_dir, img_end):
    src_img_start, start_zt, edit_mask, guidance_schedule, cached_latents, guidance_energy = data
    uncond_embed = model.module.get_learned_conditioning([""])
    cond_embed = model.module.get_learned_conditioning([cfg.prompt])

    # Calculate coefficients
    step_idx = cfg.current_step
    n_steps = cfg.total_steps

    coeff = (step_idx + 1) / (n_steps + 1)
    coeff_start = 1.0 - coeff
    coeff_end = coeff

    for idx in range(cfg.num_samples):
        print(f"Sampling {idx + 1}/{cfg.num_samples}")
        out_dir = save_dir / f'sample_{idx:03}'
        out_dir.mkdir(parents=True, exist_ok=True)

        sample, start_zt, info = sampler.sampling_bidirectional(
            num_ddim_steps=cfg.ddim_steps,
            cond_embed=cond_embed,
            uncond_embed=uncond_embed,
            batch_size=1,
            shape=[4, 64, 64],
            CFG_scale=cfg.scale,
            eta=cfg.ddim_eta,
            src_img_start=src_img_start,
            src_img_end=img_end,
            coeff_start=coeff_start,
            coeff_end=coeff_end,
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

        name = "pred_" + cfg.target_flow_name[5:9] + ".png"
        save_path = out_dir / name
        utils.save_image(img, save_path)
        for key in ['losses', 'losses_flow', 'losses_color', 'noise_norms', 'guidance_norms']:
            np.save(out_dir / f'{key}.npy', info[key])
        torch.save(start_zt, out_dir / 'start_zt.pth')


def make_results_gif(results_dir: str,
                     gif_names: list = ["flows.gif", "flows_fast.gif"],
                     durations: list = [300, 70]):
    """
    Copies the start and end images into the sample_000 directory,
    then scans for pred_XXXX.png files and saves multiple GIFs.

    Args:
        results_dir: Path to output root.
        gif_names: List of GIF filenames to save.
        durations: List of frame durations in milliseconds for each GIF.
    """
    results_path = Path(results_dir)
    sample_path = results_path / "sample_000"

    # Ensure sample_000 exists
    sample_path.mkdir(parents=True, exist_ok=True)

    # === Copy start and end images ===
    input_dir = results_path.parent  # Assume input_dir is the parent folder of save_dir
    img_start_path = input_dir / "pred.png"
    img_end_path = input_dir / "pred_altered.png"

    if img_start_path.is_file():
        img_start = Image.open(img_start_path).convert("RGB")
        img_start.save(sample_path / "pred_0000.png")
    else:
        print(f"Warning: Start image {img_start_path} not found.")

    if img_end_path.is_file():
        img_end = Image.open(img_end_path).convert("RGB")
        img_end.save(sample_path / "pred_9999.png")
    else:
        print(f"Warning: End image {img_end_path} not found.")

    # === Collect frames ===
    png_paths = [p for p in sample_path.glob("pred_*.png") if re.match(r"pred_\d{4}\.png", p.name)]
    if not png_paths:
        raise RuntimeError(f"No pred_XXXX.png files found in {sample_path}")

    png_paths = sorted(png_paths, key=lambda p: int(p.stem.split('_')[1]))
    frames = [Image.open(p) for p in png_paths]

    # === Save all requested GIFs ===
    for gif_name, duration in zip(gif_names, durations):
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

    flowInterpolater = FlowInterpolate()
    n_steps = int(cfg.n_flows)  # Total number of steps (including start and end)

    # Load and preprocess start and end images
    img_start_path = input_dir / "pred.png"
    img_end_path = input_dir / "pred_altered.png"
    img_start = to_tensor(Image.open(img_start_path).convert("RGB"))[None].cuda().float() * 2 - 1
    img_end = to_tensor(Image.open(img_end_path).convert("RGB"))[None].cuda().float() * 2 - 1

    img_current = img_start.clone()

    # Prepare flow directory
    flow_dir = input_dir / 'flows'
    flow_dir.mkdir(parents=True, exist_ok=True)

    print("Starting interpolation process...")

    for step in range(n_steps):
        print(f"\nStep {step}/{n_steps-1}")

        if step == 0:
            # For step 0, just save the start image
            out_dir = output_root / f'sample_000'
            out_dir.mkdir(parents=True, exist_ok=True)

            save_path = out_dir / f"pred_{step:04}.png"
            img_to_save = torch.clamp((img_current + 1.0) / 2.0, 0.0, 1.0)
            utils.save_image(img_to_save, save_path)
            print(f"Saved starting image at {save_path}")
            continue

        # 1. Compute bidirectional flows
        flow_forward, flow_backward = flowInterpolater.bidirectional(
            img_current, img_end, step_idx=step-1, n_steps=n_steps-1
        )

        # Save flows for debugging (optional)
        torch.save(flow_forward.cpu(), flow_dir / f"flow_forward_{step:04}.pth")
        torch.save(flow_backward.cpu(), flow_dir / f"flow_backward_{step:04}.pth")

        # 2. Set forward flow for sampling (temporary, we still only use forward flow for now)
        flow_path = flow_dir / f"flow_forward_{step:04}.pth"
        cfg.target_flow_name = flow_path.name

        cfg.current_step = step # <- this must be set correctly before calling run_sampling!
        cfg.total_steps = n_steps   # <- this must also be set correctly before calling run_sampling!

        # 3. Load guidance data based on the new flow
        data = load_guidance_data(cfg, input_dir)

        # 4. Run sampling step
        out_dir = output_root / f'sample_000'
        out_dir.mkdir(parents=True, exist_ok=True)
        run_sampling(cfg, model, sampler, data, out_dir, img_end)

        # 5. Update current image
        new_pred_path = out_dir / f"pred_{step:04}.png"
        img_current = to_tensor(Image.open(new_pred_path).convert("RGB"))[None].cuda().float() * 2 - 1

    print(f"\nInterpolation process complete. Outputs saved in {output_root}.")

    # Make GIFs
    make_results_gif(output_root, gif_names=["flows.gif", "flows_fast.gif"], durations=[300, 70])


if __name__ == '__main__':
    main()




    """
    Conceptual outline:
        1 We have start and end image, also we have the n_flows we want to generate!
            Note the start and end image should be part of the trajectory as well unchanged
        2 Generate BOTH flow:
            Both, start -> end img,   end -> start img
            Flow_forward
            Flow_backward
        3 Start by modifying loss function
            Intermediate images is generated using loss function:
                Loss function normally:
                c1*Lflow(F(start_img, gen img), flow) + mask*c2*Lcolour(start_img, warp(gen_img, F(start_img, gen_img)))

                KEY IDEA!
                    I want to use both the start_img and end_img in the loss function
                    ie. 
                        (1-0.1) * c1 * Lflow(F(start_img, gen img), flow_forward*0.1)
                        (1-0.9) * c1 * Lflow(F(end_img, gen img), flow_backward*0.9)
                            Note 2 coefficients particular to the example, 0.1, and 0.9.
                                We weight the Loss differently to emphasize the closer value. 
                        And similarly for colour loss. 
                            Use same coefficient values etc, have two terms like for the flows above!
                                Should be easy to pattern match. 
                        0.1, 0.9 coefficients corrospond to a schedule where n_flows = 8,
                        Next image would be 0.2, 0.8
                            Ie. 0.0 is the start image, 1 is the end image, and we have 8 inbetween
        4 Second key idea:
            After generating the next image, we recalculate BOTH flows using the new image
                ie. After generating image corrosponding to the loss function above, lets call it img_intermediate_1
                we calculate using raft(img_intermediate_1, end_img) 
            Then we iterate.
    """