import argparse
import os
import sys
import pygame
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import imageio
import torch
import yaml
import stat
import subprocess
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="segment_anything")

# --- CLI ARGUMENTS ---
parser = argparse.ArgumentParser()
parser.add_argument("image_name", type=str, help="Name of the image (without .png)")
parser.add_argument("output_folder", type=str, help="Output folder name (no slashes)")
args = parser.parse_args()

# --- CONFIG ---
image_path = f"./assets/{args.image_name}.png"
output_dir = f"./assets/{args.output_folder}"
os.makedirs(output_dir, exist_ok=True)

# Subdirectories
masks_dir = os.path.join(output_dir, "masks")
flows_dir = os.path.join(output_dir, "flows")
vis_dir = os.path.join(output_dir, "visualizations")
logs_dir = os.path.join(output_dir, "logs")
os.makedirs(masks_dir, exist_ok=True)
os.makedirs(flows_dir, exist_ok=True)
os.makedirs(vis_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

sam_checkpoint = "./gui/sam_files/sam_vit_b_01ec64.pth"
model_type = "vit_b"
device = "cpu"
resize_size = 512

# --- Image loading ---
im_raw = Image.open(image_path)
im_np = np.array(im_raw)

# --- Resize and pad image to 512x512 ---
def resize(image: np.ndarray, resize_size: int = 512) -> np.ndarray:
    pil_img = Image.fromarray(image)
    w, h = pil_img.size
    if w >= h:
        new_w = resize_size
        new_h = int(h * resize_size / w)
    else:
        new_h = resize_size
        new_w = int(w * resize_size / h)
    resized = pil_img.resize((new_w, new_h), Image.BILINEAR)
    padded = Image.new("RGB", (resize_size, resize_size))
    padded.paste(resized, ((resize_size - new_w) // 2, (resize_size - new_h) // 2))
    return np.array(padded)

im_np = resize(im_np, resize_size)
Image.fromarray(im_np).save(os.path.join(output_dir, "pred.png"))

from segment_anything import sam_model_registry, SamPredictor
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device)
predictor = SamPredictor(sam)
predictor.set_image(im_np)

pygame.init()
screen = pygame.display.set_mode((resize_size, resize_size))
pygame.display.set_caption("Click to Mask, Then Draw Path")
white = (255, 255, 255)
black = (0, 0, 0)

# --- State ---
drawing = False
path_points = []
mask = None
masked_overlay = None
waiting_for_mask_click = True

surf = pygame.image.fromstring(Image.fromarray(im_np).tobytes(), im_np.shape[1::-1], "RGB")
screen.blit(surf, (0, 0))
pygame.display.flip()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if waiting_for_mask_click:
                click_point = pygame.mouse.get_pos()
                masks, scores, _ = predictor.predict(
                    point_coords=np.array([click_point]),
                    point_labels=np.array([1]),
                    multimask_output=True
                )
                mask = masks[scores.argmax()].astype(np.uint8)
                masked_overlay = (0.6 * im_np * (1 - mask[..., None]) + 255 * mask[..., None]).astype(np.uint8)
                waiting_for_mask_click = False
                surf = pygame.image.fromstring(Image.fromarray(masked_overlay).tobytes(), masked_overlay.shape[1::-1], "RGB")
                screen.blit(surf, (0, 0))
                pygame.display.flip()
            else:
                drawing = True
                path_points = [pygame.mouse.get_pos()]

        elif event.type == pygame.MOUSEBUTTONUP and not waiting_for_mask_click:
            drawing = False
            if len(path_points) < 2:
                continue

            # --- GUI for config ---
            import tkinter as tk
            root = tk.Tk()
            root.title("Enter Diffusion Parameters")
            root.geometry("400x350")
            entries = {}
            defaults = {
                'n_flows': 10,
                'dilation_iterations': 10,
                'guidance_weight': 300.0,
                'num_recursive_steps': 10,
                'color_weight': 100,
                'flow_weight': 3,
                'clip_grad': 200.0,
                'prompt': "an apple on a wooden table"
            }

            def submit():
                for k in entries:
                    val = entries[k].get()
                    defaults[k] = val if k == 'prompt' else int(val) if val.isdigit() else float(val)
                root.destroy()

            for i, (k, v) in enumerate(defaults.items()):
                tk.Label(root, text=k).grid(row=i, column=0)
                entry = tk.Entry(root)
                entry.insert(0, str(v))
                entry.grid(row=i, column=1)
                entries[k] = entry

            tk.Button(root, text="Start Generation", command=submit).grid(row=len(defaults), column=0, columnspan=2)
            root.mainloop()

            n_flows = int(defaults['n_flows'])
            dilation_iterations = int(defaults['dilation_iterations'])
            mask_bool = mask > 0
            path_np = np.array(path_points)
            indices = np.linspace(1, len(path_np) - 1, n_flows).astype(int)
            origin = path_np[0]
            targets = path_np[indices]
            coords_y, coords_x = np.where(mask_bool)

            for i, target in enumerate(targets):
                dy = target[1] - origin[1]  # vertical change (row)
                dx = target[0] - origin[0]  # horizontal change (col)

                flow = np.zeros((resize_size, resize_size, 2), np.float32)
                print(f"dx = {dx}, dy = {dy}") # Debugging line

                flow[mask_bool] = [dx, dy]  # channel 0 = dx, channel 1 = dy
                torch.save(torch.from_numpy(flow.transpose(2, 0, 1)).unsqueeze(0).float(), os.path.join(flows_dir, f"flow_{i}.pth"))

                new_y = coords_y + dy
                new_x = coords_x + dx
                valid = (0 <= new_y) & (new_y < resize_size) & (0 <= new_x) & (new_x < resize_size)
                dest_mask = np.zeros_like(mask_bool)
                dest_mask[new_y[valid], new_x[valid]] = 1

                union_mask = np.logical_or(mask_bool, dest_mask)
                dilated_mask = cv2.dilate(union_mask.astype(np.uint8), np.ones((5,5), np.uint8), iterations=dilation_iterations) > 0
                final_edit_mask = ~dilated_mask
                downsampled = np.array(Image.fromarray(final_edit_mask.astype(np.uint8) * 255).resize((64, 64), Image.NEAREST)) // 255
                mask_stack = np.stack([downsampled] * 4, axis=0)
                torch.save(torch.from_numpy(mask_stack).unsqueeze(0).bool(), os.path.join(masks_dir, f"mask_{i}.pth"))

                # Visualization
                mask_background = im_np.copy()
                mask_background[mask_bool] = [255, 255, 255]

                # Flow: reshape to (H, W, 2) and extract u (dx), v (dy)
                flow_vis = flow.copy()
                u = flow_vis[:, :, 0]  # horizontal displacement (dx)
                v = flow_vis[:, :, 1]  # vertical displacement (dy)

                # Subsample grid for cleaner quiver plot
                step = 20
                H, W = u.shape
                y, x = np.mgrid[0:H:step, 0:W:step]
                u_sub = u[::step, ::step]
                v_sub = v[::step, ::step]

                # Path overlay image
                arrowed = mask_background.copy()
                for j in range(1, len(path_np)):
                    cv2.line(arrowed, tuple(path_np[j - 1]), tuple(path_np[j]), color=(255, 0, 0), thickness=2)
                cv2.arrowedLine(arrowed, tuple(origin), tuple(target), color=(0, 0, 0), thickness=4, tipLength=0.05)

                # Final edit mask visualization
                final_mask_vis = im_np.copy()
                final_mask_vis[final_edit_mask] = [255, 0, 0]

                # Plot all three panels
                fig, axs = plt.subplots(1, 3, figsize=(18, 6))

                axs[0].imshow(arrowed)
                axs[0].set_title(f"Path {i}")
                axs[0].axis("off")

                axs[1].imshow(np.ones((H, W, 3)))  # White background
                axs[1].quiver(x, y, u_sub, -v_sub, color='r', scale=100, width=0.003)
                axs[1].set_title("Optical Flow (quiver plot)")
                axs[1].axis("equal")
                axs[1].grid(True)

                axs[2].imshow(final_mask_vis)
                axs[2].set_title("Final Union Mask")
                axs[2].axis("off")

                plt.tight_layout()
                plt.savefig(os.path.join(vis_dir, f"mask_arrow_flow_vis_{i}.png"))
                plt.close()


            gif_path = os.path.join(output_dir, "flow_visualization.gif")
            plot_paths = [os.path.join(vis_dir, f"mask_arrow_flow_vis_{i}.png") for i in range(n_flows)]
            images = [imageio.imread(p) for p in plot_paths if os.path.exists(p)]
            imageio.mimsave(gif_path, images, duration=0.8)

            yaml_data = {
                'save_dir': f"results/{args.output_folder}",
                'num_samples': 1,
                'input_dir': output_dir,
                'log_freq': 0,
                'sd_model_config': "configs/stable-diffusion/v1-inference.yaml",
                'ckpt': "./chkpts/sd-v1-4.ckpt",
                'ddim_steps': 500,
                'ddim_eta': 0.0,
                'scale': 7.5,
                'target_flow_name': None,
                'edit_mask_path': "",
                'oracle_flow': False,
                'no_occlusion_masking': False,
                'no_init_startzt': False,
                'use_cached_latents': False,
                'guidance_schedule_path': "data/guidance_schedule.npy",
                'mixed_precision': True
            }
            yaml_data.update({k: defaults[k] for k in ['guidance_weight', 'num_recursive_steps', 'color_weight', 'flow_weight', 'clip_grad', 'prompt']})

            yaml_path = os.path.abspath(os.path.join(output_dir, "generation_config.yaml"))
            with open(yaml_path, "w") as f:
                yaml.dump(yaml_data, f)

            bash_script_path = os.path.abspath(os.path.join(output_dir, f"{args.image_name}_job.sh"))
            with open(bash_script_path, "w") as f:
                f.write(f"""#!/bin/bash
#BSUB -J vidgen1_{args.output_folder}
#BSUB -q gpuv100
#BSUB -W 03:00
#BSUB -R \"rusage[mem=4GB]\"
#BSUB -gpu "num=1"
#BSUB -o {logs_dir}/{args.output_folder}.out
#BSUB -e {logs_dir}/{args.output_folder}.err

#BSUB -n 4
#BSUB -R \"span[hosts=1]\"

cd /work3/s204129/adlicv/project/VideoGeneration || exit 1

module load cuda/11.7
source /work3/s204129/adlicv/project/VideoGeneration/motion_guidance_env/bin/activate

mkdir -p assets/{args.output_folder}/logs
export HOME=$PWD
mkdir -p \"$HOME/.cache\" \"$HOME/.config\" \"$HOME/.huggingface\"
export PYTHONPATH=/VideoGeneration/motion_guidance_env/src/taming-transformers
python ./generate.py assets/{args.output_folder}/generation_config.yaml

""")
            os.chmod(bash_script_path, stat.S_IRWXU)

            # Ask user if they want to submit the job
            def submit_or_exit(job_script_path):
                submit_root = tk.Tk()
                submit_root.title("Submit Job?")
                tk.Label(submit_root, text="Generation complete!").pack(pady=(10, 5))
                tk.Label(submit_root, text="Do you want to submit the job now?").pack(pady=5)

                def submit():
                    subprocess.run(f"bsub < {job_script_path}", shell=True)
                    submit_root.destroy()

                def dont_submit():
                    submit_root.destroy()

                btn_frame = tk.Frame(submit_root)
                btn_frame.pack(pady=10)
                tk.Button(btn_frame, text="Submit Job", command=submit).pack(side="left", padx=10)
                tk.Button(btn_frame, text="Don't Submit", command=dont_submit).pack(side="right", padx=10)

                submit_root.mainloop()

            submit_or_exit(bash_script_path)

        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            running = False

    if drawing:
        pos = pygame.mouse.get_pos()
        if path_points and pos != path_points[-1]:
            path_points.append(pos)

    if not waiting_for_mask_click and masked_overlay is not None:
        surf = pygame.image.fromstring(Image.fromarray(masked_overlay).tobytes(), masked_overlay.shape[1::-1], "RGB")
        screen.blit(surf, (0, 0))
        if len(path_points) > 1:
            pygame.draw.lines(screen, black, False, path_points, 3)
        pygame.display.flip()

pygame.quit()
sys.exit()