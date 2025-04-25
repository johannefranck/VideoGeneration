import pygame
import sys
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
import cv2
from segment_anything import sam_model_registry, SamPredictor
import imageio
import torch

# --- CONFIG ---
image_path = "./assets/topiary.png"
sam_checkpoint = "./gui/sam_files/sam_vit_b_01ec64.pth"
model_type = "vit_b"
device = "cpu"
resize_size = 512
n_flows = 10
output_dir = "./assets/custom_flows"
os.makedirs(output_dir, exist_ok=True)

# --- Resize + pad image to 1024x1024 ---
def resize(image: np.ndarray, resize_size: int = 1024) -> np.ndarray:
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

# --- Load + Resize Image ---
im_raw = Image.open(image_path)
im_np = resize(np.array(im_raw), resize_size)
im_pil = Image.fromarray(im_np)

# --- Load SAM ---
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device)
predictor = SamPredictor(sam)
predictor.set_image(im_np)

# --- Pygame Setup ---
pygame.init()
screen = pygame.display.set_mode((resize_size, resize_size))
pygame.display.set_caption("Click to Mask, Then Draw Path")
white = (255, 255, 255)
black = (0, 0, 0)

# --- Helpers ---
def show_image(surface, image):
    surf = pygame.image.fromstring(image.tobytes(), image.size, image.mode)
    surface.blit(surf, (0, 0))
    pygame.display.flip()

# --- State ---
drawing = False
path_points = []
mask = None
masked_overlay = None
waiting_for_mask_click = True
show_image(screen, im_pil)

# --- Main Loop ---
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if waiting_for_mask_click:
                click_point = pygame.mouse.get_pos()
                input_point = np.array([click_point])
                input_label = np.array([1])

                masks, scores, _ = predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=True
                )
                mask = masks[scores.argmax()].astype(np.uint8)

                # Resize and repeat to match repo format: [1, 4, 64, 64]
                mask_small = np.array(Image.fromarray(mask * 255).resize((64, 64), Image.NEAREST)) // 255
                mask_stack = np.stack([mask_small] * 4, axis=0)
                torch_mask = torch.from_numpy(mask_stack).unsqueeze(0).float()
                torch.save(torch_mask, os.path.join(output_dir, "mask.pth"))
                print("Mask saved.")

                # --- Visualize the saved mask for confirmation ---
                upscaled_mask = np.array(Image.fromarray(mask_small * 255).resize((resize_size, resize_size), Image.NEAREST))
                vis_overlay = im_np.copy()
                vis_overlay[upscaled_mask == 255] = [255, 0, 0]
                plt.figure(figsize=(6, 6))
                plt.imshow(vis_overlay)
                plt.title("Saved Mask Visualization")
                plt.axis('off')
                plt.savefig(os.path.join(output_dir, "mask_visualization.png"))
                plt.close()

                masked_overlay = (0.6 * im_np * (1 - mask[..., None]) + 255 * mask[..., None]).astype(np.uint8)
                waiting_for_mask_click = False
                show_image(screen, Image.fromarray(masked_overlay))

            else:
                drawing = True
                path_points = [pygame.mouse.get_pos()]

        elif event.type == pygame.MOUSEBUTTONUP and not waiting_for_mask_click:
            drawing = False
            print("Drawn path:", path_points)

            if len(path_points) < 2:
                print("⚠️ Not enough path points to generate flows.")
                continue

            # --- Flow Generation ---
            path_np = np.array(path_points)
            indices = np.linspace(1, len(path_np) - 1, n_flows).astype(int)
            origin = path_np[0]
            targets = path_np[indices]

            coords = np.meshgrid(np.arange(resize_size), np.arange(resize_size), indexing='ij')
            coords = np.stack(coords, axis=-1).astype(np.float32)

            for i, target in enumerate(targets):
                dx = target[0] - origin[0]
                dy = target[1] - origin[1]
                flow = np.zeros((resize_size, resize_size, 2), dtype=np.float32)
                flow[mask == 1] = [dy, dx]

                # Save in (2, H, W) format for compatibility
                flow_tensor = torch.from_numpy(flow.transpose(2, 0, 1)).unsqueeze(0).float()
                torch.save(flow_tensor, os.path.join(output_dir, f"flow_{i}.pth"))

                # Composite mask onto new location manually
                composited = im_np.copy()
                object_pixels = im_np.copy()

                coords_y, coords_x = np.where(mask == 1)
                new_y = coords_y + dy
                new_x = coords_x + dx

                valid = (0 <= new_y) & (new_y < resize_size) & (0 <= new_x) & (new_x < resize_size)
                new_y = new_y[valid].astype(int)
                new_x = new_x[valid].astype(int)
                src_y = coords_y[valid]
                src_x = coords_x[valid]

                # First paint original masked area black
                composited[coords_y, coords_x] = 0
                # Then paste object pixels on top at new location
                composited[new_y, new_x] = object_pixels[src_y, src_x]

                # Arrow
                mask_vis = (0.6 * im_np * (1 - mask[..., None]) + 255 * mask[..., None]).astype(np.uint8)
                arrowed = mask_vis.copy()

                # Draw full path in red with alpha
                for j in range(1, len(path_np)):
                    pt1 = tuple(path_np[j - 1])
                    pt2 = tuple(path_np[j])
                    overlay = arrowed.copy()
                    cv2.line(overlay, pt1, pt2, color=(255, 0, 0), thickness=2)
                    arrowed = cv2.addWeighted(overlay, 0.8, arrowed, 0.2, 0)

                cv2.arrowedLine(arrowed, tuple(origin), tuple(target), color=(0, 0, 0), thickness=4, tipLength=0.05)

                # Quiver
                step = 20
                y, x = np.mgrid[0:resize_size:step, 0:resize_size:step]
                u = flow[::step, ::step, 1]
                v = flow[::step, ::step, 0]

                fig, axs = plt.subplots(1, 3, figsize=(18, 6))
                axs[0].imshow(arrowed)
                axs[0].set_title(f"Mask + Arrow {i}")
                axs[0].axis("off")
                axs[1].imshow(im_np)
                axs[1].quiver(x, y, u, -v, color='r', scale=100, width=0.003)
                axs[1].set_title("Flow Field")
                axs[1].axis("off")
                axs[2].imshow(composited)
                axs[2].set_title("Warped Image")
                axs[2].axis("off")
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"mask_arrow_flow_vis_{i}.png"))
                plt.close()

            # --- Create GIF from saved plots ---
            plot_paths = [os.path.join(output_dir, f"mask_arrow_flow_vis_{i}.png") for i in range(n_flows)]
            images = [imageio.imread(p) for p in plot_paths if os.path.exists(p)]
            gif_path = os.path.join(output_dir, "flow_visualization.gif")
            imageio.mimsave(gif_path, images, duration=0.8)
            print(f"Saved GIF to {gif_path}")

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

    if drawing:
        pos = pygame.mouse.get_pos()
        if path_points and pos != path_points[-1]:
            path_points.append(pos)

    if not waiting_for_mask_click and masked_overlay is not None:
        screen.blit(pygame.image.fromstring(Image.fromarray(masked_overlay).tobytes(), im_pil.size, im_pil.mode), (0, 0))
        if len(path_points) > 1:
            pygame.draw.lines(screen, black, False, path_points, 3)
        pygame.display.flip()

pygame.quit()
sys.exit()
