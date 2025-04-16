import numpy as np
import pygame
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor

# --- CONFIG ---
image_path = "./assets/topiary.png"
sam_checkpoint = "./gui/sam_files/sam_vit_b_01ec64.pth" #can be downloaded from https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
model_type = "vit_b"
device = "cpu"
resize_size = 1024
# --------------

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

# --- Pygame helpers ---
def show_image(surface, image: Image.Image):
    surf = pygame.image.fromstring(image.tobytes(), image.size, image.mode)
    surface.blit(surf, (0, 0))
    pygame.display.flip()

def draw_arrow(surface, color, start, end, width=3):
    pygame.draw.line(surface, color, start, end, width)
    arrow_size = 10
    theta = np.arctan2(start[1] - end[1], start[0] - end[0])
    for angle in [np.pi / 6, -np.pi / 6]:
        tip = (
            end[0] + arrow_size * np.cos(theta + angle),
            end[1] + arrow_size * np.sin(theta + angle)
        )
        pygame.draw.line(surface, color, end, tip, width)

# --- Load image + SAM ---
image_raw = Image.open(image_path)
image_np = resize(np.array(image_raw), resize_size)

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device)
predictor = SamPredictor(sam)
predictor.set_image(image_np)

# --- Pygame setup ---
pygame.init()
screen = pygame.display.set_mode((resize_size, resize_size))
pygame.display.set_caption("Motion Flow Tool")
show_image(screen, Image.fromarray(image_np))

# --- Main loop ---
mask = None
flow = np.zeros((resize_size, resize_size, 2), dtype=np.float32)
click_point = None
arrow_start = None
arrow_end = None
drawing = False
done = False

while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

        elif event.type == pygame.MOUSEBUTTONUP and mask is None:
            click_point = pygame.mouse.get_pos()
            input_point = np.array([click_point])

            y, x = click_point[1], click_point[0]

            input_label = np.array([1])
            masks, scores, _ = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True
            )
            mask = masks[scores.argmax()]
            mask = mask.astype(np.uint8)
            print("Mask created. Now drag to define motion.")

            # Visualize mask on image
            overlay = (0.6 * image_np * (1 - mask[..., None]) + 255 * mask[..., None]).astype(np.uint8)
            show_image(screen, Image.fromarray(overlay))

        elif event.type == pygame.MOUSEBUTTONDOWN and mask is not None:
            arrow_start = pygame.mouse.get_pos()
            drawing = True

        elif event.type == pygame.MOUSEBUTTONUP and drawing:
            arrow_end = pygame.mouse.get_pos()
            dx = arrow_end[0] - arrow_start[0]
            dy = arrow_end[1] - arrow_start[1]
            print(f"Motion vector: dx={dx}, dy={dy}")

            # Apply motion vector to masked region
            flow[mask == 1] = [dy, dx]  # Correct order: [row offset, col offset]
            print(f"Flow field (at click): {flow[y, x]} (should be [dy, dx])")


            # Save mask + flow
            np.save("./assets/custom_flows/mask.npy", mask)
            np.save("./assets/custom_flows/flow.npy", flow)
            print("Saved mask.npy and flow.npy. Exiting.")
            done = True

        elif event.type == pygame.MOUSEMOTION and drawing:
            current = pygame.mouse.get_pos()
            overlay = (0.6 * image_np * (1 - mask[..., None]) + 255 * mask[..., None]).astype(np.uint8)
            show_image(screen, Image.fromarray(overlay))
            draw_arrow(screen, (0, 0, 0), arrow_start, current)

pygame.quit()

import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
import cv2

print("Creating visualization...")

# --- Panel 1: Mask + Arrow ---
mask_vis = (0.6 * image_np * (1 - mask[..., None]) + 255 * mask[..., None]).astype(np.uint8)
arrowed = mask_vis.copy()
cv2.arrowedLine(
    arrowed, arrow_start, arrow_end,
    color=(0, 0, 0), thickness=4, tipLength=0.05
)

# --- Panel 2: Flow Field ---
step = 20
y, x = np.mgrid[0:resize_size:step, 0:resize_size:step]
u = flow[::step, ::step, 1]
v = flow[::step, ::step, 0]

# --- Panel 3: Warped image ---
coords = np.meshgrid(np.arange(resize_size), np.arange(resize_size), indexing='ij')
coords = np.stack(coords, axis=-1).astype(np.float32)
warped_coords = coords - flow  # pull-based warping

warped_image = image_np.copy()
for c in range(3):
    channel_warped = map_coordinates(
        image_np[..., c],
        [warped_coords[..., 0], warped_coords[..., 1]],
        order=1, mode='constant', cval=0.0
    )

    # Apply only to the mask
    warped_image[..., c][mask == 1] = channel_warped[mask == 1]


# --- Plot all three ---
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

axs[0].imshow(arrowed)
axs[0].set_title("Mask + Arrow")
axs[0].axis("off")

axs[1].imshow(image_np)
axs[1].quiver(x, y, u, -v, color='r', scale=100, width=0.003)
axs[1].set_title("Flow Field")
axs[1].axis("off")

axs[2].imshow(warped_image)
axs[2].set_title("Warped Image")
axs[2].axis("off")

plt.tight_layout()
plt.savefig("./assets/custom_flows/mask_arrow_flow_vis.png")
plt.show()
print("Saved visualization to './assets/custom_flows/mask_arrow_flow_vis.png'")
