import torch

device = 'cpu'

# Load start_zt.pth
start_zt = torch.load('./data/teapot/start_zt.pth').to(device)

# Load the last cached latent (assuming 250 steps)
cached_zt = torch.load('./data/teapot/latents/zt.00249.pth').to(device)

# Compare
diff = torch.norm(start_zt - cached_zt)
print(f"Difference between start_zt.pth and zt.00249.pth: {diff.item()}")
