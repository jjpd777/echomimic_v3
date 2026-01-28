import torch
from safetensors.torch import load_file

paths = [
    "flash-pro/Wan2.1-Fun-V1.1-1.3B-InP/diffusion_pytorch_model.safetensors",
    "flash-pro/transformer/diffusion_pytorch_model.safetensors"
]

for p in paths:
    print(f"Checking {p}...")
    try:
        # This only reads the header, making it very fast
        with torch.no_grad():
            sd = load_file(p, device="cpu")
        print(f"✅ {p} is healthy and readable.")
    except Exception as e:
        print(f"❌ Error loading {p}: {e}")
