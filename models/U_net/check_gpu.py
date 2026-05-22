"""
check_gpu.py
============
Run this BEFORE training to confirm PyTorch sees the RTX 5060.

Expected on a healthy install:
  torch:       2.7.x  (or newer)
  CUDA build:  12.8   (or newer)
  available:   True
  device 0:    NVIDIA GeForce RTX 5060
  capability:  (12, 0)        <-- Blackwell = sm_120

If `available` is False or you see a 'sm_120 not supported' warning,
your PyTorch wheel was built for an older CUDA. Reinstall with:
  pip install --upgrade --force-reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu128
"""
import torch

print("torch:      ", torch.__version__)
print("CUDA build: ", torch.version.cuda)
print("available:  ", torch.cuda.is_available())

if torch.cuda.is_available():
    i = 0
    print("device 0:   ", torch.cuda.get_device_name(i))
    print("capability: ", torch.cuda.get_device_capability(i))
    free, total = torch.cuda.mem_get_info(i)
    print(f"VRAM:        {free/1e9:.1f} / {total/1e9:.1f} GB free")

    # Tiny matmul to make sure kernels actually launch on this arch
    a = torch.randn(1024, 1024, device="cuda")
    b = torch.randn(1024, 1024, device="cuda")
    c = a @ b
    torch.cuda.synchronize()
    print("matmul OK, result norm:", c.norm().item())
else:
    print("\nCUDA not available. Check the NVIDIA driver and the PyTorch wheel.")
