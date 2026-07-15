#!/usr/bin/env python3
"""Minimal PyTorch GPU allocation check for the SCC."""
import os
import torch

print("CUDA_VISIBLE_DEVICES:", os.getenv("CUDA_VISIBLE_DEVICES"))
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU count visible to job:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    x = torch.randn(1024, 1024, device="cuda")
    y = x @ x
    print("GPU smoke-test checksum:", float(y.mean().cpu()))
