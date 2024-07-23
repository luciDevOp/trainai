import torch

print(torch.__version__)
print("CUDA available:", torch.cuda.is_available())  # Should be False on macOS as CUDA is not supported
print("MPS available:", torch.backends.mps.is_available())  # Should be True on Apple Silicon Macs
