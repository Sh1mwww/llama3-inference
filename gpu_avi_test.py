import torch, os, platform
print("Python", platform.python_version())
print("Torch ", torch.__version__, "CUDA", torch.version.cuda)
print("CUDA_VISIBLE_DEVICES =", os.getenv("CUDA_VISIBLE_DEVICES"))
print("cuda.is_available()  =", torch.cuda.is_available())