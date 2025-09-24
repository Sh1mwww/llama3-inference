import torch, time

device = torch.device("cuda:0")
N = 1024 * 1024 * 256  # 256 MB
x_cpu = torch.empty(N, dtype=torch.float32, pin_memory=True)  # Pinned 内存
x_gpu = torch.empty(N, dtype=torch.float32, device=device)

# H2D
torch.cuda.synchronize()
t0 = time.time()
x_gpu.copy_(x_cpu, non_blocking=True)
torch.cuda.synchronize()
t1 = time.time()
print("H2D BW: %.2f GB/s" % (x_cpu.numel()*4/ (t1-t0) / 1e9))

# D2H
torch.cuda.synchronize()
t0 = time.time()
x_cpu.copy_(x_gpu, non_blocking=True)
torch.cuda.synchronize()
t1 = time.time()
print("D2H BW: %.2f GB/s" % (x_cpu.numel()*4/ (t1-t0) / 1e9))
