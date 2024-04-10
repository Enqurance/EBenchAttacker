import torch
import time

# 检查CUDA是否可用，如果可用则使用第一个可用的GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")

# 创建两个张量并移动到指定的设备（GPU或CPU）
tensor1 = torch.randn(3, 3).to(device)
tensor2 = torch.ones(3, 3).to(device)

# 无限循环进行计算
while True:
    # 将两个张量相加
    result = tensor1 + tensor2
    time.sleep(10)
