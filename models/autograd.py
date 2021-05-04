import torch

x = torch.randn(3, requires_grad=True)
y = x * 2

print(y)
