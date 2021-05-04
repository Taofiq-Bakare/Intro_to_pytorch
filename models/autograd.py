import numpy as np
# import torch

# x = torch.randn(3, requires_grad=True)
# y = x * 2

# print(f'y value is {y}')

# z = y * y + 3
# z = z.mean()

# print(f'z value is {z}')

# z.backward()

# print(f'dz/dx is {x.grad}')


# Compute every step manually

# Linear regression
# f = w * x

# here : f = 2 * x
X = np.array([1, 2, 3, 4], dtype=np.float32)
Y = np.array([2, 4, 6, 8], dtype=np.float32)

w = 0.0

# model output


def forward(x):
    return w * x

# loss = MSE


def loss(y, y_pred):
    return ((y_pred - y)**2).mean()

# J = MSE = 1/N * (w*x - y)**2
# dJ/dw = 1/N * 2x(w*x - y)


def gradient(x, y, y_pred):
    return np.dot(2*x, y_pred - y).mean()


print(f"Prediction before training: f(5) = {forward(5):.3f}")
