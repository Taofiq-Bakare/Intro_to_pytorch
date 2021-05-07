# Here I convert the manual gradient descent to auto with Pytorch.


import torch
import torch.nn as nn

# Recap steps

'''

1. A forward pass
2. Loss/error calculation
3. Back propagation/optimization/weight correction
4. Training loop
5. Make predictions.

'''

# linear regression
# y_pred = w * x


# Create dummy data

X = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
Y = torch.tensor([6, 7, 8, 9, 10], dtype=torch.float32)

# Initialize the weight(enable gradient calculation)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# Define the functions


def forward(x):
    return w * x

# loss = mean squared error (y_pred - y) ** 2


loss = nn.MSELoss()

# def loss(y, y_pred):
#     return ((y_pred - y) ** 2).mean()

# Backward pass


# def gradient(x, y, y_pred):
#     return np.dot(2*x, y_pred - y).mean()

# This is handled by torch.autograd.backward
# After setting require_grad = True for the tensor


# print(f"Prediction before training: f(5) = {forward(5):.3f}")
print(f"Prediction before training: f(5) = {forward(5):.3f}")


# design the forward pass

learning_rate = 0.001
n_iteration = 100


# Define an optimizer for automatic weight update
optimizer = torch.optim.ASGD([w], lr=learning_rate)

for epoch in range(n_iteration):
    '''we make y_preds
    calculate the loss and the gradient
    '''
    # make forward pass prediction
    y_pred = forward(X)

    # Calculate the loss/error
    error = loss(Y, y_pred)

    # Calculate the gradient
    # grad = gradient(X, Y, y_pred)
    error.backward()

    # update weight automatically using an optimizer
    optimizer.step()

    # We need to zero the gradient
    optimizer.zero_grad()

    if epoch % 10 == 0:
        print(f"epoch {epoch+1}: w ={w:.3f}, loss = {error:.8f}")

print(f"Prediction after training: f(5) = {forward(5):.3f}")
