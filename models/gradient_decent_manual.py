'''
I understand that gradient decent has the following steps:

1. A forward pass
2. Loss calculation
3. A backward pass

....................

4. Optimization or error correction
5. Repeat for a number of epochs.

'''
import numpy as np

# create dummy dataset

X = np.array([1, 2, 3, 4, 5], dtype=np.int32)
Y = np.array([6, 7, 8, 9, 10], dtype=np.int32)


'''
What formula to use?

linear regression
y = w * x

loss calculation(mean square error)

mse = 1/N (y_pred - y) ** 2

Backward pass = differentiate the loss with respect to the weight

where loss = mes = 1/N ( (w * x) - y) ** 2

d(loss)/dw = 1/N 2x((w * x) - y)

Optimization = correct the weight

w = learning * -w


'''
# initialise weight parameter

w = 0.0

# define the functions


def forward(x):
    return w * x

# loss = mean squared error (y_pred - y) ** 2


def loss(y, y_pred):
    return ((y_pred - y) ** 2).mean()

# Backward pass


def gradient(x, y, y_pred):
    return np.dot(2*x, y_pred - y).mean()


# print(f"Prediction before training: f(5) = {forward(5):.3f}")
print(f"Prediction before training: f(5) = {forward(5):.3f}")


# design the forward pass

learning_rate = 0.001
n_iteration = 20

for epoch in range(n_iteration):
    '''we make y_preds
    calculate the loss and the gradient
    '''
    # make forward pass prediction
    y_pred = forward(X)

    # Calculate the loss/error
    error = loss(Y, y_pred)

    # Calculate the gradient
    grad = gradient(X, Y, y_pred)

    # Perform optimization and update the weight
    w -= learning_rate * grad

    if epoch % 2 == 0:
        print(f"epoch {epoch+1}: w ={w:.3f}, loss = {error:.8f}")

print(f"Prediction after training: f(5) = {forward(5):.3f}")
