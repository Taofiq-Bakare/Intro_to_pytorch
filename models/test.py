# Time to test my understanding of the tutorial so far

import torch
# from torch._C import dtype
import torch.nn as nn


''' This is linear regression problem

y_pred = w * x

Define the processes involved in training a linear regression model

1. Make a forward prediction
2. Calculate the error
3. Perform backward propagation(differentiate the error term wrt to the variables)
4. Update the weight(optimize)

Steps in Pytorch

1. Define prediction function
2. Select error metric(MSE in this case)
3. Define optimizer to use(Stochastic gradient descent in this case)
4. Define the for loop to train
    1. Make forward prediction
    2. calculate loss between the actual and predicted values
    4. Call torch.autograd.backward on the loss function
    5. Correct/optimize the weight with optimizer.step()
    6. Zero the gradient
5. Make prediction with the updated weight.

'''

# Initialize data

X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([5, 6, 7, 8, ], dtype=torch.float32)


# Initialize constants

w = torch.tensor(0.0, requires_grad=True, dtype=torch.float32)
learning_rate = 0.01
n_iteration = 1000
# Formula for forward prediction


def forward(x,):
    '''
    Makes initial forward prediction with the
    randomly initialized weight.
    '''
    return w * x

# Initialize loss function


loss = nn.MSELoss()
optimizer = torch.optim.SGD([w], lr=learning_rate)

# Make first prediction

print(f"Prediction without training; y_pred = {forward(5):.4f} ")

# Define the training loop

for epoch in range(n_iteration):
    ''' The training is done here
    1. Make the foreward prediction
    2. Calculate the loss
    3. Differentiate the loss w.r.t the variables(backwar propagation)
    4. Correct/optimize the weights
    5. Zero the auto_grad call.

    '''
    y_pred = forward(X)

    # Calculate loss

    error = loss(Y, y_pred)

    # Perform backward propagation

    error.backward()

    # Optimize the weights

    optimizer.step()

    # Zero the autograd

    optimizer.zero_grad()

    # print some metric after certain epoch
    period = 100
    if epoch % period == 0:
        '''
        print metric after certain period
        '''
        print(f"epoch {epoch+1}: w ={w:.3f}, error = {error:.8f}")

print(f"Prediction after training: f(5) = {forward(5):.3f}")
