# Second offhand run of the logistic regression using cancer data

import torch
import torch.nn as nn
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
import numpy as np

"""Pytorch workflow

1. Generate and prepare data
2. Create the model class
3. Create constants
4. Train the model in a for loop
5. Print result

"""

"""
1. Generate and prepare data
    1. Create data from sklearn dataset
    2. Split the into data and target
    3. Get the number of samples and features
    4. Train/test split
    5. Convert to torch.tensor
    5. Normalize the data using standardscaler
    6. Reshap the input features
"""
# Load the breast cancer data
data = datasets.load_breast_cancer()

# Assign data and target

X, y = data.data, data.target

# Get the number of samples and features

n_sample, n_features = X.shape

# Train/test split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=123, test_size=0.20)

# Normalize data

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# Convert the data to torch format

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))


# Reshape input data from
# one row * n_sample columns()
# To
# n_sample rows * one column

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

'''
2. Create the model class
    1. Create class that inherits from torch.nn.Module
    2. Create super class
    3. Define the self.linear regression
    4. Create the forward method
    5. Create the loss function
'''


class Model(nn.Module):
    def __init__(self, input_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(input_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y_pred = self.linear(x)
        y_pred = self.sigmoid(y_pred)
        return y_pred


'''
3. Create constants
    1. Select optimizer
    2. Instantiate number of iterations
    3. Instantiate learning rate
    4. Instantiate the model
    5. Instantiate loss function
'''


model = Model(n_features)
n_iteration = 1000
learning_rate = 1e-2
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.BCELoss()

'''
4. Train the model in a for loop
    1. Call the forward method
    2. Call the loss method
    3. Call the optimizer
    4. Update the weights
    5. Zero the autograd function
'''

for epoch in range(n_iteration):
    forward = model(X_train)

    error = criterion(forward, y_train)

    error.backward()
    optimizer.step()
    optimizer.zero_grad()

    period = 100
    if (epoch+1) % period == 0:
        print(f"Epoch: {epoch}, loss: {error.item():.4f}")

'''
5. Print result
    1. Prediction before training
    2. Print epoch, current weights, loss
    3. Print prediction after training
    4. Print accuracy

'''
with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f" Accuracy: {acc.item():.4f}")
    print('\n')
    print(f" Actual values: {y_test[0:5].flatten()}\n")
    print(f" Predicted values: {y_predicted_cls[0:5].flatten()}")
