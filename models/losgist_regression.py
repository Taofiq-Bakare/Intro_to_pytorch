# An attempt at logistic regression offhand

import torch
import numpy as np
import torch.nn as nn
import sklearn
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

"""[Define the steps needed]

Preprocessing
    1. Create data
    2. Get the number of rows and columns
    3. Train/test split
    4. Check for null values
    5. Convert data from numpy to torch.tensor

Creat Pytorch model:
    1. Define the constants(loss function, optimizer,learning rate)
    2. Define a class for the model
    3. Define the forward method
    4.

"""

# Create the data


data = datasets.load_breast_cancer()

X, y = data.data, data.target

# Create number of data and features
n_rows, n_features = X.shape

# Train/test split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123)

# Scale data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# Convert data from numpy to torch

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))


y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

# Create model


class Model(nn.Module):
    def __init__(self, n_input_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        # sigmoid bcus it is a classification problem(0 or 1)
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


model = Model(n_features)

learning_rate = 0.01
loss = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
n_epoch = 100

# Design training loop.

for epoch in range(n_epoch):
    # Forward pass
    y_pred = model(X_train)
    error = loss(y_pred, y_train)

    # Backward pass and update
    error.backward()
    optimizer.step()

    # Zero the grad before new update
    optimizer.zero_grad()
    period = 10
    if (epoch+1) % period == 0:
        print(f'epoch: {epoch+1}, loss = {error.item():.4f}')


with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f'accuracy: {acc.item():.4f}')
