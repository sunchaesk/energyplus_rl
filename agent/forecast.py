
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import random

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train(model, inputs, labels, num_epochs, learning_rate):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# training data and labels
inputs = torch.randn(100, 7)
labels = torch.randn(100, 1)
print('inputs', inputs)
print('labels', labels)

# Define input and output
input_size = 7
output_size = 1

hidden_size = 64

model = MLP(input_size, hidden_size, output_size)

num_epochs = 1000
learning_rate = 0.001

train(model, inputs, labels, num_epochs, learning_rate)
