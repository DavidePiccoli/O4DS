import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# 1 layer dense neural network
class NeuralNetManual:
    def __init__(self, input_dim, hidden_dim, output_dim):
        # Initialize weights and biases for each layer with float32
        self.w1 = torch.randn(input_dim, hidden_dim, dtype=torch.float32, requires_grad=False)
        self.b1 = torch.zeros(hidden_dim, dtype=torch.float32, requires_grad=False)
        self.w2 = torch.randn(hidden_dim, output_dim, dtype=torch.float32, requires_grad=False)
        self.b2 = torch.zeros(output_dim, dtype=torch.float32, requires_grad=False)

    def forward(self, x):
        # Forward
        self.z1 = x @ self.w1 + self.b1
        self.a1 = torch.tanh(self.z1)  # Tanh
        self.z2 = self.a1 @ self.w2 + self.b2
        return torch.tanh(self.z2)  # Tanh activation on the output layer

    # Here we define the operations for backropgation manually
    def backward(self, x, y, output, l1_lambda=0.01):
        batch_size = x.shape[0]
        
        # output layer
        dL_dz2 = (output - y) / batch_size * (1 - output**2)  # MSE loss gradient with tanh derivative
        
        self.dL_dw2 = self.a1.T @ dL_dz2 + l1_lambda * torch.sign(self.w2)  #Add L1 regularization gradient
        self.dL_db2 = dL_dz2.sum(dim=0)
        
        # hidden layer
        dL_da1 = dL_dz2 @ self.w2.T
        dL_dz1 = dL_da1 * (1 - self.a1**2)  # Gradient of tanh
        
        # w1 and b1
        self.dL_dw1 = x.T @ dL_dz1 + l1_lambda * torch.sign(self.w1)  #Add L1 regularization gradient
        self.dL_db1 = dL_dz1.sum(dim=0)

    def update_params(self, alpha, beta, prev_update):
        # Update
        for param, grad, prev in zip(
            [self.w1, self.b1, self.w2, self.b2],
            [self.dL_dw1, self.dL_db1, self.dL_dw2, self.dL_db2],
            prev_update
        ):
            update = -alpha * grad + beta * prev
            param += update
            prev[:] = update  # Update

    def print_structure(self):
        print("Neural Network Structure:")
        print(f"Input Layer -> {self.w1.shape[0]} neurons")
        print(f"Hidden Layer 1 -> {self.w1.shape[1]} neurons (tanh activation)")
        print(f"Output Layer -> {self.w2.shape[1]} neurons (tanh activation)")



# 2 layers version
# dense neural network
class NeuralNetManual_2l:
    def __init__(self, input_dim, hidden_dim, output_dim):
        # Initialize weights and biases for each layer with float32
        self.w1 = torch.randn(input_dim, hidden_dim, dtype=torch.float32, requires_grad=False)
        self.b1 = torch.zeros(hidden_dim, dtype=torch.float32, requires_grad=False)
        self.w2 = torch.randn(hidden_dim, hidden_dim, dtype=torch.float32, requires_grad=False)
        self.b2 = torch.zeros(hidden_dim, dtype=torch.float32, requires_grad=False)
        self.w3 = torch.randn(hidden_dim, output_dim, dtype=torch.float32, requires_grad=False)
        self.b3 = torch.zeros(output_dim, dtype=torch.float32, requires_grad=False)

    def forward(self, x):
        # Forward pass
        self.z1 = x @ self.w1 + self.b1
        self.a1 = torch.tanh(self.z1)  # Tanh
        self.z2 = self.a1 @ self.w2 + self.b2
        self.a2 = torch.tanh(self.z2)  # Tanh 
        self.z3 = self.a2 @ self.w3 + self.b3
        return torch.tanh(self.z3)  # Tanh activation on the output layer

    # Here we define the operations for backropgation manually
    def backward(self, x, y, output, l1_lambda=0.01):
        batch_size = x.shape[0]
        
        # output layer
        dL_dz3 = (output - y) / batch_size * (1 - output**2)  # MSE loss gradient with tanh derivative
        
        # w3 and b3
        self.dL_dw3 = self.a2.T @ dL_dz3 + l1_lambda * torch.sign(self.w3)  # Add L1 regularization gradient
        self.dL_db3 = dL_dz3.sum(dim=0)
        
        # second hidden layer
        dL_da2 = dL_dz3 @ self.w3.T
        dL_dz2 = dL_da2 * (1 - self.a2**2)  # Gradient of tanh
        
        # w2 and b2
        self.dL_dw2 = self.a1.T @ dL_dz2 + l1_lambda * torch.sign(self.w2)
        self.dL_db2 = dL_dz2.sum(dim=0)
        
        # first hidden layer
        dL_da1 = dL_dz2 @ self.w2.T
        dL_dz1 = dL_da1 * (1 - self.a1**2)  # Gradient of tanh
        
        # w1 and b1
        self.dL_dw1 = x.T @ dL_dz1 + l1_lambda * torch.sign(self.w1)  # Add L1 regularization gradient
        self.dL_db1 = dL_dz1.sum(dim=0)

    def update_params(self, alpha, beta, prev_update):
        # Update
        for param, grad, prev in zip(
            [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3],
            [self.dL_dw1, self.dL_db1, self.dL_dw2, self.dL_db2, self.dL_dw3, self.dL_db3],
            prev_update
        ):
            update = -alpha * grad + beta * prev
            param += update
            prev[:] = update  # Update

    def print_structure(self):
        print("Neural Network Structure:")
        print(f"Input Layer -> {self.w1.shape[0]} neurons")
        print(f"Hidden Layer 1 -> {self.w1.shape[1]} neurons (tanh activation)")
        print(f"Hidden Layer 2 -> {self.w2.shape[1]} neurons (tanh activation)")
        print(f"Output Layer -> {self.w3.shape[1]} neurons (tanh activation)")