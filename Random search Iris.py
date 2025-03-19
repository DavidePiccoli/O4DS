import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time
import random

from HB_implementation import heavy_ball_manual
from PBM_implementation import proximal_bundle_method_manual
from NN_implementation import NeuralNetManual


# Load Iris dataset
iris = load_iris()
x = iris.data
y = iris.target.reshape(-1, 1)

encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=45)

# Normalize
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= np.max(x_train)
x_test /= np.max(x_test)

x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


# -- Iris Random search, HB --
# # lists of possible values for each parameter
learning_rates = [0.0001, 0.00015, 0.0005, 0.001, 0.005, 0.015, 0.01, 0.03, 0.05, 0.07, 0.1]
momentums = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def generate_params_HB(num_params_sets=10):
    params_list = []
    for _ in range(num_params_sets):
        params = {
            'learning_rate': random.choice(learning_rates),
            'momentum': random.choice(momentums),
        }
        params_list.append(params)

    return params_list

params_list = generate_params_HB(50)
results = {'Results':[], 'Startings':[], 'Times':[], 'Params':[]}

for i, params in enumerate(params_list):
    print('TESTING:', params)

    # Initialize the model
    input_dim = 4  # Number of features in the Iris dataset
    hidden_dim = 10
    output_dim = 3  # Number of classes in the Iris dataset
    model = NeuralNetManual(input_dim, hidden_dim, output_dim)

    start_time = time.time()
    # Train the model
    trained_model, loss_history = heavy_ball_manual(model, train_loader, max_iter=2000, alpha=params['learning_rate'], beta=params['momentum'], l1_lambda=0.05)
    end_time = time.time()
    training_time = end_time - start_time
    
    # store
    final_loss = loss_history[-1]
    results['Results'].append(final_loss)
    results['Startings'].append(loss_history[0])
    results['Times'].append(training_time)
    results['Params'].append(params)

    print('With:', params,
      '\nResult is:', final_loss, 'Starting point was:', loss_history[0])
    
# Create a DataFrame
df_HB = pd.DataFrame(results)
    

# -- Iris Random search, PBM --
# lists of possible values for each parameter
ts = [0.001, 0.003, 0.005, 0.008, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3]
max_subs = [5, 10, 15, 20, 25, 30, 40, 50]
m_1s = [0.0001, 0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.1]

def generate_params_PBM(num_params_sets=10):
    params_list = []
    for _ in range(num_params_sets):
        params = {
            't': random.choice(ts),
            'max_sub': random.choice(max_subs),
            'm_1': random.choice(m_1s),
        }
        params_list.append(params)

    return params_list

params_list = generate_params_PBM(50)

results = {'Results':[], 'Startings':[], 'Times':[], 'Subproblem times':[], 'Params': []}

for i, params in enumerate(params_list):
    print('TESTING:', params)

    # Initialize the model
    input_dim = 4  # Number of features in the Iris dataset
    hidden_dim = 10
    output_dim = 3  # Number of classes in the Iris dataset
    model = NeuralNetManual(input_dim, hidden_dim, output_dim)

    start_time = time.time()
    # Train the model
    trained_model, loss_history, subproblem_time = proximal_bundle_method_manual(model, train_loader, max_iter=2000, t=params['t'], 
                                                                m_1=params['m_1'], max_sub=params['max_sub'], l1_lambda=0.05, t_factor = 0.0001)
    end_time = time.time()
    training_time = end_time - start_time
    
    # store
    final_loss = loss_history[-1]
    results['Results'].append(final_loss)
    results['Startings'].append(loss_history[0])
    results['Times'].append(training_time)
    results['Subproblem times'].append(subproblem_time)
    results['Params'].append(params)

    print('With:', params,
      '\nResult is:', final_loss, 'Starting point was:', loss_history[0])
    
# Create a DataFrame
df_PBM = pd.DataFrame(results)