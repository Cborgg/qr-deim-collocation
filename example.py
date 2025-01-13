"""Example showing how to use QR-DEIM training function."""
import torch
from torch import nn
import numpy as np
from scipy import linalg
from scipy.io import loadmat
import matplotlib.pyplot as plt

# Custom imports.
from trainers import train_adaptive_qr_deim_sampling
from constants import QR_DEIMSamplingConstants as Constants
from models import PINN
import utils

# Define train and test datasets. We scale the x component of the data to be in
# the interval [-1, 1].

# Use 100 initial training points.
n_train = 100
n_test = 10000

train_data = torch.rand((n_train, 2)).to(Constants.DEVICE)
train_data[:, 0] = utils.normalize_data(train_data[:, 0])
train_data.requires_grad_(True)

test_data = torch.rand((n_test, 2)).to(Constants.DEVICE)
test_data[:, 0] = utils.normalize_data(test_data[:, 0])
test_data.requires_grad_(True)

# Define model.
model = PINN(
    hidden_size=64,
    num_layers=5,
    output_transform=utils.wave_equation_output_transform
).to(Constants.DEVICE)

# Train PINN.
results = train_adaptive_qr_deim_sampling(
    model=model,
    residual_function=utils.wave_equation_residual,
    n_steps=30000,
    train_data=train_data,
    test_data=test_data,
    point_budget=500
)
