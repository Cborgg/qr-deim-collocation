import torch
import numpy as np
import matplotlib.pyplot as plt

from trainers import train_uniform_random_sampling
from models import PINN
from utils import helmholtz_output_transform, helmholtz_residual
from constants import QR_DEIMSamplingConstants as Constants

# Problem setup
x1, x2 = 0.0, 1.0
p1, p2 = 1.0, -1.0
k = 1.0

# Training and test data
n_train = 100
n_test = 1000

train_x = torch.rand(n_train, 1).to(Constants.DEVICE) * (x2 - x1) + x1
train_x.requires_grad_(True)

test_x = torch.linspace(x1, x2, n_test).unsqueeze(1).to(Constants.DEVICE)
test_x.requires_grad_(True)

# PINN model
model = PINN(
    hidden_size=64,
    num_layers=5,
    output_transform=lambda u, x: helmholtz_output_transform(u, x, x1, x2, p1, p2),
    input_dim=1,
).to(Constants.DEVICE)

# Train using uniform random sampling
results = train_uniform_random_sampling(
    model=model,
    residual_function=lambda m, d: helmholtz_residual(m, d, k),
    n_steps=5000,
    train_data=train_x,
    test_data=test_x,
)

# Restore best model and evaluate
model.load_state_dict(results["best_model_weights"])
with torch.no_grad():
    pinn_solution = model(test_x).cpu().numpy().squeeze()

# True solution for comparison
x_np = test_x.cpu().numpy().squeeze()
L = x2 - x1
A, B = p1, p2
true_solution = (
    A * np.cos(k * x_np) + ((B - A * np.cos(k * L)) / np.sin(k * L)) * np.sin(k * x_np)
)

plt.plot(x_np, pinn_solution, label="PINN")
plt.plot(x_np, true_solution, label="True")
plt.xlabel("x")
plt.ylabel("p(x)")
plt.legend()
plt.tight_layout()
plt.show()
