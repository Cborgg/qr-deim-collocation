"""Utility functions for PINN sampling."""
from typing import List, Dict
import torch
from torch import nn
import numpy as np
from scipy import linalg

# Custom imports.
from constants import QR_DEIMSamplingConstants as Constants


def normalize_data(data: torch.Tensor) -> torch.Tensor:
    """Normalize data to be between -1 and 1."""
    return 2 * (data - data.min()) / (data.max() - data.min()) - 1


def compute_relative_errors(
    true_solution: torch.Tensor,
    predicted_solution: torch.Tensor,
) -> Dict[str, float]:
    """Compute relative l2 and l-infinity errors.

    Args:
        true_solution: True solution tensor.
        predicted_solution: Predicted solution tensor.

    Returns:
        Dictionary of relative L2 and L-infinity errors.
    """
    # Initialize dictionary to store errors.
    errors = {}

    # Compute relative l2 error.
    errors["relative_l2_error"] = (
        torch.linalg.norm(true_solution - predicted_solution) /
        torch.linalg.norm(true_solution)
    ).item()

    # Compute relative l-infinity error.
    errors["relative_linf_error"] = (
        torch.max(torch.abs(true_solution - predicted_solution)) / 
        torch.max(torch.abs(true_solution))
    ).item()
    return errors


def get_deim_pivots(
    snapshots: torch.Tensor,
    threshold: float=1e-2,
    verbose: bool=False
) -> List[int]:
    """Get pivot points for QR-DEIM sampling.

    Get the index of points to sample from the snapshots using QR-DEIM. Given
    a snapshot matrix, the function computes the SVD of the snapshots and
    then uses QR pivoting to get the index of the points to sample.

    Args:
        snapshots: Snapshot matrix of shape (n_snapshot_points, n_snapshots).
        threshold: Threshold for the singular values.
        verbose: Print the number of pivots used.

    Returns:
        List of indices of the pivot points.
    """
    [U, S, _] = linalg.svd(snapshots)
    n_pivots = np.sum(S > threshold)
    if verbose:
        print(f"Using {n_pivots} pivots...")
    [_, _, P] = linalg.qr(U[:, 0:n_pivots].T, pivoting=True)
    return list(P[0:n_pivots])


def wave_equation_output_transform(u, x, t):
    """Transform the output for the wave equation."""
    return u * (t**2 * (x**2 - 1)) + Constants.WAVE_INITIAL_CONDITION(x)


def allen_cahn_equation_output_transform(u, x, t):
    """Transform the output for the Allen-Cahn equation."""
    return u * (t * (x**2 - 1)) + Constants.ALLEN_CAHN_INITIAL_CONDITION(x)


def burgers_equation_output_transform(u, x, t):
    """Transform the output for the Burgers equation."""
    return u * (t * (x**2 - 1)) + Constants.BURGERS_INITIAL_CONDITION(x)


def helmholtz_output_transform(
    u: torch.Tensor,
    x: torch.Tensor,
    x1: float = 0.0,
    x2: float = 1.0,
    p1: float = 1.0,
    p2: float = -1.0,
):
    """Transform the output for the 1D Helmholtz equation.

    This enforces the boundary conditions ``p(x1)=p1`` and ``p(x2)=p2`` by
    adding a function that is zero at ``x1`` and ``x2``.
    """
    bc = p1 + (p2 - p1) * (x - x1) / (x2 - x1)
    return bc + (x - x1) * (x - x2) * u


def wave_equation_true_solution(x, t):
    """Compute the true solution for the wave equation."""
    return (
        torch.sin(torch.pi * x) * torch.cos(Constants.WAVE_SPEED * torch.pi * t)
    )


def wave_equation_residual(
    model: nn.Module,
    data: torch.Tensor,
    wave_speed: float=Constants.WAVE_SPEED
) -> torch.Tensor:
    """Residual function for the wave equation.

    Wave equation: utt = wave_speed^2 * uxx.

    Args:
        model: Neural network model.
        data: Input data tensor.
        wave_speed: Speed of the wave.

    Returns:
        Residual tensor for the wave equation.
    """
    u = model(data)
    u_grad = torch.autograd.grad(
        u, data, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]
    u_x = u_grad[:, 0]
    u_t = u_grad[:, 1]
    u_xx = torch.autograd.grad(
        u_x, data, grad_outputs=torch.ones_like(u_x), create_graph=True
    )[0][:, 0]
    u_tt = torch.autograd.grad(
        u_t, data, grad_outputs=torch.ones_like(u_t), create_graph=True
    )[0][:, 1]
    return u_tt - wave_speed**2 * u_xx


def allen_cahn_equation_residual(
    model: nn.Module,
    data: torch.Tensor,
    diffusion=Constants.ALLEN_CAHN_DIFFUSION,
    epsilon=Constants.ALLEN_CAHN_EPSILON,
    function=Constants.ALLEN_CAHN_FUNCTION
) -> torch.Tensor:
    """Residual function for the Allen-Cahn equation.

    Allen-Cahn equation: u_t - diffusion*u_xx = -(1/epsilon)**2 * function(u).

    Args:
        model: Neural network model.
        data: Input data tensor.
        diffusion: Diffusion coefficient.
        epsilon: Regularization parameter.
        function: Function for the Allen-Cahn equation.

    Returns:
        Residual tensor for the Allen-Cahn equation.
    """
    u = model(data)
    u_grad = torch.autograd.grad(
        u, data, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]
    u_x = u_grad[:, 0]
    u_t = u_grad[:, 1]
    u_xx = torch.autograd.grad(
        u_x, data, grad_outputs=torch.ones_like(u_x), create_graph=True
    )[0][:, 0]
    return u_t - diffusion*u_xx + (1. / epsilon)**2 * function(u).squeeze(-1)


def burgers_equation_residual(
    model: nn.Module,
    data: torch.Tensor,
    viscosity=Constants.BURGERS_VISCOSITY
) -> torch.Tensor:
    """Residual function for the Burgers' equation.

    Burgers' equation: u_t + u * u_x = viscosity*u_xx.

    Args:
        model: Neural network model.
        data: Input data tensor.
        viscosity: Viscosity coefficient.

    Returns:
        Residual tensor for the Burgers' equation.
    """
    u = model(data)
    u_grad = torch.autograd.grad(
        u, data, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]
    u_x = u_grad[:, 0]
    u_t = u_grad[:, 1]
    u_xx = torch.autograd.grad(
        u_x, data, grad_outputs=torch.ones_like(u_x), create_graph=True
    )[0][:, 0]
    return u_t + u.squeeze(-1) * u_x - viscosity * u_xx


def helmholtz_residual(
    model: nn.Module,
    data: torch.Tensor,
    k: float = 1.0,
) -> torch.Tensor:
    """Residual for the 1D Helmholtz equation ``p'' + k^2 p = 0``."""
    p = model(data)
    p_x = torch.autograd.grad(
        p, data, grad_outputs=torch.ones_like(p), create_graph=True
    )[0][:, 0]
    p_xx = torch.autograd.grad(
        p_x, data, grad_outputs=torch.ones_like(p_x), create_graph=True
    )[0][:, 0]
    return p_xx + k**2 * p.squeeze(-1)
