"""Training functions for different collocation point sampling methods."""
from collections.abc import Callable
from typing import Dict, Any
import torch
from torch import nn
import numpy as np
import pdb

# Custom imports.
from constants import QR_DEIMSamplingConstants as Constants
import utils


def train_uniform_random_sampling(
        model: nn.Module,
        residual_function: Callable[[nn.Module, torch.Tensor], torch.Tensor],
        n_steps: int,
        train_data: torch.Tensor,
        test_data: torch.Tensor,
        verbose: bool=False,
) -> Dict[str, Any]:
    """Train a PINN with uniform random sampling.

    Args:
        model: The PyTorch model to train.
        residual_function: A function that takes the model and data and returns
            the residual.
        n_steps: The number of training steps.
        train_data: The training data.
        test_data: The test data.

    Returns:
        A dictionary containing the following key/value pairs:
            - "train_loss_history": A list of training losses.
            - "test_loss_history": A list of test losses.
            - "best_model_weights": The model weights with the lowest test loss.
            - "train_data": The training data or collocation points.
    """
    # Initialize the optimizer.
    optimizer = torch.optim.Adam(model.parameters(), lr=Constants.LEARNING_RATE)

    # Train the model.
    best_test_loss = np.inf
    best_model_weights = model.state_dict()
    train_loss_history = []
    test_loss_history = []
    for step in range(n_steps):
        # Train loss.
        optimizer.zero_grad()
        residual = residual_function(model, train_data)
        loss = torch.mean(torch.square(residual))
        train_loss_history.append(loss.item())
        loss.backward()
        optimizer.step()

        # Test loss.
        test_residual = residual_function(model, test_data)
        test_loss = torch.mean(torch.square(test_residual))
        test_loss_history.append(test_loss.item())

        # Update the best model weights if the test loss is lower than the
        # previous best test loss.
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_model_weights = model.state_dict()

        # Print the loss if verbose is True.
        if (
            verbose and
            (step % Constants.REPORT_EVERY_N_STEPS == 0 or
            step == n_steps - 1)
        ):
            print(
                f"Step: {step}, Train Loss: {loss.item()}, "
                f"Test Loss: {test_loss.item()}"
            )

    return {
        "train_loss_history": train_loss_history,
        "test_loss_history": test_loss_history,
        "best_model_weights": best_model_weights,
        "train_data": train_data.detach().cpu().numpy(),
    }


def train_adaptive_greedy_sampling(
        model: nn.Module,
        residual_function: Callable[[nn.Module, torch.Tensor], torch.Tensor],
        n_steps: int,
        train_data: torch.Tensor,
        test_data: torch.Tensor,
        point_budget: int,
        n_refinement_points=10000,
        top_n_refinement=250,
        refinement_period=1000,
        verbose=False,
) -> Dict[str, Any]:
    """Train a PINN with adaptive greedy sampling.

    Args:
        model: The PyTorch model to train.
        residual_function: A function that takes the model and data and returns
            the residual.
        n_steps: The number of training steps.
        train_data: The training data.
        test_data: The test data.
        point_budget: The maximum number of collocation points.
        n_refinement_points: The number of points to sample for refinement.
        top_n_refinement: The number of points to pick for refinement.
        refinement_period: The number of steps between refinements.
        verbose: Whether to print the loss during training.

    Returns:
        A dictionary containing the following key/value pairs:
            - "train_loss_history": A list of training losses.
            - "test_loss_history": A list of test losses.
            - "best_model_weights": The model weights with the lowest test loss.
            - "train_data": The training data or collocation points.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=Constants.LEARNING_RATE)
    best_test_loss = np.inf
    best_model_weights = model.state_dict()
    train_loss_history = []
    test_loss_history = []

    # Set up the refinement points.
    refinement_points = torch.rand(n_refinement_points, 2).to(Constants.DEVICE)
    refinement_points[:, 0] = utils.normalize_data(refinement_points[:, 0])
    refinement_points.requires_grad_(True)

    for step in range(n_steps):
        # Train loss.
        optimizer.zero_grad()
        residual = residual_function(model, train_data)
        loss = torch.mean(torch.square(residual))
        train_loss_history.append(loss.item())
        loss.backward(retain_graph=True)
        optimizer.step()

        # Test loss.
        test_residual = residual_function(model, test_data)
        test_loss = torch.mean(torch.square(test_residual))
        test_loss_history.append(test_loss.item())
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_model_weights = model.state_dict()

        if (
            verbose and
            (step % Constants.REPORT_EVERY_N_STEPS == 0 or
            step == n_steps - 1)
        ):
            print(
                f"Step: {step}, Train Loss: {loss.item()}, "
                f"Test Loss: {test_loss.item()}"
            )

        # Greedy refinement based on residual.
        if (
            step % refinement_period and
            step > 0 and
            train_data.shape[0] < point_budget
        ):
            # Compute residual on refinement points.
            refinement_residual = residual_function(
                model, refinement_points
            )

            # Compute top-n points with highest residual values.
            top_indices = torch.topk(
                torch.abs(refinement_residual),
                top_n_refinement,
                dim=0
            ).indices
            top_points = refinement_points[top_indices]

            # Add these points to the training data.
            train_data = torch.cat([train_data, top_points], dim=0)

            # Reset refinement points.
            refinement_points = torch.rand(
                n_refinement_points, 2
            ).to(Constants.DEVICE)
            refinement_points[:, 0] = utils.normalize_data(
                refinement_points[:, 0]
            )
            refinement_points.requires_grad_(True)

            # Make sure that the training set size stays within the point
            # budget.
            if train_data.shape[0] > point_budget:
                train_data = train_data[:point_budget, :]

    return {
        "train_loss_history": train_loss_history,
        "test_loss_history": test_loss_history,
        "best_model_weights": best_model_weights,
        "train_data": train_data.detach().cpu().numpy()
    }


def train_adaptive_distribution_sampling(
        model: nn.Module,
        residual_function: Callable[[nn.Module, torch.Tensor], torch.Tensor],
        n_steps: int,
        train_data: torch.Tensor,
        test_data: torch.Tensor,
        point_budget: int,
        n_refinement_points=10000,
        top_n_refinement=250,
        refinement_period=1000,
        verbose=False,
) -> Dict[str, Any]:
    """Train a PINN with adaptive distribution sampling.

    Args:
        model: The PyTorch model to train.
        residual_function: A function that takes the model and data and returns
            the residual.
        n_steps: The number of training steps.
        train_data: The training data.
        test_data: The test data.
        point_budget: The maximum number of collocation points.
        n_refinement_points: The number of points to sample for refinement.
        top_n_refinement: The number of points to pick for refinement.
        refinement_period: The number of steps between refinements.
        verbose: Whether to print the loss during training.

    Returns:
        A dictionary containing the following key/value pairs:
            - "train_loss_history": A list of training losses.
            - "test_loss_history": A list of test losses.
            - "best_model_weights": The model weights with the lowest test loss.
            - "train_data": The training data or collocation points.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=Constants.LEARNING_RATE)
    best_test_loss = np.inf
    best_model_weights = model.state_dict()
    train_loss_history = []
    test_loss_history = []

    # Points to  pick from for refinement
    refinement_points = torch.rand(n_refinement_points, 2).to(Constants.DEVICE)
    refinement_points[:, 0] = utils.normalize_data(refinement_points[:, 0])
    refinement_points.requires_grad_(True)

    for step in range(n_steps):
        # Train loss.
        optimizer.zero_grad()
        residual = residual_function(model, train_data)
        loss = torch.mean(torch.square(residual))
        train_loss_history.append(loss.item())
        loss.backward(retain_graph=True)
        optimizer.step()

        # Test loss.
        test_residual = residual_function(model, test_data)
        test_loss = torch.mean(torch.square(test_residual))
        test_loss_history.append(test_loss.item())
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_model_weights = model.state_dict()

        if (
            verbose and
            (step % Constants.REPORT_EVERY_N_STEPS == 0 or
            step == n_steps - 1)
        ):
            print(
                f"Step: {step}, Train Loss: {loss.item()}, "
                f"Test Loss: {test_loss.item()}"
            )

        # Distribution refinement based on residual.
        if (
            step % refinement_period and
            step > 0 and
            train_data.shape[0] < point_budget
        ):
            # Compute residual on refinement points.
            refinement_residual = residual_function(
                model, refinement_points
            )

            # Create probability distribution based on residual.
            probabilities = (
                torch.abs(refinement_residual) /
                torch.mean(refinement_residual) + 1
            )
            probabilities /= torch.sum(probabilities)
            probabilities = torch.abs(probabilities)

            # Sample from probability distribution.
            top_indices = torch.multinomial(
                probabilities, num_samples=top_n_refinement, replacement=False
            )
            top_points = refinement_points[top_indices]

            # Add these points to the training data.
            train_data = torch.cat([train_data, top_points], dim=0)

            # Reset refinement points.
            refinement_points = torch.rand(
                n_refinement_points, 2
            ).to(Constants.DEVICE)
            refinement_points[:, 0] = utils.normalize_data(
                refinement_points[:, 0]
            )
            refinement_points.requires_grad_(True)

            # Make sure that the training set size stays within the point
            # budget.
            if train_data.shape[0] > point_budget:
                train_data = train_data[:point_budget, :]

    return {
        "train_loss_history": train_loss_history,
        "test_loss_history": test_loss_history,
        "best_model_weights": best_model_weights,
        "train_data": train_data.detach().cpu().numpy()
    }


def train_adaptive_qr_deim_sampling(
        model: nn.Module,
        residual_function: Callable[[nn.Module, torch.Tensor], torch.Tensor],
        n_steps: int,
        train_data: torch.Tensor,
        test_data: torch.Tensor,
        point_budget: int,
        n_snapshot_points: int=2000,
        n_snapshots: int=1000,
        init_threshold: float=1e-1,
        min_threshold: float=1e-3,
        min_pivots: int=50,
        verbose: bool=False,
) -> Dict[str, Any]:
    """Train a PINN with QR-DEIM adaptive sampling.

    Args:
        model: The PyTorch model to train.
        residual_function: A function that takes the model and data and returns
            the residual.
        n_steps: The number of training steps.
        train_data: The training data.
        test_data: The test data.
        point_budget: The maximum number of collocation points.
        n_snapshot_points: The number of snapshot points.
        n_snapshots: The number of snapshots to take.
        threshold: The threshold for the DEIM algorithm.
        verbose: Whether to print the loss during training.

    Returns:
        A dictionary containing the following key/value pairs:
            - "train_loss_history": A list of training losses.
            - "test_loss_history": A list of test losses.
            - "best_model_weights": The model weights with the lowest test loss.
            - "train_data": The training data or collocation points.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=Constants.LEARNING_RATE)
    best_test_loss = np.inf
    best_model_weights = model.state_dict()
    train_loss_history = []
    test_loss_history = []

    # Set up snapshots and residual history.
    snapshot_points = torch.rand(n_snapshot_points, 2).to(Constants.DEVICE)
    snapshot_points[:, 0] = utils.normalize_data(snapshot_points[:, 0])
    snapshot_points.requires_grad_(True)
    snapshot_residuals = torch.zeros(
        n_snapshot_points, n_snapshots
    ).to(Constants.DEVICE)

    # Initialize pivoting threshold from initial threshold value.
    threshold = init_threshold
    for step in range(n_steps):
        # Train loss.
        optimizer.zero_grad()

        # Get residuals on training and snapshot data.
        residual = residual_function(model, train_data)

        # If we haven't reached the point budget, store the residuals.
        if train_data.shape[0] < point_budget:
            snapshot_residuals[:, step % n_snapshots] = (
                residual_function(model, snapshot_points)
            )

        # Compute the loss and update the model.
        loss = torch.mean(torch.square(residual))
        train_loss_history.append(loss.item())
        loss.backward(retain_graph=True)
        optimizer.step()

        # Test loss.
        test_residual = residual_function(model, test_data)
        test_loss = torch.mean(torch.square(test_residual))
        test_loss_history.append(test_loss.item())
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_model_weights = model.state_dict()

        # Print the loss if verbose is True.
        if (
            verbose and
            (step % Constants.REPORT_EVERY_N_STEPS == 0 or
            step == n_steps - 1)
        ):
            print(
                f"Step: {step}, Train Loss: {loss.item()}, "
                f"Test Loss: {test_loss.item()}"
            )

        # Refinement based on QR-DEIM.
        if (
            step % n_snapshots == 0 and
            step > 0 and
            train_data.shape[0] < point_budget
        ):
            # Detach train_data to ensure it's no longer part of the computation
            # graph.
            train_data = train_data.detach()

            # Compute the DEIM pivots.
            snapshot_residuals = snapshot_residuals.detach().cpu().numpy()
            pivot_indices = utils.get_deim_pivots(
                snapshot_residuals,
                threshold=threshold,
                verbose=verbose
            )
            new_points = snapshot_points[pivot_indices, :]
            if len(pivot_indices) < min_pivots:
                threshold = max(min_threshold, threshold / 10)

            # Concatenate the detached train_data with new_points.
            train_data = torch.cat(
                [train_data, new_points], dim=0
            ).to(Constants.DEVICE)

            # Reset snapshot residuals.
            snapshot_points = torch.rand(
                n_snapshot_points, 2
            ).to(Constants.DEVICE)
            snapshot_points[:, 0] = utils.normalize_data(snapshot_points[:, 0])
            snapshot_points.requires_grad_(True)
            snapshot_residuals = torch.zeros(
                snapshot_points.shape[0], n_snapshots
            ).to(Constants.DEVICE)

            # Make sure that the training set size stays within the point
            # budget.
            if train_data.shape[0] > point_budget:
                train_data = train_data[:point_budget, :]

    return {
        "train_loss_history": train_loss_history,
        "test_loss_history": test_loss_history,
        "best_model_weights": best_model_weights,
        "train_data": train_data.detach().cpu().numpy(),
    }
