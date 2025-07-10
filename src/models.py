"""PINN models for Wave, Allen-Cahn, and Burgers' equations."""
from collections.abc import Callable
import torch
from torch import nn


class FeedForwardNet(nn.Module):
    """Generic feedforward neural network.

    Attributes:
        first_layer: First layer of the neural network. Takes ``input_dim``
            inputs and maps to ``hidden_size``.
        hidden_layers: List of hidden layers. Each hidden layer is a linear
            layer followed by a Tanh activation function. Each layer has
            hidden_size neurons.
        last_layer: Last layer of the neural network. Maps from hidden_size to
            one output.
    """
    def __init__(self, hidden_size: int, num_layers: int, input_dim: int = 2):
        super().__init__()
        self.first_layer = nn.Linear(input_dim, hidden_size)
        self.hidden_layers = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.Tanh()
                )
                for _ in range(num_layers)
            ]
        )
        self.last_layer = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first_layer(x)
        x = self.hidden_layers(x)
        x = self.last_layer(x)
        return x


class PINN(FeedForwardNet):
    """PINN that strongly enforces initial and boundary conditions.

    Inherits from FeedForwardNet. The forward method is the same as the
    FeedForwardNet, but transforms the output to enforce the initial and
    boundary conditions.

    Attributes:
        output_transform: Function that returns the initial and boundary
            conditions. For example, for some initial condition u0(x), boundary
            condition u(-1, 0) = u(1, 0) = 0, and output u, the transform would
            be u = u0(x) + t * (x^2 - 1) * u. This input would be given as a
            lambda function like lambda u, x, t: u0(x) + t * (x**2 - 1) * u.
    """
    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        output_transform: Callable[[torch.Tensor], torch.Tensor],
        input_dim: int = 2,
    ):
        super().__init__(hidden_size, num_layers, input_dim)
        self.output_transform = output_transform
        self.input_dim = input_dim

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Forward pass for the PINN."""
        # Compute the output of the neural network.
        u = super().forward(data)

        # Transform the output to enforce the initial and boundary conditions.
        if self.input_dim == 2:
            # Expand the dimensions of x and t to match the shape of ``u`` and
            # avoid broadcasting issues.
            x = data[:, 0].unsqueeze(1)
            t = data[:, 1].unsqueeze(1)
            return self.output_transform(u, x, t)
        elif self.input_dim == 1:
            x = data[:, 0].unsqueeze(1)
            return self.output_transform(u, x)
        else:
            raise ValueError(f"Unsupported input_dim: {self.input_dim}")