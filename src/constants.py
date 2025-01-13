"""Constants for QR-DEIM sampling project."""
import dataclasses
import torch
import numpy as np


@dataclasses.dataclass(frozen=True)
class QR_DEIMSamplingConstants:
    """Data class for constants for QR-DEIM sampling project."""
    # General constants.
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LEARNING_RATE = 1e-3

    # Initial conditions for PDEs.
    WAVE_INITIAL_CONDITION = lambda x: torch.sin(torch.pi * x)
    ALLEN_CAHN_INITIAL_CONDITION = lambda x: x**2 * torch.cos(torch.pi * x)
    BURGERS_INITIAL_CONDITION = lambda x: -torch.sin(torch.pi * x)

    # PDE parameters.
    WAVE_SPEED = 2.0
    ALLEN_CAHN_EPSILON = np.sqrt(0.2)
    ALLEN_CAHN_DIFFUSION = 0.001
    ALLEN_CAHN_FUNCTION = lambda u: u**3 - u
    BURGERS_VISCOSITY = 0.01 / np.pi
