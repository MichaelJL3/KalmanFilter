
"""Kalman settings."""

from dataclasses import dataclass

import numpy as np

Mat = np.array

@dataclass
class KalmanSettings:
    """Kalman settings."""
    dim_x: int # the size of the x dimension
    dim_z: int # the size of the z dimension
    X: Mat = None # The system state vector
    P: Mat = None # The estimate uncertainty matrix
    F: Mat = None # The state transition matrix
    Q: Mat = None # The process noise matrix
    H: Mat = None # The observation matrix
    R: Mat = None # The meaurement uncertainty matrix

    def __post_init__(self):
        """Fill defaults for remaining settings."""
        if (self.dim_x < 0 or self.dim_z < 0):
            dimensions = (self.dim_x, self.dim_z)
            raise ValueError(f"dimensions cannot be less than 0, actual {dimensions}")

        self.X = np.zeros(self.dim_x) if self.X is None else self.X
        self.P = np.identity(self.dim_x) if self.P is None else self.P
        self.F = np.zeros((self.dim_x, self.dim_x)) if self.F is None else self.F
        self.Q = np.zeros((self.dim_x, self.dim_x)) if self.Q is None else self.Q
        self.H = np.zeros((self.dim_x, self.dim_z)) if self.H is None else self.H
        self.R = np.zeros((self.dim_z, self.dim_z)) if self.R is None else self.R
