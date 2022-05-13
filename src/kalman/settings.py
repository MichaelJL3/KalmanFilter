
"""Kalman settings."""

from dataclasses import dataclass

import numpy as np

Mat = np.array

@dataclass
class KalmanSettings:
    """Kalman settings."""
    dim_x: int # the size of the x dimension
    dim_z: int # the size of the z dimension
    H: Mat # The observation matrix
    F: Mat # The state transition matrix
    X: Mat # The system state vector
    R: Mat # The meaurement uncertainty matrix
    P: Mat # The estimate uncertainty matrix
    Q: Mat # The process noise matrix
