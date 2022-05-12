
import numpy as np

from dataclasses import dataclass

Mat = np.array

@dataclass
class KalmanSettings:
    dim_x: int
    dim_z: int
    X: Mat
    R: Mat
    H: Mat
    P: Mat
    F: Mat
    Q: Mat
