
"""Kalman filter factory."""

from .filter import KalmanFilter
from .settings import KalmanSettings

class KalmanFilterFactory():
    """Kalman filter factory."""

    @staticmethod
    def create(settings: KalmanSettings) -> KalmanFilter:
        """Create a new kalman filter.

        Args:
            settings (KalmanSettings): The initial settings of the filter.

        Returns:
            KalmanFilter: The kalman filter.
        """
        return KalmanFilter(
            settings.dim_x,
            settings.dim_z,
            F = settings.F,
            H = settings.H,
            X = settings.X,
            Q = settings.Q,
            P = settings.P,
            R = settings.R
        )
