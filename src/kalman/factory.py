
from .filter import KalmanFilter
from .settings import KalmanSettings

class KalmanFilterFactory():
    def create(self, settings: KalmanSettings) -> KalmanFilter:
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
