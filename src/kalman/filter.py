
"""Kalman filter."""

from typing import Tuple

import numpy as np

Mat = np.array

def gain(P: Mat, R: Mat, H: Mat) -> Mat:
    """Calculate kalman gain.

    Args:
        P (Mat): The estimate uncertainty matrix.
        R (Mat): The meaurement uncertainty matrix.
        H (Mat): The observation matrix.

    Returns:
        Mat: The gain matrix.
    """
    h_t = H.transpose()

    k_temp = np.dot(P, h_t)
    K = k_temp.dot(np.linalg.inv(np.dot(H, k_temp) + R))

    return K

def covariance_update(P: Mat, K: Mat, R: Mat, H: Mat) -> Mat:
    """Calculate the covariance update.

    Args:
        P (Mat): The estimate uncertainty matrix.
        K (Mat): The gain matrix.
        R (Mat): The meaurement uncertainty matrix.
        H (Mat): The observation matrix.

    Returns:
        Mat: The covariance matrix.
    """
    k_t = K.transpose()
    I = np.identity(P.shape[0])

    p_temp = I - K.dot(H)
    p_temp_t = np.transpose(p_temp)
    P1 = p_temp.dot(P).dot(p_temp_t) + K.dot(R).dot(k_t)

    return P1

def state_update(Z: Mat, X: Mat, K: Mat, H: Mat) -> Mat:
    """Calculate the state updates.

    Args:
        Z (Mat): The measurement vector.
        X (Mat): The state vector.
        K (Mat): The gain matrix.
        H (Mat): The observation matrix.

    Returns:
        Mat: The updated state vector.
    """
    X1 = X + K.dot(Z - H.dot(X))

    return X1

def covariance_extrapolate(F: Mat, P1: Mat, Q: Mat) -> Mat:
    """Calculate the covariance prediction.

    Args:
        F (Mat):  The state transition matrix.
        P1 (Mat): The Estimate uncertainty matrix.
        Q (Mat):  The process noise matrix.

    Returns:
        Mat: Predicted uncertainty matrix.
    """
    f_t = F.transpose()
    P2 = F.dot(P1).dot(f_t) + Q

    return P2

def state_extrapolate(F: Mat, X1: Mat, G: Mat=None, U: Mat=None, W: Mat=None) -> Mat:
    """Calculate the state prediction.

    Args:
        F (Mat):  The state transition matrix.
        X1 (Mat): The state vector.
        U (Mat):  (Optional) The input vector.
        G (Mat):  (Optional) The control matrix.
        W (Mat):  (Optional) The process noise vector.

    Returns:
        Mat: Predicted state matrix.
    """
    X2 = F.dot(X1)
    if G and U and W:
        X2 += G.dot(U) + W

    return X2

def update(Z: Mat, X: Mat, P: Mat, R: Mat, H: Mat) -> Tuple[Mat, Mat]:
    """Update with next measurement.

    Args:
        Z (Mat): The measurement vector.
        X (Mat): The system state vector.
        P (Mat): The estimate uncertainty matrix.
        R (Mat): The meaurement uncertainty matrix.
        H (Mat): The observation matrix.

    Returns:
        Tuple[Mat, Mat]: The updated state and uncertainty matrices.
    """
    K  = gain(P, R, H)
    X1 = state_update(Z, X, K, H)
    P1 = covariance_update(P, K, R, H)

    return (X1, P1)

def predict(X1: Mat, P1: Mat, F: Mat, Q: Mat, \
    U: Mat=None, G: Mat=None, W: Mat=None) -> Tuple[Mat, Mat]:
    """Predict actual state.

    Args:
        X1 (Mat): The system state vector.
        P1 (Mat): The estimate uncertainty matrix.
        F (Mat): The state transition matrix.
        Q (Mat): The process noise matrix.
        U (Mat):  (Optional) The input vector.
        G (Mat):  (Optional) The control matrix.
        W (Mat):  (Optional) The process noise vector.

    Returns:
        Tuple[Mat, Mat]: The predicted state and uncertainty matrices.
    """
    X2 = state_extrapolate(F, X1, G, U, W)
    P2 = covariance_extrapolate(F, P1, Q)

    return (X2, P2)

class KalmanFilter:
    """KalmanFilter class.

    Raises:
        ValueError: If matrix dimensions are improper.
    """

    _F: Mat
    _H: Mat
    _R: Mat
    _Q: Mat
    _P: Mat
    _X: Mat
    _G: Mat
    _W: Mat

    @property
    def F(self) -> Mat:
        """Get the transition matrix.

        Returns:
            Mat: The matrix.
        """
        return self._F

    @F.setter
    def F(self, F: Mat):
        if F.shape != (self.dim_x, self.dim_x):
            KalmanFilter.__shape_error('F', (self.dim_x, self.dim_x), F.shape)
        self._F = F

    @property
    def H(self) -> Mat:
        """Get the observation matrix.

        Returns:
            Mat: The matrix.
        """
        return self._H

    @H.setter
    def H(self, H: Mat):
        if H.shape != (self.dim_z, self.dim_x):
            self.__shape_error('H', (self.dim_z, self.dim_x), H.shape)
        self._H = H

    @property
    def R(self) -> Mat:
        """Get the measurement uncertainty matrix.

        Returns:
            Mat: The matrix.
        """
        return self._R

    @R.setter
    def R(self, R: Mat):
        if R.shape != (self.dim_z, self.dim_z):
            self.__shape_error('R', (self.dim_z, self.dim_z), R.shape)
        self._R = R

    @property
    def Q(self) -> Mat:
        """Get the process noise matrix.

        Returns:
            Mat: The matrix.
        """
        return self._Q

    @Q.setter
    def Q(self, Q: Mat):
        if Q.shape != (self.dim_x, self.dim_x):
            self.__shape_error('Q', (self.dim_x, self.dim_x), Q.shape)
        self._Q = Q

    @property
    def P(self) -> Mat:
        """Get the estimate uncertainty matrix.

        Returns:
            Mat: The matrix.
        """
        return self._P

    @P.setter
    def P(self, P: Mat):
        if P.shape != (self.dim_x, self.dim_x):
            self.__shape_error('P', (self.dim_x, self.dim_x), P.shape)
        self._P = P

    @property
    def X(self) -> Mat:
        """Get the state vector.

        Returns:
            Mat: The matrix.
        """
        return self._X

    @X.setter
    def X(self, X: Mat):
        if X.shape != (self.dim_x,):
            self.__shape_error('X', (self.dim_x,), X.shape)
        self._X = X

    @property
    def W(self) -> Mat:
        """Get the process noise vector.

        Returns:
            Mat: The vector.
        """
        return self._W

    @W.setter
    def W(self, W: Mat):
        if W and W.shape != (self.dim_x,):
            self.__shape_error('W', (self.dim_x,), W.shape)
        self._W = W

    @property
    def G(self) -> Mat:
        """Get the control matrix.

        Returns:
            Mat: The matrix.
        """
        return self._G

    @G.setter
    def G(self, G: Mat):
        if G and G.shape[0] != self.dim_x:
            self.__shape_error('G', (self.dim_x,), G.shape)
        self._G = G

    @property
    def V(self) -> Mat:
        """Get the random noise vector.

        Returns:
            Mat: The vector.
        """
        return self._V

    def __init__(self, dim_x: int, dim_z: int, **kwargs):
        """Ctor

        Args:
            dim_x (int): The size of the x dimension.
            dim_z (int): The size of the z dimension.
        """
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.F = kwargs.get('F', np.zeros((dim_x, dim_x)))
        self.H = kwargs.get('H', np.zeros((dim_z, dim_x)))
        self.X = kwargs.get('X', np.zeros(dim_x))
        self.R = kwargs.get('R', np.identity(dim_z))
        self.Q = kwargs.get('Q', np.identity(dim_x))
        self.P = kwargs.get('P', np.identity(dim_x))
        self.G = kwargs.get('G')
        self.W = kwargs.get('W')
        self._V = np.zeros(dim_x)

    @staticmethod
    def __shape_error(name: str, expected: Tuple[int, int], actual: Tuple[int, int]):
        """Handle improper dimensions.

        Args:
            name (str): The name of the matrix.
            expected (Tuple[int, int]): The expected dimensions.
            actual (Tuple[int, int]): The actual dimensions.

        Raises:
            ValueError: The error for mismatched dimensionality.
        """
        raise ValueError(f'{name} should have shape {expected} cannot set to {actual}')

    def update(self, Z: Mat) -> Tuple[Mat, Mat]:
        """Update with next measurement.

        Args:
            Z (Mat): The measurement vector.

        Returns:
            Tuple[Mat, Mat]: The updated state and uncertainty matrices.
        """
        (X1, P1) = update(Z, self.X, self.P, self.R, self.H)
        self._V = Z - self.H.dot(X1)
        self.X = X1
        self.P = P1
        return (X1, P1)

    def predict(self, U: Mat=None) -> Tuple[Mat, Mat]:
        """Predict actual state.

        Args:
            U (Mat): (Optional) The control input.

        Returns:
            Tuple[Mat, Mat]: The predicted state and uncertainty matrices.
        """
        (X2, P2) = predict(self.X, self.P, self.F, self.Q, U, self.G, self.W)
        self.X = X2
        self.P = P2
        return (X2, P2)
