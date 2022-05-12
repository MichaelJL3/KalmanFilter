
import numpy as np

from typing import Tuple

Mat = np.array

def gain(P: Mat, R: Mat, H: Mat) -> Mat:
    H_T = H.transpose()

    K_temp = np.dot(P, H_T)
    K = K_temp.dot(np.linalg.inv(np.dot(H, K_temp) + R))

    return K

def covariance_update(P: Mat, K: Mat, R: Mat, H: Mat) -> np.array:
    K_T = K.transpose()
    I = np.identity(P.shape[0])

    P_temp = I - K.dot(H)
    P_temp_T = np.transpose(P_temp)
    P1 = P_temp.dot(P).dot(P_temp_T) + K.dot(R).dot(K_T)

    return P1

def state_update(Z: Mat, X: Mat, K: Mat, H: Mat) -> Mat:
    X1 = X + K.dot(Z - H.dot(X))

    return X1

def covariance_extrapolate(F: Mat, P1: Mat, Q: Mat) -> Mat:
    F_T = F.transpose()
    P2 = F.dot(P1).dot(F_T) + Q
    
    return P2

def state_extrapolate(F: Mat, X1: Mat) -> Mat:
    X2 = F.dot(X1)
    
    return X2

def update(Z: Mat, X: Mat, P: Mat, R: Mat, H: Mat) -> Tuple[Mat, Mat]:
    K  = gain(P, R, H)
    X1 = state_update(Z, X, K, H)
    P1 = covariance_update(P, K, R, H)

    return (X1, P1)

def predict(X1: Mat, P1: Mat, F: Mat, Q: Mat) -> Tuple[Mat, Mat]:
    X2 = state_extrapolate(F, X1)
    P2 = covariance_extrapolate(F, P1, Q)

    return (X2, P2)

class KalmanFilter:
    _F: Mat
    _H: Mat
    _R: Mat
    _Q: Mat
    _P: Mat
    _X: Mat

    @property
    def F(self):
        return self._F

    @F.setter
    def F(self, F: Mat):
        if F.shape != (self.dim_x, self.dim_x):
            self.__shape_error('F', (self.dim_x, self.dim_x), F.shape)
        self._F = F

    @property
    def H(self):
        return self._H

    @H.setter
    def H(self, H: Mat):
        if H.shape != (self.dim_z, self.dim_x):
            self.__shape_error('H', (self.dim_z, self.dim_x), H.shape)
        self._H = H

    @property
    def R(self):
        return self._R

    @R.setter
    def R(self, R: Mat):
        if R.shape != (self.dim_z, self.dim_z):
            self.__shape_error('R', (self.dim_z, self.dim_z), R.shape)
        self._R = R

    @property
    def Q(self):
        return self._Q

    @Q.setter
    def Q(self, Q: Mat):
        if Q.shape != (self.dim_x, self.dim_x):
            self.__shape_error('Q', (self.dim_x, self.dim_x), Q.shape)
        self._Q = Q

    @property
    def P(self):
        return self._P

    @P.setter
    def P(self, P: Mat):
        if P.shape != (self.dim_x, self.dim_x):
            self.__shape_error('P', (self.dim_x, self.dim_x), P.shape)
        self._P = P

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, X: Mat):
        if X.shape != (self.dim_x,):
            self.__shape_error('X', (self.dim_x,), X.shape)
        self._X = X

    @property
    def V(self) -> Mat:
        return self._V
    
    def __init__(self, dim_x: int, dim_z: int, **kwargs):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.F = kwargs.get('F')
        self.H = kwargs.get('H')
        self.X = kwargs.get('X', np.zeros(dim_x))
        self.R = kwargs.get('R', np.identity(dim_z))
        self.Q = kwargs.get('Q', np.identity(dim_x))
        self.P = kwargs.get('P', np.identity(dim_x))

    def __shape_error(self, name: str, shapeA: Tuple[int, int], shapeB: Tuple[int, int]):
        raise ValueError(f'{name} should have shape {shapeA} cannot set to {shapeB}')

    def update(self, Z: Mat) -> Tuple[Mat, Mat]:
        (X1, P1) = update(Z, self.X, self.P, self.R, self.H)
        self._V = Z - self.H.dot(X1)
        self.X = X1
        self.P = P1
        return (X1, P1)

    def predict(self) -> Tuple[Mat, Mat]:
        (X2, P2) = predict(self.X, self.P, self.F, self.Q)
        self.X = X2
        self.P = P2
        return (X2, P2)
