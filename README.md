
# Kalman Filter

[![codecov](https://codecov.io/gh/MichaelJL3/KalmanFilter/branch/main/graph/badge.svg?token=hX9omf3V4T)](https://codecov.io/gh/MichaelJL3/KalmanFilter)
[![Pylint](https://github.com/MichaelJL3/KalmanFilter/actions/workflows/pylint.yml/badge.svg)](https://github.com/MichaelJL3/KalmanFilter/actions/workflows/pylint.yml)
![Build](https://img.shields.io/github/checks-status/MichaelJL3/KalmanFilter/main)
[![Python 3.6](https://img.shields.io/badge/python-3.8.10-blue.svg)](https://www.python.org/downloads/release/python-3810/)

## Docs

https://en.wikipedia.org/wiki/Kalman_filter

https://www.kalmanfilter.net/background.html

## Usage Example

```python

dim_x = 6
dim_z = 2
X = np.zeros(dim_x)
F = np.zeros((dim_x, dim_x))
H = np.zeros((dim_z, dim_x))
R = np.identity(dim_z)
P = np.identity(dim_x)
Q = np.identity(dim_x)

# default construction
kalman = KalmanFilter(dim_x, dim_z)
kalman.X = X
kalman.H = H
kalman.R = R
kalman.Q = Q
kalman.F = F
kalman.P = P

# constructing with kwargs
kalman = KalmanFilter(
    dim_x,
    dim_z,
    X=X,
    H=H,
    R=R,
    Q=Q,
    F=F,
    P=P
)

# using settings + factory
settings = KalmanSettings(
    dim_x,
    dim_z,
    X=X,
    H=H,
    R=R,
    Q=Q,
    F=F,
    P=P
)

kalman = KalmanFilterFactory.create(settings)

# usage
Z = np.zeros(dim_z)
(X1, P1) = kalman.update(Z)
(X2, P2) = kalman.predict()

```