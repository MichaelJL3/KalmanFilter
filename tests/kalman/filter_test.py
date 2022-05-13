
"""Kalman settings tests."""

from unittest import TestCase
from parameterized import parameterized

import numpy as np

from src.kalman.filter import KalmanFilter
from src.kalman.filter import gain
from src.kalman.filter import covariance_update
from src.kalman.filter import covariance_extrapolate
from src.kalman.filter import state_update
from src.kalman.filter import state_extrapolate
from src.kalman.filter import update
from src.kalman.filter import predict

from tests.kalman.data import KalmanDataEx0

class TestKalmanFilter(TestCase):
    """KalmanFilter tests.

    Args:
        TestCase (TestCase): The test base class.
    """

    @parameterized.expand([
        [{ 'X': np.zeros(0) }],
        [{ 'P': np.zeros(0) }],
        [{ 'Q': np.zeros(0) }],
        [{ 'R': np.zeros(0) }],
        [{ 'H': np.zeros(0) }],
        [{ 'F': np.zeros(0) }]
    ])
    def test_incorrect_dimensions_throws(self, kwargs):
        """Incorrect dimensions throws error.

        Args:
            kwargs (dict): The dictionary of values.
        """
        with self.assertRaises(ValueError):
            KalmanFilter(1, 1, **kwargs)

    def test_overwrite_with_incorrect_dimensions_throws(self):
        """Incorrect dimensions throws error."""

        kalman = KalmanFilter(1, 1)

        with self.assertRaises(ValueError):
            kalman.X = np.zeros(0)
        with self.assertRaises(ValueError):
            kalman.P = np.zeros(0)
        with self.assertRaises(ValueError):
            kalman.Q = np.zeros(0)
        with self.assertRaises(ValueError):
            kalman.R = np.zeros(0)
        with self.assertRaises(ValueError):
            kalman.F = np.zeros(0)
        with self.assertRaises(ValueError):
            kalman.H = np.zeros(0)

    def test_properties_are_defaulted(self):
        """Properties are defaulted if not defined."""

        kalman = KalmanFilter(1, 1)
        self.assertIsNotNone(kalman.X)
        self.assertIsNotNone(kalman.P)
        self.assertIsNotNone(kalman.Q)
        self.assertIsNotNone(kalman.R)
        self.assertIsNotNone(kalman.F)
        self.assertIsNotNone(kalman.H)

    def test_properties_are_overwritable(self):
        """Properties are overwritable."""

        kalman = KalmanFilter(1, 1)
        kalman.X = np.zeros(1)
        kalman.P = np.identity(1)
        kalman.Q = np.identity(1)
        kalman.R = np.identity(1)
        kalman.H = np.zeros((1, 1))
        kalman.F = np.zeros((1, 1))

        self.assertEqual((1,), kalman.X.shape)
        self.assertEqual((1, 1), kalman.P.shape)
        self.assertEqual((1, 1), kalman.H.shape)
        self.assertEqual((1, 1), kalman.R.shape)
        self.assertEqual((1, 1), kalman.F.shape)
        self.assertEqual((1, 1), kalman.Q.shape)

    @parameterized.expand([
        [
            KalmanDataEx0.get_p_1_0(),
            KalmanDataEx0.get_r(),
            KalmanDataEx0.get_h(),
            KalmanDataEx0.get_k()
        ]
    ])
    def test_gain_function(self, P, R, H, exp_k):
        """Should calculate gain."""
        K = gain(P, R, H)

        self.assertIsNotNone(K)
        np.testing.assert_almost_equal(exp_k, K, decimal=2)

    @parameterized.expand([
        [
            KalmanDataEx0.get_p_1_0(),
            KalmanDataEx0.get_k(),
            KalmanDataEx0.get_r(),
            KalmanDataEx0.get_h(),
            KalmanDataEx0.get_p_1_1()
        ]
    ])
    def test_covariance_update_function(self, P, K, R, H, exp_p):
        """Should calculate covariance update."""
        next_p = covariance_update(P, K, R, H)
        print(next_p)

        self.assertIsNotNone(next_p)
        np.testing.assert_almost_equal(exp_p, next_p, decimal=1)

    @parameterized.expand([
        [
            KalmanDataEx0.get_p_1_1(),
            KalmanDataEx0.get_f(),
            KalmanDataEx0.get_q(),
            KalmanDataEx0.get_p_2_1(),
        ]
    ])
    def test_covariance_extrapolate_function(self, P, F, Q, exp_p):
        """Should calculate covariance extrapolation."""
        next_p = covariance_extrapolate(F, P, Q)

        self.assertIsNotNone(next_p - exp_p)
        np.testing.assert_almost_equal(exp_p, next_p, decimal=1)

    @parameterized.expand([
        [
            KalmanDataEx0.get_z(),
            KalmanDataEx0.get_x_1_0(),
            KalmanDataEx0.get_k(),
            KalmanDataEx0.get_h(),
            KalmanDataEx0.get_x_1_1()
        ]
    ])
    def test_state_update_function(self, Z, X, K, H, exp_x):
        """Should calculate state update."""
        next_x = state_update(Z, X, K, H)

        self.assertIsNotNone(next_x)
        np.testing.assert_almost_equal(exp_x, next_x, decimal=1)

    @parameterized.expand([
        [
            KalmanDataEx0.get_f(),
            KalmanDataEx0.get_x_1_1(),
            KalmanDataEx0.get_x_2_1()
        ]
    ])
    def test_state_extrapolate_function(self, F, X, exp_x):
        """Should calculate state extrapolation."""
        next_x = state_extrapolate(F, X)

        self.assertIsNotNone(next_x)
        np.testing.assert_almost_equal(exp_x, next_x, decimal=1)

    @parameterized.expand([
        [
            KalmanDataEx0.get_x_1_1(),
            KalmanDataEx0.get_p_1_1(),
            KalmanDataEx0.get_f(),
            KalmanDataEx0.get_q(),
            KalmanDataEx0.get_x_2_1(),
            KalmanDataEx0.get_p_2_1()
        ]
    ])
    def test_predict_function(self, X, P, F, Q, exp_x, exp_p):
        """Should predict current iteration."""
        (next_x, next_p) = predict(X, P, F, Q)

        self.assertIsNotNone(next_x)
        np.testing.assert_almost_equal(exp_x, next_x, decimal=1)
        np.testing.assert_almost_equal(exp_p, next_p, decimal=1)

    @parameterized.expand([
        [
            KalmanDataEx0.get_z(),
            KalmanDataEx0.get_x_1_0(),
            KalmanDataEx0.get_p_1_0(),
            KalmanDataEx0.get_r(),
            KalmanDataEx0.get_h(),
            KalmanDataEx0.get_x_1_1(),
            KalmanDataEx0.get_p_1_1()
        ]
    ])
    def test_update_function(self, Z, X, P, R, H, exp_x, exp_p):
        """Should update current iteration."""
        (next_x, next_p) = update(Z, X, P, R, H)

        self.assertIsNotNone(next_x)
        self.assertIsNotNone(next_p)
        np.testing.assert_almost_equal(exp_x, next_x, decimal=1)
        np.testing.assert_almost_equal(exp_p, next_p, decimal=1)

    @parameterized.expand([
        [
            KalmanDataEx0.get_dim_x(),
            KalmanDataEx0.get_dim_z(),
            KalmanDataEx0.get_x_1_1(),
            KalmanDataEx0.get_p_1_1(),
            KalmanDataEx0.get_r(),
            KalmanDataEx0.get_h(),
            KalmanDataEx0.get_f(),
            KalmanDataEx0.get_q(),
            KalmanDataEx0.get_x_2_1(),
            KalmanDataEx0.get_p_2_1()
        ]
    ])
    def test_stateful_predict_function(self, dim_x, dim_z, X, P, R, H, F, Q, exp_x, exp_p):
        """Should predict current iteration."""
        kalman = KalmanFilter(
            dim_x,
            dim_z,
            X=X,
            P=P,
            F=F,
            Q=Q,
            R=R,
            H=H
        )

        (next_x, next_p) = kalman.predict()

        self.assertIsNotNone(next_x)
        np.testing.assert_almost_equal(exp_x, next_x, decimal=1)
        np.testing.assert_almost_equal(exp_p, next_p, decimal=1)

    @parameterized.expand([
        [
            KalmanDataEx0.get_dim_x(),
            KalmanDataEx0.get_dim_z(),
            KalmanDataEx0.get_z(),
            KalmanDataEx0.get_x_1_0(),
            KalmanDataEx0.get_p_1_0(),
            KalmanDataEx0.get_r(),
            KalmanDataEx0.get_h(),
            KalmanDataEx0.get_x_1_1(),
            KalmanDataEx0.get_p_1_1(),
            KalmanDataEx0.get_v_1_1()
        ]
    ])
    def test_stateful_update_function(self, dim_x, dim_z, Z, X, P, R, H, exp_x, exp_p, exp_v):
        """Should update current iteration."""
        kalman = KalmanFilter(
            dim_x,
            dim_z,
            X=X,
            P=P,
            R=R,
            H=H
        )

        (next_x, next_p) = kalman.update(Z)

        self.assertIsNotNone(next_x)
        self.assertIsNotNone(next_p)
        self.assertIsNotNone(kalman.V)
        np.testing.assert_almost_equal(exp_v, kalman.V, decimal=1)
        np.testing.assert_almost_equal(exp_x, next_x, decimal=1)
        np.testing.assert_almost_equal(exp_p, next_p, decimal=1)
