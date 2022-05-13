
"""Kalman settings tests."""

from unittest import TestCase

import numpy as np

from src.kalman.factory import KalmanSettings

class TestKalmanSettings(TestCase):
    """KalmanSettings tests.

    Args:
        TestCase (TestCase): The test base class.
    """

    def test_settings_should_default(self):
        """Should fill in missing values based on dimensions."""

        settings = KalmanSettings(1, 1)
        self.assertIsNotNone(settings)
        self.assertIsNotNone(settings.X)
        self.assertIsNotNone(settings.H)
        self.assertIsNotNone(settings.R)
        self.assertIsNotNone(settings.P)
        self.assertIsNotNone(settings.F)
        self.assertIsNotNone(settings.Q)

    def test_full_settings(self):
        """Should not overwrite filled in values."""

        settings = KalmanSettings(
            dim_x = 1,
            dim_z = 1,
            X = np.zeros(1),
            H = np.zeros(1),
            R = np.zeros(1),
            P = np.zeros(1),
            F = np.zeros(1),
            Q = np.zeros(1)
        )

        self.assertIsNotNone(settings)
        self.assertEqual((1,), settings.X.shape)
        self.assertEqual((1,), settings.H.shape)
        self.assertEqual((1,), settings.R.shape)
        self.assertEqual((1,), settings.P.shape)
        self.assertEqual((1,), settings.F.shape)
        self.assertEqual((1,), settings.Q.shape)

    def test_mixed_settings(self):
        """Should overwrite missing values."""

        settings = KalmanSettings(
            dim_x = 1,
            dim_z = 1,
            X = np.zeros(1),
            H = np.zeros(1),
            R = np.zeros(1),
            P = np.zeros(1),
            F = np.zeros(1)
        )

        self.assertIsNotNone(settings)
        self.assertEqual((1,), settings.X.shape)
        self.assertEqual((1,), settings.H.shape)
        self.assertEqual((1,), settings.R.shape)
        self.assertEqual((1,), settings.P.shape)
        self.assertEqual((1,), settings.F.shape)
        self.assertEqual((1, 1), settings.Q.shape)

    def test_negative_dimensions_throws(self):
        """Should throw if dimension is negative."""

        with self.assertRaises(ValueError):
            KalmanSettings(-1, 1)
        with self.assertRaises(ValueError):
            KalmanSettings(3, -1)
