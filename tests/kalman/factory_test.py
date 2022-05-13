
"""Kalman filter factory tests."""

from unittest import TestCase

from src.kalman.factory import KalmanFilterFactory
from src.kalman.factory import KalmanSettings

class TestKalmanFilterFactory(TestCase):
    """KalmanFilterFactory tests.

    Args:
        TestCase (TestCase): The test base class.
    """

    def test_no_arg_throws(self):
        """Should throw error."""

        with self.assertRaises(AttributeError):
            KalmanFilterFactory.create(None)

    def test_create_should_return_instance(self):
        """Should create a non null filter instance."""

        settings = KalmanSettings(1, 1)
        kalman = KalmanFilterFactory.create(settings)
        self.assertIsNotNone(kalman)
