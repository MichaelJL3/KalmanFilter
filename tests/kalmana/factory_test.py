
from argparse import ArgumentError
from src.kalman.factory import KalmanFilterFactory
from src.kalman.factory import KalmanSettings

from unittest import TestCase

class TestKalmanFilterFactory(TestCase):
    def test_create_without_settings_should_throw(self):
        with self.assertRaises(ArgumentError):
            KalmanFilterFactory.create()

    def test_create_should_return_instance(self):
        settings = KalmanSettings()
        filter = KalmanFilterFactory.create(settings)
        self.assertIsNotNone(filter)
