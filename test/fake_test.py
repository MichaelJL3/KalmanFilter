
import numpy as np

from src.kalman.factory import KalmanFilterFactory
from src.kalman.factory import KalmanSettings

from unittest import TestCase
from parameterized import parameterized

class TestKalmanExamples(TestCase):
    @parameterized.expand([
        [1000,    1030,   1,  1030   ],
        [1030,     989, 1/2,  1009.5 ],
        [1009.5,  1017, 1/3,  1012   ],
        [1012,    1009, 1/4,  1011.25],
        [1011.25, 1013, 1/5,  1011.6 ],
        [1011.6,   979, 1/6,  1006.17],
        [1006.17, 1008, 1/7,  1006.43],
        [1006.43, 1042, 1/8,  1010.87],
        [1010.87, 1012, 1/9,  1011   ],
        [1011,    1011, 1/10, 1011   ]
    ])
    # alpha filter
    # dynamic model is constant (assuming gold weight doesn't change over time) 
    def test_gold_weight(self, x_n_n_1, z_n, a_n, e_x):
        predicted = x_n_n_1 + a_n * (z_n - x_n_n_1)

        self.assertAlmostEqual(e_x, predicted, delta=0.01)

    @parameterized.expand([
        [30200,   40,   30110, 30182,   38.2, 30373],
        [30373,   38.2, 30265, 30351.4, 36,   30531.6],
        [30531.6, 36,   30740, 30573.3, 40.2, 30774.3],
        [30774.3, 40.2, 30750, 30769.5, 39.7, 30968.1],
        [30968.1, 39.7, 31135, 31001.5, 43.1, 31216.8],
        [31216.8, 43.1, 31015, 31176.4, 39,   31371.5],
        [31371.5, 39,   31180, 31333.2, 35.2, 31509.2],
        [31509.2, 35.2, 31610, 31529.4, 37.2, 31715.4],
        [31715.4, 37.2, 31960, 31764.3, 42.1, 31974.8],
        [31974.8, 42.1, 31865, 31952.9, 39.9, 32152.4]
    ])
    # alpha - beta filter
    # gains are constant
    # model assumes constant velocity
    def test_plane_tracking_1d(self, x_n_n_1, v_n_n_1, z_n, e_x, e_v, u_e_x):
        alpha_gain = 0.2
        beta_gain  = 0.1
        dt = 5

        predicted_x = x_n_n_1 + alpha_gain * (z_n - x_n_n_1)
        predicted_v = v_n_n_1 + beta_gain * (z_n - x_n_n_1) / dt
        update_x = predicted_x + dt * predicted_v

        self.assertAlmostEqual(e_x, predicted_x, delta=1)
        self.assertAlmostEqual(e_v, predicted_v, delta=1)
        self.assertAlmostEqual(u_e_x, update_x, delta=1)

    @parameterized.expand([
        [30250,   50,     0,    30160, 30205,   42.8,  -0.7,  30410,   39.2],
        [30410,   39.2,  -0.7,  30365, 30387.5, 35.6,  -1.1,  30552,   30.2],
        [30552,   30.2,  -1.1,  30890, 30721,   57.2,   1.6,  31027.5, 65.4],
        [31027.5, 65.4,   1.6,  31050, 31038.8, 67.2,   1.8,  31397.1, 76.2],
        [31397.1, 76.2,   1.8,  31785, 31591.1, 107.2,  4.9,  32188.5, 131.7],
        [32188.5, 131.7,  4.9,  32215, 32201.7, 133.9,  5.1,  32935.1, 159.5],
        [32935.1, 159.5,  5.1,  33130, 33032.5, 175.1,  6.7,  33991.3, 208.5],
        [33991.3, 208.5,  6.7,  34510, 34250.7, 250,    10.8, 35635.8, 304.1],
        [35635.8, 304.1,  10.8, 36010, 35822.9, 334,    13.8, 37665.8, 403.1],
        [37665.8, 403.1,  13.8, 37265, 37465.4, 371.1,  10.6, 39453,   424]
    ])
    # alpha - beta - gamma filter
    # gains are constant
    # temporarily constant velocity
    # temporarily constant acceleration
    def test_accel_plane_tracking_1d(self, x_n_n_1, v_n_n_1, a_n_n_1, z_n, e_x, e_v, e_a, u_e_x, u_e_v):
        alpha_gain = 0.5
        beta_gain  = 0.4
        gamma_gain = 0.1
        dt = 5

        predicted_x = x_n_n_1 + alpha_gain * (z_n - x_n_n_1)
        predicted_v = v_n_n_1 + beta_gain * (z_n - x_n_n_1) / dt
        predicted_a = a_n_n_1 + gamma_gain * (z_n - x_n_n_1) / (.5 * dt ** 2)

        update_x = predicted_x + dt * predicted_v + (dt ** 2 / 2) * predicted_a
        update_v = predicted_v + dt * predicted_a
        
        self.assertAlmostEqual(e_x, predicted_x, delta=1)
        self.assertAlmostEqual(e_v, predicted_v, delta=1)
        self.assertAlmostEqual(e_a, predicted_a, delta=1)
        self.assertAlmostEqual(u_e_x, update_x, delta=1)
        self.assertAlmostEqual(u_e_v, update_v, delta=1)

    @parameterized.expand([
        [60,    225,   48.54, 49.69, 22.5 ],
        [49.69, 22.5,  47.11, 48.7,  11.84],
        [48.7,  11.84, 55.01, 50.57, 8.04 ],
        [50.57, 8.04,  55.15, 51.68, 6.08 ],
        [51.68, 6.08,  49.89, 51.33, 4.89 ],
        [51.33, 4.89,  40.85, 49.62, 4.09 ],
        [49.62, 4.09,  46.72, 49.21, 3.52 ],
        [49.21, 3.52,  50.05, 49.31, 3.08 ],
        [49.31, 3.08,  51.27, 49.53, 2.74 ],
        [49.53, 2.74,  49.95, 49.57, 2.47 ]
    ])
    # updated gain
    def test_building_height(self, x_n_n_1, p_n_n_1, z_n, e_x, u_p):
        measure_error = 5 # std dev
        variance_error = measure_error ** 2
        
        k_n = p_n_n_1 / (p_n_n_1 + variance_error)
        predicted_x = x_n_n_1 + k_n * (z_n - x_n_n_1)
        
        update_p = (1 - k_n) * p_n_n_1 
        
        self.assertAlmostEqual(e_x, predicted_x, delta=1)
        self.assertAlmostEqual(u_p, update_p, delta=1)

    @parameterized.expand([
        [10,     10000,  49.95,  49.95,  0.01  ],
        [49.95,  0.01,   49.967, 49.959, 0.005 ],
        [49.959, 0.005,  50.1,   50.007, 0.0034],
        [50.007, 0.0034, 50.106, 50.032, 0.0026],
        [50.032, 0.0026, 49.992, 50.023, 0.0021],
        [50.023, 0.0021, 49.819, 49.987, 0.0018],
        [49.987, 0.0018, 49.933, 49.978, 0.0016],
        [49.978, 0.0016, 50.007, 49.983, 0.0015],
        [49.983, 0.0015, 50.023, 49.988, 0.0014],
        [49.988, 0.0014, 49.99,  49.988, 0.0013]
    ])
    # temperature is constant
    def test_temperature_with_noise(self, x_n_n_1, p_n_n_1, z_n, e_x, u_p):
        measure_error = 0.01 # std dev
        variance_error = measure_error ** 2
        p_noise = 0.0001
        
        p_n_n_1 += p_noise
        
        k_n = p_n_n_1 / (p_n_n_1 + variance_error)
        predicted_x = x_n_n_1 + k_n * (z_n - x_n_n_1)
        
        update_p = (1 - k_n) * p_n_n_1 
        
        self.assertAlmostEqual(e_x, predicted_x, delta=1)
        self.assertAlmostEqual(u_p, update_p, delta=1)

    @parameterized.expand([
        []
    ])
    def test_multidim_airplane_noinput(self):
        dt = 1
        dt_2 = dt ** 2 / 2

        settings = KalmanSettings(
            dim_x=6,
            dim_z=2,
            R=np.array([
                [9, 0],
                [0, 9] 
            ]),
            H=np.array([
                [1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0]
            ]),
            Q=np.array([
                [.25, .5, .5,  0,   0,  0],
                [.5,   1,  1,  0,   0,  0],
                [.5,   1,  1,  0,   0,  0],
                [ 0,   0,  0, .25, .5, .5],
                [ 0,   0,  0, .5,   1,  1],
                [ 0,   0,  0, .5,   1,  1]
            ]),
            F=np.array([
                [1, dt, dt_2, 0, 0,  0   ],
                [0, 1,  dt,   0, 0,  0   ],
                [0, 0,  1,    0, 0,  0   ],
                [0, 0,  0,    1, dt, dt_2],
                [0, 0,  0,    0, 1,  dt  ],
                [0, 0,  0,    0, 0,  1   ]
            ]),
            X=np.array([0, 0, 0, 0, 0, 0]),
            P=np.identity(6) * 500
        )

        Z = np.array([-393.66, 300.4])

        kalman = KalmanFilterFactory().create(settings)
        kalman.predict()
        kalman.update(Z)
        kalman.predict()

        X_e = np.array([-694.3, -347.15, -86.8, 529.8, 264.9, 66.23])
        P_e = np.array([
            [974,  1239, 561, 0,    0,    0  ],
            [1239, 1622, 782, 0,    0,    0  ],
            [561,  782,  447, 0,    0,    0  ],
            [0,    0,    0,   974,  1239, 561],
            [0,    0,    0,   1239, 1622, 782],
            [0,    0,    0,   561,  782,  447]
        ])

        np.testing.assert_almost_equal(kalman.X, X_e, decimal=0)
        np.testing.assert_almost_equal(kalman.P, P_e, decimal=0)
