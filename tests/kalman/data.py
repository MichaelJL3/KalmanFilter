
"""data models for tests."""

import numpy as np

class KalmanDataEx0:
    """Example 0 data."""

    @staticmethod
    def get_dim_x():
        """Get dimension x size."""
        return 6

    @staticmethod
    def get_dim_z():
        """Get dimension z size."""
        return 2

    @staticmethod
    def get_z():
        """Get Z measurements."""
        return np.array([-393.66, 300.4])

    @staticmethod
    def get_x_1_0():
        """Get X first predicted state."""
        return np.zeros(6)

    @staticmethod
    def get_x_1_1():
        """Get X first updated state."""
        return np.array([-390.54, -260.36, -86.8, 298.02, 198.7, 66.23])

    @staticmethod
    def get_x_2_1():
        """Get X next predicted state."""
        return np.array([-694.3, -347.15, -86.8, 529.8, 264.9, 66.23])

    @staticmethod
    def get_v_1_1():
        """Get V first updated state."""
        return np.array([-3.1,  2.4])

    @staticmethod
    def get_p_1_0():
        """Get P first predicted state."""
        return np.array([
            [1125, 750,  250, 0,    0,    0  ],
            [750,  1000, 500, 0,    0,    0  ],
            [250,  500,  500, 0,    0,    0  ],
            [0,    0,    0,   1125, 750,  250],
            [0,    0,    0,   750,  1000, 500],
            [0,    0,    0,   250,  500,  500]
        ])

    @staticmethod
    def get_p_1_1():
        """Get P first updated state."""
        return np.array([
            [8.93, 5.95,  2,     0,    0,     0    ],
            [5.95, 504,   334.7, 0,    0,     0    ],
            [2,    334.7, 444.9, 0,    0,     0    ],
            [0,    0,     0,     8.93, 5.95,  2    ],
            [0,    0,     0,     5.95, 504,   334.7],
            [0,    0,     0,     2,    334.7, 444.9]
        ])

    @staticmethod
    def get_p_2_1():
        """Get P next predicted state."""
        return np.array([
            [972.8,  1236.5, 559.2, 0,      0,      0    ],
            [1236.5, 1618.3, 779.6, 0,      0,      0    ],
            [559.2,  779.6,  444.9, 0,      0,      0    ],
            [0,      0,      0,     972.8,  1236.5, 559.2],
            [0,      0,      0,     1236.5, 1618.3, 779.6],
            [0,      0,      0,     559.2,  779.6,  444.9]
        ])

    @staticmethod
    def get_q():
        """Get the Q state."""
        return np.array([
            [.25, .5, .5,  0,   0,  0],
            [.5,   1,  1,  0,   0,  0],
            [.5,   1,  1,  0,   0,  0],
            [ 0,   0,  0, .25, .5, .5],
            [ 0,   0,  0, .5,   1,  1],
            [ 0,   0,  0, .5,   1,  1]
        ]) * .2 ** 2

    @staticmethod
    def get_f():
        """Get the F state."""
        return np.array([
            [1, 1, .5, 0, 0,  0],
            [0, 1,  1, 0, 0,  0],
            [0, 0,  1, 0, 0,  0],
            [0, 0,  0, 1, 1, .5],
            [0, 0,  0, 0, 1,  1],
            [0, 0,  0, 0, 0,  1]
        ])

    @staticmethod
    def get_r():
        """Get the R state."""
        return np.array([
            [9, 0],
            [0, 9]
        ])

    @staticmethod
    def get_h():
        """Get the H state."""
        return np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0]
        ])

    @staticmethod
    def get_k():
        """Get K first state."""
        return np.array([
            [0.9921, 0     ],
            [0.6614, 0     ],
            [0.2205, 0     ],
            [0,      0.9921],
            [0,      0.6614],
            [0,      0.2205]
        ])
