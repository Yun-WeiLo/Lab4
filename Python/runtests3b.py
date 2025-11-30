import unittest
import os
import numpy as np
from Lab4 import *

class TEST_IKinBody(unittest.TestCase):

    def test_3(self):
        Blist = np.array([[0, 0, -1, 2, 0, 0],
                            [0, 0, 0, 0, 1, 0],
                            [0, 0, 1, 0, 0, 0.1]]).T
        M = np.array([[-1, 0, 0, 0],
                        [0, 1, 0, 6],
                        [0, 0, -1, 2],
                        [0, 0, 0, 1]])
        T = np.array([[0, 1, 0, -5],
                        [1, 0, 0, 4],
                        [0, 0, -1, 1.6858],
                        [0, 0, 0, 1]])
        thetalist0 = np.array([1.5, 2.5, 3])
        eomg = 0.01
        ev = 0.001
        actualTheta, actualSuccess = ECE569_IKinBody(Blist, M, T, thetalist0, eomg, ev)
        expectedTheta = np.array([1.5707, 2.9997, 3.1415])
        expectedSuccess = True
        self.assertTrue(np.allclose(actualTheta, expectedTheta, atol=1e-3), f'Expected {expectedTheta}, got {actualTheta}')
        self.assertEqual(actualSuccess, expectedSuccess, f'Expected {expectedSuccess}, got {actualSuccess}')

    def test_6(self):
        Blist = np.array([[0, 0, 0, 0, 0, 0],
                            [1.0000, 0, 0, 0, -1.0000, 0],
                            [0, 1.0000, 1.0000, 1.0000, 0, 1.0000],
                            [0.2232, 0.0854, 0.0854, 0.0854, -0.0921, 0],
                            [0, -0.4567, -0.2132, 0, 0, 0],
                            [0.4567, 0, 0, 0, 0, 0]])
        M = np.array([[-1.0000, 0, 0, 0.4567],
                        [0, 0, 1.0000, 0.2232],
                        [0, 1.0000, 0, 0.0665],
                        [0, 0, 0, 1.0000]])
        T = np.array([[-0.6987, -0.0569, 0.7131, 0.2154],
                        [0.7151, -0.0248, 0.6986, 0.2271],
                        [-0.0221, 0.9981, 0.0580, 0.2966],
                        [0, 0, 0, 1.0000]])
        thetalist0 = np.array([-1.6800, -1.4018, -1.8127, -2.9937, -0.8857, -0.0696]) + 0.1
        actualTheta, actualSuccess = ECE569_IKinBody(Blist, M, T, thetalist0, 1e-6, 1e-6)
        expectedTheta = np.array([-1.6800, -1.4018, -1.8127, -2.9937, -0.8857, -0.0696])
        expectedSuccess = True
        self.assertTrue(np.allclose(actualTheta, expectedTheta, atol=1e-3), f'Expected {expectedTheta}, got {actualTheta}')
        self.assertEqual(actualSuccess, expectedSuccess, f'Expected {expectedSuccess}, got {actualSuccess}')


if __name__ == '__main__':
    current_folder = os.path.basename(os.getcwd())
    if current_folder != 'Python':
        print(f'Please run the test from Lab4/Python directory. You are currently in {current_folder}')
        print(f'Hint: cd ~/ece569-fall2025/Lab4/Python')
    else:

        # run all of the unit tests using
        # python3 tests.py
        unittest.main(verbosity=2)

        # Note: if you just want to run a specific test, do something like this from the Lab4/Python directory:
        # python3 -m unittest tests.TEST_MatrixLog6 -v
    