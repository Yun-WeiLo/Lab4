import unittest
import os
import numpy as np
from Lab4 import *

class TEST_TransInv(unittest.TestCase):

    def test_identity(self):
        T = np.eye(4)
        actual = ECE569_TransInv(T)
        expected = np.eye(4)
        self.assertTrue(np.allclose(actual, expected, atol=1e-2), f'Expected {expected}, got {actual}')

    def test_nonidentity(self):
        T = np.array([[1, 0,  0, 0],
                      [0, 0, -1, 0],
                      [0, 1,  0, 3],
                      [0, 0,  0, 1]])
        actual = ECE569_TransInv(T)
        expected = np.array([[1,  0, 0,  0],
                            [0,  0, 1, -3],
                            [0, -1, 0,  0],
                            [0,  0, 0,  1]])
        self.assertTrue(np.allclose(actual, expected, atol=1e-2), f'Expected {expected}, got {actual}')

class TEST_JacobianBody(unittest.TestCase):

    def test_4(self):
        Blist = np.array([[0, 0, 1, 0, 0.2, 0.2], 
                        [1, 0, 0, 2, 0, 3], 
                        [0, 1, 0, 0, 2, 1], 
                        [1, 0, 0, 0.2, 0.3, 0.4]]).T
        thetalist = np.array([0.2, 1.1, 0.1, 1.2])
        actual = ECE569_JacobianBody(Blist, thetalist)
        expected = np.array([[-0.0453, 0.9950, 0, 1.0000], 
                            [0.7436, 0.0930, 0.3624, 0], 
                            [-0.6671, 0.0362, -0.9320, 0], 
                            [2.3259, 1.6681, 0.5641, 0.2000], 
                            [-1.4432, 2.9456, 1.4331, 0.3000], 
                            [-2.0664, 1.8288, -1.5887, 0.4000]])
        self.assertTrue(np.allclose(actual, expected, atol=1e-3), f'Expected {expected}, got {actual}')

    def test_6(self):
        Blist = np.array([[0, 0, 0, 0, 0, 0], 
                        [1.0000, 0, 0, 0, -1.0000, 0], 
                        [0, 1.0000, 1.0000, 1.0000, 0, 1.0000], 
                        [0.2232, 0.0854, 0.0854, 0.0854, -0.0921, 0], 
                        [0, -0.4567, -0.2132, 0, 0, 0], 
                        [0.4567, 0, 0, 0, 0, 0]])
        thetalist = np.array([-1.6800, -1.4018, -1.8127, -2.9937, -0.8857, -0.0696])
        actual = ECE569_JacobianBody(Blist, thetalist)
        expected = np.array([[-0.0221, -0.7725, -0.7725, -0.7725, 0.0695, 0], 
                            [0.9981, -0.0539, -0.0539, -0.0539, -0.9976, 0], 
                            [0.0580, 0.6327, 0.6327, 0.6327, 0, 1.0000], 
                            [0.3127, -0.0973, 0.0541, 0.0489, -0.0919, 0], 
                            [0.0076, 0.2532, 0.2866, 0.0749, -0.0064, 0], 
                            [-0.0114, -0.0973, 0.0905, 0.0661, 0, 0]])
        self.assertTrue(np.allclose(actual, expected, atol=1e-3), f'Expected {expected}, got {actual}')


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
    