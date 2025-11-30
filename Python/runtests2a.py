import unittest
import os
import numpy as np
from Lab4 import *

class TEST_VecTose3(unittest.TestCase):

    def test_nonzero(self):
        xi = np.array([1,2,3,4,5,6])
        actual = ECE569_VecTose3(xi)
        expected = np.array([[0, -3, 2, 4], [3, 0, -1, 5], [-2, 1, 0, 6], [0, 0, 0, 0]])
        self.assertTrue(np.allclose(actual, expected), f'Expected {expected}, got {actual}')

class TEST_se3ToVec(unittest.TestCase):

    def test_nonzero(self):
        se3mat = np.array([[0, -3, 2, 4], [3, 0, -1, 5], [-2, 1, 0, 6], [0, 0, 0, 0]])
        actual = ECE569_se3ToVec(se3mat)
        expected = np.array([1,2,3,4,5,6])
        self.assertTrue(np.allclose(actual, expected), f'Expected {expected}, got {actual}')

class TEST_MatrixLog6(unittest.TestCase):

    def test_identity(self):
        T = np.eye(4)
        actual = ECE569_MatrixLog6(T)
        expected = np.zeros((4,4))
        self.assertTrue(np.allclose(actual, expected), f'Expected {expected}, got {actual}')

    def test_nonidentity(self):
        T = np.array([[1, 0,  0, 0],
                      [0, 0, -1, 0],
                      [0, 1,  0, 3],
                      [0, 0,  0, 1]])
        actual = ECE569_MatrixLog6(T)
        expected = np.array([[0, 0, 0, 0], [0, 0, -np.pi/2, 2.3562], [0, np.pi/2, 0, 2.3562], [0, 0, 0, 0]])
        self.assertTrue(np.allclose(actual, expected), f'Expected {expected}, got {actual}')

class TEST_MatrixExp6(unittest.TestCase):

    def test_zero(self):
        se3mat = np.array([[0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 0, 3], [0, 0, 0, 0]])
        actual = ECE569_MatrixExp6(se3mat)
        expected = np.array([[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]])
        self.assertTrue(np.allclose(actual, expected), f'Expected {expected}, got {actual}')

    def test_nonzero(self):
        se3mat = np.array([[0, 0, 0, 0], [0, 0, -np.pi/2, 2.3562], [0, np.pi/2, 0, 2.3562], [0, 0, 0, 0]])
        actual = ECE569_MatrixExp6(se3mat)
        expected = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 3], [0, 0, 0, 1]])
        self.assertTrue(np.allclose(actual, expected), f'Expected {expected}, got {actual}')

class TEST_Adjoint(unittest.TestCase):

    def test_nonzero(self):
        T = np.array([[1, 0,  0, 0], 
                      [0, 0, -1, 0], 
                      [0, 1,  0, 3], 
                      [0, 0,  0, 1]])
        actual = ECE569_Adjoint(T)
        expected = np.array([[1, 0,  0, 0, 0,  0], 
                             [0, 0, -1, 0, 0,  0], 
                             [0, 1,  0, 0, 0,  0], 
                             [0, 0,  3, 1, 0,  0], 
                             [3, 0,  0, 0, 0, -1], 
                             [0, 0,  0, 0, 1,  0]])
        self.assertTrue(np.allclose(actual, expected), f'Expected {expected}, got {actual}')



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
    