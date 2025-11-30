import unittest
import os
import numpy as np
from Lab4 import *

class TEST_FKinBody(unittest.TestCase):

    def test_3(self):
        M = np.array([[-1, 0, 0, 0], 
                      [0, 1, 0, 6], 
                      [0, 0, -1, 2], 
                      [0, 0, 0, 1]])
        Blist = np.array([[0, 0, -1, 2, 0, 0], 
                          [0, 0, 0, 0, 1, 0], 
                          [0, 0, 1, 0, 0, 0.1]]).T
        thetalist = np.array([np.pi/2, 3, np.pi])
        actual = ECE569_FKinBody(M, Blist, thetalist)
        expected = np.array([[-0.0000, 1.0000, 0, -5.0000], 
                             [1.0000, 0.0000, 0, 4.0000], 
                             [0, 0, -1.0000, 1.6858], 
                             [0, 0, 0, 1.0000]])
        self.assertTrue(np.allclose(actual, expected, atol=1e-2), f'Expected {expected}, got {actual}')

    """
    function test_6(testCase)
    M =    [
        -1.0000         0         0    0.4567;
        0         0    1.0000    0.2232;
        0    1.0000         0    0.0665;
        0         0         0    1.0000];
    Blist = [
        0         0         0         0         0         0;
        1.0000         0         0         0   -1.0000         0;
            0    1.0000    1.0000    1.0000         0    1.0000;
        0.2232    0.0854    0.0854    0.0854   -0.0921         0;
            0   -0.4567   -0.2132         0         0         0;
        0.4567         0         0         0         0         0];
    thetalist = [-1.6800   -1.4018   -1.8127   -2.9937   -0.8857   -0.0696]';
    actual = ECE569_FKinBody(M, Blist, thetalist);
    expected = [
    -0.6987   -0.0569    0.7131    0.2154;
        0.7151   -0.0248    0.6986    0.2271;
    -0.0221    0.9981    0.0580    0.2966;
            0         0         0    1.0000];
    verifyEqual(testCase,actual,expected,'AbsTol', 1e-2);"""
    def test_6(self):
        M = np.array([[-1.0000, 0, 0, 0.4567], 
                      [0, 0, 1.0000, 0.2232], 
                      [0, 1.0000, 0, 0.0665], 
                      [0, 0, 0, 1.0000]])
        Blist = np.array([[0, 0, 0, 0, 0, 0], 
                          [1.0000, 0, 0, 0, -1.0000, 0], 
                          [0, 1.0000, 1.0000, 1.0000, 0, 1.0000], 
                          [0.2232, 0.0854, 0.0854, 0.0854, -0.0921, 0], 
                          [0, -0.4567, -0.2132, 0, 0, 0], 
                          [0.4567, 0, 0, 0, 0, 0]])
        thetalist = np.array([-1.6800, -1.4018, -1.8127, -2.9937, -0.8857, -0.0696])
        actual = ECE569_FKinBody(M, Blist, thetalist)
        expected = np.array([[-0.6987, -0.0569, 0.7131, 0.2154], 
                             [0.7151, -0.0248, 0.6986, 0.2271], 
                             [-0.0221, 0.9981, 0.0580, 0.2966], 
                             [0, 0, 0, 1.0000]])
        self.assertTrue(np.allclose(actual, expected, atol=1e-2), f'Expected {expected}, got {actual}')

class TEST_FKinSpace(unittest.TestCase):

    def test_3(self):
        M = np.array([[-1, 0, 0, 0], 
                      [0, 1, 0, 6], 
                      [0, 0, -1, 2], 
                      [0, 0, 0, 1]])
        Slist = np.array([[0, 0, 1, 4, 0, 0], 
                          [0, 0, 0, 0, 1, 0], 
                          [0, 0, -1, -6, 0, -0.1]]).T
        thetalist = np.array([np.pi/2, 3, np.pi])
        actual = ECE569_FKinSpace(M, Slist, thetalist)
        expected = np.array([[-0.0000, 1.0000, 0, -5.0000], 
                             [1.0000, 0.0000, 0, 4.0000], 
                             [0, 0, -1.0000, 1.6858], 
                             [0, 0, 0, 1.0000]])
        self.assertTrue(np.allclose(actual, expected, atol=1e-2), f'Expected {expected}, got {actual}')

    def test_6(self):
        M = np.array([[-1.0000, 0, 0, 0.4567], 
                      [0, 0, 1.0000, 0.2232], 
                      [0, 1.0000, 0, 0.0665], 
                      [0, 0, 0, 1.0000]])
        Slist = np.array([[0, 0, 0, 0, 0, 0], 
                          [0, 1.0000, 1.0000, 1.0000, 0, 1.0000], 
                          [1.0000, 0, 0, 0, -1.0000, 0], 
                          [0, -0.1519, -0.1519, -0.1519, -0.1311, -0.0665], 
                          [0, 0, 0, 0, 0.4567, 0], 
                          [0, 0, 0.2435, 0.4567, 0, 0.4567]])
        thetalist = np.array([-1.6800, -1.4018, -1.8127, -2.9937, -0.8857, -0.0696])
        actual = ECE569_FKinSpace(M, Slist, thetalist)
        expected = np.array([[-0.6987, -0.0569, 0.7131, 0.2154], 
                             [0.7151, -0.0248, 0.6986, 0.2271], 
                             [-0.0221, 0.9981, 0.0580, 0.2966], 
                             [0, 0, 0, 1.0000]])
        self.assertTrue(np.allclose(actual, expected, atol=1e-2), f'Expected {expected}, got {actual}')


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
    