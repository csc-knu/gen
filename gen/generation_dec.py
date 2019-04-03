#!/usr/bin/env python
import numpy as np
import unittest
import time


def __generation_dec(n: int, m: int, x_min: np.array, x_max: np.array) -> np.matrix:
    """
    :param n: num rows in returned matrix
    :param m: num cols in returned matrix
    :param x_min: float array, min possible nums in cols of returned matrix
    :param x_max: float array, max possible nums in cols of returned matrix
    :return: n times m float matrix with nums in col number i in [x_min[i], x_max[i])
    """
    assert n > 0, "n should be positive"
    assert m > 0, "m should be positive"

    assert x_min.shape == (m, ), "x_min should be of shape (m, )"   
    assert x_max.shape == (m, ), "x_max should be of shape (m, )"

    return np.random.uniform(low=x_min, high=x_max, size=(n, m))


def generation_dec(n: int, x_min: np.array, x_max: np.array) -> np.matrix:
    """
    :param n: num rows to return
    :param x_min: float arr, min possible nums in cols to return
    :param x_max: float arr, max possible nums in cols to return
    :return: float mat with n rows and nums in col number i in [x_min[i], x_max[i])
    """
    assert n > 0, "n should be positive"

    assert x_min.ndim == 1, "x_min should have dimension 1"
    assert x_max.ndim == 1, "x_max should have dimension 1"
    assert x_min.shape == x_max.shape, "x_min and x_max shapes mismatch"
    m: int = len(x_min)

    assert np.all(x_min < x_max), "x_min should be element-wise less than x_max"

    return __generation_dec(n, m, x_min, x_max)


class TestGenerationDec(unittest.TestCase):
    def setUp(self):
        self.SLOW_TEST_THRESHOLD = .1
        self._started_at = time.time()
    
    def tearDown(self):
        elapsed: float = time.time() - self._started_at
        if elapsed > self.SLOW_TEST_THRESHOLD:
            print(f'{self.id()} ({round(elapsed, 3)}s)')

    def test_not_positive_n(self):
        with self.assertRaises(AssertionError):
            n: int = -1
            x_min, x_max = np.zeros(2), np.ones(2)
            generation_dec(n, x_min, x_max)

    def test_x_min_greater_than_x_max(self):
        with self.assertRaises(AssertionError):
            n: int = 1
            x_min, x_max = np.array([0, 0, 1]), np.array([1, 1, 0])
            generation_dec(n, x_min, x_max)

    def test_x_min_x_max_shape_mismatch(self):
        with self.assertRaises(AssertionError):
            n: int = 1
            x_min, x_max = np.zeros(2), np.ones(3)
            generation_dec(n, x_min, x_max)

    def test_x_min_wrong_dimension(self):
        with self.assertRaises(AssertionError):
            n: int = 1
            x_min, x_max = np.zeros((2, 1)), np.ones(2)
            generation_dec(n, x_min, x_max)

    def test_x_max_wrong_dimension(self):
        with self.assertRaises(AssertionError):
            n: int = 1
            x_min, x_max = np.zeros(2), np.ones((2, 1))
            generation_dec(n, x_min, x_max)

    def test_x_min_and_x_max_wrong_dimensions(self):
        with self.assertRaises(AssertionError):
            n: int = 1
            x_min, x_max = np.zeros((2, 1)), np.ones((2, 1))
            generation_dec(n, x_min, x_max)

    def test_large_n_performance(self): 
        x_min, x_max = np.zeros(2), np.ones(2)
        n: int = 10**6
        generation_dec(n, x_min, x_max)

    def test_large_m_performance(self):
        n: int = 2
        m: int = 10**6
        x_min, x_max = np.zeros(m), np.ones(m)
        generation_dec(n, x_min, x_max)


if __name__ == '__main__':
    unittest.main()
