#!/usr/bin/env python
import numpy as np
from typing import Tuple
from itertools import accumulate 
import unittest
import time


def __bin_dec_param(m: int, x_min: np.array, x_max: np.array, 
                    eps: float) -> Tuple[np.array, np.array, np.array]:
    """
    :param m: dimen of x_min and x_max
    :param x_min: float array, min possible nums
    :param x_max: float array, max possible nums
    :param eps: desired precision
    :return: 
        nn - min num bin digits to encode nums in [x_min, x_max) with precision eps
        dd - discretionaries of obtained encodings, < eps
        NN - prefix sums of nn
    """
    assert m > 0, "m should be positive"

    assert x_min.shape == (m, ), "x_min should be of shape (m, )"
    assert x_max.shape == (m, ), "x_max should be of shape (m, )"
    
    nn = (np.floor(np.log2((x_max - x_min) / eps)) + 1).astype(int)
    assert np.all(nn > 0), "desired precision should be smaller than |x_max - x_min|"

    dd = np.round((x_max - x_min) / (np.power(2, nn) - 1), 7)
    NN = np.array([0,] + list(accumulate(nn)))
    
    return nn, dd, NN


def bin_dec_param(x_min: np.array, x_max: np.array, 
                  eps: float) -> Tuple[np.array, np.array, np.array]:
    """
    :param x_min: float array, min possible nums
    :param x_max: float array, max possible nums
    :param eps: desired precision
    :return: 
        nn - min num bin digits to encode nums in [x_min, x_max) with precision eps
        dd - discretionaries of obtained encodings, < eps
        NN - prefix sums of nn
    """
    assert x_min.ndim == 1, "x_min should have dimension 1"
    assert x_max.ndim == 1, "x_max should have dimension 1"
    assert x_min.shape == x_max.shape, "x_min and x_max shapes mismatch"
    m: int = len(x_min)

    return __bin_dec_param(m, x_min, x_max, eps)


class TestBinDecParam(unittest.TestCase):
    def setUp(self):
        self.SLOW_TEST_THRESHOLD = .1
        self._started_at = time.time()
    
    def tearDown(self):
        elapsed: float = time.time() - self._started_at
        if elapsed > self.SLOW_TEST_THRESHOLD:
            print(f'{self.id()} ({round(elapsed, 3)}s)')

    def test_large_m_performance(self):
        eps: float = 1e-5
        m: int = 2 * 10**5
        x_min, x_max = np.zeros(m), np.ones(m)
        bin_dec_param(x_min, x_max, eps)

    def test_different_eps_performance(self):
        m: int = 10**3
        x_min, x_max = np.zeros(m), np.ones(m)
        for eps in np.power(.1, np.arange(10)):
            with self.subTest(eps=eps):        
                bin_dec_param(x_min, x_max, eps)

    def test_x_min_x_max_shape_mismatch(self):
        with self.assertRaises(AssertionError):
            eps: float = 1e-3
            x_min, x_max = np.zeros(2), np.ones(3)
            bin_dec_param(x_min, x_max, eps)

    def test_x_min_wrong_dimension(self):
        with self.assertRaises(AssertionError):
            eps: float = 1e-3
            x_min, x_max = np.zeros((2, 1)), np.ones(2)
            bin_dec_param(x_min, x_max, eps)

    def test_x_max_wrong_dimension(self):
        with self.assertRaises(AssertionError):
            eps: float = 1e-3
            x_min, x_max = np.zeros(2), np.ones((2, 1))
            bin_dec_param(x_min, x_max, eps)

    def test_x_min_and_x_max_wrong_dimensions(self):
        with self.assertRaises(AssertionError):
            eps: float = 1e-3
            x_min, x_max = np.zeros((2, 1)), np.ones((2, 1))
            bin_dec_param(x_min, x_max, eps)

    def test_correctness_large_eps(self):
        eps: float = 1e-1
        m: int = 5
        x_min, x_max = np.zeros(m), np.array([.1, .3, .7, 1.5, 3.1])
        nn, dd, NN = bin_dec_param(x_min, x_max, eps)
        self.assertEqual(nn.tolist(), [1, 2, 3, 4, 5])
        self.assertAlmostEqual(dd.tolist(), [eps for _ in range(5)],)
        self.assertEqual(NN.tolist(), [0, 1, 3, 6, 10, 15])

    def test_correctness_small_eps(self):
        eps: float = 1e-3
        m: int = 5
        x_min, x_max = np.zeros(m), np.array([.001, .003, .007, .015, .031])
        nn, dd, NN = bin_dec_param(x_min, x_max, eps)
        self.assertEqual(nn.tolist(), [1, 2, 3, 4, 5])
        self.assertAlmostEqual(dd.tolist(), [eps for _ in range(5)],)
        self.assertEqual(NN.tolist(), [0, 1, 3, 6, 10, 15])


if __name__ == '__main__':
    unittest.main()
