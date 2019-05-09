#!/usr/bin/env python
import numpy as np
from itertools import accumulate 
from functools import reduce
import unittest
import time


def __cod_decimal(x_bin: np.array, x_min: float, d: float) -> float:
    """
    :param x_bin: bin num to decode as np.array of digits
    :param x_min: min num possible to encode
    :param d: discretionary of encoded number
    :return: decoded num as float
    """
    assert d > 0, "discretionary should be positive"

    x_dec_1 = reduce(lambda _1, _2: _1 * 2 + _2, map(int, x_bin[::-1]))

    x_bin_1 = 0
    while x_dec_1 != 0:
        x_bin_1 ^= x_dec_1
        x_dec_1 >>= 1
    return x_min + d * x_bin_1


def __a_cod_decimal(n: int, m: int, g_bin: np.matrix, x_min: np.array, NN: np.array,
                    dd: np.array) -> np.matrix:
    """
    :param n: num rows in g_bin
    :param m: num cols in g_bin
    :param g_bin: bin nums as np.matrix of np.arrays of digigts to decode
    :param x_min: min nums possible to encode
    :param NN: prefix sums of num bin digits in encoded nums
    :param dd: discretionaries of encodings
    :return: decoded nums as np.matrix of floats
    """
    _m: int = NN.shape[0] - 1
    assert x_min.shape == (_m, ), "x_min should be of shape (_m, )"
    assert NN.shape == (_m + 1, ), "dd should be of shape (_m + 1, )"
    assert dd.shape == (_m, ), "dd should be of shape (_m, )"

    return np.matrix([[
        __cod_decimal(np.asarray(g_bin[i, NN[j]:NN[j + 1]]).reshape(-1), x_min[j], dd[j])
    for j in range(_m)] for i in range(n)])


def a_cod_decimal(g_bin: np.matrix, x_min: np.array, NN: np.array, 
                  dd: np.array) -> np.matrix:
    """
    :param g_bin: bin nums as np.matrix of np.arrays of digigts to decode
    :param x_min: min nums possible to encode
    :param NN: prefix sums of num bin digits in encoded nums
    :param dd: discretionaries of encodings
    :return: decoded nums as np.matrix of floats
    """
    assert g_bin.ndim == 2, "g_bin should be a matrix (have dimension 2)"
    assert x_min.ndim == 1, "x_min should be an array (have dimension 1)"
    assert NN.ndim == 1, "NN should be an array (have dimension 1)"
    assert dd.ndim == 1, "dd should be an array (have dimension 1)"
    assert g_bin.shape[1] == NN[-1], "g_bin and NN shapes mismatch"
    assert NN.shape[0] == dd.shape[0] + 1, "NN and dd shapes mismatch"
    assert x_min.shape[0] == dd.shape[0], "x_min and dd shapes mismatch"
    n, m = g_bin.shape
    
    return __a_cod_decimal(n, m, g_bin, x_min, NN, dd)


class TestCodDecimal(unittest.TestCase):
    def setUp(self):
        self.SLOW_TEST_THRESHOLD = .1
        self._started_at = time.time()

    def tearDown(self):
        elapsed: float = time.time() - self._started_at
        if elapsed > self.SLOW_TEST_THRESHOLD:
            print(f'{self.id()} ({round(elapsed, 3)}s)')

    def test_g_bin_wrong_ndim(self):
        with self.assertRaises(AssertionError):
            g_bin = np.matrix([
                [[1, 0, 0], [1, 1, 0, 0]], 
                [[1, 1, 0], [1, 1, 1, 0]], 
                [[1, 0, 1], [0, 1, 0, 1]],
            ])
            
            x_min = np.array([1, 2])
            NN = np.array([0, 3, 7])
            dd = np.array([.1, .1])

            a_cod_decimal(g_bin, x_min, NN, dd)
    
    def test_x_min_wrong_ndim(self):
        with self.assertRaises(AssertionError):
            g_bin = np.matrix([
                [1, 0, 0, 1, 1, 0, 0], 
                [1, 1, 0, 1, 1, 1, 0], 
                [1, 0, 1, 0, 1, 0, 1],
            ])
            
            x_min = np.array([[1], [2]])
            NN = np.array([0, 3, 7])
            dd = np.array([.1, .1])

            a_cod_decimal(g_bin, x_min, NN, dd)
        
    def test_dd_wrong_ndim(self):
        with self.assertRaises(AssertionError):
            g_bin = np.matrix([
                [1, 0, 0, 1, 1, 0, 0], 
                [1, 1, 0, 1, 1, 1, 0], 
                [1, 0, 1, 0, 1, 0, 1],
            ])
            
            x_min = np.array([1, 2])
            NN = np.array([0, 3, 7])
            dd = np.array([[.1], [.1]])

            a_cod_decimal(g_bin, x_min, NN, dd)

    def test_NN_wrong_ndim(self):
        with self.assertRaises(AssertionError):
            g_bin = np.matrix([
                [1, 0, 0, 1, 1, 0, 0], 
                [1, 1, 0, 1, 1, 1, 0], 
                [1, 0, 1, 0, 1, 0, 1],
            ])
            
            x_min = np.array([1, 2])
            NN = np.array([[0], [3], [7]])
            dd = np.array([.1, .1])

            a_cod_decimal(g_bin, x_min, NN, dd)
    
    def test_NN_dd_shapes_mismatch(self):
        with self.assertRaises(AssertionError):
            g_bin = np.matrix([
                [1, 0, 0, 1, 1, 0, 0], 
                [1, 1, 0, 1, 1, 1, 0], 
                [1, 0, 1, 0, 1, 0, 1],
            ])
            
            x_min = np.array([1, 2])
            NN = np.array([3, 7])
            dd = np.array([.1, .1])

            a_cod_decimal(g_bin, x_min, NN, dd)
    
    def test_NN_g_bin_shapes_mismatch(self):
        with self.assertRaises(AssertionError):
            g_bin = np.matrix([
                [1, 0, 0, 1, 1, 0, 0], 
                [1, 1, 0, 1, 1, 1, 0], 
                [1, 0, 1, 0, 1, 0, 1],
            ])
            
            x_min = np.array([1, 2])
            NN = np.array([0, 3, 6])
            dd = np.array([.1, .1])

            a_cod_decimal(g_bin, x_min, NN, dd)
    
    def test_NN_x_min_shapes_mismatch(self):
        with self.assertRaises(AssertionError):
            g_bin = np.matrix([
                [1, 0, 0, 1, 1, 0, 0], 
                [1, 1, 0, 1, 1, 1, 0], 
                [1, 0, 1, 0, 1, 0, 1],
            ])
            
            x_min = np.array([1, 2, 3])
            NN = np.array([0, 3, 7])
            dd = np.array([.1, .1])

            a_cod_decimal(g_bin, x_min, NN, dd)
    
    def test_large_m_performance(self):
        n: int = 2
        m: int = 10**4
        g_bin = np.random.randint(low=0, high=2, size=(n, m))
        x_min = np.zeros(m)
        NN = np.arange(m + 1)
        dd = np.ones(m)
        a_cod_decimal(g_bin, x_min, NN, dd)
    
    def test_large_n_performance(self):
        n: int = 10**4
        m: int = 2
        g_bin = np.random.randint(low=0, high=2, size=(n, m))
        x_min = np.zeros(m)
        NN = np.arange(m + 1)
        dd = np.ones(m)
        a_cod_decimal(g_bin, x_min, NN, dd)

    def test_large_dd_performance(self):
        n: int = 2
        m: int = 10**3
        k: int = 10**2
        g_bin = np.random.randint(low=0, high=2, size=(n, k * m))
        x_min = np.zeros(m)
        NN = k * np.arange(m + 1)
        dd = k * np.ones(m)
        a_cod_decimal(g_bin, x_min, NN, dd)

    def test_correctness(self):
        g_bin = np.matrix([
            [1, 0, 0, 1, 1, 0, 0], 
            [1, 1, 0, 1, 1, 1, 0], 
            [1, 0, 1, 0, 1, 0, 1],
        ])
        
        x_min = np.array([1, 2])
        NN = np.array([0, 3, 7])
        dd = np.array([.1, .1])
        
        self.assertEqual(
            a_cod_decimal(g_bin, x_min, NN, dd).tolist(), 
            [[1.1, 2.3], [1.3, 2.7], [1.5, 3.0]]
        )


if __name__ == '__main__':
    unittest.main()
