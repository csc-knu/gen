#!/usr/bin/env python
import numpy as np
import unittest
import time


def __cod_binary(x_dec: float, x_min: float, l: int, d: float) -> np.array:
    """
    :param x_dec: dec num to encode
    :param x_min: min num possible to encode
    :param l: num of bin digits to return
    :param d: discretionary of encoding to return
    :return: encoded num as np.array of digits
    """
    assert l > 0, "num digits in encoding should be positive"
    assert d > 0, "discretionery should be positive"

    # either this assertion or the following assertion about xx should be removed
    if x_dec < x_min: x_dec = x_min
    if x_dec > x_min + d * ((1 << l) - 1): x_dec = x_min + d * ((1 << l) - 1)
    assert x_min <= x_dec <= x_min + d * 2**l, "x_dec cannot be encoded"
    xx = int(np.floor((x_dec - x_min) / d))
    assert xx <= 2**l - 1, "x_dec cannot be encoded"
    xx ^= xx >> 1
    
    digits = list(map(int, bin(xx)[:1:-1]))

    return np.array(digits + [0 for _ in range(l - len(digits))])


def __a_cod_binary(n: int, m: int, g_dec: np.matrix, x_min: np.array,
                   nn: np.array, dd: np.array) -> np.matrix:
    """
    :param n: num rows in g_dec
    :param m: num cols in g_dec
    :param g_dec: dec nums matrix to encode
    :param x_min: min nums possible to encode
    :param nn: num bin digits to encode nums
    :param dd: discretionaries of encodings
    :return: encoded nums as np.matrix of np.arrays of digits
    """
    assert x_min.shape == (m, ), "x_min should be of shape (m, )"
    assert nn.shape == (m, ), "nn should be of shape (m, )"
    assert dd.shape == (m, ), "dd should be of shape (m, )"
    assert g_dec.shape == (n, m), "g_dec should be of shape (n, m)"

    return np.matrix([
        np.hstack([
            __cod_binary(g_dec[i, j], x_min[j], nn[j], dd[j]) for j in range(m)
        ]) for i in range(n)
    ])


def a_cod_binary(g_dec: np.matrix, x_min: np.array, nn: np.array, 
                 dd: np.array) -> np.matrix:
    """
    :param g_dec: dec nums matrix to encode
    :param x_min: min nums possible to encode
    :param nn: num bin digits to encode nums
    :param dd: discretionaries of encodings
    :return: encoded nums as np.matrix of np.arrays of digits
    """
    assert g_dec.ndim == 2, f"g_dec should be a matrix (have dimension 2). " + \
                            f"got g_dec.ndim = {g_dec.ndim}, g_dec = {g_dec}"
    assert x_min.ndim == 1, "x_min should be an array (have dimension 1)"
    assert nn.ndim == 1, "nn should be an array (have dimension 1)"
    assert dd.ndim == 1, "dd should be an array (have dimension 1)"
    assert g_dec.shape[1] == x_min.shape[0], "g_dec and x_min shapes mismatch"
    assert g_dec.shape[1] == nn.shape[0], "g_dec and nn shapes mismatch"
    assert g_dec.shape[1] == dd.shape[0], "g_dec and dd shapes mismatch"
    n, m = g_dec.shape
    
    return __a_cod_binary(n, m, g_dec, x_min, nn, dd)


class TestCodBinary(unittest.TestCase):
    def setUp(self):
        self.SLOW_TEST_THRESHOLD = .1
        self._started_at = time.time()
    
    def tearDown(self):
        elapsed: float = time.time() - self._started_at
        if elapsed > self.SLOW_TEST_THRESHOLD:
            print(f'{self.id()} ({round(elapsed, 3)}s)')

    def test_g_dec_wrong_ndim(self):
        with self.assertRaises(AssertionError):
            nn = np.array([1, 1])
            dd = np.array([1, 1])
            x_min = np.array([0, 1])
            g_dec = np.array([[[0, 1], [0, 1]], [[0, 1], [0, 1]]])
            a_cod_binary(g_dec, x_min, nn, dd)

    def test_x_min_wrong_ndim(self):
        with self.assertRaises(AssertionError):
            nn = np.array([1, 1])
            dd = np.array([1, 1])
            x_min = np.array([[0], [1]])
            g_dec = np.array([[0, 1], [0, 1]])
            a_cod_binary(g_dec, x_min, nn, dd)

    def test_nn_wrong_ndim(self):
        with self.assertRaises(AssertionError):
            nn = np.array([[1], [1]])
            dd = np.array([1, 1])
            x_min = np.array([0, 1])
            g_dec = np.array([[0, 1], [0, 1]])
            a_cod_binary(g_dec, x_min, nn, dd)

    def test_dd_wrong_ndim(self):
        with self.assertRaises(AssertionError):
            nn = np.array([1, 1])
            dd = np.array([[1], [1]])
            x_min = np.array([0, 1])
            g_dec = np.array([[0, 1], [0, 1]])
            a_cod_binary(g_dec, x_min, nn, dd)

    def test_x_min_and_g_dec_shapes_mismatch(self):
        with self.assertRaises(AssertionError):
            nn = np.array([1, 1])
            dd = np.array([1, 1])
            x_min = np.array([0, 1])
            g_dec = np.array([[0, 1], [0, 1], [0, 1]])
            a_cod_binary(g_dec, x_min, nn, dd)

    def test_nn_and_g_dec_shapes_mismatch(self):
        with self.assertRaises(AssertionError):
            nn = np.array([1, 1, 1])
            dd = np.array([1, 1])
            x_min = np.array([0, 1])
            g_dec = np.array([[0, 1], [0, 1]])
            a_cod_binary(g_dec, x_min, nn, dd)

    def test_dd_and_g_dec_shapes_mismatch(self):
        with self.assertRaises(AssertionError):
            nn = np.array([1, 1])
            dd = np.array([1, 1, 1])
            x_min = np.array([0, 1])
            g_dec = np.array([[0, 1], [0, 1]])
            a_cod_binary(g_dec, x_min, nn, dd)

    def test_large_n(self):
        n: int = 5 * 10**3
        m: int = 2
        x_min = np.zeros(m, dtype=float)
        nn = np.ones(m, dtype=int)
        dd = np.ones(m, dtype=int)
        g_dec = np.random.uniform(low=0, high=1, size=(n, m))
        a_cod_binary(g_dec, x_min, nn, dd)

    def test_large_m(self):
        n: int = 2
        m: int = 5 * 10**3
        x_min = np.zeros(m, dtype=float)
        nn = np.ones(m, dtype=int)
        dd = np.ones(m, dtype=int)
        g_dec = np.random.uniform(low=0, high=1, size=(n, m))
        a_cod_binary(g_dec, x_min, nn, dd)

    def test_correctenss(self):
        g_dec = np.matrix([[1.1, 2.31], [1.3, 2.7]])
        x_min = np.array([1, 2])
        nn = np.array([3, 4])
        dd = np.array([.1, .1])
        self.assertEqual(a_cod_binary(g_dec, x_min, nn, dd).tolist(),
                         [[1, 0, 0, 1, 1, 0, 0], [1, 1, 0, 1, 1, 1, 0]])


if __name__ == '__main__':
    unittest.main()
