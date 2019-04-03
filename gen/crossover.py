#!/usr/bin/env python
import numpy as np


def __crossover(n: int, g: np.matrix, m_list: np.array, f_list: np.array) -> np.matrix:
	"""
	:param n: half of g.shape[0]
	:param g: bin mat of genes
	:param m_list: male nums
	:param f_list: female nums
	:return: crossed-over bin mat of genes
	"""
	cros = np.random.randint(low=0, high=g.shape[1], size=n)

	g_cros = np.copy(g)

	for m, f, c in zip(m_list, f_list, cros):
		g_cros[[m, f], :c] = g_cros[[f, m], :c]

	return g_cros


def crossover(g: np.matrix, m_list: np.array, f_list: np.array) -> np.matrix:
	"""
	:param g: bin mat of genes
	:param m_list: male nums
	:param f_list: female nums
	:return: crossed-over bin mat of genes
	"""
	return __crossover(g.shape[0] >> 1, g, m_list, f_list)


if __name__ == '__main__':
	n = 2

	g = np.matrix([
		[1, 1, 1, 1, 1, 1], 
		[1, 1, 1, 1, 1, 1], 
		[0, 0, 0, 0, 0, 0], 
		[0, 0, 0, 0, 0, 0],
	])

	m_list, f_list = np.array([0, 1]), np.array([2, 3])

	print(crossover(g, m_list, f_list))
