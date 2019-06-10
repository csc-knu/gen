#!/usr/bin/env python
import numpy as np


def __best(m: int, g_dec: np.matrix) -> int:
	"""
	:param m: num args
	:param g_dec: args and func vals mat
	:return: argmax func val index
	"""
	return np.argmax(g_dec[:, m])


def __worst(m: int, g_dec: np.matrix) -> int:
	"""
	:param m: num args
	:param g_dec: args and func vals mat
	:return: argmin func val index
	"""
	return np.argmin(g_dec[:, m])


def best(g_dec: np.matrix) -> int:
	"""
	:param g_dec: args and func vals mat
	:return: argmax func val index
	"""
	return __best(-1, g_dec)


def worst(g_dec: np.matrix) -> int:
	"""
	:param g_dec: args and func vals mat
	:return: argmin func val index
	"""
	return __worst(-1, g_dec)


if __name__ == '__main__':
	m = 3

	g_dec = np.matrix([
		[0, 0, 1, 1],
		[0, 1, 0, 2],
		[1, 0, 0, 4],
		[1, 1, 1, 7],
		[0, 0, 0, 0],
		[0, 1, 1, 3],
		[1, 0, 1, 5],
		[1, 1, 0, 6], 
	])

	print(best(m, g_dec), worst(m, g_dec))
