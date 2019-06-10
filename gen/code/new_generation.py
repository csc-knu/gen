#!/usr/bin/env python
import numpy as np


def new_generation(g_dec: np.matrix, num: np.array) -> np.matrix:
	"""
	:param g_dec: old generation mat
	:param num: repeats
	:return: new generation mat
	"""
	return np.repeat(g_dec, repeats=num, axis=0)


if __name__ == '__main__':
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

	num = np.array([0, 0, 1, 2, 0, 1, 2, 2])

	print(new_generation(g_dec, num))
