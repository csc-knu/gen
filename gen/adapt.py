#!/usr/bin/env python
import numpy as np
from itertools import accumulate 


def __adapt(periodic: np.array, n: int) -> np.array:
	"""
	:param periodic: n nums
	:param n: len periodic
	:return: adaptability
	"""
	periodic -= np.min(periodic)

	periodic /= np.sum(periodic)

	num = np.zeros(n)

	for roll in np.random.choice(np.arange(n), n, p=periodic):
		num[roll] += 1

	return num.astype(int)


def adapt(periodic: np.array) -> np.array:
	"""
	:param periodic: n nums
	:return: adaptability
	"""
	n: int = len(periodic)

	return __adapt(periodic, n)


if __name__ == '__main__':
	n = 4

	periodic = np.array([1.1, 1.3, 1.7, 1.9])

	print(adapt(periodic))
