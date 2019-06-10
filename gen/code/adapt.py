#!/usr/bin/env python
import numpy as np
from itertools import accumulate 


def __adapt(p: np.array, n: int) -> np.array:
	"""
	:param p: n nums
	:param n: len p
	:return: adaptability
	"""
	p -= np.min(p)

	p /= np.sum(p)

	num = np.zeros(n)

	for roll in np.random.choice(np.arange(n), n, p=p):
		num[roll] += 1

	return num.astype(int)


def adapt(p: np.array) -> np.array:
	"""
	:param p: n nums
	:return: adaptability
	"""
	n: int = len(p)

	return __adapt(p, n)


if __name__ == '__main__':
	n = 4

	p = np.array([1.1, 1.3, 1.7, 1.9])

	print(adapt(p))
