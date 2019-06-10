#!/usr/bin/env python
import numpy as np
from typing import Tuple


def parents(n: int) -> Tuple[np.array, np.array]:
	"""
	:param n: half of nums to div into lists
	:return: two lists with permutation of [0, 2n)
	"""
	p = np.random.permutation(np.arange(n << 1))
	return p[:n], p[n:]


if __name__ == '__main__':
	n = 7

	m_list, f_list = parents(n)

	print(m_list, f_list, sep='\n')
