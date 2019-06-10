#!/usr/bin/env python
import numpy as np
from typing import Tuple


def mutation(g: np.matrix, p: float) -> Tuple[np.matrix, int]:
	"""
	:param g: bin mat to mutate
	:param p: prob of mutation
	:return: 
		g_mut - mutated matrix
		s_mut - number of mutations
	"""
	mask = np.random.uniform(size=g.shape) < p
	return g ^ mask, np.sum(np.sum(mask))


if __name__ == '__main__':
	g = np.random.randint(low=0, high=2, size=(10, 20))

	g_mut, s_mut = mutation(g, 1e-1)

	print(g_mut, s_mut, sep='\n')
