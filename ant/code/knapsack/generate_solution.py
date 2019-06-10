#!/usr/bin/env python
import numpy as np
from typing import Tuple


def chi(w: np.array, u: np.array, f: np.array, 
		rho: float=1.) -> np.array:
	return (u / w) * (1 + rho * f)


def generate_solution(T: int, q: np.array, w: np.array, u: np.array, 
		W: int, F: np.array) -> Tuple[np.array, int, int]:
	c, _w, _u = np.repeat(0, T), 0, 0
	while True:
		_chi = chi(w, u, F) * (w <= W) * (q > c)
		if sum(_chi) == 0:
			return c, _w, _u
		_chi /= sum(_chi)
		i = np.random.choice(np.arange(T), p=_chi)
		W -= w[i]
		_w += w[i]
		_u += u[i]
		c[i] += 1
