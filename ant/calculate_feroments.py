#!/usr/bin/env python
import numpy as np

def calculate_feroments(c: np.array, _w: float, _u: float,
		w: np.array, u: np.array) -> np.array:
	f = np.log(c + 1) * (u / w) * (_u / _w)
	return np.array(f, dtype='float64')
