#!/usr/bin/env python
import numpy as np
from generate_solution import generate_solution
from calculate_feroments import calculate_feroments

np.set_printoptions(precision=3)

np.random.seed(0)

T = 100
q = np.random.choice(np.arange(1, 5), size=T)
w = np.random.choice(np.arange(10, 20), size=T)
u = np.random.choice(np.arange(100, 200), size=T)
W = 1_000
N = 100
M = 100
alpha = .1
F = np.array(np.repeat(100, T), dtype='float64')
best_u = 0

for i in range(M):
	f = np.array(np.repeat(0, T), dtype='float64')
	for j in range(N):
		c, _w, _u = generate_solution(T, q, w, u, W, F)
		best_u = max(_u, best_u)
		f += calculate_feroments(c, _w, _u, w, u)
	print(f'it #{i}: best_u = {best_u}')
	F = alpha * f + (1 - alpha) * F

print(generate_solution(T, q, w, u, W, F))
print(u/w)
print(F)