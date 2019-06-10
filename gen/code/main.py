#!/usr/bin/env python
import numpy as np
from generation_dec import generation_dec
from bin_dec_param import bin_dec_param
from cod_binary import a_cod_binary
from cod_decimal import a_cod_decimal
from mutation import mutation
from parents import parents
from crossover import crossover
from adapt import adapt
from new_generation import new_generation
from best_worst import best, worst


np.set_printoptions(linewidth=100)

np.random.seed(65537)

n, _m = 1 << 10, 20

x_min, x_max = np.repeat(-2, _m), np.repeat(2, _m)

# x_min, x_max = np.repeat(-5.12, _m), np.repeat(5.12, _m)

g_dec = generation_dec(n, x_min, x_max)

# print(f'g_dec = {g_dec}')

eps = 1e-5

nn, dd, NN = bin_dec_param(x_min, x_max, eps)

# print(
# 	f'nn = {nn}\n'
# 	f'dd = {dd}\n'
# 	f'NN = {NN}'
# )

p = 2e-3

def rastrigin(x: np.array) -> float:
	return -10 * _m - np.sum(np.power(x, 2) - 10 * np.cos(2 * np.pi * x))


for _ in range(1, 1 << 10):
	g_bin = a_cod_binary(g_dec, x_min, nn, dd)

	# print(f'g_bin = {g_bin}')

	g_bin, mutation_count = mutation(g_bin, p)

	# print(
	# 	f'g_bin = {g_bin}\n'
	# 	f'mutation_count = {mutation_count}'
	# )

	m, f = parents(n >> 1)

	# print(
	# 	f'm = {m}\n'
	# 	f'f = {f}'
	# )

	g_bin = crossover(g_bin, m, f)

	# print(f'g_bin = {g_bin}')

	g_dec = a_cod_decimal(g_bin, x_min, NN, dd)

	# print(f'g_dec = {g_dec}')

	f_vals = np.array([rastrigin(g_dec[i]) for i in range(n)]).reshape((n, 1))

	# if not (_ & 63):
	print(f'it {_:0>4}: {np.max(f_vals):>12.7f}')

	# print(f'f_vals = {f_vals}')

	g_dec = np.hstack([g_dec, f_vals])

	# print(f'g_dec[:, -1] = {g_dec[:, -1]}')

	b, w = best(g_dec), worst(g_dec)

	# print(
	# 	f'b = {b}\n'
	# 	f'w = {w}'
	# )

	g_best = np.asarray(g_dec[b, :-1]).flatten()

	# print(f'g_best = {g_best}')

	g_dec = np.delete(g_dec, w, axis=0)

	# print(f'g_dec = {g_dec}, g_dec.shape = {g_dec.shape}')

	num = adapt(np.asarray(g_dec[:, -1]).flatten())

	# print(f'num = {num}, len(num) = {len(num)}')

	g_dec = np.vstack([new_generation(g_dec[:, :-1], num), g_best])

	# print(f'g_dec = {g_dec}')

print(g_dec)
 