#!/usr/bin/env python
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from random import uniform, seed
from math import cos, pi, exp, sqrt
from typing import List


# region problem setting
n, l = 2, 50


def rastrigin(Xj: List[float]) -> float:
	""" Rastrigin function """
	return 10 * n + sum(Xj[i]**2 - 10 * cos(2 * pi * Xj[i]) 
		for i in range(n))


b_lo, b_up, f = -5.12, 5.12, rastrigin


def rosenbrock(Xj: List[float]) -> float:
	""" Rosenbrock's function """
	return sum(100 * (Xj[i + 1] - Xj[i]**2)**2 + (1 - Xj[i])**2 
		for i in range(n - 1))


# b_lo, b_up = -5, 5


def rosenbrock_penalty(Xj: List[float]) -> float:
	""" Rosenbrock's function with penalty """
	p1 = 100 * max((Xj[0] - 1)**3 - Xj[1] + 1, 0)**2
	p2 = 100 * max(Xj[0] + Xj[1] - 2, 0)**2
	return sum(100 * (Xj[i + 1] - Xj[i]**2)**2 + (1 - Xj[i])**2 
		for i in range(n - 1)) + p1 + p2


# b_lo, b_up = -5, 5


# def f(Xj: List[float]) -> float:
# 	""" Ackley function """
# 	return - 20 * exp(-.2 * sqrt(.5 * (Xj[0]**2 + Xj[1]**2))) - \
# 		exp(.5 * cos(2 * pi * Xj[0]) + cos(2 * pi * Xj[1])) + exp(1) + 20


# b_lo, b_up = 5, 5
# endregion


# region algorithm params
seed(65537)

_omega, step, max_iter, grid_sz = .99, .1, 1 << 7, 100 + 1


def sign(x: float) -> float:
	return (x > 0) - (x < 0)


def a(Xj: List[float], XLj: List[float], XG: List[float]) -> float:
	b1, b2 = f(Xj) - f(XLj), f(Xj) - f(XG)

	return sign(b1) * (.5 + b1**2) / (1 + b1**2 + b2**2), \
		sign(b2) * (.5 + b2**2) / (1 + b1**2 + b2**2)


def omega(k):
	return _omega * (1 - k / (2 * max_iter))
# endregion


# region initial state of the swarm
X = [[uniform(b_lo, b_up) for i in range(n)] for j in range(l)]
V = [[uniform(-step, step) for i in range(n)] for j in range(l)]
F = [f(X[j]) for j in range(l)]
XL, XG = X[:], X[F.index(min(F))]
# endregion


# region plotting (for n == 2 only)
def plot():
	fig = plt.figure(figsize=(10, 10))
	plt.axis([b_lo, b_up, b_lo, b_up])
	xs = [b_lo + (b_up - b_lo) * i / (grid_sz - 1) for i in range(grid_sz)]
	x1s, x2s = [[xs[i] for i in range(grid_sz)] for j in range(grid_sz)], \
		[[xs[j] for i in range(grid_sz)] for j in range(grid_sz)]
	fs = [[f([x1s[i][j], x2s[i][j]]) for j in range(grid_sz)] 
		for i in range(grid_sz)]
	fs_raw = [fs[i][j] for i in range(grid_sz) for j in range(grid_sz)]
	levels = MaxNLocator(nbins=10).tick_values(min(fs_raw), max(fs_raw))
	plt.contourf(x1s, x2s, fs, levels=levels, cmap="RdBu_r")
	plt.colorbar()
	plt.contour(x1s, x2s, fs, levels=levels, colors='k')
	px1s, px2s = [X[j][0] for j in range(l)], [X[j][1] for j in range(l)]
	plt.scatter(px1s, px2s, c='k')
	plt.quiver(px1s, px2s, [V[j][0] for j in range(l)], [V[j][1] 
		for j in range(l)], scale=10)
	plt.draw()
	plt.pause(.1)
	plt.savefig(f'img/{k}.png')
	plt.close()


def plot_err():
	ks = list(range(max_iter + 1))
	fig = plt.figure(figsize=(10, 10))
	plt.plot(ks, err_best, 'r.', label='fitness of the best particle')
	plt.plot(ks, err_local, 'g--', label='average fitness of xl_j')
	plt.plot(ks, err_mean, 'b-', label='average fitness of x_j')
	plt.legend(loc='upper right')
	plt.draw()
	plt.pause(5)
	plt.savefig(f'img/err_plot.png')
	plt.close()
# endregion


# region error tracking
err_best = [f(XG)]
xy_best = [XG]
err_local = [sum(f(XL[j]) for j in range(l)) / l]
err_mean = [sum(f(X[j]) for j in range(l)) / l]


def err():
	global err_mean, err_local, err_best
	xy_best.append(XG)
	err_best.append(f(XG))
	err_local.append(sum(f(XL[j]) for j in range(l)) / l)
	err_mean.append(sum(f(X[j]) for j in range(l)) / l)


def log_err():
	with open(f'err.log', 'w') as out:
		for k in range(max_iter):
			out.write(f'iteration number {k}:\n'
				f'\txy_best   = {xy_best[k]},\n'
				f'\terr_best  = {err_best[k]},\n'
				f'\terr_local = {err_local[k]},\n'
				f'\terr_mean  = {err_mean[k]}.\n\n')
# endregion


# region algorithm itself
for k in range(max_iter):
	rand = [uniform(0, step) for j in range(l)]

	for j in range(l):
		a1, a2 = a(X[j], XL[j], XG)
		V[j] = [omega(k) * V[j][i] + rand[j] * (a1 * (XL[j][i] - X[j][i]) + \
			a2 * (XG[i] - X[j][i])) for i in range(n)]

	if n == 2:
		plot()
	
	err()

	for j in range(l):
		X[j] = [X[j][i] + V[j][i] for i in range(n)]

	X = [X[j] if all(b_lo < X[j][i] < b_up for i in range(n)) else XL[j] 
		for j in range(l)]
	XL = [XL[j] if f(XL[j]) < f(X[j]) else X[j] for j in range(l)]
	F = [f(X[j]) for j in range(l)]
	XG = XG if f(XG) < min(F) else X[F.index(min(F))]

plot_err()
log_err()
# endregion