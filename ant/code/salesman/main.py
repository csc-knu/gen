#!/usr/bin/env python
from numpy.random import choice
import matplotlib.pyplot as plt


class Ant:
    def __init__(self, start, target):
        self.tabu_list = []
        self.vertice = start
        self.target = target
        self.path_length = 0
        self.path = []
        self.alive = True

    def step(self):
        pre_probability = { 
            to : (g[self.vertice][to].feroment + \
                1 / g[self.vertice][to].length)
            for to in (set(g[self.vertice].keys()) - set(self.tabu_list))
        }

        if not pre_probability:
            self.alive = False
            return

        sum_pre_probability = sum(pre_probability.values())

        probability = {
            to : pre_probability[to] / sum_pre_probability
            for to in pre_probability
        }

        choose_from, choice_probability = [], []

        for t in probability:
            choose_from.append(t)
            choice_probability.append(probability[t])

        step_to = choice(choose_from, p=choice_probability)

        self.path_length += g[self.vertice][step_to].length
        self.path.append((self.vertice, step_to))
        self.tabu_list.append(self.vertice)
        self.vertice = step_to

    def solve(self):
        while self.vertice != self.target and self.alive:
            self.step()

        if self.alive:
            for f, t in self.path:
                g[f][t].delta += .1 * g[f][t].length / self.path_length**2


class Edge:
    def __init__(self, length):
        self.length, self.feroment, self.delta = length, length, 0

    def __repr__(self):
        return f'Edge(length={self.length}, feroment={self.feroment}, delta={self.delta})'


if __name__ == '__main__':
    A, B, C, D, E, F, G = 0, 1, 2, 3, 4, 5, 6

    START, END = A, G

    g = {
        A: {B: Edge(2), C: Edge(3), D: Edge(6)},
        B: {E: Edge(4), F: Edge(5)},
        C: {E: Edge(2), F: Edge(3)},
        D: {E: Edge(5), F: Edge(2)},
        E: {G: Edge(2)},
        F: {G: Edge(1)},
        G: {},
    }

    h = [{f: {t: g[f][t].feroment / g[f][t].length for t in g[f]} for f in g}]

    n, m = 1000, 1000
    
    for i in range(m):
        for j in range(n):
            print(f'{i:0>3} {j:0>3}')
            ant = Ant(start=START, target=END)
            ant.solve()

        for f in g:
            for t in g[f]:
                g[f][t].feroment, g[f][t].delta = \
                    .7 * g[f][t].feroment + g[f][t].delta, 0

        h.append({f: {t: g[f][t].feroment / g[f][t].length for t in g[f]} for f in g})

    k = 0
    style = [
        {'c': 'r', 'd': '-'},
        {'c': 'r', 'd': '--'},
        {'c': 'r', 'd': '-.'},
        {'c': 'r', 'd': '.'},
        {'c': 'g', 'd': '-'},
        {'c': 'g', 'd': '--'},
        {'c': 'g', 'd': '-.'},
        {'c': 'g', 'd': '.'},
        {'c': 'b', 'd': '-'},
        {'c': 'b', 'd': '--'},
        {'c': 'b', 'd': '-.'},
        {'c': 'b', 'd': '.'},
    ]

    plt.figure(figsize=(20,10))
    plt.title('Feroment intensity on edges', fontsize=20)
    plt.xlabel('Iteration number', fontsize=16)
    plt.ylabel('Feroment intensity', fontsize=16)
    for f in g:
        for t in g[f]:
            k += 1
            plt.plot(
                list(range(m)), 
                [h[i][f][t] for i in range(m)], 
                f"{style[k]['c']}{style[k]['d']}",
                label=f'${f} \\to {t}$'
            )
    plt.legend(fontsize=16)
    plt.grid(True)
    # plt.show()
    plt.savefig(f'feroment_{m}.png', bbox_inches='tight')
