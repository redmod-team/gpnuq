""" versatile mockup Worker """

from profit.run import Worker

import numpy as np


def rosenbrock(x, y, a, b):
    return (a - x) ** 2 + b * (y - x**2) ** 2


def simulation(u, v, a=1, b=3):
    r = np.array([0, 0.5, 1])
    return rosenbrock((r - 0.5) + u - 5, 1 + 3 * (v - 0.6), a, b)


@Worker.wrap("mockup")
def mockup(u, v) -> "f":
    return simulation(u, v)


if __name__ == "__main__":
    params = np.loadtxt("input.csv", delimiter=",")
    data = simulation(*params)
    np.savetxt("output.csv")
