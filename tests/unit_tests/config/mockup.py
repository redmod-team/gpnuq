"""Mockup domain code."""

import numpy as np
from h5py import File


def rosenbrock(x, y, a, b):
    return (a - x) ** 2 + b * (y - x**2) ** 2


def f(r, u, v):
    return rosenbrock((r - 0.5) + u - 5, 1 + 3 * (v - 0.6), a=1, b=3), r


params = np.loadtxt("mockup.in")
r = np.arange(0, 1, 0.1)
result = f(r, params[0], params[1])
print(result)
with File("mockup.out", "w") as fout:
    fout["f"] = result[0]
    fout["g"] = result[1]
