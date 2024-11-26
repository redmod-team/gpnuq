""" versatile mockup Worker """

from profit.run import Worker

import numpy as np


def rosenbrock(x, y, a, b):
    return (a - x) ** 2 + b * (y - x**2) ** 2


@Worker.wrap("mockup")
def mockup(r, u, v, a, b) -> "f":
    return rosenbrock((r - 0.5) + u - 5, 1 + 3 * (v - 0.6), a, b)
