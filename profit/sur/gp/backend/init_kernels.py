"""Script to initialize kernels via SymPy.

Running this script generates Fortran code for kernels
that is written to `kernels_base.f90`. This code has to
be compiled via `make` subsequently.
"""

# %%
from sympy import symbols, sqrt, exp, diff
from sympy.utilities.codegen import codegen
import shutil

# %% # Kernel functions (kernel, 1st and second derivatives)
r, ra, rb = symbols(r"r r_a r_b", real=True)
l = symbols(r"l", positive=True)

kern = {}
kern["sqexp"] = exp(-(r**2) / 2)
kern["matern32"] = (1 + sqrt(3) * r) * exp(-sqrt(3) * r)
kern["matern52"] = (1 + sqrt(5) * r + 5 * r / 3) * exp(-sqrt(5) * r)
kern["wend4"] = (1 - r) ** 4 * (1 + 4 * r)

dkern = {}
for k, v in kern.items():
    dkern[k] = diff(v, r).simplify()

d2kern = {}
for k, v in dkern.items():
    d2kern[k] = diff(v, r).simplify()

funlist = []
for k in kern:
    funlist = funlist + [
        ("kern_{}".format(k), kern[k]),
        ("dkern_{}".format(k), dkern[k]),
        ("d2kern_{}".format(k), d2kern[k]),
    ]

[(name, code), (h_name, header)] = codegen(
    funlist, "F95", "kernels_base", header=False, empty=False
)

with open(name, "w") as f:
    f.write(code)
