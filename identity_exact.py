#!/usr/bin/env python3
"""Exact lower bound for h_F: at the identity field and Bloch momentum k = (pi, pi, pi) the Bloch matrix of
-Hess (3 copies of a scalar form) is the integer 54 x 54 matrix B^T B; its characteristic polynomial is factored
exactly and the largest root of each factor isolated to 1e-12.  Usage: python3 identity_exact.py"""
import contextlib
import io

import numpy as np
import sympy as sp

with contextlib.redirect_stdout(io.StringIO()):
    import bloch_identity as b  # builds the Bloch rows (and runs its own search, output suppressed)

B = np.zeros((len(b.rows), len(b.fi)), int)
for r, terms in enumerate(b.rows):
    for j, n, s in terms:
        B[r, j] += int(s * (-1) ** int(n.sum()))  # e^{i k.n} = (-1)^{n_1+n_2+n_3} at k = (pi, pi, pi)
x = sp.symbols("x")
for f, mult in sp.factor_list((sp.Matrix(B.T @ B)).charpoly(x).as_expr())[1]:
    (lo, hi), _ = max(sp.Poly(f, x).intervals(eps=sp.Rational(1, 10 ** 12)), key=lambda t: t[0][1])
    print(f"degree {sp.degree(f, x)}, multiplicity {mult}: largest root in [{float(lo):.12f}, {float(hi):.12f}]",
          f if sp.degree(f, x) <= 9 else "")
