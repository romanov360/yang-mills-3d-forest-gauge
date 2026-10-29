#!/usr/bin/env python3
"""Per-plaquette version of certificate3d: every plaquette of the period cell gets its own weights (periodic on Z^3).

Each free link must collect at most h - 4t from its four plaquettes.  Exact positions are taken from the torus, so no
symmetry assumption about plaquette labellings is used.
"""
import sys

import cvxpy as cp
import numpy as np

import certificate3d as c
import torus3d as t3

m = int(sys.argv[1]) if len(sys.argv) > 1 else 8
T = t3.Torus(3)
W = c.grid(m)
F = [c.qmat(c.fp, w) for w in W]
C = [c.qmat(c.cpf, w) for w in W]
t = cp.Variable()
h = cp.Variable()
coll = {e: 0 for e in T.free}
cons = []
for p in T.plaqs:
    E = [g for g, _ in p]
    free_pos = [i for i in range(4) if E[i] not in T.fixed]
    a = {i: cp.Variable() for i in free_pos}
    idx = [4 * i + k for i in free_pos for k in range(4)]
    for Fk, Ck in zip(F, C):
        M = -Fk - t * Ck
        M = M[np.ix_(idx, idx)]
        D = sum(a[i] * np.kron(np.diag([1.0 if j == i else 0.0 for j in free_pos]), np.eye(4)) for i in free_pos)
        cons.append(D + 0.5 * (M + M.T) >> 0)
    for i in free_pos:
        coll[E[i]] = coll[E[i]] + a[i]
for e in T.free:
    cons.append(coll[e] + 4 * t <= h)
pr = cp.Problem(cp.Minimize(h), cons)
pr.solve(solver="CLARABEL")
print(f"grid {m}: per-plaquette certificate h = {pr.value:.6f}, t = {t.value:.6f}")
