#!/usr/bin/env python3
"""Hill-climb the extreme eigenvalues of the d = 3 forest-gauge-fixed SU(2) Hessian over fields on a torus.

usage: climb3d.py L seed steps mode   (mode: abelian | full); prints sup lambda_max and sup (-lambda_min) found.
"""
import sys

import numpy as np

import torus3d as t
h = t.h

L, seed, steps, mode = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
rng = np.random.default_rng(seed)
T = t.Torus(L)
free = T.free


def ext(Q):
    ev = np.linalg.eigvalsh(t.free_hessian(T, Q))
    return ev[-1], -ev[0]


for which in (0, 1):
    best_all = -1
    for start in range(3):
        if mode == "abelian":
            th = rng.uniform(0, 2 * np.pi, T.nE)
            th[list(T.fixed)] = 0
            Q = t.abelian(th)
        else:
            Q = h.random_su2(rng, T.nE)
            for e in T.fixed:
                Q[e] = np.eye(2)
        best = ext(Q)[which]
        scale = 1.0
        for k in range(steps):
            e = free[rng.integers(len(free))]
            old = Q[e].copy()
            if mode == "abelian":
                z = np.exp(1j * rng.normal() * scale)
                Q[e] = np.diag([z, z.conjugate()]) @ Q[e]
            else:
                Q[e] = h.expm_su2(sum(c * b for c, b in zip(rng.normal(size=3) * scale, h.BASIS))) @ Q[e]
            v = ext(Q)[which]
            if v >= best:
                best = v
            else:
                Q[e] = old
            if k % 300 == 299:
                scale *= 0.7
        best_all = max(best_all, best)
        print(f"{['lambda_max', '-lambda_min'][which]} start {start}: {best:.6f}", flush=True)
    print(f"L={L} {mode}: sup {['lambda_max', '-lambda_min'][which]} = {best_all:.6f}", flush=True)
