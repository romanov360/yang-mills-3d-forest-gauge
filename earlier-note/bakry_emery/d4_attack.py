#!/usr/bin/env python3
"""Targeted attack on the conjecture sup_Q lambda_max(Hess S)/(N beta) = 4d, in d = 4 (SU(2)).

Starts: (a) the maximiser of ||d_A||^2, which exceeds 16 on the L = 2 torus; (b) pi-flux plus noise; (c) Haar random.
Tori with L = 2 and L = 3. Prints the best value per start.
"""
import sys

import numpy as np

import fasthess as f
import hessian as h
from dA_norm import norm2
from search import pi_flux


def climb(fun, lat, Q, rng, steps, scale=0.5):
    best = fun(lat, Q)
    for k in range(steps):
        e = rng.integers(lat.nE)
        R = h.expm_su2(sum(c * b for c, b in zip(rng.normal(size=3) * scale, h.BASIS)))
        old = Q[e].copy()
        Q[e] = R @ Q[e]
        v = fun(lat, Q)
        if v >= best:
            best = v
        else:
            Q[e] = old
        if k % 300 == 299:
            scale *= 0.85
    return best, Q


def main():
    rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    Ls = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [2, 3]
    for L in Ls:
        lat = h.Lattice(L, 4)
        bdA, QdA = climb(norm2, lat, h.random_su2(rng, lat.nE), rng, steps)
        hA = f.lam_max(lat, QdA)
        b1, _ = climb(f.lam_max, lat, QdA.copy(), rng, steps, scale=0.2)
        Qp = pi_flux(lat)
        for e in range(lat.nE):
            Qp[e] = h.expm_su2(sum(c * b for c, b in zip(rng.normal(size=3) * 0.3, h.BASIS))) @ Qp[e]
        b2, _ = climb(f.lam_max, lat, Qp, rng, steps)
        b3, _ = climb(f.lam_max, lat, h.random_su2(rng, lat.nE), rng, steps)
        best = max(b1, b2, b3)
        print(f"L={L} d=4: sup||d_A||^2 climb {bdA:.4f}; Hessian at that config {hA:.4f}; Hessian climbs: from it {b1:.4f}, "
              f"from pi-flux+noise {b2:.4f}, from Haar {b3:.4f}; 4d = 16  {'EXCEEDS 4d' if best > 16 + 1e-6 else 'within 4d'}",
              flush=True)


if __name__ == "__main__":
    main()
