#!/usr/bin/env python3
"""Adversarial search for sup_Q lambda_max(Hess S)/(N beta), SU(2), with the vectorised Hessian.

Start families: Haar-random links; random centre (Z/2) links; abelian U(1) links exp(i theta sigma_3) with random
angles; pi-flux with random perturbations. Climb: single-link multiplicative moves, annealed step, accept if
lambda_max does not decrease. Prints the best value per (L, d) with the bounds 8(d-1) (SZZ) and 4d.
"""
import sys

import numpy as np

import fasthess as f
import hessian as h
from search import pi_flux


def climb(lat, Q, rng, steps, scale):
    best = f.lam_max(lat, Q)
    for k in range(steps):
        e = rng.integers(lat.nE)
        R = h.expm_su2(sum(c * b for c, b in zip(rng.normal(size=3) * scale, h.BASIS)))
        old = Q[e].copy()
        Q[e] = R @ Q[e]
        val = f.lam_max(lat, Q)
        if val >= best:
            best = val
        else:
            Q[e] = old
        if k % 300 == 299:
            scale *= 0.85
    return best, Q


def starts(lat, rng, kind):
    n = lat.nE
    if kind == "haar":
        return h.random_su2(rng, n)
    if kind == "centre":
        return np.array([rng.choice([1, -1]) * np.eye(2, dtype=complex) for _ in range(n)])
    if kind == "piflux+noise":
        Q = pi_flux(lat)
        for e in range(n):
            Q[e] = h.expm_su2(sum(c * b for c, b in zip(rng.normal(size=3) * 0.3, h.BASIS))) @ Q[e]
        return Q
    if kind == "u1":
        th = rng.uniform(0, 2 * np.pi, n)
        return np.array([np.diag([np.exp(1j * t), np.exp(-1j * t)]) for t in th])
    raise ValueError(kind)


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    rng = np.random.default_rng(seed)
    for L, d in ((2, 3), (3, 3), (2, 4)):
        lat = h.Lattice(L, d)
        out = []
        for kind in ("haar", "haar", "centre", "u1", "piflux+noise"):
            b, Q = climb(lat, starts(lat, rng, kind), rng, steps, 0.5)
            out.append((kind, b))
        best = max(b for _, b in out)
        print(f"L={L} d={d}: " + ", ".join(f"{k} {b:.5f}" for k, b in out) +
              f"  | best {best:.5f}  4d = {4 * d}  SZZ 8(d-1) = {8 * (d - 1)}  {'EXCEEDS 4d' if best > 4 * d + 1e-9 else 'within 4d'}",
              flush=True)


if __name__ == "__main__":
    main()
