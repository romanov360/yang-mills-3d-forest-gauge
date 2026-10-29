#!/usr/bin/env python3
"""sup over SU(2) connections of ||d_A||^2, (d_A X)_p = X_{x,mu} + Ad(Q_{x,mu}) X_{x+mu,nu} - X_{x,nu} - Ad(Q_{x,nu}) X_{x+nu,mu}."""
import sys

import numpy as np

import fasthess as f
import hessian as h


def dA_matrix(lat, Q):
    Ad = f.adjoint(Q)  # (nE, 3, 3)
    P = len(lat.plaqs)
    D = np.zeros((3 * P, 3 * lat.nE))
    for k, ((e1, _), (e2, _), (e3, _), (e4, _)) in enumerate(lat.plaqs):
        r = slice(3 * k, 3 * k + 3)
        D[r, 3 * e1:3 * e1 + 3] += np.eye(3)
        D[r, 3 * e2:3 * e2 + 3] += Ad[e1]
        D[r, 3 * e4:3 * e4 + 3] -= np.eye(3)
        D[r, 3 * e3:3 * e3 + 3] -= Ad[e4]
    return D


def norm2(lat, Q):
    return float(np.linalg.norm(dA_matrix(lat, Q), 2) ** 2)


if __name__ == "__main__":
    rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    for L, d in ((2, 3), (3, 3), (2, 4)):
        lat = h.Lattice(L, d)
        best = 0
        for start in range(3):
            Q = h.random_su2(rng, lat.nE)
            b, sc = norm2(lat, Q), 0.5
            for k in range(2000):
                e = rng.integers(lat.nE)
                R = h.expm_su2(sum(c * bb for c, bb in zip(rng.normal(size=3) * sc, h.BASIS)))
                old = Q[e].copy()
                Q[e] = R @ Q[e]
                v = norm2(lat, Q)
                if v >= b:
                    b = v
                else:
                    Q[e] = old
                if k % 300 == 299:
                    sc *= 0.85
            best = max(best, b)
        I = np.array([np.eye(2, dtype=complex)] * lat.nE)
        print(f"L={L} d={d}: ||d||^2 at identity {norm2(lat, I):.4f}; sup_A ||d_A||^2 (climb) {best:.4f}; 4d = {4 * d}, 8(d-1) = {8 * (d - 1)}", flush=True)
