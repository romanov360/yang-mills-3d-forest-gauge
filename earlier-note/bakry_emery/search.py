#!/usr/bin/env python3
"""Adversarial search for sup_Q lambda_max(Hess S)/(N beta) on small SU(2) tori; compare with SZZ 8(d-1) and with 4d."""
import sys

import numpy as np

import hessian as h


def pi_flux(lat):
    """Centre-valued links with every plaquette = -I: sigma_mu(x) = (-1)^(x_0 + ... + x_{mu-1})."""
    Q = np.empty((lat.nE, 2, 2), complex)
    for x in lat.sites:
        for mu in range(lat.d):
            Q[lat.e(x, mu)] = (-1) ** (sum(x[:mu]) % 2) * np.eye(2)
    return Q


def climb(lat, Q, rng, steps, scale=0.3):
    best = h.lam_max(lat, Q)
    for k in range(steps):
        e = rng.integers(lat.nE)
        c = rng.normal(size=3) * scale
        R = h.expm_su2(sum(c[a] * h.BASIS[a] for a in range(3)))
        old = Q[e].copy()
        Q[e] = R @ Q[e]
        val = h.lam_max(lat, Q)
        if val >= best:
            best = val
        else:
            Q[e] = old
        if k % 200 == 199:
            scale *= 0.8
    return best, Q


def main():
    rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
    for L, d in ((2, 2), (4, 2), (2, 3), (2, 4)):
        lat = h.Lattice(L, d)
        pf = h.lam_max(lat, pi_flux(lat))
        res = []
        for start in range(3):
            Q = h.random_su2(rng, lat.nE)
            b, _ = climb(lat, Q, rng, steps)
            res.append(b)
        b_pf, _ = climb(lat, pi_flux(lat), rng, steps // 2, scale=0.05)
        print(f"L={L} d={d}: pi-flux {pf:.4f}; pi-flux+climb {b_pf:.4f}; random+climb {', '.join(f'{x:.4f}' for x in res)};"
              f"  SZZ 8(d-1) = {8 * (d - 1)}, 4d = {4 * d}", flush=True)


if __name__ == "__main__":
    main()
