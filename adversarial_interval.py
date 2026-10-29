#!/usr/bin/env python3
"""Adversarial tests of the interval proof (cert_interval.py); not part of the proof.

1. Lipschitz bound: on random phase pairs for all 81 plaquettes, ||M_p(phi) - M_p(phi')|| / ((4 + |k23|) max|dphi|)
   must be <= 1.
2. Tamper test on the tightest plaquette: lowering its weights by delta lowers lambda_min by exactly delta.  With
   delta = 0.016 the matrix is indefinite and both modes must reject it; with delta = 0.004 it stays positive
   definite and both modes should accept it.
Usage: python3 adversarial_interval.py   (uses cert_h9.6.npy; writes nothing but stdout and a scratch file)
"""
import os
import sys
import tempfile
from fractions import Fraction

import numpy as np

import certify3d as cz

cert = np.load("cert_h9.6.npy", allow_pickle=True).item()
G = {y: [[Fraction(v) for v in r] for r in cert["G"][y]] for y in cz.T.sites}
A = [{i: Fraction(v) for i, v in a.items()} for a in cert["A"]]
rng = np.random.default_rng(1)
worst, mins = 0.0, []
for j in range(len(cz.PL)):
    for r in (1e-3, 1e-2, 0.1, 1.0, 3.0):
        P, D = rng.uniform(0, 2 * np.pi, (400, 2)), rng.uniform(-r, r, (400, 2))
        M1, k23 = cz.numeric_matrices(G, A, j, P)
        M2, _ = cz.numeric_matrices(G, A, j, P + D)
        nrm = np.abs(np.linalg.eigvalsh(M1 - M2)).max(axis=1)
        worst = max(worst, (nrm / ((4 + k23) * np.abs(D).max(axis=1))).max())
    M, _ = cz.numeric_matrices(G, A, j, cz.grid_pts(128))
    mins.append(np.linalg.eigvalsh(M)[:, 0].min())
print(f"1. largest ||dM|| / ((4+|k23|) max|dphi|) over 81 x 2000 random pairs: {worst:.4f} (must be <= 1)")
j = int(np.argmin(mins))
print(f"2. tightest plaquette {j}: least eigenvalue on a 128 x 128 grid {mins[j]:.5f}")

tmp = os.path.join(tempfile.mkdtemp(), "tamper.npy")
for delta in (Fraction(16, 1000), Fraction(4, 1000)):
    bad = dict(cert)
    bad["A"] = list(cert["A"])
    bad["A"][j] = {i: str(Fraction(v) - delta) for i, v in cert["A"][j].items()}
    np.save(tmp, bad, allow_pickle=True)
    sys.argv = ["cert_interval.py", tmp]
    sys.modules.pop("cert_interval", None)
    import cert_interval as ci
    M, _ = cz.numeric_matrices(ci.Gq, ci.Aq, j, cz.grid_pts(128))
    print(f"   delta = {float(delta)}: least grid eigenvalue {np.linalg.eigvalsh(M)[:, 0].min():.5f}")
    for mode in ("centre", "box"):
        try:
            n, d = ci.prove(mode, 64, max_depth=8, plaquettes=[j])
            print(f"     {mode}: accepted (depth {d}, {n} squares)")
        except AssertionError as e:
            print(f"     {mode}: rejected ({e})")
