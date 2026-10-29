#!/usr/bin/env python3
"""Adversarial sup of lambda_max of the gauge-fixed Hessian for the optimal forests (method ceiling), SU(2)."""
import sys

import numpy as np

import hessian as h
from verify_forest import PATTERNS, Box, coefficients, free_hessian


def lam(box, Q, fixed_idx):
    Hf, _ = free_hessian(box, Q, fixed_idx)
    return np.linalg.eigvalsh(Hf)[-1]


def main():
    rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
    for d, L in ((3, 6), (4, 4)):
        pat, P, cF = PATTERNS[d]
        box = Box(L, d, pat, P)
        _, _, fixed_idx = coefficients(box)
        free = [e for e in range(box.nE) if e not in fixed_idx]
        best_all = -1
        for start in range(3):
            Q = h.random_su2(rng, box.nE)
            for e in fixed_idx:
                Q[e] = np.eye(2)
            best, sc = lam(box, Q, fixed_idx), 0.5
            for k in range(steps):
                e = free[rng.integers(len(free))]
                R = h.expm_su2(sum(c * b for c, b in zip(rng.normal(size=3) * sc, h.BASIS)))
                old = Q[e].copy()
                Q[e] = R @ Q[e]
                v = lam(box, Q, fixed_idx)
                if v >= best:
                    best = v
                else:
                    Q[e] = old
                if k % 300 == 299:
                    sc *= 0.85
            best_all = max(best_all, best)
        print(f"d={d} box {L}^{d}: adversarial sup lambda_max (gauge-fixed, optimal forest) ~ {best_all:.4f}; "
              f"rigorous c_F = {cF}; 4d = {4 * d}", flush=True)


if __name__ == "__main__":
    main()
