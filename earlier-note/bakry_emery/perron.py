#!/usr/bin/env python3
"""Perron eigenvalue rho(M_F) of the free-edge 'all-plus' matrix M_F[e,e'] = #{plaquettes containing both e and e'}
(diagonal = #plaquettes containing e, weighted by nothing else), which gives the sharper rigorous bound
|Hess S^F| <= rho(M_F) N |beta| |v|^2 (from Lemma 3.1: sum_p (sum_{e in p} |X_e|)^2 = <|X|, M_F |X|>).
For a periodic pattern on Z^d, rho is attained at Bloch momentum 0, i.e. on the period torus (we use a torus of side
2P or 3P, a multiple of the period, so that all edges are distinct); boxes have rho <= this (principal submatrix)."""
import itertools
import sys

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigsh

PATTERNS = {4: ("0212331001332120", 2), 3: ("210102120102120210120210102", 3)}


def rho(d, pat, P, Lmul=2):
    L = P * Lmul
    sites = list(itertools.product(range(L), repeat=d))
    cells = sorted(itertools.product(range(P), repeat=d))
    mu_of = {c: int(pat[i]) for i, c in enumerate(cells)}
    mu = {x: mu_of[tuple(xi % P for xi in x)] for x in sites}
    sh = lambda x, a: tuple((x[i] + (1 if i == a else 0)) % L for i in range(d))
    eidx = {}
    for x in sites:
        for a in range(d):
            eidx[(x, a)] = len(eidx)
    fixed = {eidx[(y, a)] for (y, a) in eidx if mu[sh(y, a)] == a}
    M = lil_matrix((len(eidx), len(eidx)))
    rows = np.zeros(len(eidx))
    for y in sites:
        for a in range(d):
            for b in range(a + 1, d):
                es = [eidx[(y, a)], eidx[(sh(y, a), b)], eidx[(sh(y, b), a)], eidx[(y, b)]]
                free = [e for e in es if e not in fixed]
                for e in free:
                    for e2 in free:
                        M[e, e2] += 1
    free_all = [e for e in range(len(eidx)) if e not in fixed]
    Mf = M.tocsr()[free_all][:, free_all]
    rs = np.asarray(Mf.sum(axis=1)).ravel()
    val = eigsh(Mf.astype(float), k=1, which="LA")[0][0]
    return val, rs.min(), rs.max(), rs.mean()


if __name__ == "__main__":
    for d in (3, 4):
        pat, P = PATTERNS[d]
        r, lo, hi, avg = rho(d, pat, P)
        print(f"d={d}: Perron rho(M_F) = {r:.6f}; row sums in [{lo:.0f}, {hi:.0f}], mean {avg:.4f}; lower bound 8(d-1)^2/d = {8 * (d - 1) ** 2 / d:.4f}")
    # temporal gauge control in d = 4: rho should be 20
    r, lo, hi, avg = rho(4, "0" * 16, 2)
    print(f"control temporal gauge d=4: rho = {r:.6f} (expected 20)")
