#!/usr/bin/env python3
"""Three-dimensional forest gauge on periodic tori: partner counts and the gauge-fixed SU(2) Hessian.

The forest F = {(x - e_mu(x), mu(x))} of the d = 3 pattern (period 3) is a forest on Z^3.  On a torus of side L (a
multiple of 3) the same rule fixes the same links of every period cell; the torus only serves to evaluate the
gauge-fixed Hessian at periodic fields, and a periodic field on the torus is a periodic field on Z^3, so every torus
eigenvalue of the free-link Hessian is a lower bound for the Z^3 constant h_F (Rayleigh quotients of cut-off periodic
vectors).
"""
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "earlier-note", "bakry_emery"))
import fasthess as f  # noqa: E402
import hessian as h  # noqa: E402

PAT3 = "210102120102120210120210102"


class Torus:
    def __init__(self, L, d=3, pattern=PAT3, P=3):
        assert L % P == 0
        self.L, self.d = L, d
        self.sites = list(itertools.product(range(L), repeat=d))
        cells = sorted(itertools.product(range(P), repeat=d))
        mu_of = {c: int(pattern[i]) for i, c in enumerate(cells)}
        self.mu = {x: mu_of[tuple(xi % P for xi in x)] for x in self.sites}
        self.sh = lambda x, a, s=1: tuple((x[i] + (s if i == a else 0)) % L for i in range(d))
        self.edges = [(x, a) for x in self.sites for a in range(d)]
        self.eidx = {e: i for i, e in enumerate(self.edges)}
        self.nE = len(self.edges)
        self.fixed = {self.eidx[(y, a)] for (y, a) in self.edges if self.mu[self.sh(y, a)] == a}
        self.free = [e for e in range(self.nE) if e not in self.fixed]
        self.plaqs = []
        for x in self.sites:
            for a in range(d):
                for b in range(a + 1, d):
                    self.plaqs.append(((self.eidx[(x, a)], True), (self.eidx[(self.sh(x, a), b)], True),
                                       (self.eidx[(self.sh(x, b), a)], False), (self.eidx[(x, b)], False)))


def partners(T):
    """For each free link: (number of fixed links in each of its plaquettes, number of distinct free partners)."""
    out = {}
    for e in T.free:
        ps = [p for p in T.plaqs if e in [g for g, _ in p]]
        part = set()
        types = []
        for p in ps:
            others = [g for g, _ in p if g != e]
            types.append(sum(g in T.fixed for g in others))
            part |= {g for g in others if g not in T.fixed}
        out[e] = (tuple(sorted(types)), len(part), sum(3 - t for t in types))
    return out


def free_hessian(T, Q):
    H = f.hess_matrix(T, Q)
    idx = np.array([3 * e + a for e in T.free for a in range(3)])
    return H[np.ix_(idx, idx)]


def lam(T, Q):
    return float(np.linalg.eigvalsh(free_hessian(T, Q))[-1])


def abelian(theta):
    Q = np.zeros((len(theta), 2, 2), dtype=complex)
    Q[:, 0, 0] = np.exp(1j * theta)
    Q[:, 1, 1] = np.exp(-1j * theta)
    return Q


if __name__ == "__main__":
    T = Torus(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
    pr = partners(T)
    from collections import Counter
    print("fixed links:", len(T.fixed), "of", T.nE)
    print("(plaquette fixed-count types, #partners, sum of free others):", Counter(pr.values()))
    Q = np.array([np.eye(2, dtype=complex)] * T.nE)
    print("lambda_max at identity:", lam(T, Q))
