#!/usr/bin/env python3
"""Bloch Hessian of the d = 3 forest-gauge-fixed SU(2) action at fields of period 3, and a search for its sup.

A field Q on the 3^3 cell, extended periodically, is a field on Z^3 with F-links equal to I.  For a Bloch momentum k
the Hessian acts on X_e(x + 3n) = X_e e^{i k.n}; the top eigenvalue of the Hermitian Bloch matrix lies in the spectrum
of the Hessian on l^2(Z^3), so every value printed here is a lower bound for h_F (both signs are searched).
"""
import sys

import numpy as np

import torus3d as t
fh = t.f

T = t.Torus(3)
fi = {e: i for i, e in enumerate(T.free)}
nF = len(fi)
# unwrapped shifts of the 4 links of every plaquette (which period cell each link lies in)
SHIFT = []
for x in T.sites:
    for a in range(3):
        for b in range(a + 1, 3):
            ys = [x, tuple(x[i] + (i == a) for i in range(3)), tuple(x[i] + (i == b) for i in range(3)), x]
            SHIFT.append([tuple((v - v % 3) // 3 for v in y) for y in ys])
SHIFT = np.array(SHIFT)  # (P, 4, 3)
E = np.array([[e for e, _ in p] for p in T.plaqs])
FREE = np.array([[fi.get(e, -1) for e in row] for row in E])


def blocks(Q):
    """Per-plaquette 12x12 quadratic forms K[p, i, a, j, b] (same algebra as fasthess.hess_matrix)."""
    _, Tm, s, Qp = fh.plaquette_data(T, Q)
    B = np.array(fh.h.BASIS)
    y = s[None, :, None, None, None] * np.einsum("pikl,alm,pinm->piakn", Tm, B, np.conj(Tm))
    yy = np.einsum("piakl,pjblm->piajbkm", y, y)
    sq = np.einsum("piajbkm,pmk->piajb", yy, Qp).real
    comm = sq - np.swapaxes(np.swapaxes(sq, 1, 3), 2, 4)
    ilj = (np.arange(4)[:, None] < np.arange(4)[None, :]).astype(float)
    K = sq + ilj[None, :, None, :, None] * comm
    return 0.5 * (K + np.swapaxes(np.swapaxes(K, 1, 3), 2, 4))


_P, _I, _J = np.nonzero((FREE[:, :, None] >= 0) & (FREE[:, None, :] >= 0))
_R, _C = FREE[_P, _I], FREE[_P, _J]
_ROWS = (3 * _R[:, None, None] + np.arange(3)[None, :, None]).repeat(3, axis=2)
_COLS = (3 * _C[:, None, None] + np.arange(3)[None, None, :]).repeat(3, axis=1)


def bloch(K, k):
    ph = np.exp(1j * SHIFT @ k)  # (P, 4)
    vals = (np.conj(ph[_P, _I]) * ph[_P, _J])[:, None, None] * K[_P, _I, :, _J, :]
    H = np.zeros((3 * nF, 3 * nF), complex)
    np.add.at(H, (_ROWS.ravel(), _COLS.ravel()), vals.ravel())
    return H


KS = [np.array(v) for v in np.ndindex(2, 2, 2)]


def score(Q, sign, ks=None):
    K = blocks(Q)
    best = -np.inf
    for k in (ks if ks is not None else KS):
        ev = np.linalg.eigvalsh(bloch(K, np.pi * np.asarray(k, float)))
        best = max(best, ev[-1] if sign > 0 else -ev[0])
    return best


if __name__ == "__main__":
    seed, steps, mode = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    rng = np.random.default_rng(seed)
    I = np.array([np.eye(2, dtype=complex)] * T.nE)
    print("identity: +", score(I, 1), " -", score(I, -1), flush=True)
    for sign in (1, -1):
        for start in range(int(sys.argv[4]) if len(sys.argv) > 4 else 2):
            if mode == "abelian":
                th = rng.uniform(0, 2 * np.pi, T.nE)
                th[list(T.fixed)] = 0
                Q = t.abelian(th)
            elif mode == "nearI":
                Q = np.array([fh.h.expm_su2(sum(c * b for c, b in zip(rng.normal(size=3) * 0.3, fh.h.BASIS)))
                              for _ in range(T.nE)])
                Q[list(T.fixed)] = np.eye(2)
            else:
                Q = fh.h.random_su2(rng, T.nE)
                Q[list(T.fixed)] = np.eye(2)
            best = score(Q, sign)
            sc = 0.8
            for it in range(steps):
                e = T.free[rng.integers(nF)]
                old = Q[e].copy()
                if mode == "abelian":
                    z = np.exp(1j * rng.normal() * sc)
                    Q[e] = np.diag([z, np.conj(z)]) @ Q[e]
                else:
                    Q[e] = fh.h.expm_su2(sum(c * b for c, b in zip(rng.normal(size=3) * sc, fh.h.BASIS))) @ Q[e]
                v = score(Q, sign)
                if v >= best:
                    best = v
                else:
                    Q[e] = old
                if it % 250 == 249:
                    sc *= 0.75
            print(f"sign {'+' if sign > 0 else '-'} start {start}: {best:.6f}", flush=True)
