#!/usr/bin/env python3
"""Independent numerical checks of the conventions behind certify3d.py (not part of the proof; adversarial tests).

1. Plaquette rewriting: for Q_e -> e^{tX_e} Q_e, Re Tr Q_p(t) = Re Tr(e^{tY1} e^{tY2} W e^{-tY3} e^{-tY4}) with
   Y1 = X1, Y2 = Ad(Q1) X2, Y3 = Ad(Q4) X3, Y4 = X4, W = Q_p.  We check sum_p f_p(Y, W) against the Hessian of
   fasthess.py (an independent formula) on random SU(2) fields of the 3-periodic torus.
2. Site identity: with ends o_mu(y) = X_(y,mu), i_mu(y) = Ad(Q_(y-mu,mu)^*) X_(y-mu,mu), the corner terms of all
   plaquettes plus the diagonal site terms equal sum_y <ends, G_y ends> >= 0.
3. Spectrum: every eigenvalue of the gauge-fixed Hessian at random, abelian and identity fields lies in [-h, h].
4. Block reduction: the plaquette inequality f_p + corners <= sum a |Y_r|^2 holds at random U(3) holonomies and
   random u(3) vectors (the proof checks U(2) only).
Usage: python3 validate_cert.py cert_h10.npy
"""
import sys
from fractions import Fraction

import numpy as np

import certificate3d as c
import certify3d as cz
import torus3d as t3

rng = np.random.default_rng(0)
T = cz.T
cert = np.load(sys.argv[1] if len(sys.argv) > 1 else "cert_h10.npy", allow_pickle=True).item()
G = {y: np.array([[float(Fraction(v)) for v in r] for r in cert["G"][y]]) for y in T.sites}
A = [{i: float(Fraction(v)) for i, v in a.items()} for a in cert["A"]]
h = float(Fraction(cert["h"]))
B3 = np.array(t3.h.BASIS)  # orthonormal basis of su(2)


def rand_su2(n):
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    a, b = q[:, 0] + 1j * q[:, 1], q[:, 2] + 1j * q[:, 3]
    return np.stack([np.stack([a, -b.conj()], -1), np.stack([b, a.conj()], -1)], -2)


Ad = lambda U, X: U @ X @ U.conj().T
ip = c.ip


def check_fields(Q, X):
    Xm = np.einsum("ea,aij->eij", X, B3)
    fsum, corners = 0.0, 0.0
    for j, (p, cor, free_pos) in enumerate(cz.PL):
        e1, e2, e3, e4 = p
        Wp = Q[e1] @ Q[e2] @ Q[e3].conj().T @ Q[e4].conj().T
        Y = [Xm[e1], Ad(Q[e1], Xm[e2]), Ad(Q[e4], Xm[e3]), Xm[e4]]
        fsum += c.fp(Y, Wp)
        k = [2 * G[s][r, q] for s, r, q in cor]
        corners += (k[0] * ip(Y[0], Y[3]) + k[1] * ip(Y[0], Y[1]) + k[2] * ip(Y[1], Wp @ Y[2] @ Wp.conj().T)
                    + k[3] * ip(Y[2], Y[3]))
    H = t3.f.hess_quadform(T, Q, X)
    site, diag = 0.0, 0.0
    for y in T.sites:
        ends = [Xm[T.eidx[(y, m)]] for m in range(3)]
        for m in range(3):
            e = T.eidx[(T.sh(y, m, -1), m)]
            ends.append(Ad(Q[e].conj().T, Xm[e]))
        Gm = np.array([[ip(a, b) for b in ends] for a in ends])
        site += float((G[y] * Gm).sum())
        diag += float(sum(G[y][a, a] * Gm[a, a] for a in range(6)))
    return H, fsum, corners + diag, site


worst = 0.0
for trial in range(5):
    Q = rand_su2(T.nE)
    X = rng.normal(size=(T.nE, 3))
    X[list(T.fixed)] = 0
    H, fsum, cd, site = check_fields(Q, X)
    worst = max(worst, abs(H - fsum), abs(cd - site))
print(f"1-2. rewriting and site identity: max discrepancy {worst:.2e}")

ev = []
for trial in range(20):
    for Q in (rand_su2(T.nE), t3.abelian(rng.uniform(0, 2 * np.pi, T.nE))):
        ev.append(np.linalg.eigvalsh(t3.free_hessian(T, Q))[[0, -1]])
ev.append(np.linalg.eigvalsh(t3.free_hessian(T, np.array([np.eye(2, dtype=complex)] * T.nE)))[[0, -1]])
ev = np.array(ev)
print(f"3. eigenvalues of the gauge-fixed Hessian on 41 fields in [{ev.min():.4f}, {ev.max():.4f}], h = {h:.6f}")


def rand_u(n):
    Z = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    q, r = np.linalg.qr(Z)
    return q * (np.diag(r) / abs(np.diag(r)))


def rand_un_alg(n):
    Z = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    return (Z - Z.conj().T) / 2


low = np.inf
for trial in range(4000):
    j = rng.integers(len(cz.PL))
    p, cor, free_pos = cz.PL[j]
    W = rand_u(3)
    Y = [rand_un_alg(3) if i in free_pos else np.zeros((3, 3)) for i in range(4)]
    k = [2 * G[s][r, q] for s, r, q in cor]
    val = (sum(A[j][i] * ip(Y[i], Y[i]) for i in free_pos) - c.fp(Y, W)
           - (k[0] * ip(Y[0], Y[3]) + k[1] * ip(Y[0], Y[1]) + k[2] * ip(Y[1], W @ Y[2] @ W.conj().T)
              + k[3] * ip(Y[2], Y[3])))
    low = min(low, val / sum(ip(y, y) for y in Y))
print(f"4. U(3) plaquette inequality at 4000 random points: least normalised slack {low:.4f} (must be > 0)")
