#!/usr/bin/env python3
"""Rigorous version of the site-plus-plaquette certificate (see certificate3d_site.py) for the d = 3 forest gauge.

Step 1 (search, floating point): on finitely many W = diag(e^{i phi_1}, e^{i phi_2}) per plaquette (cutting planes)
find site matrices G_y and plaquette weights a_{p,r} >= 0 with per-link totals <= h0 that maximise a uniform margin eps in every plaquette LMI.
Step 2 (rounding): round G_y and a to rationals with denominator DEN.  Check exactly (fractions) that each G_y is
positive semidefinite (LDL^T pivots), that G_y[o_mu, i_mu] = 0, that a >= 0, and compute the exact per-link totals;
h is their maximum.
Step 3 (all phases): each plaquette matrix M_p(W) = D_p - F(W) - corners(W) is checked on squares of phases, refined
adaptively.  For W, W' with |phi_a - phi'_a| <= r one has ||W - W'||_op <= r and
    ||M_p(W) - M_p(W')|| <= (4 + |k23|) ||W - W'||_op,
because f_p is linear in W with |f_p(W) - f_p(W')| <= (sum_i |Y_i|)^2 ||W - W'|| <= 4 |Y|^2 ||W - W'|| and the far
corner term moves by at most |k23| * 2 |Y_2| |Y_3| ||W - W'||.  So lambda_min(M_p(centre)) > (4 + |k23|) r for
every square of a cover proves M_p(W) > 0 for all W.  The U(2) blocks cover every U(N) (block reduction), and the cover
contains W and -W, so both signs of f_p are covered.  Floating-point eigenvalue errors are below 1e-12 and are
included in the reported slack.

Usage: python3 certify3d.py h0 m_check   (the search uses cutting planes from a 4 x 4 grid)
"""
import sys
from fractions import Fraction

import cvxpy as cp
import numpy as np

import certificate3d as c
import torus3d as t3

DEN = 10 ** 6
T = t3.Torus(3)


def cm(i, j):
    M = np.zeros((16, 16))
    M[4 * i:4 * i + 4, 4 * j:4 * j + 4] = 0.5 * np.eye(4)
    return M + M.T


C14, C12, C34 = cm(0, 3), cm(0, 1), cm(2, 3)


def c23(w):
    return c.qmat(lambda Y, W: c.ip(Y[1], W @ Y[2] @ W.conj().T), w)


# f_p is real-linear in W and the far corner depends on W = diag(e^{i phi_1}, e^{i phi_2}) only through
# e^{i(phi_1 - phi_2)}, so both are trigonometric polynomials of degree one in the phases.
_E = lambda a, b: np.diag([a, b]).astype(complex)
F_BASIS = [c.qmat(c.fp, _E(1, 0)), c.qmat(c.fp, _E(1j, 0)), c.qmat(c.fp, _E(0, 1)), c.qmat(c.fp, _E(0, 1j))]
_C0, _Cpi, _Chalf = c23(_E(1, 1)), c23(_E(1, -1)), c23(_E(1j, 1))
C_BASIS = [(_C0 + _Cpi) / 2, (_C0 - _Cpi) / 2, _Chalf - (_C0 + _Cpi) / 2]


def F_of(p1, p2):
    return (np.cos(p1)[:, None, None] * F_BASIS[0] + np.sin(p1)[:, None, None] * F_BASIS[1]
            + np.cos(p2)[:, None, None] * F_BASIS[2] + np.sin(p2)[:, None, None] * F_BASIS[3])


def C_of(p1, p2):
    d = p1 - p2
    return C_BASIS[0][None] + np.cos(d)[:, None, None] * C_BASIS[1] + np.sin(d)[:, None, None] * C_BASIS[2]


def plaquettes():
    out = []
    for x in T.sites:
        for mu in range(3):
            for nu in range(mu + 1, 3):
                p = (T.eidx[(x, mu)], T.eidx[(T.sh(x, mu), nu)], T.eidx[(T.sh(x, nu), mu)], T.eidx[(x, nu)])
                corners = ((x, mu, nu), (T.sh(x, mu), 3 + mu, nu), (T.sh(T.sh(x, mu), nu), 3 + nu, 3 + mu),
                           (T.sh(x, nu), mu, 3 + nu))  # site and (row, col) of G for k14, k12, k23, k34
                free_pos = [i for i in range(4) if p[i] not in T.fixed]
                out.append((p, corners, free_pos))
    return out


PL = plaquettes()


def plaq_matrix(F, C23, k, a, free_pos):
    k14, k12, k23, k34 = k
    idx = [4 * i + q for i in free_pos for q in range(4)]
    M = -F - (k14 * C14 + k12 * C12 + k23 * C23 + k34 * C34)
    M = sum(a[i] * np.kron(np.diag([1.0 if j == i else 0.0 for j in range(4)]), np.eye(4)) for i in free_pos) + M
    M = 0.5 * (M + M.T)
    return M[np.ix_(idx, idx)]


def search(h0, pts, gmargin=1e-3):
    """Maximise the uniform margin eps; pts[j] is the array of phase pairs imposed on plaquette j."""
    G = {y: cp.Variable((6, 6), symmetric=True) for y in T.sites}
    eps = cp.Variable()
    cons = []
    for y in T.sites:
        cons.append(G[y] >> gmargin * np.eye(6))
        for mu in range(3):
            cons.append(G[y][mu, 3 + mu] == 0)
    coll = {e: 0 for e in T.free}
    for (x, mu) in T.edges:
        e = T.eidx[(x, mu)]
        if e not in T.fixed:
            coll[e] = coll[e] + G[x][mu, mu] + G[T.sh(x, mu)][3 + mu, 3 + mu]
    A = []
    for (p, corners, free_pos), P in zip(PL, pts):
        k = [2 * G[s][r, q] for s, r, q in corners]
        a = {i: cp.Variable(nonneg=True) for i in free_pos}
        A.append(a)
        n = 4 * len(free_pos)
        for Fk, Ck in zip(F_of(P[:, 0], P[:, 1]), C_of(P[:, 0], P[:, 1])):
            cons.append(plaq_matrix(Fk, Ck, k, a, free_pos) >> eps * np.eye(n))
        for i in free_pos:
            coll[p[i]] = coll[p[i]] + a[i]
    for e in T.free:
        cons.append(coll[e] <= h0)
    cp.Problem(cp.Maximize(eps), cons).solve(solver="CLARABEL")
    return {y: G[y].value for y in T.sites}, [{i: v.value for i, v in a.items()} for a in A], eps.value


def numeric_matrices(G, A, j, P):
    """Plaquette j's matrices at phase pairs P (n x 2), for numeric (float or Fraction) G, A."""
    p, corners, free_pos = PL[j]
    k = [2 * float(G[s][r][q]) for s, r, q in corners]
    idx = [4 * i + q for i in free_pos for q in range(4)]
    D = np.zeros((16, 16))
    for i in free_pos:
        D[4 * i:4 * i + 4, 4 * i:4 * i + 4] = float(A[j][i]) * np.eye(4)
    M = D[None] - F_of(P[:, 0], P[:, 1]) - (k[0] * C14 + k[1] * C12 + k[3] * C34)[None] - k[2] * C_of(P[:, 0], P[:, 1])
    M = 0.5 * (M + np.swapaxes(M, 1, 2))
    return M[:, idx][:, :, idx], abs(k[2])


def grid_pts(m):
    ph = 2 * np.pi * np.arange(m) / m
    return np.array([(x, y) for x in ph for y in ph])


def cutting_plane(h0, m0=4, mfine=96, rounds=8, add=6):
    pts = [grid_pts(m0) for _ in PL]
    fine = grid_pts(mfine)
    for r in range(rounds):
        Gf, Af, eps = search(h0, pts)
        worst = np.inf
        for j in range(len(PL)):
            M, _ = numeric_matrices(Gf, Af, j, fine)
            lam = np.linalg.eigvalsh(M)[:, 0]
            worst = min(worst, lam.min())
            new = fine[np.argsort(lam)[:add]]
            pts[j] = np.vstack([pts[j], new])
        print(f"  round {r}: eps on cuts = {eps:.5f}, min on grid {mfine} = {worst:.5f}", flush=True)
        if worst > 0.8 * eps:
            break
    return Gf, Af, eps


def rat(v):
    return Fraction(round(float(v) * DEN), DEN)


def ldl_psd(M):
    """Exact test that a symmetric rational matrix is positive definite (all LDL^T pivots > 0)."""
    M = [row[:] for row in M]
    n = len(M)
    for j in range(n):
        if M[j][j] <= 0:
            return False
        for i in range(j + 1, n):
            f = M[i][j] / M[j][j]
            for k in range(j, n):
                M[i][k] -= f * M[j][k]
    return True


def round_and_check(Gf, Af):
    G = {}
    for y in T.sites:
        g = [[rat(0.5 * (Gf[y][i, j] + Gf[y][j, i])) for j in range(6)] for i in range(6)]
        for mu in range(3):
            g[mu][3 + mu] = g[3 + mu][mu] = Fraction(0)
        assert ldl_psd(g), f"G at {y} not PD after rounding"
        G[y] = g
    A = [{i: max(Fraction(0), rat(v)) for i, v in a.items()} for a in Af]
    coll = {e: Fraction(0) for e in T.free}
    for (x, mu) in T.edges:
        e = T.eidx[(x, mu)]
        if e not in T.fixed:
            coll[e] += G[x][mu][mu] + G[T.sh(x, mu)][3 + mu][3 + mu]
    for (p, corners, free_pos), a in zip(PL, A):
        for i in free_pos:
            coll[p[i]] += a[i]
    return G, A, max(coll.values())


def check_all_phases(G, A, m, max_depth=12):
    """Adaptive proof that every plaquette matrix is positive definite for all phases.

    A square of half-width r around a centre is accepted when lambda_min(M(centre)) > (4 + |k23|) r + 1e-12
    (Lipschitz bound in the docstring with ||W - W'||_op <= r); otherwise it is split into four.  Returns the least
    accepted slack and the number of squares evaluated, or raises if some square cannot be accepted.
    """
    worst, count = np.inf, 0
    r0 = np.pi / m
    for j in range(len(PL)):
        cen, r = grid_pts(m) + r0, r0
        for depth in range(max_depth):
            M, k23 = numeric_matrices(G, A, j, cen)
            lam = np.linalg.eigvalsh(M)[:, 0]
            count += len(cen)
            slack = lam - (4 + k23) * r - 1e-12
            if np.any(lam <= 0):
                raise AssertionError(f"plaquette {j}: negative eigenvalue {lam.min()} at {cen[np.argmin(lam)]}")
            ok = slack > 0
            if ok.any():
                worst = min(worst, slack[ok].min())
            cen = cen[~ok]
            if len(cen) == 0:
                break
            r /= 2
            cen = np.concatenate([cen + np.array([sx, sy]) * r for sx in (-1, 1) for sy in (-1, 1)])
        else:
            raise AssertionError(f"plaquette {j}: not resolved at depth {max_depth}")
    return worst, count


if __name__ == "__main__":
    h0, mc = float(sys.argv[1]), int(sys.argv[2])
    Gf, Af, eps = cutting_plane(h0)
    print(f"search: h0 = {h0}, margin eps = {eps:.6f}", flush=True)
    G, A, h = round_and_check(Gf, Af)
    print(f"rounded: exact per-link maximum h = {h} = {float(h):.6f}", flush=True)
    s, n = check_all_phases(G, A, mc)
    print(f"all phases: every plaquette matrix positive definite ({n} squares, least slack {s:.3e}) -> PROVED",
          flush=True)
    np.save(f"cert_h{h0}.npy", {"G": {y: [[str(v) for v in r] for r in G[y]] for y in T.sites},
                                "A": [{i: str(v) for i, v in a.items()} for a in A], "h": str(h)}, allow_pickle=True)
