#!/usr/bin/env python3
"""Search for a local certificate of the d = 3 forest-gauge Hessian bound (corner-form method).

For a plaquette with transported vectors Y_1..Y_4 in u(N) and holonomy W, f_p is the second derivative of Re Tr Q_p
and c_p the corner form; on Z^d,  sum_p c_p >= -4|X|^2  (divergence identity).  If for every plaquette type
    sum_{free positions r} a_r |Y_r|^2 - (+-f_p) - t c_p >= 0         (local lemma)
then  +-Hess <= (max over free links of the weights it collects + 4t) |X|^2.
By conjugation W = diag(w_a); the form splits into blocks depending on one w_a or on a pair (w_a, w_b), so U(2) with
W = diag(e^{i phi_1}, e^{i phi_2}) covers every U(N).  Here the LMIs are imposed on a grid of (phi_1, phi_2); the
resulting weights are then checked on a much finer grid (and later proved exactly).
"""
import itertools
import sys

import cvxpy as cp
import numpy as np

s = [np.array([[0, 1], [1, 0]], complex), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]], complex)]
BAS = [1j * np.eye(2) / np.sqrt(2)] + [1j * m / np.sqrt(2) for m in s]  # orthonormal basis of u(2), <X,Y> = -Tr XY


def ip(X, Y):
    return -np.trace(X @ Y).real


def fp(Y, W):
    Y1, Y2, Y3, Y4 = Y
    rt = lambda M: np.trace(M).real
    v = sum(rt(y @ y @ W) for y in Y) + 2 * rt(Y1 @ Y2 @ W) + 2 * rt(Y3 @ Y4 @ W)
    v -= 2 * sum(rt(Yi @ W @ Yj) for Yi in (Y1, Y2) for Yj in (Y3, Y4))
    return v


def cpf(Y, W):
    Y1, Y2, Y3, Y4 = Y
    return 2 * ip(Y1, Y4) - 2 * ip(Y1, Y2) + 2 * ip(Y2, W @ Y3 @ W.conj().T) - 2 * ip(Y3, Y4)


def qmat(form, W):
    """16x16 symmetric matrix of a quadratic form in (Y_1..Y_4), coordinates 4*i + basis index."""
    n = 16
    vec = lambda v: [sum(v[4 * i + a] * BAS[a] for a in range(4)) for i in range(4)]
    E = np.eye(n)
    d = np.array([form(vec(E[i]), W) for i in range(n)])
    M = np.zeros((n, n))
    for i in range(n):
        M[i, i] = d[i]
        for j in range(i + 1, n):
            M[i, j] = M[j, i] = (form(vec(E[i] + E[j]), W) - d[i] - d[j]) / 2
    return M


def grid(m):
    ph = 2 * np.pi * np.arange(m) / m
    return [np.diag([np.exp(1j * a), np.exp(1j * b)]) for a in ph for b in ph]


PLACEMENTS = {  # type -> list of (fixed positions, {free position: role})
    "1": [((k,), {(k + 1) % 4: "A", (k + 3) % 4: "A", (k + 2) % 4: "O"}) for k in range(4)],
    "2a": [((k, (k + 1) % 4), {(k + 2) % 4: "B", (k + 3) % 4: "B"}) for k in range(4)],
    "2o": [((k, (k + 2) % 4), {(k + 1) % 4: "C", (k + 3) % 4: "C"}) for k in range(2)],
}
CLASSES = {"1A,1A,1O,2o": ("A", "A", "O", "C"), "1A,1A,1O,2a": ("A", "A", "O", "B"),
           "1A,1O,1O,2a": ("A", "O", "O", "B"), "1A,1A,1A,2a": ("A", "A", "A", "B")}


def solve(m=12, t_fixed=None):
    W = grid(m)
    F = [qmat(fp, w) for w in W]
    C = [qmat(cpf, w) for w in W]
    wt = {r: cp.Variable() for r in "AOBC"}
    t = cp.Variable() if t_fixed is None else t_fixed
    h = cp.Variable()
    cons = []
    for typ, pls in PLACEMENTS.items():
        for fixed, roles in pls:
            idx = [4 * i + a for i in sorted(roles) for a in range(4)]
            D = {i: np.zeros((16, 16)) for i in roles}
            for i in roles:
                D[i][4 * i:4 * i + 4, 4 * i:4 * i + 4] = np.eye(4)
            for Fk, Ck in zip(F, C):
                M = sum(wt[r] * D[i] for i, r in roles.items()) - Fk - t * Ck
                Ms = M[np.ix_(idx, idx)]
                cons.append(0.5 * (Ms + Ms.T) >> 0)
    for cl, roles in CLASSES.items():
        cons.append(sum(wt[r] for r in roles) + 4 * t <= h)
    pr = cp.Problem(cp.Minimize(h), cons)
    pr.solve(solver="CLARABEL")
    tv = t.value if t_fixed is None else t_fixed
    return pr.value, float(tv), {r: float(v.value) for r, v in wt.items()}


def check(weights, t, m=60):
    """Minimum eigenvalue of every local form on a fine grid (should be >= 0)."""
    worst = np.inf
    for w in grid(m):
        Fk, Ck = qmat(fp, w), qmat(cpf, w)
        for typ, pls in PLACEMENTS.items():
            for fixed, roles in pls:
                idx = [4 * i + a for i in sorted(roles) for a in range(4)]
                M = -Fk - t * Ck
                for i, r in roles.items():
                    M[4 * i:4 * i + 4, 4 * i:4 * i + 4] += weights[r] * np.eye(4)
                worst = min(worst, np.linalg.eigvalsh(M[np.ix_(idx, idx)])[0])
    return worst


if __name__ == "__main__":
    m = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    h, t, w = solve(m)
    print(f"grid {m}: h = {h:.6f}, t = {t:.6f}, weights = { {k: round(v, 6) for k, v in w.items()} }")
    totals = {cl: sum(w[r] for r in roles) + 4 * t for cl, roles in CLASSES.items()}
    print("class totals:", {k: round(v, 5) for k, v in totals.items()})
    print("fine-grid min eigenvalue:", check(w, t, 48))
