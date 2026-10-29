#!/usr/bin/env python3
"""General site-plus-plaquette certificate for the d = 3 forest-gauge Hessian bound.

At a site y the link ends are o_mu (outgoing link (y, mu)) and i_mu (incoming link (y - e_mu, mu), transported to y).
For any symmetric G_y >= 0 on the six ends with G_y[o_mu, i_mu] = 0,
    0 <= sum_y sum_{a,b} G_y[a,b] <end_a, end_b>,
and every perpendicular pair of ends at y is a corner of exactly one plaquette, where it equals
    corner x:            <Y1,Y4> = <o_mu, o_nu>        corner x + e_mu:       <Y1,Y2> = <i_mu, o_nu>
    corner x+e_mu+e_nu:  <Y2,W Y3 W^*> = <i_nu, i_mu>  corner x + e_nu:       <Y3,Y4> = <o_mu, i_nu>.
So if every plaquette satisfies  +-f_p + sum_corners 2 G[corner] * corner  <=  sum_free a_{p,r} |Y_r|^2,
then +-Hess <= max_e (sum_p a_{p,e} + G_tail(e)[o,o] + G_head(e)[i,i]) |X|^2.  (Prior work: G = t(|D|^2 + sum|o+i|^2).)
"""
import sys

import cvxpy as cp
import numpy as np

import certificate3d as c
import torus3d as t3

m = int(sys.argv[1]) if len(sys.argv) > 1 else 6
L = 3
T = t3.Torus(L)
W = c.grid(m)
F = [c.qmat(c.fp, w) for w in W]
# corner forms as 16x16 matrices (W-dependent only for the far corner)
def cm(i, j):
    M = np.zeros((16, 16))
    M[4 * i:4 * i + 4, 4 * j:4 * j + 4] = 0.5 * np.eye(4)
    return M + M.T  # <Y_i, Y_j>
C14, C12, C34 = cm(0, 3), cm(0, 1), cm(2, 3)
C23 = [c.qmat(lambda Y, w: c.ip(Y[1], w @ Y[2] @ w.conj().T), w) for w in W]

G = {y: cp.Variable((6, 6), symmetric=True) for y in T.sites}  # ends: 0..2 = o_mu, 3..5 = i_mu
cons = []
for y in T.sites:
    cons.append(G[y] >> 0)
    for mu in range(3):
        cons.append(G[y][mu, 3 + mu] == 0)
coll = {e: 0 for e in T.free}
for (x, mu) in T.edges:
    e = T.eidx[(x, mu)]
    if e in T.fixed:
        continue
    coll[e] = coll[e] + G[x][mu, mu] + G[T.sh(x, mu)][3 + mu, 3 + mu]
h = cp.Variable()
for x in T.sites:
    for mu in range(3):
        for nu in range(mu + 1, 3):
            p = (T.eidx[(x, mu)], T.eidx[(T.sh(x, mu), nu)], T.eidx[(T.sh(x, nu), mu)], T.eidx[(x, nu)])
            xm, xn, xmn = T.sh(x, mu), T.sh(x, nu), T.sh(T.sh(x, mu), nu)
            k14 = 2 * G[x][mu, nu]                 # <o_mu, o_nu> at x
            k12 = 2 * G[xm][3 + mu, nu]            # <i_mu, o_nu> at x + e_mu
            k23 = 2 * G[xmn][3 + nu, 3 + mu]       # <i_nu, i_mu> at x + e_mu + e_nu
            k34 = 2 * G[xn][mu, 3 + nu]            # <o_mu, i_nu> at x + e_nu
            free_pos = [i for i in range(4) if p[i] not in T.fixed]
            a = {i: cp.Variable() for i in free_pos}
            idx = [4 * i + k for i in free_pos for k in range(4)]
            D = sum(a[i] * np.kron(np.diag([1.0 if j == i else 0.0 for j in free_pos]), np.eye(4)) for i in free_pos)
            for Fk, Ck in zip(F, C23):
                M = -Fk - (k14 * C14 + k12 * C12 + k23 * Ck + k34 * C34)
                M = M[np.ix_(idx, idx)]
                cons.append(D + 0.5 * (M + M.T) >> 0)
            for i in free_pos:
                coll[p[i]] = coll[p[i]] + a[i]
for e in T.free:
    cons.append(coll[e] <= h)
pr = cp.Problem(cp.Minimize(h), cons)
pr.solve(solver="CLARABEL")
print(f"grid {m}: site+plaquette certificate h = {pr.value:.6f}")
np.save(f"site_cert_m{m}.npy", {"G": {y: G[y].value for y in T.sites}}, allow_pickle=True)
