#!/usr/bin/env python3
"""Covariant version of the free identity 4d|X|^2 - ||dX||^2 = ||delta X||^2 + sum_{x,mu,nu} |X_nu(x+mu) + X_nu(x)|^2.
Computes the remainder C(A) = 4d I - D^T D - G G^T - P^T P for SU(2) connections (adjoint transports)."""
import numpy as np

import fasthess as f
import hessian as h
from dA_norm import dA_matrix


def grad_matrix(lat, Q):
    """(nabla_A xi)_{x,mu} = Ad(Q_{x,mu}) xi(x+mu) - xi(x): maps 3V -> 3E."""
    Ad = f.adjoint(Q)
    V = len(lat.sites)
    G = np.zeros((3 * lat.nE, 3 * V))
    for x in lat.sites:
        for mu in range(lat.d):
            e = lat.e(x, mu)
            G[3 * e:3 * e + 3, 3 * lat.sidx[lat.sh(x, mu)]:3 * lat.sidx[lat.sh(x, mu)] + 3] += Ad[e]
            G[3 * e:3 * e + 3, 3 * lat.sidx[x]:3 * lat.sidx[x] + 3] -= np.eye(3)
    return G


def pair_matrix(lat, Q, sign=+1):
    """rows (x, mu, nu): Ad(Q_{x,mu}) X_{x+mu,nu} + sign * X_{x,nu}."""
    Ad = f.adjoint(Q)
    rows = []
    for x in lat.sites:
        for mu in range(lat.d):
            for nu in range(lat.d):
                R = np.zeros((3, 3 * lat.nE))
                e_far = lat.e(lat.sh(x, mu), nu)
                R[:, 3 * e_far:3 * e_far + 3] += Ad[lat.e(x, mu)]
                e0 = lat.e(x, nu)
                R[:, 3 * e0:3 * e0 + 3] += sign * np.eye(3)
                rows.append(R)
    return np.vstack(rows)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for L, d in ((3, 2), (3, 3)):
        lat = h.Lattice(L, d)
        for name, Q in (("identity", np.array([np.eye(2, dtype=complex)] * lat.nE)), ("random", h.random_su2(rng, lat.nE))):
            D, G, P = dA_matrix(lat, Q), grad_matrix(lat, Q), pair_matrix(lat, Q)
            C = 4 * d * np.eye(3 * lat.nE) - D.T @ D - G @ G.T - P.T @ P
            ev = np.linalg.eigvalsh(C)
            print(f"L={L} d={d} {name}: remainder C eigenvalues in [{ev[0]:.4f}, {ev[-1]:.4f}], ||C||_F = {np.linalg.norm(C):.4f}")
