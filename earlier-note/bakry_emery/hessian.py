#!/usr/bin/env python3
"""Exact Hessian of the Wilson action for SU(2) lattice gauge theory on a periodic L^d torus.

Conventions follow Shen-Zhu-Zhu (arXiv:2204.12737):
  S(Q) = N beta sum_p Re Tr(Q_p),  Q_p = Q_{x,mu} Q_{x+mu,nu} Q_{x+nu,mu}^* Q_{x,nu}^*,
  tangent vectors v_e = X_e Q_e with X_e in su(N), |v|^2 = sum_e Tr(X_e X_e^*),
  curves Q_e(t) = exp(t X_e) Q_e (geodesics for the bi-invariant metric), so Hess S(v, v) = d^2/dt^2 S(Q(t)) at t = 0.
We report the quadratic form with N beta = 1, i.e. H(v, v) = Hess S(v, v)/(N beta), as a dense symmetric matrix in an
orthonormal basis X_a = i sigma_a/sqrt(2) of su(2) at every edge.

SZZ Lemma 4.1: |H(v, v)| <= 8(d-1)|v|^2. The Bakry-Emery condition for beta > 0 only needs an upper bound on the
largest eigenvalue lambda_max(H). This module computes lambda_max(H) exactly for a given configuration.
"""
import itertools

import numpy as np

SIG = [np.array([[0, 1], [1, 0]], complex), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]], complex)]
BASIS = [1j * s / np.sqrt(2) for s in SIG]  # Tr(X X^*) = 1
I2 = np.eye(2, dtype=complex)


class Lattice:
    def __init__(self, L, d):
        self.L, self.d = L, d
        self.sites = list(itertools.product(range(L), repeat=d))
        self.sidx = {x: i for i, x in enumerate(self.sites)}
        self.nE = len(self.sites) * d
        self.plaqs = []  # list of 4 (edge index, forward?) in product order
        for x in self.sites:
            for mu in range(d):
                for nu in range(mu + 1, d):
                    self.plaqs.append(((self.e(x, mu), True), (self.e(self.sh(x, mu), nu), True),
                                       (self.e(self.sh(x, nu), mu), False), (self.e(x, nu), False)))

    def sh(self, x, mu, s=1):
        y = list(x)
        y[mu] = (y[mu] + s) % self.L
        return tuple(y)

    def e(self, x, mu):
        return self.sidx[x] * self.d + mu


def factor(Q, fwd):
    return Q if fwd else Q.conj().T


def d1(X, M, fwd):
    """First derivative of the factor M (= Q or Q^*) along X at t = 0."""
    return X @ M if fwd else -M @ X


def d2(XY, M, fwd):
    """Second derivative (symmetrised product XY = (X_a X_b + X_b X_a)/2)."""
    return XY @ M if fwd else M @ XY


def hessian(lat, Q):
    """Dense matrix H[(e,a),(f,b)] with N beta = 1; Q has shape (nE, 2, 2)."""
    n = lat.nE * 3
    H = np.zeros((n, n))
    sym = [[(BASIS[a] @ BASIS[b] + BASIS[b] @ BASIS[a]) / 2 for b in range(3)] for a in range(3)]
    for plaq in lat.plaqs:
        Ms = [factor(Q[e], f) for e, f in plaq]
        for i, (ei, fi) in enumerate(plaq):
            for j, (ej, fj) in enumerate(plaq):
                for a in range(3):
                    for b in range(3):
                        if i == j:
                            prod = I2
                            for k in range(4):
                                prod = prod @ (d2(sym[a][b], Ms[k], fi) if k == i else Ms[k])
                            val = np.trace(prod).real
                        else:
                            prod = I2
                            for k in range(4):
                                if k == i:
                                    prod = prod @ d1(BASIS[a], Ms[k], fi)
                                elif k == j:
                                    prod = prod @ d1(BASIS[b], Ms[k], fj)
                                else:
                                    prod = prod @ Ms[k]
                            val = np.trace(prod).real
                        H[ei * 3 + a, ej * 3 + b] += val
    # i != j pairs were counted in both orders (i,j) and (j,i): the second derivative of a product f(t) with two
    # perturbed factors contributes 2 * (cross term); summing over ordered pairs gives exactly that.
    return (H + H.T) / 2


def hessian_fd(lat, Q, v, h=1e-4):
    """Finite-difference second derivative of sum_p Re Tr Q_p along v (control for hessian())."""
    def S(t):
        Qt = np.array([expm_su2(t * v[e]) @ Q[e] for e in range(lat.nE)])
        return sum(np.trace(factor(Qt[a[0]], a[1]) @ factor(Qt[b[0]], b[1]) @ factor(Qt[c[0]], c[1]) @ factor(Qt[dd[0]], dd[1])).real
                   for a, b, c, dd in lat.plaqs)
    return (S(h) - 2 * S(0) + S(-h)) / h ** 2


def expm_su2(X):
    """exp of X in su(2) (anti-Hermitian, traceless)."""
    # X = i (a . sigma); |a| = theta; exp = cos(theta) I + i sin(theta) (a/theta) . sigma
    a = np.array([np.trace(X @ s).imag / 2 for s in SIG])
    th = np.linalg.norm(a)
    if th < 1e-15:
        return I2 + X
    return np.cos(th) * I2 + 1j * np.sin(th) / th * sum(a[k] * SIG[k] for k in range(3))


def random_su2(rng, n):
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return np.array([[[q[i, 0] + 1j * q[i, 3], q[i, 2] + 1j * q[i, 1]], [-q[i, 2] + 1j * q[i, 1], q[i, 0] - 1j * q[i, 3]]]
                     for i in range(n)])


def lam_max(lat, Q):
    return float(np.linalg.eigvalsh(hessian(lat, Q))[-1])
