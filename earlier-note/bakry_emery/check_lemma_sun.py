#!/usr/bin/env python3
"""Check Lemma 3.1 for SU(N), N = 2, 3, 4: for a loop with k free links (k = 1..4, some inverted) and fixed unitary
insertions, |d^2/dt^2 Re Tr(holonomy)| <= (sum |X_e|)^2, at random points, by finite differences; also check that the
bound is attained for k = 4 at Q_p = diag(-1,-1,1,..) with aligned X (sharpness); and Ric(SU(2)) = 1 = N/2 in the
metric Tr(XY^*)."""
import numpy as np
from scipy.linalg import expm

rng = np.random.default_rng(0)


def rand_su(N):
    Z = (rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))) / np.sqrt(2)
    Qm, R = np.linalg.qr(Z)
    Qm = Qm * (np.diag(R) / abs(np.diag(R)))
    return Qm / np.linalg.det(Qm) ** (1 / N)


def rand_alg(N):
    A = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    A = (A - A.conj().T) / 2
    return A - np.trace(A) / N * np.eye(N)


def hs(X):
    return np.sqrt(np.trace(X @ X.conj().T).real)


worst = 0
for N in (2, 3, 4):
    for k in (1, 2, 3, 4):
        for trial in range(200):
            Us = [rand_su(N) for _ in range(k)]
            W = [rand_su(N) for _ in range(k)]  # fixed insertions (e.g. gauge-fixed links / other matrices)
            inv = rng.integers(0, 2, k)
            X = [rand_alg(N) * rng.uniform(0.1, 2) for _ in range(k)]

            def phi(t):
                M = np.eye(N, dtype=complex)
                for i in range(k):
                    Ui = expm(t * X[i]) @ Us[i]
                    M = M @ (Ui.conj().T if inv[i] else Ui) @ W[i]
                return np.trace(M).real
            h = 1e-4
            d2 = (phi(h) - 2 * phi(0) + phi(-h)) / h ** 2
            bound = sum(hs(x) for x in X) ** 2
            worst = max(worst, abs(d2) / bound)
print(f"Lemma 3.1: max |Hess| / (sum|X|)^2 over 2400 random SU(2..4) loops = {worst:.4f}  (must be <= 1)")
# sharpness k = 4, SU(3): Q_p = diag(-1,-1,1), X aligned in the su(2) block, signs as in a plaquette
N = 3
J = np.diag([-1, -1, 1]).astype(complex)
Z = np.zeros((3, 3), complex)
Z[:2, :2] = 1j * np.array([[1, 0], [0, -1]]) / np.sqrt(2)
Qs = [J, np.eye(3), np.eye(3), np.eye(3)]
def phi(t):
    U1, U2, U3, U4 = (expm(t * Z) @ Qs[0], expm(t * Z) @ Qs[1], expm(-t * Z) @ Qs[2], expm(-t * Z) @ Qs[3])
    return np.trace(U1 @ U2 @ U3.conj().T @ U4.conj().T).real
d2 = (phi(1e-4) - 2 * phi(0) + phi(-1e-4)) / 1e-8
print(f"sharpness: plaquette with Q_p = diag(-1,-1,1), aligned X: Hess = {d2:.5f}, bound 4 * sum|X|^2 = {4 * 4 * hs(Z) ** 2:.5f}, (sum|X|)^2 = {(4 * hs(Z)) ** 2:.5f}")
# Ricci of SU(2) in metric Tr(X Y^*): Ric(X,X) = 1/4 sum_b |[X, X_b]|^2 for bi-invariant metric
B = [1j * s / np.sqrt(2) for s in (np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]]))]
X = B[0]
ric = 0.25 * sum(hs(X @ b - b @ X) ** 2 for b in B)
print(f"Ric_SU(2)(X,X) for unit X = {ric:.5f} (= N/2 = 1)")
