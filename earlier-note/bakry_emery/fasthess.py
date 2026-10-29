#!/usr/bin/env python3
"""Vectorised exact Hessian of sum_p Re Tr Q_p for SU(2) (quaternion form), plus a joint (Q, v) ascent.

Per plaquette, with partial products P_0 = I, P_1 = M_1, P_2 = M_1 M_2, P_3 = M_1 M_2 M_3, Q_p = P_4 and the four
perturbations moved to the left as Y_1 = X_1, Y_2 = Ad(P_1) X_2, Y_3 = -Ad(P_3) X_3, Y_4 = -Ad(P_4) X_4, one has
  d^2/dt^2 Re Tr(e^{tY_1} e^{tY_2} e^{tY_3} e^{tY_4} Q_p) = Re Tr((sum_i Y_i)^2 Q_p) + Re Tr(sum_{i<j} [Y_i, Y_j] Q_p).
This file evaluates that formula (and checks it against hessian.py, which uses no such rewriting).
"""
import numpy as np

import hessian as h


def adjoint(U):
    """3x3 real matrix of Ad(U) on su(2) in the basis X_a = i sigma_a / sqrt(2); U has shape (..., 2, 2)."""
    B = np.array(h.BASIS)  # (3, 2, 2)
    UB = np.einsum("...ij,ajk,...lk->...ail", U, B, U.conj())  # U X_a U^*
    # coefficient on X_b: <X_b, Y> = Re Tr(X_b^* Y) (orthonormal basis)
    return np.einsum("bji,...ajk->...ba", B.conj(), UB).real.transpose(*range(U.ndim - 2), -1, -2).swapaxes(-1, -2)


def plaquette_data(lat, Q):
    E = np.array([[e for e, _ in p] for p in lat.plaqs])  # (P, 4)
    F = np.array([[f for _, f in p] for p in lat.plaqs])
    M = Q[E]  # (P, 4, 2, 2)
    M = np.where(F[..., None, None], M, np.conj(np.swapaxes(M, -1, -2)))
    P1 = M[:, 0]
    P2 = P1 @ M[:, 1]
    P3 = P2 @ M[:, 2]
    P4 = P3 @ M[:, 3]
    I = np.broadcast_to(np.eye(2, dtype=complex), P1.shape)
    # Y_i = s_i Ad(T_i) X_i with T = (I, P1, P3, P4), s = (+, +, -, -)
    T = np.stack([I, P1, P3, P4], axis=1)
    s = np.array([1, 1, -1, -1])
    return E, T, s, P4


def hess_quadform(lat, Q, X):
    """sum_p d^2/dt^2 Re Tr Q_p(t) along X (shape (nE, 3) coefficients), using the Y-rewriting."""
    E, T, s, Qp = plaquette_data(lat, Q)
    B = np.array(h.BASIS)
    Xm = np.einsum("ea,aij->eij", X, B)[E]  # (P, 4, 2, 2)
    Y = s[None, :, None, None] * (T @ Xm @ np.conj(np.swapaxes(T, -1, -2)))
    Z = Y.sum(axis=1)
    W = np.zeros_like(Z)
    for i in range(4):
        for j in range(i + 1, 4):
            W += Y[:, i] @ Y[:, j] - Y[:, j] @ Y[:, i]
    return float((np.trace(Z @ Z @ Qp, axis1=-2, axis2=-1).real + np.trace(W @ Qp, axis1=-2, axis2=-1).real).sum())


def rayleigh(lat, Q, X):
    return hess_quadform(lat, Q, X) / float((X ** 2).sum())


def hess_matrix(lat, Q):
    """Dense Hessian matrix (N beta = 1), vectorised over plaquettes; basis index 3*e + a."""
    E, T, s, Qp = plaquette_data(lat, Q)
    B = np.array(h.BASIS)
    # y[p, i, a] = s_i T_i B_a T_i^*
    y = s[None, :, None, None, None] * np.einsum("pikl,alm,pinm->piakn", T, B, np.conj(T))
    yy = np.einsum("piakl,pjblm->piajbkm", y, y)  # y_ia y_jb
    sq = np.einsum("piajbkm,pmk->piajb", yy, Qp).real  # Re Tr(y_ia y_jb Q)
    comm = sq - np.swapaxes(np.swapaxes(sq, 1, 3), 2, 4)  # Re Tr((y_ia y_jb - y_jb y_ia) Q)
    i_lt_j = (np.arange(4)[:, None] < np.arange(4)[None, :]).astype(float)
    K = sq + i_lt_j[None, :, None, :, None] * comm  # quadratic form X^T K X (not yet symmetric)
    K = 0.5 * (K + np.swapaxes(np.swapaxes(K, 1, 3), 2, 4))
    n = 3 * lat.nE
    H = np.zeros((n, n))
    idx = (3 * E[:, :, None] + np.arange(3)[None, None, :])  # (P, 4, 3)
    rows = np.broadcast_to(idx[:, :, :, None, None], K.shape)
    cols = np.broadcast_to(idx[:, None, None, :, :], K.shape)
    np.add.at(H, (rows.ravel(), cols.ravel()), K.ravel())
    return H


def lam_max(lat, Q):
    return float(np.linalg.eigvalsh(hess_matrix(lat, Q))[-1])


def split_matrices(lat, Q):
    """(T2, T1): diagonal part (both derivatives on the same edge, i.e. Re Tr with X_e^2 inserted) and cross part."""
    H = hess_matrix(lat, Q)
    T2 = np.zeros_like(H)
    E, T, s, Qp = plaquette_data(lat, Q)
    B = np.array(h.BASIS)
    y = s[None, :, None, None, None] * np.einsum("pikl,alm,pinm->piakn", T, B, np.conj(T))
    yy = np.einsum("piakl,piblm->piabkm", y, y)
    yy = 0.5 * (yy + np.swapaxes(yy, 2, 3))
    loc = np.einsum("piabkm,pmk->piab", yy, Qp).real  # Re Tr(Y_i^2 Q) block, i = j only
    for p in range(len(E)):
        for i in range(4):
            e = E[p, i]
            T2[3 * e:3 * e + 3, 3 * e:3 * e + 3] += loc[p, i]
    return T2, H - T2
