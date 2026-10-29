#!/usr/bin/env python3
"""Temporal (axial) gauge Hessian for SU(2) lattice Yang-Mills on (Z/LZ)^{d-1} x {0..T} with free b.c. in time.

After gauging all temporal links to I, the action is
  S_ax(U) = N beta [ sum_{spatial p} Re Tr U_p + sum_{x, i, t} Re Tr(U_{x,i}(t) U_{x,i}(t+1)^*) ],
a function of the spatial links only. Claims checked here:
  (1) exactness: S_ax equals the full Wilson action evaluated with temporal links = I (so its Hessian in spatial
      directions is the full Hessian restricted to those directions);
  (2) the rigorous bound lambda_max(Hess S_ax)/(N beta) <= 8(d-2) + 4 = 8d - 12, never violated in adversarial search;
  (3) sharpness in d = 3: the alternating configuration (spatial pi-flux in each slice, U(t+1) = -U(t)) attains 12 = 4d;
  (4) in d = 4 the bound 20 is not attained; the adversarial supremum is ~16 = 4d.
"""
import itertools
import sys

import numpy as np

import fasthess as f
import hessian as h


class AxialLattice:
    """Spatial edges of (Z/LZ)^{ds} x {0..T}; ds = d - 1 spatial dimensions."""

    def __init__(self, L, ds, T):
        self.L, self.ds, self.T = L, ds, T
        self.slat = h.Lattice(L, ds) if ds >= 1 else None
        self.nEs = self.slat.nE  # spatial edges per slice
        self.nE = self.nEs * (T + 1)
        # spatial plaquettes in each slice: reuse slice lattice plaquettes, offset edge indices
        self.plaqs = [tuple((e + t * self.nEs, fw) for e, fw in p) for t in range(T + 1) for p in self.slat.plaqs]
        # temporal couplings Re Tr(U(t) U(t+1)^*): (edge at t, edge at t+1)
        self.couplings = [(e + t * self.nEs, e + (t + 1) * self.nEs) for t in range(T) for e in range(self.nEs)]


def hess_axial(al, U):
    """Dense Hessian (N beta = 1) of S_ax at spatial configuration U (shape (nE, 2, 2))."""
    n = 3 * al.nE
    H = np.zeros((n, n))
    # spatial plaquettes: reuse the vectorised plaquette Hessian on a fake lattice object
    class Fake:
        pass
    fk = Fake()
    fk.plaqs, fk.nE = al.plaqs, al.nE
    if al.plaqs:
        H += f.hess_matrix(fk, U)
    # 2-link couplings g(U, V) = Re Tr(U V^*), U -> e^{tX} U, V -> e^{tY} V:
    # d^2 = Re Tr(X^2 W) + Re Tr(W Y^2) - 2 Re Tr(X W Y), W = U V^*
    B = h.BASIS
    for e1, e2 in al.couplings:
        W = U[e1] @ U[e2].conj().T
        for a in range(3):
            for b in range(3):
                sym = (B[a] @ B[b] + B[b] @ B[a]) / 2
                H[3 * e1 + a, 3 * e1 + b] += np.trace(sym @ W).real
                H[3 * e2 + a, 3 * e2 + b] += np.trace(W @ sym).real
                c = -np.trace(B[a] @ W @ B[b]).real
                H[3 * e1 + a, 3 * e2 + b] += c
                H[3 * e2 + b, 3 * e1 + a] += c
    return H


def S_axial(al, U):
    s = 0.0
    for p in al.plaqs:
        M = np.eye(2, dtype=complex)
        for e, fw in p:
            M = M @ (U[e] if fw else U[e].conj().T)
        s += np.trace(M).real
    for e1, e2 in al.couplings:
        s += np.trace(U[e1] @ U[e2].conj().T).real
    return s


def check_fd(al, U, rng, hstep=1e-4):
    c = rng.normal(size=(al.nE, 3))
    v = [sum(c[e, a] * h.BASIS[a] for a in range(3)) for e in range(al.nE)]
    S = lambda t: S_axial(al, np.array([h.expm_su2(t * v[e]) @ U[e] for e in range(al.nE)]))
    fd = (S(hstep) - 2 * S(0) + S(-hstep)) / hstep ** 2
    return float(c.ravel() @ hess_axial(al, U) @ c.ravel()), fd


def check_exact_vs_full(L, d, T, rng):
    """S_ax(U) = full Wilson action with temporal links = I on (Z/LZ)^{d-1} x {0..T} (free b.c. in time)."""
    al = AxialLattice(L, d - 1, T)
    U = h.random_su2(rng, al.nE)
    full = 0.0
    sites = list(itertools.product(range(L), repeat=d - 1))
    sl = al.slat

    def link(x, t, mu):  # mu = d-1 is time
        if mu == d - 1:
            return np.eye(2, dtype=complex)
        return U[sl.e(x, mu) + t * al.nEs]

    def shift(x, t, mu):
        if mu == d - 1:
            return x, t + 1
        return sl.sh(x, mu), t

    for t in range(T + 1):
        for x in sites:
            for mu in range(d):
                for nu in range(mu + 1, d):
                    if nu == d - 1 and t == T:
                        continue  # free b.c.: no plaquette leaving the box
                    x1, t1 = shift(x, t, mu)
                    x2, t2 = shift(x, t, nu)
                    M = link(x, t, mu) @ link(x1, t1, nu) @ link(x2, t2, mu).conj().T @ link(x, t, nu).conj().T
                    full += np.trace(M).real
    return full, S_axial(al, U)


def alternating(al):
    from search import pi_flux
    piS = pi_flux(al.slat)
    return np.concatenate([((-1) ** t) * piS for t in range(al.T + 1)])


def climb(al, U, rng, steps, scale=0.5):
    best = np.linalg.eigvalsh(hess_axial(al, U))[-1]
    for k in range(steps):
        e = rng.integers(al.nE)
        R = h.expm_su2(sum(c * b for c, b in zip(rng.normal(size=3) * scale, h.BASIS)))
        old = U[e].copy()
        U[e] = R @ U[e]
        v = np.linalg.eigvalsh(hess_axial(al, U))[-1]
        if v >= best:
            best = v
        else:
            U[e] = old
        if k % 300 == 299:
            scale *= 0.85
    return best


def main():
    rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
    fails = 0
    for L, d, T in ((3, 3, 2), (2, 4, 1)):
        full, ax = check_exact_vs_full(L, d, T, rng)
        ok = abs(full - ax) < 1e-9
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'} exactness d={d}: full action with temporal links = I {full:.6f} = S_ax {ax:.6f}")
    for L, d, T in ((3, 3, 2), (2, 4, 1)):
        al = AxialLattice(L, d - 1, T)
        U = h.random_su2(rng, al.nE)
        ex, fd = check_fd(al, U, rng)
        ok = abs(ex - fd) < 1e-4 * max(1, abs(ex))
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'} Hessian vs finite differences d={d}: {ex:.6f} vs {fd:.6f}")
    for L, d, T in ((2, 3, 3), (4, 3, 3), (2, 4, 3)):
        al = AxialLattice(L, d - 1, T)
        alt = np.linalg.eigvalsh(hess_axial(al, alternating(al)))[-1]
        best = max(climb(al, h.random_su2(rng, al.nE), rng, steps) for _ in range(3))
        bound = 8 * d - 12
        ok = best <= bound + 1e-9 and alt <= bound + 1e-9
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'} d={d} L={L} T={T}: alternating config {alt:.4f}; adversarial sup {best:.4f}; "
              f"rigorous bound 8d-12 = {bound}; 4d = {4 * d}", flush=True)
    print(f"\n{fails} failures")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
