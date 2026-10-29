#!/usr/bin/env python3
"""Interval-arithmetic proof that every plaquette matrix of a certificate is positive definite for all phases.

This replaces the floating-point eigenvalues of certify3d.check_all_phases by rigorous interval computations.

1. Exact data.  The coefficient matrices of f_p and of the corner forms in an orthonormal basis of u(2) are computed
   exactly in sympy (they are rational).  G and a are the rationals of the certificate.  So
       M_p(phi) = D_p - sum_t trig_t(phi) F_t - (k14 C14 + k12 C12 + k34 C34) - k23 (C0 + cos(phi1-phi2) C1
                  + sin(phi1-phi2) C2)
   with rational matrices and trig_t in {cos phi1, sin phi1, cos phi2, sin phi2}.
2. Interval arithmetic.  Every real number is carried as a pair of doubles [lo, hi].  After each +, -, * and / the
   endpoints are moved outward by one ulp with nextafter; since IEEE 754 rounds these operations to nearest, the
   exact result lies in the widened interval.  Rationals are converted with correct rounding and widened the same
   way.  Cosines and sines come from arb (python-flint) as rigorous balls and are widened to doubles outward.
3. Positive definiteness.  Gaussian elimination without pivoting on an interval matrix: if every pivot interval is
   strictly positive, every real symmetric matrix inside the interval matrix has positive pivots and is positive
   definite.
4. Cover.  Two independent modes.
     box:    the trig functions are enclosed over the whole square (arb balls of radius r), so a successful
             elimination proves M_p > 0 on the square with no Lipschitz estimate at all.
     centre: the trig functions are taken at the centre and (4 + |k23|) r is subtracted from the diagonal; this is the
             Lipschitz argument of the paper, made rigorous at the centre.
   Squares that fail are split into four, down to max_depth.

Usage: python3 cert_interval.py cert_h9.6.npy [box|centre|finite] [m]   (finite: only the exact finite part)
"""
import sys
import time
from fractions import Fraction

import flint
import numpy as np
import sympy as sp

import certify3d as cz

flint.ctx.prec = 80
T = cz.T
PL = cz.PL

# ---------- exact coefficient matrices ----------
_s = [sp.Matrix([[0, 1], [1, 0]]), sp.Matrix([[0, -sp.I], [sp.I, 0]]), sp.Matrix([[1, 0], [0, -1]])]
BAS = [sp.I * sp.eye(2) / sp.sqrt(2)] + [sp.I * m / sp.sqrt(2) for m in _s]


def rtr(M):
    return sp.re(sp.expand(M.trace()))


def ipx(X, Y):
    return -rtr(X * Y)


def fp(Y, W):
    Y1, Y2, Y3, Y4 = Y
    v = sum(rtr(y * y * W) for y in Y) + 2 * rtr(Y1 * Y2 * W) + 2 * rtr(Y3 * Y4 * W)
    v -= 2 * sum(rtr(Yi * W * Yj) for Yi in (Y1, Y2) for Yj in (Y3, Y4))
    return v


def qmat(form, rational=True):
    """Exact 16 x 16 matrix of the quadratic form, by polarisation on symbolic coordinates."""
    v = sp.symbols("v0:16", real=True)
    Y = [sum((v[4 * i + a] * BAS[a] for a in range(4)), sp.zeros(2)) for i in range(4)]
    q = sp.expand(form(Y))
    M = sp.zeros(16)
    for i in range(16):
        for j in range(16):
            M[i, j] = sp.nsimplify(sp.diff(q, v[i], v[j]) / 2)
    assert not rational or all(x.is_rational for x in M), "coefficient matrix not rational"
    return M


def diagW(a, b):
    return sp.diag(a, b)


print("building exact coefficient matrices ...", flush=True)
t0 = time.time()
F_EX = [qmat(lambda Y, W=diagW(1, 0): fp(Y, W)), qmat(lambda Y, W=diagW(sp.I, 0): fp(Y, W)),
        qmat(lambda Y, W=diagW(0, 1): fp(Y, W)), qmat(lambda Y, W=diagW(0, sp.I): fp(Y, W))]


def c23(W, rational=True):
    return qmat(lambda Y: ipx(Y[1], W * Y[2] * W.H), rational)


_C0, _Cpi, _Chalf = c23(diagW(1, 1)), c23(diagW(1, -1)), c23(diagW(sp.I, 1))
C_EX = [(_C0 + _Cpi) / 2, (_C0 - _Cpi) / 2, _Chalf - (_C0 + _Cpi) / 2]


def cmx(i, j):
    M = sp.zeros(16)
    for q in range(4):
        M[4 * i + q, 4 * j + q] += sp.Rational(1, 2)
        M[4 * j + q, 4 * i + q] += sp.Rational(1, 2)
    return M


CORN_EX = [cmx(0, 3), cmx(0, 1), None, cmx(2, 3)]  # k14, k12, (k23 separate), k34
print(f"  done in {time.time() - t0:.1f}s", flush=True)

# sanity: the exact matrices agree with the floating ones used in the search, and the C decomposition is exact
for E, Fl in zip(F_EX, cz.F_BASIS):
    assert np.abs(np.array(E, dtype=float) - Fl).max() < 1e-12
for E, Fl in zip(C_EX, cz.C_BASIS):
    assert np.abs(np.array(E, dtype=float) - Fl).max() < 1e-12
th = sp.symbols("th", real=True)
assert sp.simplify(c23(diagW(sp.exp(sp.I * th), 1), False) - (C_EX[0] + sp.cos(th) * C_EX[1] + sp.sin(th) * C_EX[2])) \
    == sp.zeros(16)


# ---------- interval arithmetic on numpy arrays ----------
DN, UP = -np.inf, np.inf


def iv(lo, hi=None):
    lo = np.asarray(lo, float)
    return (lo, lo if hi is None else np.asarray(hi, float))


def frac_iv(x):
    f = float(Fraction(x))
    return (np.nextafter(f, DN), np.nextafter(f, UP))


def add(a, b):
    return np.nextafter(a[0] + b[0], DN), np.nextafter(a[1] + b[1], UP)


def sub(a, b):
    return np.nextafter(a[0] - b[1], DN), np.nextafter(a[1] - b[0], UP)


def mul(a, b):
    p = (a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1])
    return (np.nextafter(np.minimum(np.minimum(p[0], p[1]), np.minimum(p[2], p[3])), DN),
            np.nextafter(np.maximum(np.maximum(p[0], p[1]), np.maximum(p[2], p[3])), UP))


def div_pos(a, b):
    """a / b for b > 0 (caller guarantees b[0] > 0)."""
    p = (a[0] / b[0], a[0] / b[1], a[1] / b[0], a[1] / b[1])
    return (np.nextafter(np.minimum(np.minimum(p[0], p[1]), np.minimum(p[2], p[3])), DN),
            np.nextafter(np.maximum(np.maximum(p[0], p[1]), np.maximum(p[2], p[3])), UP))


def arb_to_iv(x):
    """Rigorous double enclosure of an arb ball."""
    m, r = x.mid(), x.rad()
    lo, hi = m - r, m + r  # arb arithmetic, rigorous
    flo, fhi = float(lo.lower()), float(hi.upper())
    return np.nextafter(flo, DN), np.nextafter(fhi, UP)


_trig_cache = {}


def trig_1d(k, depth, m, box):
    """Enclosures of cos and sin of phi = (2k+1) pi / (m 2^depth), over [phi - r, phi + r] if box, r = pi/(m 2^depth)."""
    key = (k, depth, box)
    if key not in _trig_cache:
        den = m * 2 ** depth
        phi = flint.arb.pi() * (2 * k + 1) / den
        if box:
            phi = phi.union(phi - flint.arb.pi() / den).union(phi + flint.arb.pi() / den)
        _trig_cache[key] = (arb_to_iv(phi.cos()), arb_to_iv(phi.sin()))
    return _trig_cache[key]


def trig_diff(k1, k2, depth, m, box):
    """cos and sin of phi1 - phi2; centres are exact multiples of pi so this is a single arb evaluation."""
    key = ("d", k1 - k2, depth, box)
    if key not in _trig_cache:
        den = m * 2 ** depth
        phi = flint.arb.pi() * (2 * (k1 - k2)) / den
        if box:
            phi = phi.union(phi - 2 * flint.arb.pi() / den).union(phi + 2 * flint.arb.pi() / den)
        _trig_cache[key] = (arb_to_iv(phi.cos()), arb_to_iv(phi.sin()))
    return _trig_cache[key]


def gather(ks, depth, m, box):
    """Interval arrays (lo, hi) of shape (n,) for the six trig values of a batch of squares."""
    out = [[np.empty(len(ks)), np.empty(len(ks))] for _ in range(6)]
    for n, (k1, k2) in enumerate(ks):
        (c1, s1), (c2, s2) = trig_1d(k1, depth, m, box), trig_1d(k2, depth, m, box)
        cd, sd = trig_diff(k1, k2, depth, m, box)
        for t, v in enumerate((c1, s1, c2, s2, cd, sd)):
            out[t][0][n], out[t][1][n] = v
    return [tuple(o) for o in out]


# ---------- the plaquette matrices ----------
cert = np.load(sys.argv[1], allow_pickle=True).item()
Gq = {y: [[Fraction(v) for v in r] for r in cert["G"][y]] for y in T.sites}
Aq = [{i: Fraction(v) for i, v in a.items()} for a in cert["A"]]


def check_finite():
    """Exact rational check of the finite part, independent of the search: every G_y is positive definite with
    G_y[mu, 3 + mu] = 0, every weight is >= 0, and the per-link totals have maximum exactly the stated h."""
    for y in T.sites:
        g = Gq[y]
        assert all(g[i][j] == g[j][i] for i in range(6) for j in range(6)), f"G at {y} not symmetric"
        assert all(g[mu][3 + mu] == 0 for mu in range(3)), f"G at {y}: end-to-end entry not zero"
        assert cz.ldl_psd(g), f"G at {y} not positive definite"
    assert all(v >= 0 for a in Aq for v in a.values()), "negative plaquette weight"
    assert all(set(a) == set(fp_) for a, (_, _, fp_) in zip(Aq, PL)), "weights not on the free positions"
    coll = {e: Fraction(0) for e in T.free}
    for (x, mu) in T.edges:
        e = T.eidx[(x, mu)]
        if e not in T.fixed:
            coll[e] += Gq[x][mu][mu] + Gq[T.sh(x, mu)][3 + mu][3 + mu]
    for (p, _, free_pos), a in zip(PL, Aq):
        for i in free_pos:
            coll[p[i]] += a[i]
    h = max(coll.values())
    assert h == Fraction(cert["h"]), f"stated h {cert['h']} but per-link maximum is {h}"
    return h, len(coll)


def plaquette_terms(j):
    """Exact rational matrices (restricted to free coordinates): constant part and the six trig coefficients."""
    p, corners, free_pos = PL[j]
    k = [2 * Gq[s][r][q] for s, r, q in corners]
    idx = [4 * i + q for i in free_pos for q in range(4)]
    const = sp.zeros(16)
    for i in free_pos:
        for q in range(4):
            const[4 * i + q, 4 * i + q] += sp.Rational(Aq[j][i].numerator, Aq[j][i].denominator)
    kk = [sp.Rational(x.numerator, x.denominator) for x in k]
    const -= kk[0] * CORN_EX[0] + kk[1] * CORN_EX[1] + kk[3] * CORN_EX[3] + kk[2] * C_EX[0]
    trig = [-F_EX[0], -F_EX[1], -F_EX[2], -F_EX[3], -kk[2] * C_EX[1], -kk[2] * C_EX[2]]
    sub_ = lambda M: [[M[a, b] for b in idx] for a in idx]
    return sub_(const), [sub_(M) for M in trig], abs(k[2])


def build(const, trig, tv, shift):
    """Interval matrix (lo, hi) arrays of shape (n, d, d)."""
    d = len(const)
    nb = len(tv[0][0])
    lo, hi = np.empty((nb, d, d)), np.empty((nb, d, d))
    for a in range(d):
        for b in range(d):
            acc = iv(np.full(nb, frac_iv(const[a][b])[0]), np.full(nb, frac_iv(const[a][b])[1]))
            if const[a][b] == 0:
                acc = iv(np.zeros(nb))
            for t in range(6):
                cf = trig[t][a][b]
                if cf != 0:
                    ci = frac_iv(cf)
                    acc = add(acc, mul(iv(np.full(nb, ci[0]), np.full(nb, ci[1])), tv[t]))
            if a == b:
                acc = sub(acc, shift)
            lo[:, a, b], hi[:, a, b] = acc
    return lo, hi


def pd_interval(lo, hi):
    """Interval Gaussian elimination; returns a boolean array: True where every pivot is certified positive."""
    lo, hi = lo.copy(), hi.copy()
    n = lo.shape[1]
    ok = np.ones(lo.shape[0], bool)
    for j in range(n):
        ok &= lo[:, j, j] > 0
        piv = (np.where(ok, lo[:, j, j], 1.0), np.where(ok, hi[:, j, j], 1.0))
        for i in range(j + 1, n):
            f = div_pos((lo[:, i, j], hi[:, i, j]), piv)
            for k in range(j + 1, n):
                r = sub((lo[:, i, k], hi[:, i, k]), mul(f, (lo[:, j, k], hi[:, j, k])))
                lo[:, i, k], hi[:, i, k] = r
    return ok


CHUNK = 100000


def prove(mode, m=64, max_depth=14, plaquettes=None):
    box = mode == "box"
    total, worst_depth = 0, 0
    for j in (range(len(PL)) if plaquettes is None else plaquettes):
        const, trig, k23 = plaquette_terms(j)
        lip = 4 + float(k23)
        sq = [(k1, k2) for k1 in range(m) for k2 in range(m)]
        for depth in range(max_depth + 1):
            r = np.pi / (m * 2 ** depth)
            if not box:
                s = np.nextafter(np.nextafter(lip * 1.0000001, UP) * np.nextafter(r * 1.0000001, UP), UP)
            ok = []
            for c0 in range(0, len(sq), CHUNK):  # bounded memory
                part = sq[c0:c0 + CHUNK]
                shift = iv(np.zeros(len(part))) if box else iv(np.full(len(part), s))
                ok.append(pd_interval(*build(const, trig, gather(part, depth, m, box), shift)))
            ok = np.concatenate(ok)
            total += len(sq)
            bad = [sq[n] for n in np.flatnonzero(~ok)]
            if not bad:
                worst_depth = max(worst_depth, depth)
                break
            sq = [(2 * k1 + a, 2 * k2 + b) for k1, k2 in bad for a in (0, 1) for b in (0, 1)]
        else:
            raise AssertionError(f"plaquette {j}: not proved at depth {max_depth} ({len(bad)} squares left)")
        print(f"  plaquette {j:2d}: proved, depth {depth}, squares so far {total}", flush=True)
    return total, worst_depth


if __name__ == "__main__":
    mode = sys.argv[2] if len(sys.argv) > 2 else "box"
    m = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    h = Fraction(cert["h"])
    print(f"certificate {sys.argv[1]}: h = {h} = {float(h)}; mode {mode}; initial grid {m} x {m}", flush=True)
    hf, nl = check_finite()
    print(f"finite part (exact): {len(T.sites)} site matrices G_y positive definite, weights >= 0, "
          f"maximum of {nl} per-link totals = {hf} (matches the stated h)", flush=True)
    if mode == "finite":
        sys.exit(0)
    t0 = time.time()
    n, dmax = prove(mode, m)
    print(f"PROVED (interval arithmetic, mode {mode}): all 81 plaquette matrices positive definite for all phases; "
          f"{n} squares, max depth {dmax}, {time.time() - t0:.0f}s", flush=True)
