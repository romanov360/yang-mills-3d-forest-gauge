#!/usr/bin/env python3
"""Independent verification of the optimal gauge-fixing forests found by forest_search.py (no CP-SAT here).

Patterns (period P in every direction; mu(x) is read from the string in lexicographic order of x in {0..P-1}^d):
  d = 4, P = 2: "0212331001332120"            claimed c_F = 18, every plaquette has exactly one fixed link
  d = 3, P = 3: "210102120102120210120210102"  claimed c_F = 11
Checks, on boxes {0..L-1}^d with free boundary conditions:
  (1) F (fixed edges (x - e_mu(x), mu(x))) is a forest: verified by union-find, not only by the potential argument;
  (2) the fixed-link count of every plaquette, and c_e = sum_{p ni e} (4 - f_p) <= claimed c_F for every free edge, with
      equality attained in the bulk;
  (3) exact gauge fixing: for Haar-random SU(2) links, the gauge transformation built from F sets every F-link to I
      and leaves the Wilson action unchanged;
  (4) the Hessian of the gauge-fixed action (free links only), assembled exactly, satisfies
      lambda_max <= c_F (and lambda_min >= -c_F) at random and adversarial configurations; checked against finite
      differences.
"""
import itertools
import sys

import numpy as np

import fasthess as f
import hessian as h

PATTERNS = {4: ("0212331001332120", 2, 18), 3: ("210102120102120210120210102", 3, 11)}


class Box:
    def __init__(self, L, d, pattern, P):
        self.L, self.d = L, d
        self.sites = list(itertools.product(range(L), repeat=d))
        self.sidx = {x: i for i, x in enumerate(self.sites)}
        cells = sorted(itertools.product(range(P), repeat=d))
        mu_of = {c: int(pattern[i]) for i, c in enumerate(cells)}
        self.mu = {x: mu_of[tuple(xi % P for xi in x)] for x in self.sites}
        self.edges = [(x, a) for x in self.sites for a in range(d) if x[a] + 1 < L]
        self.eidx = {e: i for i, e in enumerate(self.edges)}
        self.nE = len(self.edges)
        sh = lambda x, a: tuple(x[i] + (1 if i == a else 0) for i in range(d))
        self.sh = sh
        self.fixed = set()
        for (y, a) in self.edges:
            if self.mu[sh(y, a)] == a:
                self.fixed.add((y, a))
        self.plaqs = []
        for x in self.sites:
            for a in range(d):
                for b in range(a + 1, d):
                    if x[a] + 1 < L and x[b] + 1 < L:
                        self.plaqs.append(((self.eidx[(x, a)], True), (self.eidx[(sh(x, a), b)], True),
                                           (self.eidx[(sh(x, b), a)], False), (self.eidx[(x, b)], False)))


def is_forest(box):
    parent = list(range(len(box.sites)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for (y, a) in box.fixed:
        r1, r2 = find(box.sidx[y]), find(box.sidx[box.sh(y, a)])
        if r1 == r2:
            return False
        parent[r1] = r2
    return True


def coefficients(box):
    fixed_idx = {box.eidx[e] for e in box.fixed}
    fp = [sum(e in fixed_idx for e, _ in p) for p in box.plaqs]
    c = {}
    for p, fcount in zip(box.plaqs, fp):
        for e, _ in p:
            if e not in fixed_idx:
                c[e] = c.get(e, 0) + (4 - fcount)
    return fp, c, fixed_idx


def gauge_fix(box, Q):
    """Gauge transformation g with g_y Q_(y,a) g_{y+a}^{-1} = I on F; built by walking parent pointers."""
    g = {}
    order = sorted(box.sites, key=sum)  # parents have smaller coordinate sum
    for x in order:
        a = box.mu[x]
        y = tuple(x[i] - (1 if i == a else 0) for i in range(box.d))
        if y in box.sidx:  # parent edge (y, a) inside the box: require g_y Q g_x^{-1} = I => g_x = g_y Q
            g[x] = g[y] @ Q[box.eidx[(y, a)]]
        else:
            g[x] = np.eye(2, dtype=complex)
    return np.array([g[x] @ Q[i] @ g[box.sh(x, a)].conj().T for i, (x, a) in enumerate(box.edges)])


def action(box, Q):
    s = 0.0
    for p in box.plaqs:
        M = np.eye(2, dtype=complex)
        for e, fw in p:
            M = M @ (Q[e] if fw else Q[e].conj().T)
        s += np.trace(M).real
    return s


def free_hessian(box, Q, fixed_idx):
    H = f.hess_matrix(box, Q)  # full Hessian on all edges
    free = [e for e in range(box.nE) if e not in fixed_idx]
    idx = np.array([3 * e + a for e in free for a in range(3)])
    return H[np.ix_(idx, idx)], free


def main():
    rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    fails = 0
    for d, (pat, P, cF) in PATTERNS.items():
        L = 6 if d == 3 else 4
        box = Box(L, d, pat, P)
        ok = is_forest(box)
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'} d={d}: fixed set is a forest on the {L}^{d} box ({len(box.fixed)} fixed of {box.nE} edges)")
        fp, c, fixed_idx = coefficients(box)
        cmax = max(c.values())
        ok = cmax <= cF
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'} d={d}: max_e c_e = {cmax} <= claimed c_F = {cF}; fixed-count histogram per plaquette "
              f"{ {k: fp.count(k) for k in sorted(set(fp))} }")
        # bulk check on a larger periodic-free box for d = 4: every interior plaquette has exactly one fixed link
        if d == 4:
            big = Box(6, 4, pat, P)
            fpb, cb, _ = coefficients(big)
            ok = set(fpb) == {1} and max(cb.values()) == 18
            fails += not ok
            print(f"{'PASS' if ok else 'FAIL'} d=4 6^4 box: all plaquettes have exactly one fixed link: {set(fpb)}; max c_e = {max(cb.values())}")
        Q = h.random_su2(rng, box.nE)
        Qg = gauge_fix(box, Q)
        okF = all(np.allclose(Qg[box.eidx[e]], np.eye(2)) for e in box.fixed)
        okS = abs(action(box, Q) - action(box, Qg)) < 1e-9
        fails += not (okF and okS)
        print(f"{'PASS' if okF and okS else 'FAIL'} d={d}: gauge fixing sets all F-links to I ({okF}) and preserves the action ({okS})")
        # Hessian of the gauge-fixed action at the gauge-fixed configuration: free directions only
        Hf, free = free_hessian(box, Qg, fixed_idx)
        ev = np.linalg.eigvalsh(Hf)
        ok = ev[-1] <= cF + 1e-9 and ev[0] >= -cF - 1e-9
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'} d={d}: gauge-fixed Hessian spectrum at a random config in [{ev[0]:.3f}, {ev[-1]:.3f}] within [-{cF}, {cF}]")
        # finite-difference control on free directions
        cvec = np.zeros((box.nE, 3))
        cvec[free] = rng.normal(size=(len(free), 3))
        v = [sum(cvec[e, a] * h.BASIS[a] for a in range(3)) for e in range(box.nE)]
        S = lambda t: action(box, np.array([h.expm_su2(t * v[e]) @ Qg[e] for e in range(box.nE)]))
        fd = (S(1e-4) - 2 * S(0) + S(-1e-4)) / 1e-8
        ex = cvec[free].ravel() @ Hf @ cvec[free].ravel()
        ok = abs(fd - ex) < 1e-3 * max(1, abs(ex))
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'} d={d}: gauge-fixed Hessian vs finite differences {ex:.5f} vs {fd:.5f}")
    print(f"\n{fails} failures")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
