#!/usr/bin/env python3
"""Search for gauge-fixing forests that minimise the Bakry-Emery Hessian coefficient c_F (CP-SAT).

Family: every site x picks a "parent" direction mu(x) in {0..d-1}; the fixed (gauged-to-identity) link is the edge
(x - e_mu, mu) joining x to x - e_mu. Along parent pointers the coordinate sum strictly decreases, so the fixed set F
is a forest for EVERY choice function mu (exact gauge fixing on any box with free boundary conditions), with one fixed
link per site. Temporal gauge is mu = const.

For a plaquette p with f_p fixed links, k_p = 4 - f_p links remain; SZZ's Cauchy-Schwarz counting gives each remaining
edge e a coefficient c_e = sum_{p ni e} k_p, and Hess S <= (max_e c_e) N |beta| |v|^2. We minimise max_{e not in F} c_e
over mu periodic with period P in every direction.
"""
import itertools
import sys

from ortools.sat.python import cp_model


def solve(d, P, time_limit=120.0, target=None):
    sites = list(itertools.product(range(P), repeat=d))
    sh = lambda x, mu, s=1: tuple((x[i] + (s if i == mu else 0)) % P for i in range(d))
    m = cp_model.CpModel()
    # mu(x) one-hot
    ch = {(x, a): m.NewBoolVar(f"c{x}{a}") for x in sites for a in range(d)}
    for x in sites:
        m.AddExactlyOne(ch[x, a] for a in range(d))
    # edge (y, a) is fixed iff mu(y + e_a) = a
    fixed = {(y, a): ch[sh(y, a), a] for y in sites for a in range(d)}
    # plaquettes (y; a < b): edges (y,a), (y+e_a, b), (y+e_b, a), (y, b)
    plaqs = [(y, a, b) for y in sites for a in range(d) for b in range(a + 1, d)]
    pedges = {p: [(p[0], p[1]), (sh(p[0], p[1]), p[2]), (sh(p[0], p[2]), p[1]), (p[0], p[2])] for p in plaqs}
    k = {}
    for p in plaqs:
        k[p] = m.NewIntVar(0, 4, f"k{p}")
        m.Add(k[p] == 4 - sum(fixed[e] for e in pedges[p]))
    incid = {e: [p for p in plaqs if e in pedges[p]] for e in fixed}
    c_max = m.NewIntVar(0, 8 * (d - 1), "cmax")
    for e, ps in incid.items():
        # only constrain free edges: c_e <= c_max unless e fixed
        m.Add(sum(k[p] for p in ps) <= c_max).OnlyEnforceIf(fixed[e].Not())
    if target is not None:
        m.Add(c_max <= target)
    m.Minimize(c_max)
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = time_limit
    s.parameters.num_workers = 4
    r = s.Solve(m)
    if r not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, s.StatusName(r), None
    mu = {x: next(a for a in range(d) if s.Value(ch[x, a])) for x in sites}
    return s.Value(c_max), s.StatusName(r), mu


if __name__ == "__main__":
    for d in (3, 4):
        lb = 8 * (d - 1) ** 2 / d
        for P in (2, 3, 4, 6):
            if d == 4 and P > 4:
                continue
            c, status, mu = solve(d, P, time_limit=float(sys.argv[1]) if len(sys.argv) > 1 else 120)
            print(f"d={d} period={P}: best c_F = {c} ({status}); temporal gauge 8d-12 = {8 * d - 12}; "
                  f"edge (SZZ) 8(d-1) = {8 * (d - 1)}; averaging lower bound {lb:.2f}", flush=True)
            if mu is not None and P <= 4:
                print("   mu pattern:", "".join(str(mu[x]) for x in sorted(mu)), flush=True)
