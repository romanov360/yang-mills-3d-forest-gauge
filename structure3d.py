#!/usr/bin/env python3
"""Plaquette structure of the d = 3 forest seen from each free link.

For a free link e and a plaquette p containing it, the type is
  '1A' : p has one fixed link, adjacent to e         '1O' : p has one fixed link, opposite to e
  '2a' : p has two fixed links, adjacent to each other (the two free links are adjacent)
  '2o' : p has two fixed links, opposite (the two free links are opposite)
  '0'  : no fixed link.
"""
from collections import Counter

import torus3d as t


def ptype(T, p, e):
    E = [g for g, _ in p]
    k = E.index(e)
    fx = [i for i in range(4) if E[i] in T.fixed]
    if len(fx) == 0:
        return "0"
    if len(fx) == 1:
        return "1O" if (fx[0] - k) % 4 == 2 else "1A"
    if len(fx) == 2:
        return "2a" if (fx[0] - fx[1]) % 4 in (1, 3) else "2o"
    return str(len(fx))


def table(T):
    per_link = Counter()
    per_plaq = Counter()
    for p in T.plaqs:
        E = [g for g, _ in p]
        nf = sum(g in T.fixed for g in E)
        per_plaq[(nf, "adj" if nf == 2 and (lambda fx: (fx[0] - fx[1]) % 4 in (1, 3))([i for i in range(4) if E[i] in T.fixed]) else "")] += 1
    for e in T.free:
        types = sorted(ptype(T, p, e) for p in T.plaqs if e in [g for g, _ in p])
        per_link[tuple(types)] += 1
    return per_plaq, per_link


if __name__ == "__main__":
    for L in (3, 6):
        pp, pl = table(t.Torus(L))
        print(f"L={L} plaquettes by (#fixed, kind):", dict(pp))
        print(f"L={L} free links by multiset of plaquette types:", dict(pl))
