# Three-dimensional lattice Yang–Mills in a forest gauge

This repository holds the paper and every computation it uses:

> Tristan Romanov, *Three-dimensional lattice Yang–Mills theory in a forest gauge: a Dobrushin range and a certified
> curvature bound*, October 2026. PDF: [`paper/paper.pdf`](paper/paper.pdf), source: [`paper/paper.tex`](paper/paper.tex).

It also holds the earlier note the paper builds on (`earlier-note/`, see below).

**What it is.** This is a strong-coupling result on Z^3. It proves a range of couplings where the lattice theory has
a positive mass gap. It does **not** touch the continuum Clay problem.

**What is new.** Earlier work treats d = 4, the temporal gauge, or no gauge fixing. Here we work in d = 3 with a
period-3 forest gauge in which every site is the head of exactly one fixed link. Each free link then has 7 plaquette
partners. The temporal gauge gives 8 and no gauge fixing gives 12.

## Results

Conventions: the Wilson weight is exp(Nβ Σ Re Tr Q_p) and β_W = N²β.

| | statement | how it is checked |
|---|---|---|
| Theorem A | SU(2) Dobrushin uniqueness and exponential decay for β_W < 4/7, with m ≥ −log(7β_W/4). Every forest has row sum at least 7\|β\|, so 4/7 is the best this method can give. | proof in the paper; the counts are kernel-checked in Lean |
| Theorem B | The gauge-fixed Hessian obeys \|Hess S₀\| ≤ h with h = 9.599991 (exact rational 9599991/10⁶). So SU(N) has a gap for \|β\| < 1/(2h) ≈ 1/19.20: SU(2) β_W < 0.208, SU(3) β_W < 0.468. | rational site/plaquette certificate (exact PSD and totals) plus an adaptive all-phase cover with a Lipschitz bound, proved in outward-rounded interval arithmetic (`cert_interval_centre.log`) |
| Lower bound | h_F ≥ 9.347955586999. This is the top root of an integer nonic, found at the identity field with k = (π,π,π). | exact factorisation in sympy |

For comparison, d = 3 SU(2)/SU(3) β_W ranges:

| method | SU(2) | SU(3) |
|---|---|---|
| SZZ | 0.125 | 0.281 |
| sharp constant 12 | 0.167 | 0.375 |
| Theorem B | 0.208 | 0.468 |
| ceiling from h_F ≥ 9.348 | 0.214 | 0.481 |

## Files

**Lattice and gauge.**
- `torus3d.py`: the 3-periodic torus, the forest and the Hessians.
- `structure3d.py`: the partner counts.

**Certificate.**
- `certificate3d*.py`: the search prototypes.
- `certify3d.py`: the search, exact rounding and a floating-point all-phase cover. Usage: `python3 certify3d.py h0 m_check`.
- `cert_interval.py`: the proof that every plaquette matrix is positive definite for all phases, in outward-rounded
  interval arithmetic (exact rational coefficient matrices, arb enclosures of the trigonometric values, interval
  Gaussian elimination). Usage: `python3 cert_interval.py cert_h9.6.npy [box|centre|finite] 64`. Logs: `cert_interval_*.log`. Every run first
  re-checks the finite part of the file exactly (G_y positive definite, weights ≥ 0, per-link maximum = stated h);
  `finite` stops there (`cert_finite.out`).
- `adversarial_interval.py` → `adversarial_interval.out`: random search for violations of the Lipschitz bound (worst
  ratio 0.75 ≤ 1) and a tamper test (a plaquette made indefinite is rejected by both modes; a still-definite one is accepted).

**Certificate outputs.**
- `cert_h9.6.npy` (the one the paper uses), `cert_h9.7.npy` and `cert_h10.npy`: the certificates.
- `cert_h*.log`: the run logs.
- `cert_h9.5.log`: a search at 9.5 (margin 1e-3, finitely many cutting planes) that found no certificate. This is evidence, not proof, that 9.5 is out of reach for this ansatz.

**Independent checks.**
- `validate_cert.py` → `validate_h9.6.out` and `validate_h9.7.out`. It checks the rewriting identity, the site identity, the spectra, and
  the plaquette inequality at random U(3) points.

**Lower bound.**
- `bloch_identity.py` and `identity_exact.py` → `identity_exact.out`.

**Numerical searches.** These are evidence only; none went above 9.348.
- `bloch3d.py`, `climb3d.py`.
- Logs: `bl_*.log`, `ab3.log`, `full3.log`.

**Lean.**
- `lean/ForestGauge3.lean`: Mathlib-free kernel checks of the 54/27/54 counts (`one_count`, `two_count`,
  `free_count`), the plaquette pattern (`plaq_ok`), the coefficient identity (`coeff_eq`) and the temporal comparison
  (`temporal3_ok`). Acyclicity of the forest is checked in Python (`torus3d.py`), not in Lean.
- `lean/ForestGauge3_v4.19.0.out`, `lean/ForestGauge3_v4.35.0.out`: the output of `lean ForestGauge3.lean` under
  Lean 4.19.0 and 4.35.0 (current stable), a few seconds each. `#print axioms` for all six theorems reports no axioms
  under 4.19.0 and only `propext` under 4.35.0. Changing a count (e.g. 54 to 55) makes the check fail.

**Paper.**
- `paper/paper.tex` and `paper/paper.pdf`, with the linter `paper/lint_tex.py`. Build with two runs of
  `pdflatex paper.tex`.

**Earlier note (`earlier-note/`).** *Gauge-fixing forests and the Bakry–Émery regime for lattice Yang–Mills theory: an
independent derivation* (reference [7] of the paper): optimal forests in d = 3 and d = 4 for the plaquette-wise
Cauchy–Schwarz bound.
- `earlier-note/paper/`: the note (`paper.tex`, `paper.pdf`).
- `earlier-note/07-bakry-emery-forests.md`: a summary with a table of every check.
- `earlier-note/bakry_emery/`: the code, including `hessian.py` and `fasthess.py`, which `torus3d.py` imports, and
  `ForestCounting.lean`.
- `earlier-note/results/`: the outputs.

## Reproducing

Run scripts from the repository root, e.g. `python3 cert_interval.py cert_h9.6.npy finite` (a few seconds) or
`python3 cert_interval.py cert_h9.6.npy centre 64` (the full all-phase proof, about an hour). Requirements: Python 3
with numpy and scipy; sympy for the lower bound; cvxpy for the certificate search; python-flint (arb) for
`cert_interval.py`; ortools for `earlier-note/bakry_emery/forest_search.py`; Lean 4 for the `.lean` files.

## Caveats

- The all-phase cover for h = 9.599991 is proved in interval arithmetic (26,371,504 squares, all 81 plaquettes,
  about one hour). The finite part (PSD of G, per-link totals) is exact rational arithmetic.
- Whether the optimal constant h_F equals 9.3479 is a conjecture.
- For SU(3) alone, the four-dimensional single-link Kantorovich estimate of Bhattacharjee–Bhattacharya–Lockwood
  (Thm. 6, giving β_W ≤ 3/4 in d = 4) would probably give a larger range in d = 3 too. We have not checked
  this and claim nothing for SU(3) beyond Theorem B; the interest of Theorem B is uniformity in the group, the
  log-Sobolev inequality and the certificate method.

## Citing

```bibtex
@misc{Romanov2026forest3d,
  author = {Tristan Romanov},
  title  = {Three-dimensional lattice {Yang--Mills} theory in a forest gauge: a {Dobrushin} range and a certified
            curvature bound},
  year   = {2026},
  note   = {Code and certificates: \url{https://github.com/romanov360/yang-mills-3d-forest-gauge}}
}
```

## License

Code: MIT (see `LICENSE`). Paper text and figures: CC BY 4.0.
