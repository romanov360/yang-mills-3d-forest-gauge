# A real, small contribution: gauge-fixing forests enlarge the rigorous mass-gap regime

**What it is.**
- A rigorous enlargement of the region of couplings where 4D (and 3D) lattice Yang–Mills is *proved* to have a mass gap.
- A constant-factor improvement at **strong coupling**.
- It is not the continuum mass gap.

The paper is `paper/paper.tex`, compiled to `paper/paper.pdf`. The code and the Lean file are in `bakry_emery/`, and the outputs are in `results/`.

## Result

Shen–Zhu–Zhu (arXiv:2204.12737) prove Bakry–Émery, and hence log-Sobolev, uniqueness and a mass gap, for SU(N) at
|β| < 1/(16(d−1)). Their Hessian constant is 8(d−1). Every link lies in 2(d−1) plaquettes of 4 links, and Cauchy–Schwarz
charges each link k = 4 per plaquette.

**Observation.**
- Fixing the gauge on a **forest** F of links (Q_e = I) is exact for gauge-invariant observables.
- It turns plaquettes into loops with k_p = 4 − |p ∩ F| free links.
- The constant becomes c_F = max_e Σ_{p∋e} k_p.

| d | SZZ (no gauge fixing) | temporal gauge | **optimal forest** | lower bound (any forest) | threshold |
| --- | --- | --- | --- | --- | --- |
| 3 | 16 | 12 | **11** (period-3 pattern) | **11** (integrality: φ(8/3)/(8/3)·4) | \|β\| < **1/22** (was 1/32) |
| 4 | 24 | 20 | **18** (period-2; every plaquette has exactly one fixed link) | 18 | \|β\| < **1/36** (was 1/48) |
| d | 8(d−1) | 8d − 12 | — | 2(d−1)φ(s*)/s* ≥ 8(d−1)²/d | 1/(16d − 24) via temporal gauge |

The lower bound column applies to every forest, even with the sharpest constant the plaquette-wise Cauchy–Schwarz can
give (the Perron eigenvalue ρ(M_F); `perron.py` gives ρ = 11 and 18 for our patterns). So the patterns are optimal for
this method in d = 3 and d = 4.

In the standard normalisation β_W = N²β, d = 4:

| Group | Before | Now |
| --- | --- | --- |
| SU(2) | β_W < 0.083 | β_W < 0.111 |
| SU(3) | β_W < 0.19 | β_W < 0.25 |

**Limit of the method.**
- *Without* gauge fixing, Bakry–Émery fails for |β| ≥ 1/(8d). A π-flux configuration in an SU(2) block gives Hessian
  eigenvalue exactly 4dN|β|, for every N ≥ 2 and both signs of β.
- Adversarial numerics in d = 2, 3 suggest that 4d is the exact supremum on Z^d; this is a conjecture, not proved. The d = 4 test on the L = 2
  torus also stayed below 16 (best 15.955).

## Verification

| Check | Where | Result |
| --- | --- | --- |
| Exact SU(2) Hessian against finite differences | `hessian.py`, `fasthess.py` | agree to 10⁻⁶ |
| π-flux gives exactly 4d | `search.py` | 8, 12, 16 for d = 2, 3, 4 |
| Adversarial sup of the un-gauge-fixed Hessian | `search2.py`, `results/be_search*` | approaches 4d from below in d = 2, 3 (including the L = 3, d = 3 torus: best 11.918) and, on the L = 2 torus, d = 4 (best 15.955 < 16; `search2.py`, `d4_attack.py`, `results/be_search2_seed1.out`, `results/d4_attack_L2.out`). The L = 3, d = 4 run stayed below 16 but had not converged (best 13.1; `results/d4_attack_L3.out`), so it is weak evidence |
| Naive covariant lemma ‖d_A‖² ≤ 4d | `dA_norm.py` | **false**: about 16.2 > 16 on the L = 2, d = 4 torus; removed from the paper and recorded there as a caution |
| Optimal forests (CP-SAT) | `forest_search.py`, `results/forest_search.out` | d = 4: 18 (OPTIMAL); d = 3: 11 (OPTIMAL) |
| Independent check: forest property, counts, gauge-fixing exactness, Hessian bound, finite differences | `verify_forest.py`, `results/verify_forest.out` | 0 failures |
| Lean 4 kernel check of the cell combinatorics (d = 4 exactly one fixed link per plaquette and c = 18; d = 3 c ≤ 11; temporal 20) | `ForestCounting.lean`, `results/forest_counting_lean.out` | passes, axioms: propext only; two tamper controls rejected |
| Temporal gauge: exactness, Hessian, bound 8d − 12 | `axial.py`, `results/axial.out` | every completed line passes (the run was stopped early) |

## Adversarial self-checks and what they changed

- **"Sharp edge constant 4d ⇒ 1/32"** was my first target. The proof did not close: the curvature/commutator term is
  indefinite. It is now stated as a conjecture with evidence, not claimed.
- **Prior work.** Cao–Nissim–Sheffield (arXiv:2509.04688) already reach 1/(8(d−1)), but for **area law** via slab
  σ-models, not for the mass gap of the full measure. The comparison is stated as such.
- **Piazza–Shen (arXiv:2610.35286, 26 Oct 2026)** is 2D only. I found no prior use of gauge-fixing forests with
  Bakry–Émery in d ≥ 3.
- **Boundary conditions.** On a time-periodic torus the temporal gauge leaves Polyakov links with the full count, so the
  theorem is stated on boxes with free boundary conditions (or on Z^d).
- **Bugs caught before release.**
  - A wrong d = 4 comparison (the conjecture's 1/32 beats the proven 1/36).
  - The sign case of the obstruction.
  - `\newcommand{\Re}` (would not compile).
  - A missing `\begin{document}`.
- **Scope of the proof.**
  - Proved in full: Bakry–Émery, log-Sobolev and Poincaré for the gauge-fixed measure, and the gauge-invariant
    variance bound.
  - By adaptation of SZZ's proofs (Lemmas 4.9–4.10, Theorem 1.2, Corollary 4.11): uniqueness and decay of correlations.
    The paper says so explicitly.

## Honest size of the contribution (revised 2026-10-28 after a wider literature search)

**The d = 4 result is pre-empted.** Unrefereed Zenodo/GitHub preprints by Bhattacharjee, Bhattacharya and Lockwood,
dated before this work, already contain the forest idea and stronger results:

- *Yang–Mills Mass Gap in 4D: Lattice SU(N) and Every Compact Simple Group*, v3.19.0, doi:10.5281/zenodo.23729004.
  Earlier title: *Mass Gap of Four-Dimensional Lattice Yang–Mills Theory in a Forest Gauge*. Code:
  github.com/creelie/4Y-M. Same d = 4 forest (one fixed link per plaquette). A mixed-block bound gives
  |Hess S^F| ≤ 29/2, hence |β| < 1/29 (ours: 18 → 1/36). They report the optimal constant lies in [9+√17, 29/2]; our
  identity-field check (10.44, 11.54, 12.11 for L = 3, 4, 5 → 13.12) agrees.
- *Lattice Yang–Mills theory: spacing-uniform obstructions, the mass gap up to the Dobrushin ceiling and in a forest
  gauge, and Lean certificate*, v3.4.0, 21 Oct 2026, doi:10.5281/zenodo.23541099 (29/2 already present here).
- *Sharp Curvature and Finite-Box Criteria for Lattice Yang–Mills Theory* (18 Oct 2026, in the same repo): claims
  |Hess S| ≤ 4d in every d. That is our Conjecture 5.2, and it gives |β| < 1/(8d) = 1/32 in d = 4, also better than our 1/36.
- Also doi:10.5281/zenodo.23449466 (Dobrushin ceiling).

**What survives** (all minor):
- the optimality of 18 / 11 *within plaquette-wise Cauchy–Schwarz* (this explains why their block bound is needed);
- the d = 3 forest, giving |β| < 1/22 > 1/24. We found no d = 3 forest in their sources, but their block method would likely beat it;
- the Lean certificate for the forest combinatorics;
- independent numerical evidence for 4d and for the 9+√17 lower estimate.

**Recommendation.** Do not submit this as a new result. The paper has been rewritten as an independent derivation and
verification with full attribution. Their claims are unrefereed, so an independent check has some value. Options:
(a) don't post; (b) post as a short verification note; (c) contact the authors. Do not describe it as improving the best known bound.

## Before sending anywhere

- Author set to Tristan Romanov.
- Compile `paper.tex` with pdflatex (for example on Overleaf) and read the proofs yourself, especially Lemma 3.1 and
  Proposition 4.4.
- Keep the AI-disclosure paragraph; arXiv and journals expect it.
- If you send it to the Yang–Mills group address, describe it accurately: a strong-coupling lattice result, not a
  contribution to the continuum existence or mass-gap question.
