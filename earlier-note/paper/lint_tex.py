#!/usr/bin/env python3
"""Structural LaTeX lint for paper.tex (no TeX installation here): begin/end nesting, brace balance, label/ref and
cite/bibitem consistency, environments defined, and control sequences checked against a whitelist (standard LaTeX,
amsmath/amssymb/amsthm, hyperref) plus the macros the paper defines. Exits non-zero on any problem."""
import pathlib
import re
import sys

TEX = (pathlib.Path(__file__).resolve().parent / "paper.tex").read_text()
KNOWN = set("""
documentclass usepackage newtheorem theoremstyle newcommand renewcommand operatorname mathrm mathbb mathfrak
title author date begin end maketitle section subsection label ref eqref cite item emph textbf texttt textit
frac tfrac sqrt sum prod int le ge leq geq neq ne in notin subset setminus cdot cdots dots ldots infty partial nabla
alpha beta gamma delta epsilon varepsilon zeta eta theta lambda mu nu xi pi rho sigma tau phi varphi chi psi omega
Lambda Delta Gamma Theta Phi Psi Omega Sigma Pi Xi
big Big bigl bigr Bigl Bigr left right langle rangle lfloor rfloor lceil rceil quad qquad mapsto to rightarrow
times mid ldots binom approx sim lesssim gtrsim equiv pm mp ast star circ ell exp log max min sup inf lim det
ni cos sin cap cup mathcal text bibitem appendix noindent thebibliography and textrm hat tilde bar vec dot overline underline
operatorname url href hfill vspace hspace newline par smallskip medskip bigskip footnote noindent
"""
.split())
fails = []
# environments
pre, body = TEX.split("\\begin{document}")
defined_env = set(re.findall(r"\\newtheorem\{(\w+)\}", pre)) | {"equation", "itemize", "enumerate", "proof", "abstract",
                                                                    "thebibliography", "document"}
stack = []
for m in re.finditer(r"\\(begin|end)\{(\w+\*?)\}", TEX):
    kind, env = m.groups()
    if kind == "begin":
        stack.append(env)
        if env not in defined_env and env != "document":
            fails.append(f"undefined environment {env}")
    else:
        if not stack or stack[-1] != env:
            fails.append(f"mismatched \\end{{{env}}} (open: {stack[-1] if stack else None})")
        else:
            stack.pop()
if stack:
    fails.append(f"unclosed environments {stack}")
# braces (ignore escaped \{ \})
depth = 0
for i, ch in enumerate(re.sub(r"\\[{}]", "", TEX)):
    depth += ch == "{"
    depth -= ch == "}"
    if depth < 0:
        fails.append(f"unbalanced closing brace near char {i}")
        break
if depth != 0:
    fails.append(f"brace imbalance {depth}")
# math delimiters $ balanced per paragraph (ignore \$)
for k, para in enumerate(re.split(r"\n\s*\n", body)):
    if para.replace("\\$", "").count("$") % 2:
        fails.append(f"odd number of $ in paragraph {k}: {para[:60]!r}")
# labels / refs / cites
labels = set(re.findall(r"\\label\{([^}]*)\}", TEX))
for r in re.findall(r"\\(?:eq)?ref\{([^}]*)\}", TEX):
    if r not in labels:
        fails.append(f"undefined reference {r}")
bib = set(re.findall(r"\\bibitem\{([^}]*)\}", TEX))
for c in re.findall(r"\\cite(?:\[[^\]]*\])?\{([^}]*)\}", TEX):
    for k in c.split(","):
        if k.strip() not in bib:
            fails.append(f"undefined citation {k}")
unused = bib - {k.strip() for c in re.findall(r"\\cite(?:\[[^\]]*\])?\{([^}]*)\}", TEX) for k in c.split(",")}
if unused:
    print("note: uncited bibitems", unused)
# control sequences
macros = set(re.findall(r"\\(?:re)?newcommand\{\\(\w+)\}", pre))
for cs in set(re.findall(r"\\([A-Za-z]+)", TEX)):
    if cs not in KNOWN and cs not in macros and cs not in defined_env:
        fails.append(f"unknown control sequence \\{cs}")
# \newcommand of an existing LaTeX command is an error
for m in re.findall(r"\\newcommand\{\\(\w+)\}", pre):
    if m in {"Re", "Im", "det", "max", "min", "exp", "log", "sup", "inf", "dim", "ker"}:
        fails.append(f"\\newcommand redefines existing \\{m} (use \\renewcommand)")
for f in fails:
    print("FAIL", f)
print(f"{len(fails)} problems")
sys.exit(1 if fails else 0)
