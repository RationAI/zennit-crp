"""SUPERSEDED by :mod:`experiments.scripts.emit_rank_outputs`. Do not run.

This emitted ``cdet_head_table.tex`` and ``cdet_normalised_table.tex``. Both
are statistically unsound and the numbers in them must not be reported:

* It pools every block into one sample -- ``stat()`` concatenates 12 blocks x
  16 images and divides by sqrt(192). The same 16 images recur in every block
  and blocks are not replicates of one another, so the reported standard error
  is too small by about sqrt(12), and the mean mixes collapsed early blocks
  with late ones.
* The normalisation divides by ``greedy - random`` per cell. That denominator
  approaches zero exactly where the cell collapses, with no guard, and flips
  sign where the greedy search scores below random -- in both cases the ratio
  is meaningless and is printed anyway.
* It subtracts an empirical 5-draw random estimate of a quantity that is zero
  in expectation by construction (DAPC is antisymmetric), which adds variance
  instead of removing bias.
* It takes a ratio of pooled means rather than aggregating a per-image ratio,
  and reports no interval at all.
* Its denominator ignores the dual-greedy search even where that search found
  the better ordering, and labels the greedy result as if it were the
  attainable optimum.

The replacement fixes each of these; see that module's docstring. The two old
fragments are left on disk so the paper still builds, and should be replaced
by ``cdet_normdapc_attention.tex`` / ``cdet_overlap_attention.tex`` and their
supplementary counterparts.
"""
import sys

raise SystemExit(
    "emit_detector_tables.py is superseded by emit_rank_outputs.py and its "
    "output is statistically unsound; see this module's docstring.")
from pathlib import Path
import numpy as np

RES = Path(__file__).resolve().parents[2] / "data" / "results" / "benchmark"
OUT = Path("/home/claude/workspaces/crp-paper/iclr2026/tables")
MODELS = [("vit_small_funny_birds", "M1 ViT-S/FB"),
          ("vit_base_imagenet", "M2 ViT-B/IN")]
SITES = ["residual", "proj_drop", "value", "qk"]
METHODS = ["cp_lrp", "chefer", "optimal", "optimal_dual", "random"]
LABEL = {"cp_lrp": "CP-LRP", "chefer": "Chefer", "optimal": "Greedy",
         "optimal_dual": "Dual", "random": "Random"}
SITE_TEX = {"residual": r"\texttt{residual}", "proj_drop": r"\texttt{proj\_drop}",
            "value": r"\texttt{value}", "qk": r"\texttt{qk}"}


def store(key, concept):
    suf = "__head" if concept == "head" else ""
    osuf = "__head_optimal" if concept == "head" else "__optimal"
    z = {}
    for f in (RES / f"cdet_dapc_{key}{suf}.npz", RES / f"cdet_dapc_{key}{osuf}.npz"):
        d = np.load(f, allow_pickle=True)
        z.update({k: d[k] for k in d.files if k != "meta"})
    return z


def stat(z, meth, site):
    v = [z[f"{meth}__{site}__b{b}__dapc"] for b in range(12)
         if f"{meth}__{site}__b{b}__dapc" in z]
    if not v:
        return None
    a = np.concatenate([np.atleast_1d(x) for x in v]).astype(float)
    return a.mean(), a.std(ddof=1) / np.sqrt(a.size), a.size


def fmt(s):
    return "---" if s is None else f"${s[0]:+.3f}$"


def head_table():
    L = [r"\begin{tabular}{llccccc}", r"\toprule",
         "model & site & " + " & ".join(LABEL[m] for m in METHODS) + r" \\",
         r"\midrule"]
    for key, lab in MODELS:
        z = store(key, "head")
        for i, site in enumerate(SITES):
            cells = [fmt(stat(z, m, site)) for m in METHODS]
            L.append(f"{lab if i == 0 else ''} & {SITE_TEX[site]} & "
                     + " & ".join(cells) + r" \\")
        L.append(r"\midrule" if key == MODELS[0][0] else r"\bottomrule")
    L.append(r"\end{tabular}")
    return "\n".join(L)


def normalised_table():
    L = [r"\begin{tabular}{llccc}", r"\toprule",
         r"model & site & CP-LRP & Chefer & Dual \\", r"\midrule"]
    for concept, cname in (("embed_dim", "embedding dimension"), ("head", "head")):
        L.append(r"\multicolumn{5}{l}{\emph{" + cname + r" detectors}} \\")
        for key, lab in MODELS:
            z = store(key, concept)
            for i, site in enumerate(SITES):
                r, g = stat(z, "random", site), stat(z, "optimal", site)
                if r is None or g is None:
                    continue
                den = g[0] - r[0]
                cells = []
                for m in ("cp_lrp", "chefer", "optimal_dual"):
                    s = stat(z, m, site)
                    cells.append("---" if s is None else f"${(s[0]-r[0])/den:+.2f}$")
                L.append(f"{lab if i == 0 else ''} & {SITE_TEX[site]} & "
                         + " & ".join(cells) + r" \\")
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L.append(r"\end{tabular}")
    return "\n".join(L)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cdet_head_table.tex").write_text(head_table() + "\n")
    (OUT / "cdet_normalised_table.tex").write_text(normalised_table() + "\n")
    print("wrote", OUT / "cdet_head_table.tex")
    print("wrote", OUT / "cdet_normalised_table.tex")
