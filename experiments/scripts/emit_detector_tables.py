"""Emit the LaTeX result tables of the concept-detector benchmark from the
stored npz files, so the paper never carries hand-copied numbers.

Writes two fragments next to the paper:
  cdet_head_table.tex        mean DAPC per site, head concept, both models
  cdet_normalised_table.tex  (DAPC - random) / (DAPC_greedy - random)
"""
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
