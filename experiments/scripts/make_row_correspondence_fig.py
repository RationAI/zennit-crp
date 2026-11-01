"""Paper figure: which parameter rows own which coordinates.

Visual language mirrors webapp/attention_unfolded (same palette, same tiny
model: N=3 tokens, D=4, H=2 heads of 2 dims, cell bar = owning head).
Writes row_correspondence.{png,pdf} into the crp-paper journal-figures tree.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

# ---- model, as in the web app -------------------------------------------
N, D, H, HD = 3, 4, 2, 2
D_HL, E_HL = 1, 1            # highlighted query-key index and value index

# ---- palette, copied from the web app -----------------------------------
PANEL, LINE, INK, MUTED = "#FFFFFF", "#D5DAD3", "#1C2126", "#69737D"
H0, H0S = "#137F8E", "#D6EEF1"
H1, H1S = "#C2651C", "#F8E3D0"
NONE_S = "#E9ECE8"
CW, CH = 0.44, 0.30          # cell size, web app ratio 52x28

head_of = lambda c: 0 if c < HD else 1
HCOL = {0: H0, 1: H1}
HSOFT = {0: H0S, 1: H1S}


def grid(ax, x, y, nrows, ncols, *, title="", sub="", hl_row=None, hl_col=None,
         bar_by_col=False, bar_by_row=False, tint=None, dim=False,
         rowlab=None, collab=None):
    """Top-left anchored grid of cells. Returns (left, right, ymid)."""
    for r in range(nrows):
        for c in range(ncols):
            fc = PANEL
            if dim:
                fc = NONE_S
            if hl_row is not None and r == hl_row:
                fc = HSOFT[head_of(r)]
            if hl_col is not None and c == hl_col:
                fc = HSOFT[head_of(c)]
            if tint is not None:
                fc = HSOFT[tint]
            ax.add_patch(Rectangle((x + c * CW, y - (r + 1) * CH), CW, CH,
                                   facecolor=fc, edgecolor=LINE, lw=0.6))
            bar = None
            if bar_by_col:
                bar = head_of(c)
            elif bar_by_row:
                bar = head_of(r)
            if bar is not None and not dim:
                ax.add_patch(Rectangle((x + c * CW, y - (r + 1) * CH),
                                       CW, CH * 0.09,
                                       facecolor=HCOL[bar], edgecolor="none"))
    if hl_row is not None:
        ax.add_patch(Rectangle((x, y - (hl_row + 1) * CH), ncols * CW, CH,
                               fill=False, edgecolor=HCOL[head_of(hl_row)], lw=1.7))
        ax.text(x - 0.12, y - (hl_row + 0.5) * CH, rowlab or "", fontsize=8,
                ha="right", va="center", color=HCOL[head_of(hl_row)])
    if hl_col is not None:
        ax.add_patch(Rectangle((x + hl_col * CW, y - nrows * CH), CW, nrows * CH,
                               fill=False, edgecolor=HCOL[head_of(hl_col)], lw=1.7))
        ax.text(x + (hl_col + 0.5) * CW, y - nrows * CH - 0.17, collab or "",
                fontsize=8, ha="center", va="center", color=HCOL[head_of(hl_col)])
    if title:
        ax.text(x, y + 0.30, title, fontsize=9.5, ha="left", va="center",
                color=INK, weight="bold")
    if sub:
        ax.text(x, y + 0.11, sub, fontsize=7.6, ha="left", va="center",
                color=MUTED, family="monospace")
    return x, x + ncols * CW, y - nrows * CH / 2


def arrow(ax, x0, x1, y, label="", label2=""):
    ax.add_patch(FancyArrowPatch((x0 + 0.08, y), (x1 - 0.08, y),
                                 arrowstyle="-|>", mutation_scale=8,
                                 lw=0.9, color=MUTED, shrinkA=0, shrinkB=0))
    mid = (x0 + x1) / 2
    if label:
        ax.text(mid, y + 0.17, label, fontsize=7.2, ha="center", color=MUTED,
                family="monospace")
    if label2:
        ax.text(mid, y - 0.19, label2, fontsize=7.2, ha="center", color=MUTED,
                family="monospace")


fig, (axA, axB) = plt.subplots(
    2, 1, figsize=(10.2, 5.8), gridspec_kw={"height_ratios": [1.25, 1.0]})
for a in (axA, axB):
    a.set_aspect("equal"); a.axis("off"); a.set_facecolor("none")

# ========================= (a) query-key path =========================
yq, yk = 3.05, 1.35
wq = grid(axA, 0, yq, D, D, title=r"$\mathbf{W}_Q$", sub="(D_q, D)",
          hl_row=D_HL, bar_by_row=True, rowlab="d")
wk = grid(axA, 0, yk, D, D, title=r"$\mathbf{W}_K$", sub="(D_q, D)",
          hl_row=D_HL, bar_by_row=True, rowlab="d")

xQ = 3.30
q = grid(axA, xQ, yq, N, D, title=r"$\mathbf{Q}$", sub="(N, D_q)",
         hl_col=D_HL, bar_by_col=True, collab="d")
k = grid(axA, xQ, yk, N, D, title=r"$\mathbf{K}$", sub="(N, D_q)",
         hl_col=D_HL, bar_by_col=True, collab="d")

arrow(axA, wq[1], xQ, yq - D_HL * CH - CH / 2, "row d", "column d")
arrow(axA, wk[1], xQ, yk - D_HL * CH - CH / 2, "row d", "column d")

ymid = (q[2] + k[2]) / 2
xop = 5.55
axA.add_patch(Rectangle((xop, ymid - 0.26), 1.85, 0.52, facecolor="#F7FAFA",
                        edgecolor=H0, lw=0.9))
axA.text(xop + 0.92, ymid, r"$\sum_{d\in\mathcal{D}^h_q} Q_{n,d} K_{m,d}$",
         fontsize=8.5, ha="center", va="center", color=INK)
for yy in (q[2], k[2]):
    axA.add_patch(FancyArrowPatch((q[1] + 0.08, yy), (xop - 0.08, ymid),
                                  arrowstyle="-|>", mutation_scale=8, lw=0.9,
                                  color=MUTED, shrinkA=0, shrinkB=0))
xS = 8.05
sm = grid(axA, xS, ymid + N * CH / 2, N, N, title=r"$\mathbf{S}^{h}$",
          sub="(N, N)", tint=0)
arrow(axA, xop + 1.85, xS, ymid)
axA.text(xS, ymid - N * CH / 2 - 0.30,
         "only head h's columns enter", fontsize=7.6, ha="left", color=H0)

axA.text(0, yq + 0.68, "(a)  query-key dimension detector: row d of "
         r"$\mathbf{W}_Q$ and row d of $\mathbf{W}_K$",
         fontsize=10, ha="left", color=INK, weight="bold")
axA.set_xlim(-0.45, 12.4); axA.set_ylim(-0.45, 3.95)

# ========================= (b) value path =========================
yv = 1.45
wv = grid(axB, 0, yv, D, D, title=r"$\mathbf{W}_V$", sub="(D_v, D)",
          hl_row=E_HL, bar_by_row=True, rowlab="e")
xV = 3.30
v = grid(axB, xV, yv - 0.15, N, D, title=r"$\mathbf{V}$", sub="(N, D_v)",
         hl_col=E_HL, bar_by_col=True, collab="e")
arrow(axB, wv[1], xV, v[2], "row e", "column e")

xZh = 6.15
zh = grid(axB, xZh, yv - 0.15, N, HD, title=r"$\mathbf{Z}^{h}$", sub="(N, D_v/H)",
          hl_col=E_HL, bar_by_col=True, collab="e")
arrow(axB, v[1], xZh, v[2], "mix over", "tokens m")

xZ = 8.35
z = grid(axB, xZ, yv - 0.15, N, D, title=r"$\mathbf{Z}$", sub="(N, D_v)",
         hl_col=E_HL, bar_by_col=True, collab="e")
arrow(axB, zh[1], xZ, v[2], "concat", "heads")

xP = 10.65
p = grid(axB, xP, yv - 0.15, D, D, title=r"$\mathbf{W}_O$", sub="(D, D_v)",
         bar_by_col=True, dim=True)
arrow(axB, z[1], xP, v[2])
axB.text(xP, v[2] - D * CH / 2 - 0.30, "heads first mix here", fontsize=7.6,
         ha="left", color=MUTED)

axB.text(xZ, v[2] - N * CH / 2 - 0.46, "index e unchanged", fontsize=7.6,
         ha="left", color=H0)
axB.text(0, yv + 0.68, "(b)  value dimension detector: row e of "
         r"$\mathbf{W}_V$", fontsize=10, ha="left", color=INK,
         weight="bold")
axB.text(0, -0.82, "cell bar = owning head  ·  "
         "head 0 teal, head 1 orange  ·  "
         "N=3 tokens, D=4, H=2 heads of 2 dimensions",
         fontsize=7.6, ha="left", color=MUTED)
axB.set_xlim(-0.45, 12.4); axB.set_ylim(-1.05, 2.35)

out = "/home/claude/workspaces/crp-paper/iclr2026/journal-figures/row_correspondence"
fig.subplots_adjust(hspace=0.05)
fig.savefig(out + ".png", dpi=170, bbox_inches="tight", facecolor="white")
fig.savefig(out + ".pdf", bbox_inches="tight", facecolor="white")
print("saved")
