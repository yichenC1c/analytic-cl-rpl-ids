"""Generate the paper figures.

Styling follows the body text: Times-metric fonts via STIX so the figures do not read as
pasted in, no top or right spines, a faint grid, and unframed legends outside the axes.
Series are distinguished by both colour and dash pattern so they survive greyscale
printing.
"""
import json, glob, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C

# ---------------------------------------------------------------- shared style
plt.rcParams.update({
    # Times metrics, matching the IEEEtran body text
    "font.family": "STIXGeneral", "mathtext.fontset": "stix",
    "font.size": 7, "axes.labelsize": 7.5, "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8, "legend.fontsize": 6.5,
    # drop top and right spines
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.5, "axes.edgecolor": "0.3",
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.color": "0.3", "ytick.color": "0.3",
    "xtick.labelcolor": "black", "ytick.labelcolor": "black",
    "xtick.minor.visible": False, "ytick.minor.visible": False,
    # faint grid, reading aid only
    "axes.grid": True, "axes.grid.axis": "y",
    "grid.linestyle": "-", "grid.linewidth": 0.4, "grid.color": "0.88",
    "lines.linewidth": 1.15, "lines.markersize": 2.6,
    # unframed legend
    "legend.frameon": False, "legend.handlelength": 2.0,
    "legend.columnspacing": 1.4, "legend.handletextpad": 0.5,
    "legend.borderaxespad": 0.0,
    "figure.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.015,
})
os.makedirs(C.FIGS, exist_ok=True)
SC  = ["random", "b2w", "w2b", "toggle"]
LBL = {"random": "Random", "b2w": "B2W", "w2b": "W2B", "toggle": "Toggle"}
# pale blue / pink / green, plus a matching lavender for the fourth series
STY = {"random": ("#8AB4D8", "-"), "b2w": ("#E2A5B4", "--"),
       "w2b": ("#9CC79C", "-."), "toggle": ("#BAADD6", (0, (1, 1.2)))}
ACC = "#8AB4D8"          # single-series accent
REF = "#B8A48C"          # reference lines


# ---------------------------------------------------------------- Fig 2: equivalence
fig, ax = plt.subplots(figsize=(3.35, 1.95))
for sc in SC:
    d = json.load(open(f"{C.RESULTS}/E4_equivalence_{sc}_s0_D{C.RFF_D}.json"))
    c, ls = STY[sc]
    ax.semilogy([r["t"] for r in d["curve"]], [max(r["rel"], 1e-18) for r in d["curve"]],
                color=c, ls=ls, label=LBL[sc])
ax.axhline(2.22e-16, color="0.45", ls=(0, (4, 2)), lw=0.7)
ax.set_xlabel("Domain Index")
ax.set_ylabel(r"Relative deviation from $W^{\rm joint}$")
ax.set_ylim(1e-17, 1e-7); ax.set_xlim(0, 49)
ax.set_yticks([1e-16, 1e-14, 1e-12, 1e-10, 1e-8])
ax.set_xticks([10, 20, 30, 40])
ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.005),
          frameon=False, borderaxespad=0.0)
fig.savefig(f"{C.FIGS}/fig2_equivalence.pdf"); plt.close(fig)


# ---------------------------------------------------------------- Fig 3: memory wall + trade-off
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.15))
fig.subplots_adjust(wspace=0.24)

# (a) Grouped bars. The exact-kernel baseline yields only two to four points per
#     ordering, so bars read better than a line across a 48-wide axis. Bars disappear as
#     orderings exhaust the budget, which is itself the message.
ours = (C.RFF_D ** 2 * 8 + C.RFF_D * 2 * 8) / 2**20
tr = {sc: {r["t"]: r for r in json.load(open(f"{C.RESULTS}/E3_kelm_{sc}_s0.json"))["trace"]}
      for sc in SC}
doms = [1, 2, 3, 4, 5]
w = 0.19
for j, sc in enumerate(SC):
    c, _ = STY[sc]
    xs, ys = [], []
    for k, t in enumerate(doms):
        r = tr[sc].get(t)
        if r and r["status"] == "ok":
            xs.append(k + (j - 1.5) * w); ys.append(r["kernel_gb"] * 1024)
    a1.bar(xs, ys, width=w, color=c, edgecolor="0.45", linewidth=0.35,
           label=LBL[sc], zorder=3)
    if j == 3:
        a1.text(-0.42, 3.0e4, "$\\times$ = budget exhausted", fontsize=6,
                color="0.35", ha="left", va="center")
    dead = [t for t in doms if tr[sc].get(t, {}).get("status") == "OOM"]
    if dead:
        k = doms.index(dead[0])
        a1.plot(k + (j - 1.5) * w, 3.0e4, marker="x", color=c, ms=5, mew=1.3,
                ls="none", zorder=4)
a1.axhline(18 * 1024, color="0.2", ls=(0, (4, 2)), lw=0.8, zorder=2)
a1.text(4.42, 18 * 1024 * 0.42, "18 GB budget", fontsize=5.8, color="0.2",
        ha="right", va="top")
a1.axhline(ours, color="0.25", lw=1.6, zorder=3,
           label=f"This work ({ours*1024:.0f}\u2009KB, constant)")
a1.set_yscale("log"); a1.set_ylim(1e-2, 1.2e5)
a1.set_xticks(range(len(doms))); a1.set_xticklabels(doms); a1.set_xlim(-0.5, 4.5)
a1.set_xlabel("Domain Index"); a1.set_ylabel("Adaptation state (MB)")
a1.grid(axis="x", visible=False)
h, lb = a1.get_legend_handles_labels()
order = [lb.index(LBL[sc]) for sc in SC] + [i for i, x in enumerate(lb) if x.startswith("This work")]
a1.legend([h[i] for i in order], [lb[i] for i in order],
          ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.005))
a1.text(.965, .92, "(a)", transform=a1.transAxes, fontsize=7, weight="bold", ha="right")

# (b) State against detection performance
sw = {}
for f in glob.glob(f"{C.RESULTS}/E5_dsweep_*.json"):
    for r in json.load(open(f))["sweep"]:
        sw.setdefault(r["D"], []).append(r)
Ds = sorted(sw)
f1 = np.array([np.mean([x["f1"] for x in sw[D]]) for D in Ds])
er = np.array([np.std([x["f1"] for x in sw[D]]) for D in Ds])
kb = np.array([sw[D][0]["state_kb"] for D in Ds])
a2.fill_between(kb, f1 - er, f1 + er, color="#1f77b4", alpha=0.15, lw=0)
a2.plot(kb, f1, color="#1f77b4", ls="-", marker="o", label=r"Sweep over $D$")
sel = Ds.index(C.RFF_D)
a2.plot(kb[sel], f1[sel], marker="o", ms=6.5, mfc="none", mec="#E2A5B4", mew=1.6,
        ls="none", label=f"Selected, $D={C.RFF_D}$")
a2.axvline(4000 * 140 * 4 / 1024, color="0.35", ls=(0, (4, 2)), lw=0.7)
a2.text(4000 * 140 * 4 / 1024, 0.6695, "Replay\nbuffer", fontsize=5.6, color="0.35",
        ha="center", va="top", linespacing=0.95,
        bbox=dict(fc="white", ec="none", pad=0.8))
for D, x, y in zip(Ds, kb, f1):
    dy = 8.0 if D not in (C.RFF_D, 512) else -11.0
    a2.annotate(f"{D}", (x, y), textcoords="offset points", xytext=(0, dy),
                fontsize=5.4, ha="center", color="0.35")
a2.set_xscale("log"); a2.set_xlim(1.3, 4e4); a2.set_ylim(0.545, 0.672)
a2.set_xlabel("Adaptation state (KB)"); a2.set_ylabel(r"Mean $F_1$ over 48 domains")
a2.legend(loc="lower right")
a2.text(.965, .92, "(b)", transform=a2.transAxes, fontsize=7, weight="bold", ha="right")
fig.savefig(f"{C.FIGS}/fig3_memory.pdf"); plt.close(fig)


# ---------------------------------------------------------------- Fig 4: per-domain BWT
rep = json.load(open(f"{C.RESULTS}/E2_summary.json"))["mean_S_bar"]
fig, ax = plt.subplots(figsize=(3.35, 1.95))
for sc in SC:
    curves = []
    for f in sorted(glob.glob(f"{C.RESULTS}/E1_ours_{sc}_s*_D{C.RFF_D}.json")):
        M = np.array(json.load(open(f))["matrix_f1"]); T = M.shape[0]
        curves.append([np.nanmean([M[t, i] - M[i, i] for i in range(t)]) for t in range(1, T)])
    m = np.mean(curves, 0)
    c, ls = STY[sc]
    ax.plot(np.arange(2, len(m) + 2), m, color=c, ls=ls, label=LBL[sc])
ax.axhline(rep, color=REF, ls=(0, (5, 2)), lw=0.9)
ax.axhline(-0.06, color="0.40", ls=(0, (2.5, 2)), lw=0.9)
ax.axhline(0, color="0.55", lw=0.5)
ax.set_xlim(1, 49); ax.set_ylim(-0.155, 0.035)
ax.set_xticks([10, 20, 30, 40])
ax.set_yticks([0.0, -0.05, -0.10, -0.15])
ax.set_xlabel("Domain Index"); ax.set_ylabel(r"$\mathrm{BWT}_t$ ($F_1$)")
ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.005),
          frameon=False, borderaxespad=0.0)
fig.savefig(f"{C.FIGS}/fig4_bwt.pdf"); plt.close(fig)

print("written:")
for f in sorted(glob.glob(f"{C.FIGS}/*.pdf")):
    print("  ", os.path.basename(f), f"{os.path.getsize(f)/1024:.0f} KB")
