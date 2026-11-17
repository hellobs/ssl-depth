"""论文配图，统一视觉规范。"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 9

ROOT = Path("D:/zzr/ssl-depth")
OUT = ROOT / "paper" / "figs"
OUT.mkdir(parents=True, exist_ok=True)

WARM = "#a8452f"
COOL = "#3a6b8c"
MID = "#9fb0c0"
NEUTRAL = "#8a8f96"
INK = "#22262b"


def bare(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#c8ccd2")
    ax.tick_params(colors=NEUTRAL, labelsize=8)


def fig_distance():
    d = json.loads((ROOT / "runs/v2_imagenet/error_analysis.json").read_text(encoding="utf-8"))
    bins = list(json.loads((ROOT / "runs/v2_imagenet/error_analysis.json").read_text(encoding="utf-8"))["by_bin"].keys())
    d1 = [d["by_bin"][b]["d1"] for b in bins]
    n = [d["by_bin"][b]["n"] for b in bins]
    bad = [d["by_bin"][b]["frac_bad"] for b in bins]

    fig, ax = plt.subplots(figsize=(6.2, 3.1), dpi=200)
    x = np.arange(len(bins))
    bars = ax.bar(x, d1, width=0.56, color=COOL, zorder=3)
    for i, (b, v, nn, bd) in enumerate(zip(bars, d1, n, bad)):
        ax.text(i, v + 0.02, "%.3f" % v, ha="center", va="bottom", fontsize=9,
                color=INK, fontweight="bold")
        ax.text(i, 0.05, "n=%d" % nn, ha="center", va="bottom", fontsize=7.5, color="white")
        if bd > 0.25:
            ax.text(i, v / 2, "%.0f%%\n低于 0.5" % (100 * bd), ha="center", va="center",
                    fontsize=7.5, color="white", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["%s m" % b for b in bins], color=INK, fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("δ1", color=INK, fontsize=9)
    ax.axhline(0.5, color=NEUTRAL, lw=0.8, ls=(0, (4, 3)), zorder=1)
    bare(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig_distance.png", bbox_inches="tight")
    print("fig_distance ok")


def fig_per_flight():
    d = json.loads((ROOT / "runs/v2_imagenet/error_analysis.json").read_text(encoding="utf-8"))
    items = sorted(d["by_ep"].items(), key=lambda kv: kv[1]["absrel"])
    names = [k.split("/")[0][-8:] for k, _ in items]
    vals = [v["absrel"] for _, v in items]

    fig, ax = plt.subplots(figsize=(6.2, 3.2), dpi=200)
    y = np.arange(len(vals))
    colors = [COOL if v < 0.10 else (WARM if v > 0.30 else MID) for v in vals]
    ax.barh(y, vals, height=0.62, color=colors, zorder=3)
    for i, v in enumerate(vals):
        ax.text(v + 0.004, i, "%.3f" % v, va="center", fontsize=7.5,
                color=WARM if v > 0.30 else INK)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7, color=NEUTRAL)
    ax.set_xlim(0, 0.38)
    ax.set_xlabel("绝对相对误差", color=INK, fontsize=9)
    ax.invert_yaxis()
    bare(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig_per_flight.png", bbox_inches="tight")
    print("fig_per_flight ok")


def fig_confusion():
    d = json.loads((ROOT / "runs/v2_imagenet/task_metric.json").read_text(encoding="utf-8"))
    r = d["by_threshold"]["9.0"]
    cells = [[r["hit"], r["miss"]], [r["false"], r["pass"]]]
    labels = [["该悬停 已悬停", "该悬停 未悬停"], ["不该悬停 却悬停", "不该悬停 未悬停"]]
    face = np.array([[0.0, 1.0], [1.0, 0.0]])
    cmap = matplotlib.colors.ListedColormap(["#e2e8ee", "#ecd0c8"])

    fig, ax = plt.subplots(figsize=(4.8, 3.2), dpi=200)
    ax.imshow(face, cmap=cmap, vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            bad = (i == 0 and j == 1) or (i == 1 and j == 0)
            ax.text(j, i - 0.13, labels[i][j], ha="center", va="center",
                    fontsize=7.5, color=NEUTRAL)
            ax.text(j, i + 0.20, "%d" % int(cells[i][j]), ha="center", va="center",
                    fontsize=16, color=WARM if bad else INK, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_confusion.png", bbox_inches="tight")
    print("fig_confusion ok")


if __name__ == "__main__":
    fig_distance(); fig_per_flight(); fig_confusion()
