"""论文配图 v2：A4 单栏、白底、无网格、统一学术视觉规范。

fig_distance.png   按真值中位距离分档的 delta_1 均值 + 低质量帧占比
fig_per_flight.png 各架次的绝对相对误差
fig_confusion.png  9 米判据下的 2x2 决策矩阵（格内横条长度 = 样本量的对数刻度）

SimHei 只有常规字重，全部文字不加粗，层级靠字号与颜色深浅区分。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"
plt.rcParams["font.size"] = 9
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"

ROOT = Path("D:/zzr/ssl-depth")
OUT = ROOT / "paper" / "figs"
DATA = ROOT / "runs" / "v4_imagenet"
TMF = DATA / "task_metric_held.json"   # 留出集任务指标
OUT.mkdir(parents=True, exist_ok=True)

INK = "#1a1d21"        # 主文字
SUB = "#3a4046"        # 次级文字
AXIS = "#4d545b"       # 轴线
FAINT = "#93a0ab"      # 参考线
BLUE = "#3a6b8c"       # 判对
BLUE_T = "#e0eaf2"     # 判对底色
WARM = "#a8452f"       # 判错
WARM_T = "#f7e1d6"     # 判错底色
TRACK = "#dde3e8"      # 横条轨道
SOFT = "#a8bece"       # 弱化柱

DPI = 300
W = 6.2


def bare(ax: plt.Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.7)
    ax.tick_params(axis="both", which="major", length=3.0, width=0.7,
                   color=AXIS, labelcolor=SUB, labelsize=8, pad=2.5)


# --------------------------------------------------------------------------
# 图 1：距离分档
# --------------------------------------------------------------------------
def fig_distance() -> None:
    d = json.loads((DATA / "error_analysis.json").read_text(encoding="utf-8"))
    bins = list(d["by_bin"].keys())
    d1 = [d["by_bin"][b]["d1"] for b in bins]
    n = [d["by_bin"][b]["n"] for b in bins]
    bad = [d["by_bin"][b]["frac_bad"] for b in bins]

    x = np.arange(len(bins))
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(W, 4.0), sharex=True,
        gridspec_kw={"height_ratios": [1.65, 1.0], "hspace": 0.13})

    # (上) 各档 delta_1 均值
    ax1.bar(x, d1, width=0.50, color=BLUE, zorder=3)
    for xi, v in zip(x, d1):
        ax1.text(xi, v + 0.025, "%.3f" % v, ha="center", va="bottom",
                 fontsize=9, color=INK)
    ref = ax1.axhline(0.5, color=FAINT, lw=0.9, ls=(0, (5, 3)), zorder=1,
                      label="参考线 $\\delta_1=0.5$（低质量帧界线）")
    ax1.legend(handles=[ref], loc="upper left", bbox_to_anchor=(0.0, 1.02),
               frameon=False, fontsize=7.5, handlelength=1.5,
               handletextpad=0.6, borderpad=0.15, labelspacing=0.3)
    ax1.set_ylim(0, 1.0)
    ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax1.set_xlim(-0.62, 3.62)
    ax1.set_ylabel("$\\delta_1$ 均值", color=INK, fontsize=9, labelpad=5)
    bare(ax1)
    ax1.spines["bottom"].set_visible(False)
    ax1.tick_params(axis="x", length=0, labelbottom=False)

    # (下) 各档中 delta_1 低于 0.5 的帧所占比例
    ax2.bar(x, [100 * v for v in bad], width=0.50, color=SOFT, zorder=3)
    for xi, v in zip(x, bad):
        ax2.text(xi, 100 * v + 1.0, "%.0f%%" % (100 * v), ha="center",
                 va="bottom", fontsize=8.5, color=INK)
    ax2.set_ylim(0, 40)
    ax2.set_yticks([0, 10, 20, 30, 40])
    ax2.set_xlim(-0.62, 3.62)
    ax2.set_ylabel("低质量帧占比（%）", color=INK, fontsize=9, labelpad=5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(["%s 米\nn=%d" % (b, nn) for b, nn in zip(bins, n)],
                        fontsize=8.5, color=INK)
    bare(ax2)
    ax2.tick_params(axis="x", length=0, pad=4)

    fig.subplots_adjust(left=0.105, right=0.985, top=0.955, bottom=0.165)
    fig.savefig(OUT / "fig_distance.png", dpi=DPI)
    print("fig_distance ok")


# --------------------------------------------------------------------------
# 图 2：逐架次误差
# --------------------------------------------------------------------------
def flight_label(key: str) -> str:
    """run_20261117_112646_46bcd8c3/episode_0001 -> '11:26 第2趟'。"""
    run, ep = key.split("/")
    stamp = run.split("_")[2]
    idx = int(ep.split("_")[1]) + 1
    return "%s:%s 第%d趟" % (stamp[0:2], stamp[2:4], idx)


def fig_per_flight() -> None:
    """error_analysis.json 只存了最好 4 趟与最差 4 趟，中间留一道断口。"""
    d = json.loads((DATA / "error_analysis.json").read_text(encoding="utf-8"))
    items = sorted(d["by_ep"].items(), key=lambda kv: kv[1]["absrel"])
    labels = [flight_label(k) for k, _ in items]
    vals = [v["absrel"] for _, v in items]
    ns = [v["n"] for _, v in items]
    half = len(vals) // 2

    def bucket(v: float) -> str:
        return BLUE if v < 0.10 else (SOFT if v <= 0.30 else WARM)

    # 中间留一道空隙，标出省略的航段（用行距表示，不画符号）
    pos = [float(i) if i < half else i + 0.85 for i in range(len(vals))]
    fig, ax = plt.subplots(figsize=(W, 3.35))
    ax.barh(pos, vals, height=0.62, color=[bucket(v) for v in vals], zorder=3)
    for y, v in zip(pos, vals):
        ax.text(v + 0.007, y, "%.3f" % v, va="center", ha="left",
                fontsize=8.5, color=INK)
    ax.set_yticks(pos)
    ax.set_yticklabels(["%s\nn=%d" % (l, nn) for l, nn in zip(labels, ns)],
                       fontsize=7.2, color=INK)
    ax.set_ylim(pos[-1] + 0.62, -0.72)
    ax.set_xlim(0, 0.465)
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4])
    ax.set_xlabel("绝对相对误差", color=INK, fontsize=9, labelpad=5)
    bare(ax)
    ax.tick_params(axis="y", length=0, pad=3)

    handles = [Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, SOFT, WARM)]
    fig.legend(handles, ["低（< 0.10）", "中（0.10-0.30）", "高（> 0.30）"],
               loc="upper right", bbox_to_anchor=(0.985, 0.995), ncol=3,
               frameon=False, fontsize=7.5, handlelength=0.9,
               handleheight=0.7, handletextpad=0.5, columnspacing=1.4,
               borderpad=0.1)

    fig.subplots_adjust(left=0.165, right=0.985, top=0.915, bottom=0.145)
    fig.savefig(OUT / "fig_per_flight.png", dpi=DPI)
    print("fig_per_flight ok")


# --------------------------------------------------------------------------
# 图 3：2x2 决策矩阵
# --------------------------------------------------------------------------
def fig_confusion() -> None:
    d = json.loads(TMF.read_text(encoding="utf-8"))
    r = d["by_threshold"]["9.0"]
    hit, miss, false, pas = r["hit"], r["miss"], r["false"], r["pass"]
    total = hit + miss + false + pas
    recall = hit / (hit + miss)
    precision = hit / (hit + false)
    lmax = math.log10(1 + max(hit, miss, false, pas))

    cells = {                       # (列, 行) -> (数量, 判对?, 说明)
        (0, 0): (hit, True, "占该类 %.1f%%" % (100 * recall)),
        (1, 0): (miss, False, "占该类 %.1f%%，漏判" % (100 * (1 - recall))),
        (0, 1): (false, False, "占该类 %.1f%%，误报"
                 % (100 * false / (false + pas))),
        (1, 1): (pas, True, "占该类 %.1f%%" % (100 * pas / (false + pas))),
    }

    fig = plt.figure(figsize=(W, 3.3))
    side = 0.4135                     # 方格边长（figure 比例，保证为正方形）
    ax = fig.add_axes([0.263, 0.115, side, side * W / 3.3])
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_aspect("equal")
    ax.axis("off")

    for (c, rw), (n, ok, note) in cells.items():
        y0 = 1 - rw
        tint = BLUE_T if ok else WARM_T
        bar_c = BLUE if ok else WARM
        ax.add_patch(Rectangle((c, y0), 1, 1, facecolor=tint,
                               edgecolor="white", linewidth=2.8, zorder=2))
        ax.text(c + 0.5, y0 + 0.665, "%d" % n, ha="center", va="center",
                fontsize=21, color=INK, zorder=4)
        ax.text(c + 0.5, y0 + 0.445, note, ha="center", va="center",
                fontsize=8, color=SUB, zorder=4)
        # 格内横条：长度按对数刻度映射样本量
        ax.add_patch(Rectangle((c + 0.115, y0 + 0.205), 0.77, 0.075,
                               facecolor=TRACK, edgecolor="none", zorder=3))
        w = 0.77 * math.log10(1 + n) / lmax
        if w > 0:
            ax.add_patch(Rectangle((c + 0.115, y0 + 0.205), max(w, 0.008),
                                   0.075, facecolor=bar_c, edgecolor="none",
                                   zorder=4))

    ax.add_patch(Rectangle((0, 0), 2, 2, facecolor="none", edgecolor=AXIS,
                           linewidth=0.8, zorder=5))

    ax.text(0.5, 2.06, "判为悬停", ha="center", va="bottom", fontsize=9,
            color=INK)
    ax.text(1.5, 2.06, "判为未悬停", ha="center", va="bottom", fontsize=9,
            color=INK)
    for rw, (title, cnt) in enumerate([("实际应悬停", hit + miss),
                                       ("实际不应悬停", false + pas)]):
        yc = 1 - rw + 0.5
        ax.text(-0.06, yc + 0.075, title, ha="right", va="center", fontsize=9,
                color=INK)
        ax.text(-0.06, yc - 0.105, "%d 帧" % cnt, ha="right", va="center",
                fontsize=8, color=SUB)

    # 右侧指标
    tx = 0.735
    fig.text(tx, 0.725, "阈值 9 米", fontsize=8, color=SUB, ha="left")
    fig.add_artist(plt.Line2D([tx, 0.99], [0.688, 0.688], color="#d5dade",
                              lw=0.7))
    fig.text(tx, 0.625, "召回率", fontsize=8.5, color=SUB, ha="left")
    fig.text(0.99, 0.625, "%.1f%%" % (100 * recall), fontsize=8.5, color=INK,
             ha="right")
    fig.text(tx, 0.545, "精确率", fontsize=8.5, color=SUB, ha="left")
    fig.text(0.99, 0.545, "%.0f%%" % (100 * precision), fontsize=8.5,
             color=INK, ha="right")
    fig.text(tx, 0.465, "样本量", fontsize=8.5, color=SUB, ha="left")
    fig.text(0.99, 0.465, "%d 帧" % total, fontsize=8.5, color=INK, ha="right")
    fig.text(tx, 0.345, "格内横条长度按\n对数刻度表示样本量", fontsize=7.5,
             color=SUB, ha="left", va="top", linespacing=1.5)

    fig.savefig(OUT / "fig_confusion.png", dpi=DPI)
    print("fig_confusion ok")


# --------------------------------------------------------------------------
# 图 4：阈值—召回—误触发
# --------------------------------------------------------------------------
def fig_threshold() -> None:
    d = json.loads(TMF.read_text(encoding="utf-8"))
    ts = sorted(d["by_threshold"], key=float)
    rec = [100 * d["by_threshold"][t]["recall"] for t in ts]
    fal = [d["by_threshold"][t]["false"] for t in ts]
    x = np.arange(len(ts))

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(W, 3.7), sharex=True,
        gridspec_kw={"height_ratios": [1.5, 1.0], "hspace": 0.16})

    ax1.plot(x, rec, color=WARM, lw=1.6, marker="o", ms=4.5, zorder=5)
    for xi, v in zip(x, rec):
        ax1.text(xi, v + 2.2, "%.1f%%" % v, ha="center", va="bottom",
                 fontsize=8.5, color=INK, zorder=6)
    ax1.set_ylim(45, 108)
    ax1.set_yticks([50, 60, 70, 80, 90, 100])
    ax1.set_ylabel("该触发帧的识别率（%）", color=INK, fontsize=9, labelpad=6)
    bare(ax1)
    ax1.spines["bottom"].set_visible(False)
    ax1.tick_params(axis="x", length=0, labelbottom=False)

    ax2.bar(x, fal, width=0.44, color=TRACK, zorder=3)
    for xi, v in zip(x, fal):
        ax2.text(xi, v + 2.0, "%d" % v, ha="center", va="bottom",
                 fontsize=8, color=SUB)
    ax2.set_ylim(0, 125)
    ax2.set_yticks([0, 40, 80, 120])
    ax2.set_ylabel("误触发帧数", color=INK, fontsize=9, labelpad=6)
    ax2.set_xticks(x)
    ax2.set_xticklabels(["%g 米" % float(t) for t in ts], fontsize=8.5,
                        color=INK)
    ax2.tick_params(axis="x", length=0, pad=4)
    bare(ax2)

    for a in (ax1, ax2):
        a.axvline(1, color=FAINT, lw=0.9, ls=(0, (5, 3)), zorder=1)
    ax1.text(1.08, 102, "设计判据 9 米", fontsize=7.5, color=SUB,
             ha="left", va="center")

    fig.subplots_adjust(left=0.135, right=0.985, top=0.955, bottom=0.155)
    fig.savefig(OUT / "fig_threshold.png", dpi=DPI)
    print("fig_threshold ok")


if __name__ == "__main__":
    fig_distance()
    fig_per_flight()
    fig_confusion()
    fig_threshold()
