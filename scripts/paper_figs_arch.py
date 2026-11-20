# -*- coding: utf-8 -*-
"""生成论文的架构示意图：Transformer、JEPA、残差块、本文的部署链路。

横向布局，宋体（与正文一致），字号放大。输出 300 dpi PNG 到 paper/figs/。
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["font.family"] = "SimSun"          # 宋体，与正文 ctex 一致
plt.rcParams["font.sans-serif"] = ["SimSun"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path("paper/figs")
OUT.mkdir(parents=True, exist_ok=True)

BLUE = "#dce9f7"
BLUE_E = "#4a7fb5"
GRAY = "#eeeeee"
GRAY_E = "#8a8a8a"
ORANGE = "#fbe3d0"
ORANGE_E = "#c47a3d"
GREEN = "#e2f0e2"
GREEN_E = "#5a9a5a"

FS = 14          # 框内主字号
FS_S = 12        # 标注字号


def box(ax, x, y, w, h, text, fc=BLUE, ec=BLUE_E, fs=FS, lw=1.5, ls="-"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.015,rounding_size=0.06",
                                facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, linespacing=1.7)


def arrow(ax, x1, y1, x2, y2, color="#333333", lw=1.5, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=18, color=color,
                                 linewidth=lw, linestyle=ls,
                                 shrinkA=0, shrinkB=0))


def canvas(w, h, xlim, ylim):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    return fig, ax


def save(fig, name):
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- 残差块（横向）
def draw_resblock():
    fig, ax = canvas(10.0, 2.8, (0, 10.2), (0.5, 3.5))
    ys = 1.6
    ax.text(0.30, ys, "$x$", fontsize=18, ha="center", va="center")
    box(ax, 1.55, ys, 1.40, 0.90, "卷积\n$3\\times3$")
    box(ax, 3.20, ys, 1.55, 0.90, "批归一化\n+ ReLU", GRAY, GRAY_E, FS_S)
    box(ax, 4.95, ys, 1.40, 0.90, "卷积\n$3\\times3$")
    box(ax, 6.60, ys, 1.40, 0.90, "批归一化", GRAY, GRAY_E, FS_S)
    for x1, x2 in [(0.40, 0.82), (2.25, 2.40), (3.98, 4.22), (5.65, 5.87), (7.30, 7.45)]:
        arrow(ax, x1, ys, x2, ys)
    ax.plot(7.72, ys, "o", color="#333", ms=7)
    ax.text(7.72, ys + 0.45, "$+$", fontsize=19, ha="center", va="center")
    arrow(ax, 7.78, ys, 8.22, ys)
    box(ax, 8.95, ys, 1.25, 0.90, "ReLU", GRAY, GRAY_E)
    arrow(ax, 9.60, ys, 9.92, ys)
    ax.text(9.98, ys, "$y$", fontsize=18, ha="left", va="center")

    ax.plot([0.55, 0.55], [ys + 0.46, 2.95], color="#333", lw=1.5)
    ax.plot([0.55, 7.72], [2.95, 2.95], color="#333", lw=1.5)
    ax.plot([7.72, 7.72], [2.95, ys + 0.07], color="#333", lw=1.5)
    ax.text(4.2, 3.15, "直连通道", fontsize=FS_S, color="#555", ha="center")
    save(fig, "fig_resblock.png")


# ---------------------------------------------------------------- Transformer（横向）
def draw_transformer():
    fig, ax = canvas(12.0, 2.5, (0, 12.2), (0.3, 2.7))
    ys = 1.35
    ax.text(0.50, ys, "图像\n$192{\\times}108$", fontsize=FS_S, ha="center",
            va="center", linespacing=1.6)
    box(ax, 2.05, ys, 1.65, 0.95, "切块\n$6{\\times}6$ 投影")
    box(ax, 3.95, ys, 1.65, 0.95, "位置编码\n576 块")
    for x1, x2 in [(0.95, 1.20), (2.90, 3.10), (4.80, 5.05)]:
        arrow(ax, x1, ys, x2, ys)

    ax.add_patch(FancyBboxPatch((5.20, 0.50), 4.60, 1.80,
                                boxstyle="round,pad=0.02,rounding_size=0.06",
                                facecolor="white", edgecolor=GRAY_E,
                                linewidth=1.2, linestyle="--"))
    ax.text(5.40, 2.48, "编码器层 $\\times6$（只画一层）", fontsize=FS_S,
            color="#555", ha="left", va="center")
    box(ax, 6.25, ys, 1.55, 0.85, "多头\n自注意力", ORANGE, ORANGE_E, FS_S)
    box(ax, 7.90, ys, 1.25, 0.85, "残差\n+ LN", GRAY, GRAY_E, FS_S)
    box(ax, 9.35, ys, 1.05, 0.85, "前馈\n网络", ORANGE, ORANGE_E, FS_S)
    arrow(ax, 7.05, ys, 7.25, ys)
    arrow(ax, 8.55, ys, 8.80, ys)

    arrow(ax, 9.90, ys, 10.15, ys)
    box(ax, 11.05, ys, 1.55, 0.95, "表征\n$576{\\times}192$", GREEN, GREEN_E, FS_S)
    save(fig, "fig_transformer.png")


# ---------------------------------------------------------------- JEPA（横向双路）
def draw_jepa():
    fig, ax = canvas(12.0, 3.2, (0, 12.6), (0.2, 3.6))
    yt, yb = 2.60, 0.95

    ax.text(0.45, yt, "图像 $t$", fontsize=FS, ha="center", va="center")
    ax.text(0.45, yb, "图像 $t{+}1$", fontsize=FS, ha="center", va="center")
    arrow(ax, 0.95, yt, 1.25, yt)
    arrow(ax, 1.10, yb, 1.25, yb)

    box(ax, 2.40, yt, 2.00, 1.05, "上下文编码器\n（更新参数）", BLUE, BLUE_E, FS_S)
    box(ax, 2.40, yb, 2.00, 1.05, "目标编码器\n（EMA 跟随）", GREEN, GREEN_E, FS_S)
    arrow(ax, 3.42, yt, 4.15, yt)
    arrow(ax, 3.42, yb, 4.15, yb)

    box(ax, 4.80, yt, 1.05, 0.90, "$z_t$", BLUE, BLUE_E, FS)
    box(ax, 4.80, yb, 1.30, 0.90, "$z_{t+1}$", GREEN, GREEN_E, FS)
    arrow(ax, 5.35, yt, 5.70, yt)
    box(ax, 6.55, yt, 1.60, 1.05, "预测器\n（更新参数）", ORANGE, ORANGE_E, FS_S)
    arrow(ax, 7.38, yt, 7.85, yt)
    box(ax, 8.50, yt, 1.10, 0.90, "$\\hat{z}_{t+1}$", ORANGE, ORANGE_E, FS)

    box(ax, 11.15, (yt + yb) / 2, 1.70, 1.15, "损失\nL1（仅表征）", GRAY, GRAY_E, FS_S)
    arrow(ax, 9.08, yt - 0.10, 10.25, yt - 0.55, color=ORANGE_E)
    arrow(ax, 5.48, yb, 10.25, yb, color=GREEN_E, ls="--")
    ax.text(7.40, yb + 0.22, "不反传梯度", fontsize=FS_S, color=GREEN_E, ha="center")

    arrow(ax, 3.42, yt - 0.55, 2.40, yb + 0.55, color=BLUE_E, ls=":", lw=1.3)
    save(fig, "fig_jepa.png")


# ---------------------------------------------------------------- 两条回路（横向）
def draw_loops():
    fig, ax = canvas(12.0, 2.9, (0, 12.2), (0.2, 3.2))
    ys = 2.35
    ax.text(0.45, ys, "相机", fontsize=FS, ha="center", va="center")
    box(ax, 2.20, ys, 2.05, 0.95, "快回路\nMicroJEPA")
    box(ax, 4.70, ys, 1.60, 0.95, "避障判据\n（9 米）", ORANGE, ORANGE_E, FS_S)
    box(ax, 6.95, ys, 1.95, 0.95, "控制指令\n（绕行/减速）", ORANGE, ORANGE_E, FS_S)
    box(ax, 9.40, ys, 1.55, 0.95, "飞行器", GRAY, GRAY_E, FS_S)
    for x1, x2 in [(0.90, 1.15), (3.25, 3.88), (5.52, 5.95), (7.95, 8.60)]:
        arrow(ax, x1, ys, x2, ys)

    ax.text(2.20, ys - 0.72, "5.76 至 22.67 毫秒", fontsize=FS_S, color=BLUE_E, ha="center")

    # 慢回路：飞行器状态回到大模型，大模型给避障判据送航点
    yb = 0.78
    box(ax, 7.30, yb, 2.55, 0.85, "慢回路　大模型\n规划与解释", GREEN, GREEN_E, FS_S)
    ax.text(8.75, yb + 0.0, "", fontsize=FS_S)
    ax.text(7.30, yb - 0.62, "秒到分钟", fontsize=FS_S, color=GREEN_E, ha="center", va="center")

    ax.plot([9.40, 9.40], [ys - 0.50, yb], color=GREEN_E, lw=1.4, ls="--")
    arrow(ax, 9.40, yb, 8.60, yb, color=GREEN_E, ls="--")
    ax.text(9.55, 1.55, "回报状态", fontsize=FS_S, color=GREEN_E, ha="left", va="center")

    ax.plot([6.00, 4.70], [yb, yb], color=GREEN_E, lw=1.4, ls="--")
    arrow(ax, 4.70, yb, 4.70, ys - 0.50, color=GREEN_E, ls="--")
    ax.text(4.20, 1.45, "给出航点", fontsize=FS_S, color=GREEN_E, ha="center")
    save(fig, "fig_loops.png")


for fn in (draw_transformer, draw_jepa, draw_resblock, draw_loops):
    fn()
    print("已生成", fn.__name__)
