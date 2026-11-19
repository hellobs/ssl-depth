"""论文配图：精度对比、避障决策、深度示例。

输出到 paper/figs/。三张图对应正文的三处结论：
  fig1 五种表征的精度对比
  fig2 避障决策正确率
  fig3 深度预测示例（原图 / 真值 / 预测）
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path("D:/zzr/ssl-depth")
FIGS = ROOT / "paper" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

NAME = {
    "imagenet": "ImageNet\n有监督预训练",
    "resnet_scratch": "ResNet18\n随机初始化",
    "jepa": "JEPA 自监督\n航拍数据",
    "scratch": "ViT\n随机初始化",
}
ORDER = ["imagenet", "resnet_scratch", "jepa", "scratch"]
COLORS = ["#c0392b", "#e08a7a", "#2e86c1", "#95a5a6"]


def load(backbone: str, kind: str):
    p = ROOT / "runs" / ("v2_%s" % backbone) / kind
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def fig1():
    names, d1, absrel, near = [], [], [], []
    for b in ORDER:
        m = load(b, "metrics.json")
        if m is None:
            continue
        names.append(NAME[b]); t = m["test"]
        d1.append(t["d1"]); absrel.append(t["absrel_raw"])
        near.append(json.loads((ROOT / "runs" / ("v2_%s" % b) / "task_metric.json")
                               .read_text(encoding="utf-8"))["pr_median"]
                    if (ROOT / "runs" / ("v2_%s" % b) / "task_metric.json").exists() else np.nan)
    if not names:
        print("fig1 跳过：还没有 v2 结果"); return
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].bar(x, d1, color=COLORS[:len(names)])
    axes[0].set_ylabel("δ1（越高越好）"); axes[0].set_ylim(0, 1)
    axes[0].set_title("像素级精度")
    axes[0].axhline(0.5, ls="--", lw=1, c="gray")
    for i, v in enumerate(d1):
        axes[0].text(i, v + 0.02, "%.3f" % v, ha="center", fontsize=9)
    axes[1].bar(x, near, color=COLORS[:len(names)])
    axes[1].set_ylabel("预测前方距离中位数（米）")
    axes[1].axhline(29.7, ls="--", lw=1, c="k")
    axes[1].text(len(names) - 0.5, 30.5, "真值 29.7 米", ha="right", fontsize=9)
    axes[1].set_title("米制距离的绝对准确度")
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGS / "fig1_backbone.png", dpi=180)
    print("fig1 完成", len(names), "组")


def fig2():
    names, rec, mis = [], [], []
    for b in ORDER:
        t = load(b, "task_metric.json")
        if t is None:
            continue
        names.append(NAME[b])
        r = t["by_threshold"]["9.0"]
        rec.append(r["recall"] * 100)
        tot_miss = r["miss"] + r["hit"]
        mis.append(100.0 * r["miss"] / max(1, tot_miss))
    if not names:
        print("fig2 跳过"); return
    x = np.arange(len(names)); w = 0.38
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar(x - w / 2, rec, w, label="该刹车时判断正确", color="#c0392b")
    ax.bar(x + w / 2, mis, w, label="该刹车却漏判（危险）", color="#7f8c8d")
    ax.set_ylabel("百分比（%）"); ax.set_ylim(0, 100)
    ax.set_title("9 米刹车区间内的决策正确率")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    for i, (a, b_) in enumerate(zip(rec, mis)):
        ax.text(i - w / 2, a + 1.5, "%.0f" % a, ha="center", fontsize=9)
        ax.text(i + w / 2, b_ + 1.5, "%.0f" % b_, ha="center", fontsize=9)
    fig.tight_layout(); fig.savefig(FIGS / "fig2_task.png", dpi=180)
    print("fig2 完成")


def fig3(n=3):
    """原图 / 真值 / 预测 三联图。用 test 集里前方距离最近的帧，最能看出差别。"""
    import cv2
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from diag_backbone import CACHE, DepthHead, Frames, ResnetBackbone, split

    run = ROOT / "runs" / "v2_imagenet"
    if not (run / "head.pt").exists():
        print("fig3 跳过：缺少 v2_imagenet/head.pt"); return
    dev = "cuda"
    bb = ResnetBackbone(True).to(dev)
    head = DepthHead(192).to(dev)
    sd = torch.load(run / "head.pt", map_location="cpu", weights_only=False)
    head.load_state_dict(sd["head"])
    if "proj" in sd:
        bb.load_state_dict(sd["proj"], strict=False)
    bb.eval(); head.eval()

    idx = json.loads(CACHE.read_text(encoding="utf-8"))
    _, _, te = split(idx, 0)
    dl = torch.utils.data.DataLoader(Frames(idx, te), batch_size=8, num_workers=0)

    from task_metric import front_near
    cand = []
    with torch.no_grad():
        for rgb, disp_gt, mask in dl:
            tok, th, tw = bb(rgb.to(dev).unsqueeze(1))
            disp = head(tok, th, tw)
            pd = (1.0 / torch.clamp(disp.squeeze(1), min=1e-3)).cpu().numpy()
            gd = (1.0 / torch.clamp(disp_gt.squeeze(1), min=1e-3)).numpy()
            mm = (mask.squeeze(1) > 0.5).numpy()
            for j in range(pd.shape[0]):
                g = np.where(mm[j], gd[j], np.nan)
                p = np.where(mm[j], pd[j], np.nan)
                cand.append((front_near(g), rgb[j], g, p))
    cand.sort(key=lambda z: z[0])
    picks = [cand[i] for i in np.linspace(0, len(cand) - 1, n).astype(int)]

    fig, axes = plt.subplots(n, 3, figsize=(9, 3.1 * n))
    vmax = 60
    for r, (near, rgb, g, p) in enumerate(picks):
        img = (rgb.permute(1, 2, 0).numpy() * 0.5 + 0.5).clip(0, 1)
        axes[r, 0].imshow(img); axes[r, 0].set_title("前视图像", fontsize=10)
        for ax, d, ttl in ((axes[r, 1], g, "真实距离"), (axes[r, 2], p, "模型预测")):
            im = ax.imshow(np.clip(d, 0, vmax), cmap="inferno", vmin=0, vmax=vmax)
            ax.set_title("%s  前方最近 %.1f 米" % (ttl, near if ttl == "真实距离" else front_near(p)),
                         fontsize=10)
        for ax in axes[r]:
            ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="距离（米）")
    fig.savefig(FIGS / "fig3_depth.png", dpi=180, bbox_inches="tight")
    print("fig3 完成")


if __name__ == "__main__":
    fig1(); fig2(); fig3()
