"""逐航段诊断：把测试集拆开看，定位整体 AbsRel 被谁拖累。

背景：evaluate() 的 absrel_med 用「全集 p/g 中位数」作单一尺度对齐，
所以测试集里只要有一趟的尺度偏差不同，全局尺度就会被拉偏，所有帧一起受损。
δ1 不受影响（逐像素比值），这就是 δ1 稳定而 AbsRel 翻倍的原因。

用法：
  python scripts/diag_per_episode.py --run runs/v4_jepa_domain_gpu --split test
"""
from __future__ import annotations

# --- 路径解析（不写死本机绝对路径）---
import os as _os
from pathlib import Path as _Path
REPO = str(_Path(__file__).resolve().parents[1])
DATA = str(_Path(_os.environ.get("WM_DATA_ROOT", _Path(__file__).resolve().parents[2] / "wmdata")))
MICROW = str(_Path(_os.environ.get("WM_CODE_ROOT", _Path(__file__).resolve().parents[2] / "microwm")))
# ---

import argparse
import json
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, f"{REPO}/scripts")
sys.path.insert(0, f"{MICROW}/src")

from diag_backbone import (CACHE, CKPTS, DepthHead, Frames, JepaBackbone,  # noqa: E402
                           ResnetBackbone, load_split)


def met(p, g):
    s = float(np.median(p / g))
    ratio = np.maximum(p / g, g / p)
    return {"n": int(p.size), "absrel_raw": float(np.mean(np.abs(p - g) / g)),
            "absrel_med": float(np.mean(np.abs(p / s - g) / g)), "scale": s,
            "d1": float((ratio < 1.25).mean()),
            "rmse": float(np.sqrt(np.mean((p - g) ** 2))),
            "gt_med": float(np.median(g))}


def collect(bb, head, dl, dev):
    head.eval()
    ps, gs = [], []
    with torch.no_grad():
        for rgb, disp_gt, mask in dl:
            rgb, mask = rgb.to(dev), mask.to(dev)
            tok, th, tw = bb(rgb.unsqueeze(1))
            disp = head(tok, th, tw)
            pd = 1.0 / torch.clamp(disp.squeeze(1), min=1e-3)
            gd = 1.0 / torch.clamp(disp_gt.squeeze(1).to(dev), min=1e-3)
            m = mask.squeeze(1) > 0.5
            ps.append(pd[m].cpu().numpy())
            gs.append(gd[m].cpu().numpy())
    return np.concatenate(ps), np.concatenate(gs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="结果目录，如 runs/v4_jepa_domain_gpu")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--backbone", default="jepa_domain")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0,
                    help="随机初始化骨干必须用与训练相同的种子重建，否则权重对不上")
    a = ap.parse_args()

    dev = a.device
    torch.manual_seed(a.seed)   # 必须早于骨干构造：resnet_scratch 的权重靠种子复现
    idx = json.loads(CACHE.read_text(encoding="utf-8"))
    tr, va, te = load_split(idx)
    eps = te if a.split == "test" else va

    if a.backbone in ("imagenet", "resnet_scratch"):
        bb = ResnetBackbone(a.backbone == "imagenet").to(dev)
    else:
        bb = JepaBackbone(CKPTS[a.backbone]).to(dev)
    head = DepthHead(192).to(dev)
    sd = torch.load("%s/head.pt" % a.run, map_location="cpu", weights_only=False)
    head.load_state_dict(sd["head"])
    bb.load_state_dict(sd["proj"], strict=False)
    bb.eval()

    per = []
    for e in eps:
        dl = DataLoader(Frames(idx, [e]), batch_size=8, num_workers=0)
        p, g = collect(bb, head, dl, dev)
        if p.size:
            per.append((e, met(p, g)))
    # 全集口径（与 evaluate 等价）
    dl = DataLoader(Frames(idx, eps), batch_size=8, num_workers=0)
    p, g = collect(bb, head, dl, dev)
    glob = met(p, g)

    print("%s 全集: n_px %d | absrel_raw %.3f absrel_med %.3f | scale %.3f | d1 %.3f | rmse %.2f"
          % (a.split, glob["n"], glob["absrel_raw"], glob["absrel_med"], glob["scale"],
             glob["d1"], glob["rmse"]))
    print()
    print("%-46s %7s %7s %7s %7s %7s" % ("航段", "帧px", "AbsRel", "尺度", "d1", "真值中位"))
    for e, m in sorted(per, key=lambda x: -x[1]["absrel_med"]):
        print("%-46s %7d %7.3f %7.3f %7.3f %7.2f"
              % (e, m["n"], m["absrel_med"], m["scale"], m["d1"], m["gt_med"]))
    worst = max(per, key=lambda x: x[1]["absrel_med"])
    print()
    print("最差航段 %s AbsRel %.3f，占全集像素 %.1f%%"
          % (worst[0], worst[1]["absrel_med"], 100.0 * worst[1]["n"] / glob["n"]))
    # 去掉最差一趟后的全集
    keep = [x for x in per if x[0] != worst[0]]
    print("去掉该趟后（粗估，按像素加权）：")
    tot = sum(x[1]["n"] for x in keep)
    print("  AbsRel %.3f | d1 %.3f"
          % (sum(x[1]["absrel_med"] * x[1]["n"] for x in keep) / tot,
             sum(x[1]["d1"] * x[1]["n"] for x in keep) / tot))


if __name__ == "__main__":
    main()
