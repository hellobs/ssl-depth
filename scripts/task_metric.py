"""任务级评估：把深度误差翻译成避障决策。

像素误差（δ1、AbsRel）非技术看不懂，但它有个等价说法：
真值说前面 7 米该刹车，模型看出来是 12 米于是没刹，这就是漏判，也是撞机的直接原因。

前方最近障碍距离取中心区域的低分位数（不是最小值，避免单个噪点）：
  横向中心 1/3、纵向中间 1/2，取 5% 分位。
阈值默认 9 米，与 AvoidController 的硬停距离一致。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "D:/zzr/ssl-depth/scripts")
from diag_backbone import (CACHE, DepthHead, Frames, JepaBackbone,  # noqa: E402
                           ResnetBackbone, split)

THRESHOLDS = [6.0, 9.0, 12.0, 15.0, 20.0]


def front_near(depth: np.ndarray) -> float:
    h, w = depth.shape
    roi = depth[h // 4: 3 * h // 4, w // 3: 2 * w // 3]
    roi = roi[np.isfinite(roi) & (roi > 0)]
    return float(np.percentile(roi, 5)) if roi.size else float("nan")


def build(backbone: str, run_dir: str):
    dev = "cuda"
    torch.cuda.set_per_process_memory_fraction(0.4, 0)
    if backbone in ("imagenet", "resnet_scratch"):
        bb = ResnetBackbone(backbone == "imagenet").to(dev)
    else:
        from diag_backbone import CKPTS
        bb = JepaBackbone(None if backbone == "scratch" else CKPTS[backbone]).to(dev)
    head = DepthHead(192).to(dev)
    sd = torch.load(Path(run_dir) / "head.pt", map_location="cpu", weights_only=False)
    head.load_state_dict(sd["head"])
    if "proj" in sd and hasattr(bb, "proj"):
        bb.load_state_dict(sd["proj"], strict=False)
    bb.eval(); head.eval()
    return bb, head, dev


@torch.no_grad()
def collect(bb, head, dev, idx, eps):
    dl = DataLoader(Frames(idx, eps), batch_size=8, num_workers=0)
    gt_near, pr_near = [], []
    for rgb, disp_gt, mask in dl:
        rgb = rgb.to(dev)
        tok, th, tw = bb(rgb.unsqueeze(1))
        disp = head(tok, th, tw)
        pred_d = 1.0 / torch.clamp(disp.squeeze(1), min=1e-3)
        gt_d = 1.0 / torch.clamp(disp_gt.squeeze(1), min=1e-3)
        m = mask.squeeze(1) > 0.5
        for j in range(pred_d.shape[0]):
            g = gt_d[j].cpu().numpy(); p = pred_d[j].cpu().numpy(); mm = m[j].cpu().numpy()
            g = np.where(mm, g, np.nan); p = np.where(mm, p, np.nan)
            gt_near.append(front_near(g)); pr_near.append(front_near(p))
    return np.array(gt_near), np.array(pr_near)


def main():
    idx = json.loads(CACHE.read_text(encoding="utf-8"))
    tr, va, te = split(idx, 0)
    eps_all = sorted(idx["kept"])
    print("全量 %d 趟 %d 帧"
          % (len(eps_all), sum(len(idx["kept"][e]) for e in eps_all)), flush=True)

    for backbone, run_dir in [("imagenet", "runs/v2_imagenet"),
                              ("resnet_scratch", "runs/v2_resnet_scratch"),
                              ("jepa", "runs/v2_jepa"),
                              ("scratch", "runs/v2_scratch")]:
        if not Path(run_dir).exists():
            print("跳过", backbone, "(缺少", run_dir, ")"); continue
        bb, head, dev = build(backbone, run_dir)
        g, p = collect(bb, head, dev, idx, eps_all)
        ok = np.isfinite(g) & np.isfinite(p)
        g, p = g[ok], p[ok]
        print("\n=== %s (n=%d 帧) ===" % (backbone, g.size))
        print("  真值前方距离：中位 %.1fm  5%%分位 %.1fm  <9m 占 %.0f%%"
              % (np.median(g), np.percentile(g, 5), 100 * (g < 9).mean()))
        print("  预测前方距离：中位 %.1fm  5%%分位 %.1fm  <9m 占 %.0f%%"
              % (np.median(p), np.percentile(p, 5), 100 * (p < 9).mean()))
        print("  绝对误差：中位 %.2fm  均值 %.2fm"
              % (np.median(np.abs(p - g)), np.abs(p - g).mean()))
        print("  %5s %8s %8s %8s %8s %10s" %
              ("阈值", "该刹且刹", "漏判", "误刹", "正确放行", "该刹识别率"))
        rec = {}
        for t in THRESHOLDS:
            dg, dp = g < t, p < t
            a = int((dg & dp).sum()); b = int((dg & ~dp).sum())
            c = int((~dg & dp).sum()); d = int((~dg & ~dp).sum())
            r = a / max(1, a + b)
            print("  %5.0f %8d %8d %8d %8d %9.1f%%" % (t, a, b, c, d, 100 * r))
            rec[t] = {"hit": a, "miss": b, "false": c, "pass": d, "recall": r}
        out = Path(run_dir) / "task_metric.json"
        out.write_text(json.dumps({"backbone": backbone, "n": int(g.size),
                                   "gt_median": float(np.median(g)),
                                   "pr_median": float(np.median(p)),
                                   "by_threshold": rec}, ensure_ascii=False, indent=1),
                       encoding="utf-8")


if __name__ == "__main__":
    main()
