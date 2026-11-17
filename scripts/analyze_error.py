"""误差分层分析：按距离、按架次、按画面构成。

用法：python analyze_error.py --backbone imagenet
输出：runs/v3_<backbone>/error_analysis.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.utils.data as tud

sys.path.insert(0, "D:/zzr/ssl-depth/scripts")
from diag_backbone import CACHE, DepthHead, Frames, JepaBackbone, ResnetBackbone, split  # noqa

ROOT = Path("D:/zzr/ssl-depth")
BINS = [(0, 10), (10, 20), (20, 40), (40, 81)]


def build(backbone):
    dev = "cuda"
    torch.cuda.set_per_process_memory_fraction(0.35, 0)
    if backbone in ("imagenet", "resnet_scratch"):
        bb = ResnetBackbone(backbone == "imagenet").to(dev)
    else:
        from diag_backbone import CKPTS
        bb = JepaBackbone(None if backbone == "scratch" else CKPTS[backbone]).to(dev)
    head = DepthHead(192).to(dev)
    sd = torch.load(ROOT / "runs" / ("v3_" + backbone) / "head.pt",
                    map_location="cpu", weights_only=False)
    head.load_state_dict(sd["head"])
    if "proj" in sd and hasattr(bb, "proj"):
        bb.load_state_dict(sd["proj"], strict=False)
    bb.eval(); head.eval()
    return bb, head, dev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="imagenet")
    a = ap.parse_args()

    bb, head, dev = build(a.backbone)
    idx = json.loads(CACHE.read_text(encoding="utf-8"))
    tr, va, te = split(idx, 0)
    held = va + te
    ds = Frames(idx, held)
    dl = tud.DataLoader(ds, batch_size=8, num_workers=0)

    rows, k = [], 0
    with torch.no_grad():
        for rgb, disp_gt, mask in dl:
            tok, th, tw = bb(rgb.to(dev).unsqueeze(1))
            disp = head(tok, th, tw)
            pd = (1.0 / torch.clamp(disp.squeeze(1), min=1e-3)).cpu().numpy()
            gd = (1.0 / torch.clamp(disp_gt.squeeze(1), min=1e-3)).numpy()
            mm = (mask.squeeze(1) > 0.5).numpy()
            for j in range(pd.shape[0]):
                ep, stem = ds.items[k]; k += 1
                m = mm[j] & np.isfinite(gd[j])
                if m.sum() < 100:
                    continue
                p, g = pd[j][m], gd[j][m]
                arr = ((rgb[j].permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255
                       ).clip(0, 255).astype(np.uint8)
                gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
                rows.append({
                    "ep": ep, "stem": stem, "n": int(m.sum()),
                    "absrel": float(np.mean(np.abs(p - g) / g)),
                    "d1": float((np.maximum(p / g, g / p) < 1.25).mean()),
                    "rmse": float(np.sqrt(np.mean((p - g) ** 2))),
                    "gt_med": float(np.median(g)),
                    "sky": float((g > 55).mean()),
                    "tex": float(cv2.Laplacian(gray, cv2.CV_32F).var())})

    print("留出集 %d 帧 %d 趟" % (len(rows), len(set(r["ep"] for r in rows))))

    print("\n=== 按真值中位距离分档 ===")
    print("%-10s %5s %9s %7s %8s %8s" % ("区间", "帧数", "AbsRel", "δ1", "RMSE", "δ1<0.5占比"))
    out = {"n": len(rows), "by_bin": {}, "by_ep": {}, "worst": []}
    for lo, hi in BINS:
        sub = [r for r in rows if lo <= r["gt_med"] < hi]
        if not sub:
            continue
        e = np.mean([r["absrel"] for r in sub]); d = np.mean([r["d1"] for r in sub])
        m = np.mean([r["rmse"] for r in sub]); bad = np.mean([r["d1"] < 0.5 for r in sub])
        print("%-10s %5d %9.3f %7.3f %8.2f %11.0f%%"
              % ("%d-%d m" % (lo, hi), len(sub), e, d, m, 100 * bad))
        out["by_bin"]["%d-%d" % (lo, hi)] = {
            "n": len(sub), "absrel": float(e), "d1": float(d),
            "rmse": float(m), "frac_bad": float(bad)}

    print("\n=== 按架次（AbsRel 最好与最差各 4）===")
    by_ep = {}
    for r in rows:
        by_ep.setdefault(r["ep"], []).append(r)
    st = sorted(((e, float(np.mean([x["absrel"] for x in v])),
                  float(np.mean([x["d1"] for x in v])), len(v))
                 for e, v in by_ep.items()), key=lambda z: z[1])
    for e, ab, d1, n in st[:4] + st[-4:]:
        print("  %-42s n=%3d  AbsRel %.3f  δ1 %.3f" % (e, n, ab, d1))
        out["by_ep"][e] = {"n": n, "absrel": ab, "d1": d1}
    sp = [s[1] for s in st]
    print("  逐趟分布：最好 %.3f  最差 %.3f  中位 %.3f  标准差 %.3f"
          % (min(sp), max(sp), float(np.median(sp)), float(np.std(sp))))

    print("\n=== 误差最大的 8 帧 ===")
    print("%-9s %-8s %9s %9s %7s %7s" % ("架次", "帧", "AbsRel", "真值中位", "天空占比", "纹理"))
    for r in sorted(rows, key=lambda z: -z["absrel"])[:8]:
        print("%-9s %-8s %9.3f %9.1f %6.0f%% %7.0f"
              % (r["ep"].split("/")[0][-8:], r["stem"][-6:], r["absrel"],
                 r["gt_med"], 100 * r["sky"], r["tex"]))
        out["worst"].append(r)

    p = ROOT / "runs" / ("v3_" + a.backbone) / "error_analysis.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n写入", p)


if __name__ == "__main__":
    main()
