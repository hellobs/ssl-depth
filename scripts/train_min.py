"""第 3 步：最小闭环（3 轮）。

口径（与 HANDOFF 一致，不擅自更改）：
- 按 episode 划分 7:1.5:1.5，固定随机种子 0；
- 只保留 front_depth_valid 有效率 >= 60% 的帧；有效像素且深度落在 [0.5,30] m 才计入损失；
- 目标为视差 1/d，RGB 归一化到 [-1,1]；
- 验证集同时报「原始米制」与「中位尺度对齐」两种 AbsRel。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, "D:/zzr/microwm/src")
sys.path.insert(0, str(Path(__file__).parent))
import probe_data as P  # noqa: E402
from microwm.models.microwm import MicroJEPA  # noqa: E402

CACHE = Path("D:/zzr/ssl-depth/cache/kept80.json")
MIN_VALID = 0.50
MAX_DEPTH = 80.0


def build_index() -> dict:
    if CACHE.exists():
        return json.loads(CACHE.read_text(encoding="utf-8"))
    rows = P.collect()
    kept, total = {}, {}
    for r in rows:
        key = r["run"] + "/" + r["episode"]
        dep = np.load(r["depth"])
        val = np.load(r["valid"]).astype(bool)
        m = val & np.isfinite(dep) & (dep >= 0.5) & (dep <= MAX_DEPTH)
        total[key] = total.get(key, 0) + 1
        if m.mean() >= MIN_VALID:
            kept.setdefault(key, []).append(r["stem"])
    idx = {"kept": kept, "total": total, "frames": sum(len(v) for v in kept.values())}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(idx, ensure_ascii=False, indent=1), encoding="utf-8")
    return idx


def split(idx: dict, seed: int = 0):
    eps = sorted(idx["kept"])
    random.Random(seed).shuffle(eps)
    n = len(eps)
    a, b = int(round(0.7 * n)), int(round(0.85 * n))
    return eps[:a], eps[a:b], eps[b:]


class DepthFrames(Dataset):
    def __init__(self, idx: dict, episodes: list[str]):
        self.items = [(e, s) for e in episodes for s in idx["kept"][e]]
        self.root = Path("D:/zzr/wmdata/runs")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ep, stem = self.items[i]
        run, episode = ep.split("/")
        obs = self.root / run / episode / "observations"
        bgr = cv2.imread(str(obs / f"{stem}_front_rgb.png"), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
        dep = np.load(obs / f"{stem}_front_depth.npy").astype(np.float32)
        val = np.load(obs / f"{stem}_front_depth_valid.npy").astype(bool)
        val &= np.isfinite(dep) & (dep >= 0.5) & (dep <= MAX_DEPTH)
        disp = np.where(val, 1.0 / np.maximum(dep, 1e-3), 0.0).astype(np.float32)
        return (torch.from_numpy(rgb).permute(2, 0, 1),
                torch.from_numpy(disp)[None], torch.from_numpy(val.astype(np.float32))[None])


class DepthDecoder(nn.Module):
    """16×16 token → 256×256 视差图（4 次转置卷积上采样）。"""

    def __init__(self, dim: int = 192, width: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, width, 3, padding=1), nn.GELU(),
            nn.ConvTranspose2d(width, width, 4, 2, 1), nn.GELU(),      # 16 -> 32
            nn.Conv2d(width, width // 2, 3, padding=1), nn.GELU(),
            nn.ConvTranspose2d(width // 2, width // 2, 4, 2, 1), nn.GELU(),  # 32 -> 64
            nn.Conv2d(width // 2, width // 4, 3, padding=1), nn.GELU(),
            nn.ConvTranspose2d(width // 4, width // 4, 4, 2, 1), nn.GELU(),  # 64 -> 128
            nn.Conv2d(width // 4, width // 8, 3, padding=1), nn.GELU(),
            nn.ConvTranspose2d(width // 8, width // 8, 4, 2, 1), nn.GELU(),  # 128 -> 256
            nn.Conv2d(width // 8, 1, 1),
        )

    def forward(self, tokens, th, tw):
        b, n, d = tokens.shape
        x = tokens.transpose(1, 2).reshape(b, d, th, tw)
        return self.net(x)


@torch.no_grad()
def evaluate(model, head, dl, dev):
    model.eval(); head.eval()
    ps, gs, stds, rel = [], [], [], []
    for rgb, disp_gt, mask in dl:
        rgb, mask = rgb.to(dev), mask.to(dev)
        z = model.encode(rgb.unsqueeze(1))[:, 0]
        disp = head(z, model.tok_h, model.tok_w)
        pred_d = 1.0 / torch.clamp(disp.squeeze(1), min=1e-3)
        gt_d = 1.0 / torch.clamp(disp_gt.squeeze(1).to(dev), min=1e-3)
        m = mask.squeeze(1) > 0.5
        for j in range(pred_d.shape[0]):
            mm = m[j]
            if int(mm.sum()) > 100:
                pd = pred_d[j][mm]
                stds.append(float(pd.std()))
                rel.append(float(pd.std() / torch.clamp(pd.mean(), min=1e-6)))
        ps.append(pred_d[m].cpu().numpy())
        gs.append(gt_d[m].cpu().numpy())
    p = np.concatenate(ps); g = np.concatenate(gs)

    def metrics(p, g):
        raw = float(np.mean(np.abs(p - g) / g))
        s = float(np.median(p / g))
        al = float(np.mean(np.abs(p / s - g) / g))
        ratio = np.maximum(p / g, g / p)
        return {"n_px": int(p.size), "absrel_raw": raw, "scale": s, "absrel_med": al,
                "d1": float((ratio < 1.25).mean()),
                "rmse_raw": float(np.sqrt(np.mean((p - g) ** 2)))}

    out = metrics(p, g)
    near = g <= 30.0
    out["near30"] = metrics(p[near], g[near]) if near.sum() else None
    out["spatial_std"] = {"mean_m": float(np.mean(stds)), "median_m": float(np.median(stds)),
                          "mean_rel": float(np.mean(rel))} if stds else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="D:/zzr/ssl-depth/runs/min3")
    a = ap.parse_args()

    dev = "cuda"
    idx = build_index()
    tr, va, te = split(idx)
    print("split: train %d ep / %d frames | val %d / %d | test %d / %d"
          % (len(tr), sum(len(idx["kept"][e]) for e in tr),
             len(va), sum(len(idx["kept"][e]) for e in va),
             len(te), sum(len(idx["kept"][e]) for e in te)), flush=True)

    cfg = {"model": {"dim": 192, "depth": 4, "heads": 6, "patch": 16,
                     "mlp_ratio": 4.0, "in_ch": 3, "pred_depth": 2},
           "data": {"img_h": 256, "img_w": 256}}
    model = MicroJEPA(cfg).to(dev)
    head = DepthDecoder(cfg["model"]["dim"]).to(dev)
    print("params: encoder %.2fM | head %.2fM"
          % (sum(p.numel() for p in model.encoder.parameters()) / 1e6,
             sum(p.numel() for p in head.parameters()) / 1e6), flush=True)

    dl_tr = DataLoader(DepthFrames(idx, tr), batch_size=a.batch, shuffle=True,
                       num_workers=0, drop_last=True)
    dl_va = DataLoader(DepthFrames(idx, va), batch_size=a.batch, shuffle=False, num_workers=0)
    dl_te = DataLoader(DepthFrames(idx, te), batch_size=a.batch, shuffle=False, num_workers=0)

    params = list(model.encoder.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    steps = a.epochs * len(dl_tr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps))

    for ep in range(a.epochs):
        model.train(); head.train()
        t0, run = time.time(), 0.0
        for i, (rgb, disp_gt, mask) in enumerate(dl_tr):
            rgb, disp_gt, mask = rgb.to(dev), disp_gt.to(dev), mask.to(dev)
            z = model.encode(rgb.unsqueeze(1))[:, 0]
            disp = head(z, model.tok_h, model.tok_w)
            loss = (torch.abs(disp - disp_gt) * mask).sum() / mask.sum().clamp(min=1.0)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sch.step()
            run = 0.9 * run + 0.1 * float(loss) if i else float(loss)
        vm = evaluate(model, head, dl_va, dev)
        print("epoch %2d/%d  loss %.4f | val AbsRel(med) %.3f d1 %.3f | near30 d1 %.3f "
              "| pred std %.2fm (rel %.2f) | %.1fs"
              % (ep + 1, a.epochs, run, vm["absrel_med"], vm["d1"],
                 (vm["near30"] or {}).get("d1", float("nan")),
                 vm["spatial_std"]["mean_m"], vm["spatial_std"]["mean_rel"],
                 time.time() - t0), flush=True)

    tm = evaluate(model, head, dl_te, dev)
    print("test:", json.dumps(tm, ensure_ascii=False), flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    torch.save({"encoder": model.encoder.state_dict(), "head": head.state_dict(), "cfg": cfg},
               Path(a.out) / "min3.pt")
    (Path(a.out) / "metrics.json").write_text(json.dumps(
        {"val": vm, "test": tm, "split": {"train": tr, "val": va, "test": te}}, indent=1),
        encoding="utf-8")


if __name__ == "__main__":
    main()
