"""域内 JEPA 自监督预训练（只用 RGB，不用深度标签）。

与 diag_backbone.py 的分工：
- diag_backbone 只**加载**现成权重（VisDrone / dronelab），测「已有表征有没有用」；
- 本脚本在**本数据**上从头自监督，产出域内权重，供三臂对照里的 jepa 臂使用。

复用 microwm 的 MicroJEPA：编码器 + EMA teacher + 预测器 + VICReg/时间方差正则，
不重写任何模型结构。数据只取 episode 内相邻帧对 (t, t+1)，不读 depth。

用法（先冒烟，不占 GPU）：
    python scripts/pretrain_jepa.py --smoke
正式（显式给 run 列表，避免把测试趟喂进预训练）：
    python scripts/pretrain_jepa.py --runs 91b26e13,1d3aec79,... --epochs 60
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
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, "D:/zzr/microwm/src")
from microwm.models.microwm import MicroJEPA  # noqa: E402

ROOT = Path("D:/zzr/wmdata/runs")
OBS = "observations"


def list_episodes(runs: list[str] | None) -> list[tuple[str, str]]:
    out = []
    for run in sorted(p for p in ROOT.iterdir() if p.is_dir()):
        if runs and not any(run.name.endswith(r) for r in runs):
            continue
        for ep in sorted(run.glob("episode_*")):
            if (ep / OBS).is_dir():
                out.append((run.name, ep.name))
    return out


class FramePairs(Dataset):
    """同一 episode 内相邻两帧；跨 episode 不组对，避免把场景切换当成运动。"""

    def __init__(self, eps: list[tuple[str, str]], img: int, limit: int = 0):
        self.items, self.img = [], img
        for run, ep in eps:
            stems = sorted(p.name[:6] for p in (ROOT / run / ep / OBS).glob("*_front_rgb.png"))
            for a, b in zip(stems, stems[1:]):
                self.items.append((run, ep, a, b))
        if limit:
            random.Random(0).shuffle(self.items)
            self.items = self.items[:limit]
        self.mean = np.array([0.5, 0.5, 0.5], np.float32)

    def __len__(self):
        return len(self.items)

    def _read(self, run, ep, stem):
        p = ROOT / run / ep / OBS / f"{stem}_front_rgb.png"
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(cv2.resize(bgr, (self.img, self.img), interpolation=cv2.INTER_AREA),
                           cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(rgb).permute(2, 0, 1)

    def __getitem__(self, i):
        run, ep, a, b = self.items[i]
        return self._read(run, ep, a), self._read(run, ep, b)


def make_cfg(img: int, patch: int, dim: int, depth: int, heads: int) -> dict:
    return {"model": {"patch": patch, "dim": dim, "depth": depth, "heads": heads,
                      "mlp_ratio": 4, "in_ch": 3, "pred_depth": 4,
                      "ema_tau": 0.996, "vicreg_var": 0.04, "vicreg_cov": 0.01,
                      "vicreg_tvar": 0.01, "pred_metric": "l1"},
            "data": {"img_h": img, "img_w": img}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="", help="逗号分隔的 run 名后缀；空表示全部")
    ap.add_argument("--img", type=int, default=256)
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--limit", type=int, default=0, help="只用前 N 个帧对（调试）")
    ap.add_argument("--smoke", action="store_true", help="3 步、64px、CPU，只验通路")
    ap.add_argument("--out", default="D:/zzr/ssl-depth/runs/jepa_domain")
    a = ap.parse_args()

    if a.smoke:
        a.img, a.patch, a.dim, a.depth, a.heads = 64, 16, 96, 2, 4
        a.epochs, a.batch, a.limit, a.out = 1, 2, 6, "D:/zzr/ssl-depth/runs/smoke_jepa"
        dev = "cpu"
    else:
        dev = "cuda"

    runs = [s.strip() for s in a.runs.split(",") if s.strip()]
    eps = list_episodes(runs or None)
    ds = FramePairs(eps, a.img, a.limit)
    if len(ds) == 0:
        raise SystemExit("没有可用帧对：检查 --runs 与数据目录")
    print("runs %d | episodes %d | pairs %d | img %d | patch %d"
          % (len({r for r, _ in eps}), len(eps), len(ds), a.img, a.patch), flush=True)

    cfg = make_cfg(a.img, a.patch, a.dim, a.depth, a.heads)
    model = MicroJEPA(cfg).to(dev)
    n_enc = sum(p.numel() for p in model.encoder.parameters())
    print("encoder %.2fM | total(model incl. teacher/predictor) %.2fM"
          % (n_enc / 1e6, sum(p.numel() for p in model.parameters()) / 1e6), flush=True)

    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=0, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    steps = max(1, a.epochs * len(dl))
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    for ep in range(a.epochs):
        t0, run_loss = time.time(), 0.0
        for i, (past, future) in enumerate(dl):
            past = past.unsqueeze(1).to(dev)      # [B,1,3,H,W]
            future = future.unsqueeze(1).to(dev)
            out = model(past, future)
            loss = out["loss"]
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step()
            model.update_teacher()
            run_loss = 0.9 * run_loss + 0.1 * float(loss.detach()) if i else float(loss.detach())
            if a.smoke:
                print("  step %d loss %.4f pred %.4f"
                      % (i + 1, float(loss.detach()), float(out["pred"])), flush=True)
        print("epoch %d/%d loss %.4f  %.1fs" % (ep + 1, a.epochs, run_loss, time.time() - t0), flush=True)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": cfg, "step": steps,
                "runs": runs or "all"}, out / "jepa.pt")
    (out / "meta.json").write_text(json.dumps(
        {"cfg": cfg, "runs": runs or "all", "pairs": len(ds), "steps": steps},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved", out / "jepa.pt", flush=True)


if __name__ == "__main__":
    main()
