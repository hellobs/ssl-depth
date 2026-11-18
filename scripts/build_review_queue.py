"""组件一：触发提取器。

从留出集里找出"模型判定触发避障"的帧（前方最近距离低于阈值），
打包成待复核条目，写入 review_queue.jsonl。

复核条目字段：
  item_id    帧 id（run/episode/stem）
  image      观测图路径
  mode       A（显示判定与先验）或 B（只显示已触发），按复核者序号自动分配
  pred_dist  模型预测的前方距离（米），仅 A 模式展示
  bin_label  距离区间（用于 A 模式的误差先验）
  gt_dist    仿真真值距离——只入队供事后分析，界面绝不显示
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as tud

sys.path.insert(0, str(Path(__file__).parent))
from diag_backbone import CACHE, ROOTS, DepthHead, Frames, ResnetBackbone, split  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "review" / "review_queue.jsonl"
THRESHOLD = 9.0
N_PER_MODE = 80          # 每种模式抽多少条
BIN_EDGES = [0.5, 2, 5, 10, 20, 40, 81]
BIN_NAMES = ["0.5-2 m", "2-5 m", "5-10 m", "10-20 m", "20-40 m", "40+ m"]


BIN_PAIRS = [(0.5, 2, "0.5-2 m"), (2, 5, "2-5 m"), (5, 10, "5-10 m"),
             (10, 20, "10-20 m"), (20, 40, "20-40 m")]


def bin_of(d: float) -> str:
    for lo, hi, name in BIN_PAIRS:
        if lo <= d < hi:
            return name
    return "40+ m"


def main() -> None:
    dev = "cuda"
    torch.cuda.set_per_process_memory_fraction(0.35, 0)
    bb = ResnetBackbone(True).to(dev)
    head = DepthHead(192).to(dev)
    sd = torch.load(ROOT / "runs/v4_imagenet/head.pt", map_location="cpu", weights_only=False)
    head.load_state_dict(sd["head"])
    bb.load_state_dict(sd["proj"], strict=False)
    bb.eval(); head.eval()

    idx = json.loads(CACHE.read_text(encoding="utf-8"))
    _, va, te = split(idx, 0)
    ds = Frames(idx, va + te)
    dl = tud.DataLoader(ds, batch_size=8, num_workers=0)

    hits = []  # (rel_path, pred_front, gt_front, gt_med)
    k = 0
    with torch.no_grad():
        for rgb, dg, mk in dl:
            tok, th, tw = bb(rgb.to(dev).unsqueeze(1))
            disp = head(tok, th, tw)
            pd = (1 / torch.clamp(disp.squeeze(1), min=1e-3)).cpu().numpy()
            gd = (1 / torch.clamp(dg.squeeze(1), min=1e-3)).numpy()
            mm = (mk.squeeze(1) > 0.5).numpy()
            for j in range(pd.shape[0]):
                m = mm[j] & np.isfinite(gd[j])
                if m.sum() < 100:
                    continue
                roi = (int(pd.shape[2] * 0.05), int(pd.shape[2] * 0.45),
                       int(pd.shape[1] * 0.3), int(pd.shape[1] * 0.7))
                pr = pd[j][roi[2]:roi[3], roi[0]:roi[1]]
                gr = gd[j][roi[2]:roi[3], roi[0]:roi[1]]
                mr = m[roi[2]:roi[3], roi[0]:roi[1]]
                if mr.sum() < 30:
                    continue
                pred_front = float(np.quantile(pr[mr], 0.05))
                gt_front = float(np.quantile(gr[mr], 0.05))
                if pred_front < THRESHOLD:
                    ep_name, stem = ds.items[k]
                    hits.append((ep_name + '/' + stem, pred_front, gt_front, float(np.median(gd[j][m]))))
                k += 1

    print("触发帧 %d / 全部 %d" % (len(hits), k))

    # 按距离区间分层抽取，保证 A/B 各覆盖不同区间
    by_bin: dict[str, list] = {}
    for h in hits:
        by_bin.setdefault(bin_of(h[2]), []).append(h)
    picked = []
    half = N_PER_MODE // 2
    for name in BIN_NAMES:
        pool = sorted(by_bin.get(name, []), key=lambda x: abs(x[1] - THRESHOLD))
        picked += pool[:half]
    picked = picked[: N_PER_MODE * 2]

    # A/B 交替分配；A/B 各自的队列在复核时按 reviewer 序号决定先做哪个模式
    items = []
    for i, (ep, pred, gt, med) in enumerate(picked):
        items.append({
            "item_id": ep,
            "mode": "A" if i % 2 == 0 else "B",
            "pred_dist": round(pred, 1),
            "bin_label": bin_of(gt),
            "gt_dist": round(gt, 1),
            "gt_med": round(med, 1),
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for it in items:
            parts = it["item_id"].split("/")
            obs = None
            for root in ROOTS:
                cand = root / parts[0] / parts[1] / "observations"
                if cand.is_dir():
                    obs = cand
                    break
            if obs is None:
                continue
            it["image_dir"] = str(obs)
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print("写入 %d 条 -> %s" % (len(items), OUT))


if __name__ == "__main__":
    main()
