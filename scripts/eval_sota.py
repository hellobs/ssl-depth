"""零样本 SOTA 深度模型在本文留出集上的评测。

为什么需要对齐：Depth Anything 这类模型输出**仿射不变的视差**，不是米制深度。
只取倒数再按中位数缩放不够（p 接近 0 的像素会变成上千米），必须拟合
    depth = 1 / (a * p + b)
其中 a、b 在**验证集**上最小二乘拟合，再原样套到留出集，不碰测试真值。
ZoeDepth 这类直接输出米制的模型跳过这一步。

像素级口径与 diag_backbone.evaluate 一致：
    absrel_raw / absrel_med（全集 p/g 中位数对齐）/ d1 / rmse
这里额外给 d1_med 与 absrel_med，因为相对深度模型的原始尺度任意。

帧级口径：前方最近障碍 p05（ROI 横向中间三分之一、纵向中间二分之一）。

与表里四行的差别：那四行是「冻结骨干 + 0.81M 头，在 17k 帧上训练」，
这里是零样本；分辨率用原始 256x256，不做 108x192 拉伸。两处都对 SOTA 更有利。

用法：
  python scripts/eval_sota.py --model depth-anything/Depth-Anything-V2-Small-hf \
      --out runs/sota_dav2s.json
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
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.stats import spearmanr

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

ROOT = Path(f"{REPO}")
SPLIT = ROOT / "runs/v5_split.json"
ROOTS = [Path(p) for p in [
    f"{DATA}/runs", f"{DATA}/runs_batch2", f"{DATA}/runs_batch3",
    f"{DATA}/runs_batch4", f"{DATA}/runs_batch5", f"{DATA}/runs_batch6",
    f"{DATA}/runs_batch7", f"{DATA}/runs_b8_newgoal", f"{DATA}/runs_nearfield",
    f"{DATA}/runs_other_ai", f"{DATA}/near_out"]]
MIN_D, MAX_D = 0.5, 80.0


def find_obs(ep: str) -> Path | None:
    run, name = ep.split("/")
    for r in ROOTS:
        p = r / run / name / "observations"
        if p.is_dir():
            return p
    return None


def roi_mask(h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), bool)
    m[h // 4: 3 * h // 4, w // 3: 2 * w // 3] = True
    return m


def load_model(name: str, dev: str):
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    proc = AutoImageProcessor.from_pretrained(name)
    model = AutoModelForDepthEstimation.from_pretrained(name).to(dev).eval()
    return proc, model


def frames_of(obs: Path, stride: int):
    stems = sorted(p.name[:-len("_front_rgb.png")] for p in obs.glob("*_front_rgb.png"))
    return stems[::stride]


@torch.no_grad()
def infer(proc, model, obs: Path, stems, dev):
    """返回 [(p_raw, g, valid_mask)]，p_raw 未做任何变换。"""
    out = []
    for s in stems:
        rgb_p = obs / (s + "_front_rgb.png")
        gt_p = obs / (s + "_front_depth.npy")
        vp = obs / (s + "_front_depth_valid.npy")
        if not (rgb_p.exists() and gt_p.exists() and vp.exists()):
            continue
        im = Image.open(rgb_p)
        im = im.convert("RGB") if im.mode != "RGB" else im
        g = np.load(gt_p).astype(np.float32)
        v = np.load(vp).astype(bool)
        inp = proc(images=im, return_tensors="pt").to(dev)
        pred = model(**inp).predicted_depth
        p = torch.nn.functional.interpolate(
            pred.unsqueeze(1), size=g.shape, mode="bicubic", align_corners=False
        ).squeeze().cpu().numpy().astype(np.float32)
        m = v & np.isfinite(g) & (g >= MIN_D) & (g <= MAX_D) & np.isfinite(p)
        if m.sum() < 100:
            continue
        out.append((p, g, m))
    return out


def fit_affine(samples):
    """视差空间最小二乘拟合 d_gt = a*p + b，返回 (a, b)。"""
    ps, dgs = [], []
    for p, g, m in samples:
        ps.append(p[m]); dgs.append(1.0 / g[m])
    p = np.concatenate(ps); dg = np.concatenate(dgs)
    if p.size > 400_000:
        sel = np.random.RandomState(0).choice(p.size, 400_000, replace=False)
        p, dg = p[sel], dg[sel]
    A = np.stack([p, np.ones_like(p)], 1)
    (a, b), *_ = np.linalg.lstsq(A, dg, rcond=None)
    return float(a), float(b)


def to_depth(p: np.ndarray, invert: bool, ab) -> np.ndarray:
    if not invert:
        return p
    a, b = ab
    d = a * p + b
    with np.errstate(divide="ignore", invalid="ignore"):
        dep = 1.0 / d
    dep = np.where(np.isfinite(dep), dep, MAX_D)
    return np.clip(dep, 0.1, 200.0)


def pixel_metrics(p: np.ndarray, g: np.ndarray) -> dict:
    s = float(np.median(p / g))
    pm = p / s
    return {"n_px": int(p.size),
            "absrel_raw": float(np.mean(np.abs(p - g) / g)),
            "absrel_med": float(np.mean(np.abs(pm - g) / g)),
            "scale": s,
            "d1": float((np.maximum(p / g, g / p) < 1.25).mean()),
            "d1_med": float((np.maximum(pm / g, g / pm) < 1.25).mean()),
            "rmse": float(np.sqrt(np.mean((p - g) ** 2)))}


def task_metrics(fp: np.ndarray, fg: np.ndarray) -> dict:
    out = {"n_frame": int(fg.size),
           "gt_median": float(np.median(fg)),
           "pr_median": float(np.median(fp))}
    for t in (6.0, 9.0, 12.0, 15.0, 20.0):
        dg, dp = fg < t, fp < t
        tp = int((dg & dp).sum()); fn = int((dg & ~dp).sum())
        fpra = int((~dg & dp).sum()); tn = int((~dg & ~dp).sum())
        out["t%d" % int(t)] = {"hit": tp, "miss": fn, "false": fpra, "pass": tn,
                               "recall": tp / max(1, tp + fn)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="held", choices=["held", "test", "val"])
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--fit-stride", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 帧，冒烟用")
    ap.add_argument("--align", default="per-image", choices=["per-image", "global"],
                    help="仿射对齐方式；仿射不变模型每帧尺度不同，默认逐帧（MiDaS 口径）")
    a = ap.parse_args()

    sp = json.loads(SPLIT.read_text(encoding="utf-8"))["split"]
    eps = sp["val"] + sp["test"] if a.split == "held" else sp[a.split]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("%s | %d 航段 | device %s" % (a.model, len(eps), dev), flush=True)
    proc, model = load_model(a.model, dev)

    # 1) 方向判定 + 仿射拟合，只用验证集
    fit_samples = []
    for ep in sp["val"]:
        obs = find_obs(ep)
        if obs is None:
            continue
        fit_samples += infer(proc, model, obs, frames_of(obs, a.fit_stride), dev)
        if len(fit_samples) >= 24:
            break
    if not fit_samples:
        raise SystemExit("验证集没有可用帧")
    p0, g0, m0 = fit_samples[0]
    rho = spearmanr(p0[m0], g0[m0]).correlation
    invert = bool(rho < 0)
    ab = fit_affine(fit_samples) if invert else None
    print("  方向 spearman %+.3f -> %s%s" % (
        rho, "视差（仿射对齐）" if invert else "米制",
        "" if not invert else "  a=%.4f b=%.4f" % ab), flush=True)

    # 2) 留出集评测
    ps, gs, fronts_p, fronts_g, n = [], [], [], [], 0
    for ep in eps:
        obs = find_obs(ep)
        if obs is None:
            continue
        for p_raw, g, m in infer(proc, model, obs, frames_of(obs, a.stride), dev):
            if invert and a.align == "per-image":
                ab_use = fit_affine([(p_raw, g, m)])   # 逐帧最小二乘，用该帧真值
            else:
                ab_use = ab
            p = to_depth(p_raw, invert, ab_use)
            ps.append(p[m]); gs.append(g[m])
            mm = m & roi_mask(*g.shape)
            if mm.sum() > 100:
                fronts_p.append(float(np.percentile(p[mm], 5)))
                fronts_g.append(float(np.percentile(g[mm], 5)))
            n += 1
            if a.limit and n >= a.limit:
                break
        if a.limit and n >= a.limit:
            break

    p = np.concatenate(ps); g = np.concatenate(gs)
    res = {"model": a.model, "split": a.split, "invert": invert,
           "align": a.align,
           "affine": None if ab is None else {"a": ab[0], "b": ab[1]},
           "pixel": pixel_metrics(p, g),
           "task": task_metrics(np.array(fronts_p), np.array(fronts_g))}
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("  %d 帧 | d1_med %.3f | AbsRel_med %.3f | RMSE %.1f"
          % (n, res["pixel"]["d1_med"], res["pixel"]["absrel_med"], res["pixel"]["rmse"]))
    print("  帧级 %d | 预测中位 %.1f m | 9 米识别 %.1f%% 误刹 %d"
          % (res["task"]["n_frame"], res["task"]["pr_median"],
             100 * res["task"]["t9"]["recall"], res["task"]["t9"]["false"]))


if __name__ == "__main__":
    main()
