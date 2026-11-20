"""数据体检（80 m 主口径 + 30 m 子集对照）。

用法：
    python probe_data.py                 # 默认 80 m 窗口
    python probe_data.py --max-depth 30

口径：有效率 = valid==1 且 0.5 <= d <= max_depth 的像素占比；
同时报「原始 valid==1 占比」，用于区分「窗口太窄」与「采集故障」。
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
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch

RUNS = Path(f"{DATA}/runs")
OBS_RE = re.compile(r"^(\d{6})_front_rgb\.png$")


def cuda_report() -> None:
    print("== 1. 环境 ==")
    print("torch", torch.__version__, "| cuda build", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print("device:", p.name, "| sm_%d%d" % (p.major, p.minor))
        print("VRAM total: %.2f GB | free now: %.2f GB"
              % (p.total_memory / 1024 ** 3, torch.cuda.mem_get_info()[0] / 1024 ** 3))


def collect() -> list[dict]:
    rows, missing = [], Counter()
    for run in sorted(p for p in RUNS.iterdir() if p.is_dir()):
        for ep in sorted(run.glob("episode_*")):
            obs = ep / "observations"
            if not obs.is_dir():
                missing["no_observations"] += 1
                continue
            for f in sorted(obs.iterdir()):
                m = OBS_RE.match(f.name)
                if not m:
                    continue
                stem = m.group(1)
                d = obs / f"{stem}_front_depth.npy"
                v = obs / f"{stem}_front_depth_valid.npy"
                if not (d.exists() and v.exists()):
                    missing["unpaired"] += 1
                    continue
                rows.append({"run": run.name, "episode": ep.name, "stem": stem,
                             "depth": d, "valid": v})
    if missing:
        print("missing:", dict(missing))
    return rows


def ratio(dep: np.ndarray, val: np.ndarray, max_depth: float) -> tuple[float, float]:
    raw = val.astype(bool) & np.isfinite(dep)
    m = raw & (dep >= 0.5) & (dep <= max_depth)
    return float(m.mean()), float(raw.mean())


def hist_text(r: np.ndarray) -> str:
    edges = [0, .2, .3, .4, .5, .6, .7, .8, .9, 1.0001]
    h, _ = np.histogram(r, bins=edges)
    return " ".join("[%.1f,%.1f)=%d" % (edges[i], edges[i + 1], h[i]) for i in range(len(h)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-depth", type=float, default=80.0)
    a = ap.parse_args()

    cuda_report()
    rows = collect()
    print("\n== 2. 数据 ==")
    print("runs with observations: %d | episodes: %d | paired frames: %d"
          % (len({r["run"] for r in rows}), len({(r["run"], r["episode"]) for r in rows}), len(rows)))

    r80, rraw, per_run, per_ep = [], [], {}, {}
    for r in rows:
        dep = np.load(r["depth"]).astype(np.float32)
        val = np.load(r["valid"])
        v, raw = ratio(dep, val, a.max_depth)
        r80.append(v); rraw.append(raw)
        per_run.setdefault(r["run"], []).append(v)
        per_ep.setdefault((r["run"], r["episode"]), []).append(v)
    r80, rraw = np.asarray(r80), np.asarray(rraw)

    print("window <= %.0f m" % a.max_depth)
    print("valid-pixel ratio: mean %.3f | p05 %.3f | p25 %.3f | median %.3f | p75 %.3f"
          % (r80.mean(), *np.percentile(r80, [5, 25, 50, 75])))
    print("raw valid==1 ratio (no depth clip): median %.3f | mean %.3f"
          % (np.median(rraw), rraw.mean()))
    print("histogram:", hist_text(r80))
    keep = r80 >= 0.50
    print("frames >= 50%%: %d / %d (%.1f%%)" % (keep.sum(), len(r80), 100 * keep.mean()))
    print("usable episodes (>=1 kept frame): %d / %d"
          % (sum(any(v >= .5 for v in a) for a in per_ep.values()), len(per_ep)))

    print("\nper-run (median, min, max, frames, kept@50%%):")
    for k in sorted(per_run):
        x = np.asarray(per_run[k])
        print("  %s  n=%3d  med=%.3f  min=%.3f  max=%.3f  kept=%d"
              % (k[-8:], len(x), np.median(x), x.min(), x.max(), int((x >= .5).sum())))

    print("\nJSON", json.dumps({"max_depth": a.max_depth, "frames": len(rows),
                                "kept": int(keep.sum())}))


if __name__ == "__main__":
    main()
