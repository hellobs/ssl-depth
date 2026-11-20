# -*- coding: utf-8 -*-
"""统计一个采集目录里的近场帧覆盖情况。

前方距离的口径与 task_metric.front_near 一致：中心 ROI（h/4–3h/4、w/3–2w/3）
内有效深度的 5% 分位。阈值取避障判据常用的 6/9/12/15/20 米。

用法：
  python scripts/nearfield_report.py --root D:/zzr/wmdata/near_out2
  python scripts/nearfield_report.py --root D:/zzr/wmdata/near_out2 --json out.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

THRESHOLDS = (6.0, 9.0, 12.0, 15.0, 20.0)


def front_near(depth: np.ndarray) -> float:
    h, w = depth.shape
    roi = depth[h // 4: 3 * h // 4, w // 3: 2 * w // 3]
    roi = roi[np.isfinite(roi) & (roi > 0)]
    return float(np.percentile(roi, 5)) if roi.size else float("nan")


def scan_run(run: Path) -> list[dict]:
    out = []
    for ep in sorted(run.glob("episode_*")):
        obs_dir = ep / "observations"
        if not obs_dir.is_dir():
            continue
        for depth_path in sorted(obs_dir.glob("*_front_depth.npy")):
            stem = depth_path.name.replace("_front_depth.npy", "")
            d = np.load(depth_path)
            out.append({"run": run.name, "episode": ep.name, "stem": stem,
                        "front_m": front_near(d)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    rows: list[dict] = []
    runs = sorted(p for p in root.glob("run_*") if p.is_dir())
    for run in runs:
        rj = run / "run.json"
        status = None
        if rj.exists():
            try:
                status = json.loads(rj.read_text(encoding="utf-8")).get("status")
            except Exception:
                status = "unreadable"
        rows.extend(scan_run(run))
        print("%s  %s  %d 帧" % (run.name, status or "-", sum(1 for r in rows if r["run"] == run.name)))

    v = np.array([r["front_m"] for r in rows], dtype=float)
    ok = np.isfinite(v)
    print("\n合计 %d 帧（有效 %d），%d 个航次" % (len(v), ok.sum(), len(runs)))
    if ok.sum() == 0:
        return 1
    print("前方距离  p05 %.2f m  中位 %.2f m  p95 %.2f m  最大 %.2f m"
          % (np.percentile(v[ok], 5), np.median(v[ok]), np.percentile(v[ok], 95), v[ok].max()))
    print("\n阈值   帧数    占比")
    for t in THRESHOLDS:
        n = int((v[ok] < t).sum())
        print("%4.0f m %5d   %5.1f%%" % (t, n, 100 * n / ok.sum()))

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"root": str(root), "frames": len(rows),
             "near": {str(t): int((v[ok] < t).sum()) for t in THRESHOLDS},
             "front_p05": float(np.percentile(v[ok], 5)),
             "front_median": float(np.median(v[ok])),
             "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
        print("\n已写 %s" % args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
