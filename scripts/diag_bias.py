"""方向性偏差：失效到底是读远还是读近。

按真值前方距离分档，统计预测值减真值的带符号误差，
以及高估帧所占比例。误判方向决定了后果是漏判还是误触发。

  python scripts/diag_bias.py --run runs/v5_imagenet \
      --index cache/kept_all_v5.json --split-file runs/v5_split.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "D:/zzr/ssl-depth/scripts")
from task_metric import CACHE, ROOT, build, collect  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/v4_imagenet")
    ap.add_argument("--index", default=None)
    ap.add_argument("--split-file", default=None)
    a = ap.parse_args()

    idx = json.loads(Path(a.index or CACHE).read_text(encoding="utf-8"))
    sp_path = Path(a.split_file) if a.split_file else (ROOT / "runs" / "v4_imagenet" / "metrics.json")
    sp = json.loads(sp_path.read_text(encoding="utf-8"))["split"]
    eps = sorted(set(sp["val"]) | set(sp["test"]))

    bb, head, dev = build("imagenet", a.run)
    g, p = collect(bb, head, dev, idx, eps)
    ok = np.isfinite(g) & np.isfinite(p)
    g, p = g[ok], p[ok]
    d = p - g

    print("n=%d | 真值中位 %.2f | 预测中位 %.2f" % (g.size, np.median(g), np.median(p)))
    print("%-12s %5s %9s %9s %8s %8s" %
          ("真值区间", "帧数", "中位误差", "均值误差", "高估占比", ">2m占比"))
    for lo, hi in [(0, 6), (6, 9), (9, 12), (12, 20), (20, 40), (40, 99)]:
        m = (g >= lo) & (g < hi)
        if m.sum() == 0:
            continue
        dd = d[m]
        print("%-12s %5d %9.2f %9.2f %7.0f%% %7.0f%%"
              % ("%d-%d m" % (lo, hi), m.sum(), np.median(dd), dd.mean(),
                 100 * (dd > 0).mean(), 100 * (dd > 2).mean()))


if __name__ == "__main__":
    main()
