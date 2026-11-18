"""方向性偏差：失效到底是读远还是读近。

按真值前方距离分档，统计预测值减真值的带符号误差，
以及高估帧所占比例。误判方向决定了后果是漏判还是误触发。
"""
from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, "D:/zzr/ssl-depth/scripts")
from task_metric import CACHE, ROOT, build, collect  # noqa: E402


def main():
    idx = json.loads(CACHE.read_text(encoding="utf-8"))
    sp = json.loads((ROOT / "runs" / "v4_imagenet" / "metrics.json")
                    .read_text(encoding="utf-8"))["split"]
    eps = sorted(set(sp["val"]) | set(sp["test"]))

    bb, head, dev = build("imagenet", "runs/v4_imagenet")
    g, p = collect(bb, head, dev, idx, eps)
    ok = np.isfinite(g) & np.isfinite(p)
    g, p = g[ok], p[ok]
    d = p - g

    print("n=%d" % g.size)
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
