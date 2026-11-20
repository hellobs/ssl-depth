"""合并数据目录，重建帧索引。

规则与之前一致：有效范围 0.5 至 80 米，帧有效率不低于 50%。

默认只扫 v4 的 6 个根、写到 `cache/kept_all.json`（保持 v4 可复现）；
加 `--extra` 会把 batch5/6/7、b8_newgoal、other_ai 一起扫进来，即 v5 口径。
v5 索引写到 `cache/kept_all_v5.json`，不覆盖 v4 的。
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

ROOTS = [Path("D:/zzr/wmdata/runs"), Path("D:/zzr/wmdata/runs_batch2"),
         Path("D:/zzr/wmdata/runs_batch3"), Path("D:/zzr/wmdata/runs_batch4"),
         Path("D:/zzr/wmdata/runs_nearfield"), Path("D:/zzr/near_out")]
EXTRA_ROOTS = [Path("D:/zzr/wmdata/runs_batch5"), Path("D:/zzr/wmdata/runs_batch6"),
               Path("D:/zzr/wmdata/runs_batch7"), Path("D:/zzr/wmdata/runs_b8_newgoal"),
               Path("D:/zzr/wmdata/runs_other_ai")]
OUT = Path("D:/zzr/ssl-depth/cache/kept_all.json")
MIN_VALID, MAX_DEPTH = 0.50, 80.0

ap = argparse.ArgumentParser()
ap.add_argument("--extra", action="store_true",
                help="连 batch5/6/7、b8_newgoal、other_ai 一起扫（v5 口径）")
ap.add_argument("--out", type=Path, default=OUT)
a = ap.parse_args()
roots = ROOTS + EXTRA_ROOTS if a.extra else ROOTS

kept, total = {}, {}
for root in roots:
    for ep in sorted(glob.glob(str(root / "run_*" / "episode_*"))):
        p = Path(ep)
        key = p.parent.name + "/" + p.name
        stems = []
        for f in sorted((p / "observations").glob("*_front_depth.npy")):
            d = np.load(f)
            v = np.load(str(f).replace("_depth.npy", "_depth_valid.npy")).astype(bool)
            m = v & np.isfinite(d) & (d >= 0.5) & (d <= MAX_DEPTH)
            if m.mean() >= MIN_VALID:
                stems.append(f.name.replace("_front_depth.npy", ""))
        if stems:
            kept[key] = stems
            total[key] = len(list((p / "observations").glob("*_front_depth.npy")))

idx = {"kept": kept, "total": total, "frames": sum(len(v) for v in kept.values())}
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(idx, ensure_ascii=False, indent=1), encoding="utf-8")

print("保留 %d 个架次，%d 帧 -> %s" % (len(kept), idx["frames"], a.out))
print("根目录 %d 个（%s）" % (len(roots), "含 extra" if a.extra else "仅 v4"))
print("总采样 %d 帧，保留率 %.0f%%" % (
    sum(total.values()), 100 * idx["frames"] / max(1, sum(total.values()))))
