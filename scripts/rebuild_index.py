"""合并两批数据，重建帧索引。

规则与之前一致：有效范围 0.5 至 80 米，帧有效率不低于 50%。
区别是现在扫两个根目录。
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

ROOTS = [Path("D:/zzr/wmdata/runs"), Path("D:/zzr/wmdata/runs_batch2"),
         Path("D:/zzr/wmdata/runs_batch3"), Path("D:/zzr/wmdata/runs_batch4"),
         Path("D:/zzr/wmdata/runs_nearfield"), Path("D:/zzr/near_out")]
OUT = Path("D:/zzr/ssl-depth/cache/kept_all.json")
MIN_VALID, MAX_DEPTH = 0.50, 80.0

kept, total = {}, {}
for root in ROOTS:
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
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(idx, ensure_ascii=False, indent=1), encoding="utf-8")

print("保留 %d 个架次，%d 帧" % (len(kept), idx["frames"]))
b2 = [k for k in kept if any(str(r).endswith("runs_batch2") for r in ROOTS if (r / k / "observations").is_dir())]
keep_rate = idx["frames"] / max(1, sum(total.values()))
print("总采样 %d 帧，保留率 %.0f%%" % (sum(total.values()), 100 * keep_rate))
