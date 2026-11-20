"""按目录分批建帧索引，可增量合并。

环境里的长进程会被打断，一次扫 11 个根（3.6 万帧、7 万个文件读）风险太高，
所以拆成几次调用：每次只扫指定的根，把结果合并进同一个 out 文件。

用法：
  python scripts/index_roots.py --roots runs,runs_batch2 --out cache/kept_all_v5.json
  python scripts/index_roots.py --roots runs_batch3,runs_batch4 --out cache/kept_all_v5.json
  ...
最后：
  python scripts/index_roots.py --report --out cache/kept_all_v5.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

BASE = {
    "runs": "D:/zzr/wmdata/runs",
    "runs_batch2": "D:/zzr/wmdata/runs_batch2",
    "runs_batch3": "D:/zzr/wmdata/runs_batch3",
    "runs_batch4": "D:/zzr/wmdata/runs_batch4",
    "runs_batch5": "D:/zzr/wmdata/runs_batch5",
    "runs_batch6": "D:/zzr/wmdata/runs_batch6",
    "runs_batch7": "D:/zzr/wmdata/runs_batch7",
    "runs_b8_newgoal": "D:/zzr/wmdata/runs_b8_newgoal",
    "runs_nearfield": "D:/zzr/wmdata/runs_nearfield",
    "runs_other_ai": "D:/zzr/wmdata/runs_other_ai",
    "near_out": "D:/zzr/near_out",
    "near_out2": "D:/zzr/near_out2",
}
MIN_VALID, MAX_DEPTH = 0.50, 80.0
OBS = "observations"


def scan_root(root: Path) -> tuple[dict, dict]:
    kept, total = {}, {}
    if not root.is_dir():
        return kept, total
    for rd in sorted(root.iterdir()):
        if not rd.is_dir() or not rd.name.startswith("run_"):
            continue
        for ep in sorted(rd.iterdir()):
            obs = ep / OBS
            if not obs.is_dir():
                continue
            key = "%s/%s" % (rd.name, ep.name)
            files = sorted(obs.glob("*_front_depth.npy"))
            stems = []
            for f in files:
                d = np.load(f)
                vp = Path(str(f).replace("_depth.npy", "_depth_valid.npy"))
                if not vp.exists():
                    continue
                v = np.load(vp).astype(bool)
                m = v & np.isfinite(d) & (d >= 0.5) & (d <= MAX_DEPTH)
                if m.mean() >= MIN_VALID:
                    stems.append(f.name.replace("_front_depth.npy", ""))
            if stems:
                kept[key] = stems
                total[key] = len(files)
    return kept, total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", default="", help="逗号分隔的目录名，见 BASE")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--report", action="store_true", help="只汇总已有 out，不扫描")
    a = ap.parse_args()

    if a.report:
        idx = json.loads(a.out.read_text(encoding="utf-8"))
        print("索引 %d 航段 / %d 帧" % (len(idx["kept"]), sum(len(v) for v in idx["kept"].values())))
        return

    if a.out.exists():
        idx = json.loads(a.out.read_text(encoding="utf-8"))
    else:
        idx = {"kept": {}, "total": {}, "frames": 0}

    names = [s.strip() for s in a.roots.split(",") if s.strip()]
    for n in names:
        if n not in BASE:
            raise SystemExit("未知目录 %s，可选：%s" % (n, ",".join(BASE)))
        kept, total = scan_root(Path(BASE[n]))
        idx["kept"].update(kept)
        idx["total"].update(total)
        print("  %-16s +%3d 航段 / %6d 帧" % (n, len(kept), sum(len(v) for v in kept.values())))

    idx["frames"] = sum(len(v) for v in idx["kept"].values())
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(idx, ensure_ascii=False, indent=1), encoding="utf-8")
    print("累计 %d 航段 / %d 帧 -> %s" % (len(idx["kept"]), idx["frames"], a.out.name))


if __name__ == "__main__":
    main()
