# -*- coding: utf-8 -*-
"""闭环第三环（一）：把复核记录变成训练标签。

设计要点
--------
1. **标签不绑队列条目，绑图像与深度。** item_id 形如 run/episode/stem，
   直接在各数据根下按存在性定位 observations 目录。队列重建过一次，
   52% 的记录已经不是当前队列的成员，如果按队列查表就全丢了。
2. **真值自己算。** 用 GT 深度按 task_metric.front_near 的口径算前方距离，
   不依赖队列里存的 gt_dist（那是生成队列时的快照）。
3. **冲突要标出来，不能默认标签是对的。** 复核记录里的人判"不该触发"、
   而真值深度显示障碍就在 9 米内，这类记录与深度监督直接矛盾；训练时可以
   丢掉（--drop-conflict），也可以降权，但必须可统计。

产出
----
cache/loop_labels.jsonl        每条一个可训练样本（含权重与冲突标记）
cache/loop_labels_report.json  统计：解决率、冲突率、重复标注一致性

用法
----
  python scripts/loop_labels.py
  python scripts/loop_labels.py --threshold 9 --margin 1.5 --drop-conflict
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
import statistics
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(f"{REPO}")
REVIEWS = ROOT / "review" / "reviews.jsonl"
OUT_LABELS = ROOT / "cache" / "loop_labels.jsonl"
OUT_REPORT = ROOT / "cache" / "loop_labels_report.json"

ROOTS = [Path(f"{DATA}/runs"), Path(f"{DATA}/runs_batch2"),
         Path(f"{DATA}/runs_batch3"), Path(f"{DATA}/runs_batch4"),
         Path(f"{DATA}/runs_nearfield"), Path(f"{DATA}/near_out"),
         Path(f"{DATA}/near_out2")]

POSITIVE = {"accept"}          # 人认为这次触发成立
NEGATIVE = {"reject"}          # 人认为不该触发
ABSTAIN = {"hold", "unknown"}  # 不产生监督


def index_runs() -> dict[str, Path]:
    """run/episode -> observations 目录。只扫一层目录，够快。"""
    idx: dict[str, Path] = {}
    for root in ROOTS:
        if not root.is_dir():
            continue
        for run in root.glob("run_*"):
            for ep in run.glob("episode_*"):
                obs = ep / "observations"
                if obs.is_dir():
                    idx["%s/%s" % (run.name, ep.name)] = obs
    return idx


def front_near(depth: np.ndarray) -> float:
    h, w = depth.shape
    roi = depth[h // 4: 3 * h // 4, w // 3: 2 * w // 3]
    roi = roi[np.isfinite(roi) & (roi > 0)]
    return float(np.percentile(roi, 5)) if roi.size else float("nan")


def resolve(item_id: str, idx: dict[str, Path]):
    """item_id -> (obs_dir, stem)；定位不到返回 (None, None)。"""
    parts = item_id.split("/")
    if len(parts) != 3:
        return None, None
    key = "%s/%s" % (parts[0], parts[1])
    return idx.get(key), parts[2]


def kappa(pairs: list[tuple[int, int]]) -> float | None:
    """两个评审都给出明确判定时的 Cohen's kappa。"""
    if len(pairs) < 2:
        return None
    n = len(pairs)
    po = sum(1 for a, b in pairs if a == b) / n
    pa = sum(a for a, _ in pairs) / n
    pb = sum(b for _, b in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe >= 1 else (po - pe) / (1 - pe)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=9.0, help="避障判据（米）")
    ap.add_argument("--margin", type=float, default=1.5, help="铰链约束的安全余量（米）")
    ap.add_argument("--drop-conflict", action="store_true",
                    help="把与真值矛盾的标签权重置 0")
    ap.add_argument("--reviews", default=str(REVIEWS))
    args = ap.parse_args()

    idx = index_runs()
    print("数据根内可用航段 %d 个" % len(idx))

    reviews = [json.loads(l) for l in Path(args.reviews).read_text(encoding="utf-8").splitlines()
               if l.strip()]
    print("复核记录 %d 条，%d 位评审" % (len(reviews), len({r["reviewer"] for r in reviews})))

    # 1) 逐条解析
    per_item: dict[str, list[dict]] = defaultdict(list)
    status = Counter()
    for r in reviews:
        obs, stem = resolve(r["item_id"], idx)
        if obs is None:
            status["找不到航段"] += 1
            per_item[r["item_id"]].append(dict(r, resolved=False))
            continue
        img = obs / ("%s_front_rgb.png" % stem)
        dep = obs / ("%s_front_depth.npy" % stem)
        if not (img.exists() and dep.exists()):
            status["缺图像或深度"] += 1
            per_item[r["item_id"]].append(dict(r, resolved=False))
            continue
        status["可用"] += 1
        per_item[r["item_id"]].append(dict(r, resolved=True, obs=str(obs),
                                           image=str(img), depth=str(dep), stem=stem))

    # 2) 每条样本：真值、聚合判定、冲突
    rows = []
    for item_id, recs in per_item.items():
        good = [r for r in recs if r.get("resolved")]
        if not good:
            continue
        obs = Path(good[0]["obs"])
        stem = good[0]["stem"]
        gt_front = front_near(np.load(obs / ("%s_front_depth.npy" % stem)))
        if not np.isfinite(gt_front):
            continue

        votes = {}
        for r in good:
            v = r.get("verdict")
            votes[v] = votes.get(v, 0) + 1
        # 明确判定取多数；平票则放弃
        pos = sum(c for v, c in votes.items() if v in POSITIVE)
        neg = sum(c for v, c in votes.items() if v in NEGATIVE)
        abst = sum(c for v, c in votes.items() if v in ABSTAIN)
        if pos == neg:
            label = None
            weight = 0.0
        else:
            label = 1 if pos > neg else 0
            agree = max(pos, neg) / max(1, pos + neg)
            weight = 1.0 if (pos + neg) == 1 else round(0.5 + 0.5 * agree, 3)
        conflict = label is not None and ((label == 0) == (gt_front < args.threshold))
        if label is None and abst == 0:
            weight = 0.0
        if conflict and args.drop_conflict:
            weight = 0.0

        rows.append({
            "item_id": item_id, "run": item_id.split("/")[0],
            "episode": item_id.split("/")[1], "stem": stem,
            "obs_dir": str(obs), "image": good[0]["image"], "depth": good[0]["depth"],
            "gt_front": round(gt_front, 3),
            "label": label,                      # 1=该触发, 0=不该触发, None=放弃
            "weight": weight if label is not None else 0.0,
            "conflict": bool(conflict),
            "n_rater": len(good), "votes": votes,
            "seconds": int(statistics.median([r.get("seconds", 0) for r in good])),
            "reasons": sorted({x for r in good for x in (r.get("reasons") or [])}),
        })

    # 3) 一致性：同一帧被两位以上评审判过时的两两比较
    pair_votes = []
    for item_id, recs in per_item.items():
        vals = []
        for r in recs:
            if not r.get("resolved"):
                continue
            v = r.get("verdict")
            if v in POSITIVE:
                vals.append(1)
            elif v in NEGATIVE:
                vals.append(0)
        pair_votes.extend(combinations(vals, 2))
    k = kappa(pair_votes)

    # 4) 落盘
    OUT_LABELS.parent.mkdir(parents=True, exist_ok=True)
    with OUT_LABELS.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    usable = [r for r in rows if r["weight"] > 0]
    report = {
        "reviews": len(reviews), "reviewers": len({r["reviewer"] for r in reviews}),
        "resolve": dict(status),
        "items": len(rows), "usable": len(usable),
        "label_balance": dict(Counter(r["label"] for r in rows)),
        "conflict": sum(1 for r in rows if r["conflict"]),
        "conflict_rate": round(sum(1 for r in rows if r["conflict"]) / max(1, len(rows)), 3),
        "kappa": None if k is None else round(k, 3),
        "double_annotated_items": sum(1 for c in Counter(r["item_id"] for r in reviews).values() if c > 1),
        "threshold": args.threshold, "margin": args.margin,
        "drop_conflict": bool(args.drop_conflict),
        "near_gt": sum(1 for r in rows if r["gt_front"] < args.threshold),
    }
    OUT_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    print("\n解析: " + " / ".join("%s %d" % (k2, v) for k2, v in status.items()))
    print("可用样本 %d（其中权重>0 的 %d）" % (report["items"], report["usable"]))
    print("标签分布: %s" % report["label_balance"])
    print("与真值冲突 %d 条（%.0f%%）；真值<%.0f m 的 %d 条"
          % (report["conflict"], 100 * report["conflict_rate"], args.threshold, report["near_gt"]))
    print("重复标注一致性 kappa: %s（%d 对）" % (report["kappa"], len(pair_votes)))
    print("已写 %s 与 %s" % (OUT_LABELS, OUT_REPORT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
