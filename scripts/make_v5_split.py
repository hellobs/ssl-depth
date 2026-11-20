"""生成 v5 划分：训练集扩到全盘，val / test 原封不动沿用 v4。

为什么只扩训练集：v4 的四行结果都是在 14/15 航段的 val/test 上评的，
换掉留出集就等于把整张表连同第七章的误差分层全部作废。只扩训练集，
评测口径不变，四处改动能收敛在「训练数据变多以后各骨干怎么变」这一个问题上。

run 级排除：预训练脚本的泄漏断言是 run 级的（某个 run 只要有一个航段
落在 val/test，整个 run 都不许进语料）。这里提前按同一口径切，免得
预训练跑到一半被断言拦下。

产物：
  runs/v5_split.json          train/val/test 航段清单
  runs/pure_train_runs_v5.txt 预训练用的 run 全名（每行一个）
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("D:/zzr/ssl-depth")
V4 = ROOT / "runs/v4_imagenet/metrics.json"
IDX = ROOT / "cache/kept_all_v5.json"
HELD = ROOT / "runs/held_out_only.txt"
OUT_SPLIT = ROOT / "runs/v5_split.json"
OUT_RUNS = ROOT / "runs/pure_train_runs_v5.txt"


def main() -> None:
    idx = json.loads(IDX.read_text(encoding="utf-8"))["kept"]
    v4 = json.loads(V4.read_text(encoding="utf-8"))["split"]

    val = [e for e in v4["val"] if e in idx]
    test = [e for e in v4["test"] if e in idx]
    blocked = ({e.split("/")[0] for e in val} | {e.split("/")[0] for e in test})

    held_runs = set()
    if HELD.exists():
        for line in HELD.read_text(encoding="utf-8").splitlines():
            s = line.split("#")[0].strip()
            if s and "/" not in s:
                held_runs.add(s)

    train = sorted(e for e in idx
                   if e.split("/")[0] not in blocked and e.split("/")[0] not in held_runs)

    n_val = len(val)
    n_test = len(test)
    assert n_val == 14 and n_test == 15, "留出集不完整：val %d / test %d" % (n_val, n_test)
    assert not (set(train) & (set(val) | set(test))), "训练集与留出集重叠"

    runs = sorted({e.split("/")[0] for e in train})
    OUT_SPLIT.write_text(json.dumps(
        {"split": {"train": train, "val": val, "test": test},
         "note": "v5：训练集扩到全盘可用航段；val/test 沿用 v4 的 14/15，评测口径不变"},
        ensure_ascii=False, indent=1), encoding="utf-8")
    OUT_RUNS.write_text("\n".join(runs) + "\n", encoding="utf-8")

    def nf(eps):
        return sum(len(idx[e]) for e in eps)
    print("v5 划分：train %d 航段 / %d 帧 | val %d / %d | test %d / %d"
          % (len(train), nf(train), n_val, nf(val), n_test, nf(test)))
    print("训练 run 数 %d -> %s" % (len(runs), OUT_RUNS.name))
    print("run 级排除：val/test 涉及 %d 个 run，held_out %d 个 run"
          % (len(blocked), len(held_runs)))


if __name__ == "__main__":
    main()
