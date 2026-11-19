"""域内 JEPA 自监督预训练（只用 RGB，不用深度标签）。

与 diag_backbone.py 的分工：
- diag_backbone 只**加载**现成权重（VisDrone），测「已有表征有没有用」；
- 本脚本在**本数据**上从头自监督，产出域内权重，作为语料这一维的第三个点。

默认超参直接取下游规格（108×192 / patch 6 / dim 192 / depth 6 / heads 3），
因为 MicroJEPA 的位置编码是**定长相加**、不插值：token 数差一个，pos 就会在
加载时被形状过滤掉，冻结主干带着随机位置先验跑且不报错。存盘后会用下游配置
实例化一次做自检（verify_loadable）。

用法（先冒烟，不占 GPU）：
    python scripts/pretrain_jepa.py --smoke
正式（排除测试趟，按 base_s6 的 4000 步对齐）：
    python scripts/pretrain_jepa.py --skip-runs 1d3aec79,cf2e5d5e --steps 4000
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, "D:/zzr/microwm/src")
from microwm.models.microwm import MicroJEPA  # noqa: E402

# 与 diag_backbone.py 同源：六批数据分处不同根目录，按存在性定位。
ROOTS = [Path("D:/zzr/wmdata/runs"), Path("D:/zzr/wmdata/runs_batch2"),
         Path("D:/zzr/wmdata/runs_batch3"), Path("D:/zzr/wmdata/runs_batch4"),
         Path("D:/zzr/wmdata/runs_nearfield"), Path("D:/zzr/near_out")]
OBS = "observations"

# 下游 JepaBackbone 的规格（= diag_backbone.JEPA_CFG）。位置编码是定长相加，
# 没有插值，所以预训练产出的 token 数必须与下游逐位一致；否则 pos 会在加载时
# 被形状过滤掉，冻结主干带着随机位置先验跑，而且全程不报错。
DOWNSTREAM = {"img_h": 108, "img_w": 192, "patch": 6, "dim": 192,
              "depth": 6, "heads": 3}

# 评测划分的唯一权威来源：v4 训练时写下的 split。
# 不能现算，也不能用 cache/kept_all.json —— 索引在训练之后重建过
# （补采 9 段、清理 3 段），kept_all 是 104 段，而论文用的是 98 段，两套不同。
V4_SPLIT = Path("D:/zzr/ssl-depth/runs/v4_imagenet/metrics.json")

# 只评测、永不训练的数据（例如新采的外部验收集）。v4 的留出集管不到它们，
# 所以必须单独声明，否则"断言通过"只是假象。
HELD_OUT_ONLY = Path("D:/zzr/ssl-depth/runs/held_out_only.txt")


def load_held_out_only(p: Path = HELD_OUT_ONLY) -> tuple[set[str], set[str]]:
    eps, runs = set(), set()
    if not p.exists():
        return eps, runs
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.split("#")[0].strip()
        if not s:
            continue
        (eps if "/" in s else runs).add(s)
    return eps, runs


def assert_no_leak(eps: list[tuple[str, str]], split_path: Path = V4_SPLIT) -> None:
    """硬断言：预训练语料与评测留出集零重叠，否则拒绝开跑。

    比"按 episode 求交"更严一档：只要某个 run 下**任何**一个 episode 落在
    留出集，就判失败。因为 list_episodes 会读取该 run 的全部 episode，
    而不是只读进划分的那些。
    """
    sp = json.loads(split_path.read_text(encoding="utf-8"))["split"]
    held = set(sp["val"]) | set(sp["test"])
    used = {"%s/%s" % (r, e) for r, e in eps}
    used_runs = {r for r, _ in eps}
    held_runs = {e.split("/")[0] for e in held}
    hit_eps = sorted(used & held)
    hit_runs = sorted(used_runs & held_runs)

    print("泄漏自检（划分来源 %s）" % split_path.name, flush=True)
    print("  语料 %d 航段 / 留出 %d 航段 | 航段交集 %d | run 交集 %d"
          % (len(used), len(held), len(hit_eps), len(hit_runs)), flush=True)
    if hit_eps or hit_runs:
        for h in hit_eps[:5]:
            print("  冲突航段:", h, flush=True)
        for r in hit_runs[:5]:
            print("  冲突 run:", r, flush=True)
        raise SystemExit("预训练语料与评测留出集重叠，拒绝开跑。")

    # 第二道：仅评测名单（v4 划分之外的数据）
    h_eps, h_runs = load_held_out_only()
    hit2 = (used & h_eps) | {e for e in used if e.split("/")[0] in h_runs}
    print("  仅评测名单 %d 条 run / %d 条航段 | 交集 %d"
          % (len(h_runs), len(h_eps), len(hit2)), flush=True)
    if hit2:
        for h in sorted(hit2)[:5]:
            print("  冲突航段:", h, flush=True)
        raise SystemExit("预训练语料含仅评测数据（held_out_only.txt），拒绝开跑。")
    print("  通过：零重叠", flush=True)


def find_obs(run: str, episode: str) -> Path:
    for r in ROOTS:
        p = r / run / episode / OBS
        if p.is_dir():
            return p
    raise FileNotFoundError("%s/%s" % (run, episode))


def all_runs() -> list[str]:
    names = []
    for r in ROOTS:
        if r.is_dir():
            names += [p.name for p in sorted(r.iterdir()) if p.is_dir()]
    return names


def list_episodes(runs: list[str] | None, skip: list[str] | None = None):
    out = []
    for name in all_runs():
        if runs and not any(name.endswith(r) for r in runs):
            continue
        if skip and any(name.endswith(s) for s in skip):
            continue
        root = next(r for r in ROOTS if (r / name).is_dir())
        for ep in sorted((root / name).glob("episode_*")):
            if (ep / OBS).is_dir():
                out.append((name, ep.name))
    return out


class FramePairs(Dataset):
    """同一 episode 内相邻两帧；跨 episode 不组对，避免把场景切换当成运动。

    注意输入**不是正方形**：下游按 108×192 喂图，这里必须一致，否则 token 数
    （108/6 × 192/6 = 576）对不上，位置编码会被静默丢弃。
    """

    def __init__(self, eps: list[tuple[str, str]], img_h: int, img_w: int,
                 limit: int = 0, gap: int = 1):
        self.items, self.img_h, self.img_w = [], img_h, img_w
        for run, ep in eps:
            obs = find_obs(run, ep)
            stems = sorted(p.name[:6] for p in obs.glob("*_front_rgb.png"))
            # gap=1 时相邻帧只隔 0.2 秒、位移 0.3 米，预测目标几乎等于输入，
            # 损失可以被"复制上一帧"最小化，表征学不到几何。拉大到 4~5 帧
            # （0.8~1.0 秒）才逼模型编码运动与结构。
            for a, b in zip(stems, stems[gap:]):
                self.items.append((run, ep, a, b))
        if limit:
            random.Random(0).shuffle(self.items)
            self.items = self.items[:limit]
        self.mean = np.array([0.5, 0.5, 0.5], np.float32)

    def __len__(self):
        return len(self.items)

    def _read(self, run, ep, stem):
        p = find_obs(run, ep) / f"{stem}_front_rgb.png"
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        rgb = cv2.resize(bgr, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(rgb).permute(2, 0, 1)

    def __getitem__(self, i):
        run, ep, a, b = self.items[i]
        return self._read(run, ep, a), self._read(run, ep, b)


def make_cfg(img_h: int, img_w: int, patch: int, dim: int, depth: int, heads: int) -> dict:
    return {"model": {"patch": patch, "dim": dim, "depth": depth, "heads": heads,
                      "mlp_ratio": 4, "in_ch": 3, "pred_depth": 4,
                      "ema_tau": 0.996, "vicreg_var": 0.04, "vicreg_cov": 0.01,
                      "vicreg_tvar": 0.01, "pred_metric": "l1"},
            "data": {"img_h": img_h, "img_w": img_w,
                     "grid_h": img_h // patch, "grid_w": img_w // patch}}


def verify_loadable(ckpt: Path) -> bool:
    """用下游配置实例化一次，按形状过滤统计命中率——直接验"这份权重能不能用"。"""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from diag_backbone import JEPA_CFG
    from microwm.models.microwm import MicroJEPA

    full = MicroJEPA(JEPA_CFG)
    own = full.state_dict()
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
    keep = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
    missing = [k for k in own if k not in keep]
    pos_ok = "encoder.pos" in keep
    print("下游加载自检: %d/%d 张量命中 | encoder.pos %s | 未命中 %d 个"
          % (len(keep), len(own), "OK" if pos_ok else "**丢失**", len(missing)),
          flush=True)
    if missing[:3]:
        print("  未命中样例:", missing[:3], flush=True)
    return pos_ok and len(keep) > 0.9 * len(own)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="", help="逗号分隔的 run 名后缀；空表示全部")
    ap.add_argument("--skip-runs", default="", help="排除的 run 名后缀（默认排除测试趟）")
    ap.add_argument("--img-h", type=int, default=DOWNSTREAM["img_h"])
    ap.add_argument("--img-w", type=int, default=DOWNSTREAM["img_w"])
    ap.add_argument("--patch", type=int, default=DOWNSTREAM["patch"])
    ap.add_argument("--dim", type=int, default=DOWNSTREAM["dim"])
    ap.add_argument("--depth", type=int, default=DOWNSTREAM["depth"])
    ap.add_argument("--heads", type=int, default=DOWNSTREAM["heads"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--steps", type=int, default=0,
                    help="0 表示按 epochs 跑；给了就按总步数跑（对齐 base_s6 的 4000 步）")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4,
                    help="数据加载进程数。0 时 GPU 会有一半时间在等磁盘读 PNG")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--limit", type=int, default=0, help="只用前 N 个帧对（调试）")
    ap.add_argument("--gap", type=int, default=1,
                    help="帧对间隔。1=相邻帧（0.2 秒），4~5 才逼模型学运动")
    ap.add_argument("--smoke", action="store_true", help="3 步、64px、CPU，只验通路")
    ap.add_argument("--out", default="D:/zzr/ssl-depth/runs/jepa_domain_clean",
                    help="默认写到 clean 目录；别写回 runs/jepa_domain，那个路径曾是泄漏版")
    a = ap.parse_args()

    if a.smoke:
        a.img_h, a.img_w, a.patch, a.dim, a.depth, a.heads = 108, 192, 6, 96, 2, 4
        a.epochs, a.steps, a.batch, a.limit = 1, 0, 2, 6
        a.out = "D:/zzr/ssl-depth/runs/smoke_jepa"
        dev = "cpu"
    else:
        dev = "cuda" if torch.cuda.is_available() else "cpu"

    tok = (a.img_h // a.patch) * (a.img_w // a.patch)
    want = (DOWNSTREAM["img_h"] // DOWNSTREAM["patch"]) * (DOWNSTREAM["img_w"] // DOWNSTREAM["patch"])
    if tok != want:
        print("警告: 当前配置 %d 个 token，下游要 %d 个；位置编码是定长的，"
              "这份权重加载时会丢掉 pos。" % (tok, want), flush=True)

    runs = [s.strip() for s in a.runs.split(",") if s.strip()]
    skip = [s.strip() for s in a.skip_runs.split(",") if s.strip()]
    eps = list_episodes(runs or None, skip or None)
    assert_no_leak(eps)
    ds = FramePairs(eps, a.img_h, a.img_w, a.limit, a.gap)
    if len(ds) == 0:
        raise SystemExit("没有可用帧对：检查 --runs 与数据目录")
    print("runs %d | episodes %d | pairs %d | gap %d | img %dx%d | patch %d | token %d"
          % (len({r for r, _ in eps}), len(eps), len(ds), a.gap,
             a.img_h, a.img_w, a.patch, tok), flush=True)

    cfg = make_cfg(a.img_h, a.img_w, a.patch, a.dim, a.depth, a.heads)
    model = MicroJEPA(cfg).to(dev)
    n_enc = sum(p.numel() for p in model.encoder.parameters())
    print("encoder %.2fM | total(model incl. teacher/predictor) %.2fM"
          % (n_enc / 1e6, sum(p.numel() for p in model.parameters()) / 1e6), flush=True)

    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=a.workers,
                    drop_last=True, persistent_workers=a.workers > 0,
                    prefetch_factor=4 if a.workers > 0 else None,
                    pin_memory=(dev == "cuda"))
    if len(dl) == 0:
        raise SystemExit("帧对数（%d）少于 batch（%d），一个完整批次都凑不齐" % (len(ds), a.batch))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    if a.steps:
        steps = a.steps
        epochs = max(1, (steps + len(dl) - 1) // len(dl))
    else:
        epochs = a.epochs
        steps = epochs * len(dl)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    print("steps %d (~%d epoch x %d batch)" % (steps, epochs, len(dl)), flush=True)

    done = 0
    for ep in range(epochs):
        t0, run_loss = time.time(), 0.0
        for i, (past, future) in enumerate(dl):
            if done >= steps:
                break
            past = past.unsqueeze(1).to(dev)      # [B,1,3,H,W]
            future = future.unsqueeze(1).to(dev)
            out = model(past, future)
            loss = out["loss"]
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step()
            model.update_teacher()
            done += 1
            run_loss = 0.9 * run_loss + 0.1 * float(loss.detach()) if i else float(loss.detach())
            if a.smoke:
                print("  step %d loss %.4f pred %.4f"
                      % (done, float(loss.detach()), float(out["pred"])), flush=True)
            if done % 100 == 0:
                print("  step %d/%d loss %.4f  %.0fs" % (done, steps, run_loss, time.time() - t0),
                      flush=True)
        print("epoch %d/%d loss %.4f  %.1fs" % (ep + 1, epochs, run_loss, time.time() - t0), flush=True)
        if done >= steps:
            break

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": cfg, "step": done,
                "runs": runs or "all"}, out / "jepa.pt")
    (out / "meta.json").write_text(json.dumps(
        {"cfg": cfg, "runs": runs or "all", "skip": skip or None, "pairs": len(ds),
         "steps": done, "episodes": len(eps)}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved", out / "jepa.pt", flush=True)
    verify_loadable(out / "jepa.pt")


if __name__ == "__main__":
    main()
