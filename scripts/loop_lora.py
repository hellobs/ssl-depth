# -*- coding: utf-8 -*-
"""闭环第三环：MicroJEPA 的 LoRA 微调管线。

旧版（ResNet 卷积 LoRA）已删除：论文主线转向世界模型，微调对象随之
换成 MicroJEPA——它正是本文检验的那条域内世界模型路线，LoRA 回答的
问题是「冻结是不是在亏待它的表征」。

结构
----
    本文训练集（17k 帧，视差监督）────────────┐
    近场新数据（可选，视差监督，加权采样）────┤
    复核标签（可选，帧级铰链）────────────────┤
                                              ▼
        MicroJEPA encoder（冻结） + ViT LoRA(r) → DepthHead(192)
                                              │
              掩码视差 L1  +  w_hinge × 帧级铰链（可选）
                                              ▼
                    held 集评估（与 tab:backbone 同口径）+ 外部近场集

为什么 LoRA 加在这三个位置
--------------------------
MicroJEPA 的 encoder 是 6 层 Block（nn.MultiheadAttention + MLP，dim 192）。
LoRA 注入每层的 mlp.fc1、mlp.fc2 与 attn.out_proj。MHA 的 in_proj 是单个
Parameter（不是模块），无法无损旁路，故 QKV 不做 LoRA——fc1/fc2/out_proj
已覆盖 MLP 与输出投影的主要容量。r=8 时全部 LoRA 约 0.11M 参数。

与 ResnetBackbone 的关键差异：JepaBackbone.forward 用 no_grad 包住 encoder，
LoRA 参数拿不到梯度；JepaLoRA 重写 forward 去掉 no_grad，冻结参数本身
requires_grad=False，不浪费计算。

用法
----
  # 纯视差 LoRA（回答表征问题，与 tab:backbone 可比）
  python scripts/loop_lora.py --train --epochs 8 --w-hinge 0

  # 带帧级铰链（决策级目标）
  python scripts/loop_lora.py --train --epochs 8 --w-hinge 0.5

  # 只做外部近场验收
  python scripts/loop_lora.py --eval-only --ckpt runs/loop_xxx/ckpt.pt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_backbone import IMG_H, IMG_W, MAX_DEPTH, MIN_DEPTH, DepthHead, Frames, find_obs  # noqa: E402
from task_metric import front_near  # noqa: E402

ROOT = Path("D:/zzr/ssl-depth")
LABELS = ROOT / "cache" / "loop_labels.jsonl"


# ---------------------------------------------------------------- LoRA
class LinearLoRA(nn.Module):
    """给冻结的线性层加低秩旁路：y = W x + scale * B(A x)。"""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.scale = alpha / r
        self.A = nn.Linear(base.in_features, r, bias=False)
        self.B = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.B.weight)          # 初始为恒等，不扰动预训练表征

    def forward(self, x):
        return self.base(x) + self.scale * self.B(self.A(x))


def inject_vit_lora(encoder: nn.Module, r: int = 8, alpha: int = 16) -> int:
    """把 encoder 每个 Block 的 mlp.fc1/fc2 换成 LoRA，返回替换个数。

    MicroJEPA 的 Block 结构（microwm/models/microwm.py）：
        norm1 → attn(nn.MultiheadAttention) → norm2 → mlp(nn.Sequential[Linear, GELU, Linear])

    注意 attn.out_proj 不注入：nn.MultiheadAttention.forward 内部用
    `linear(x, self.out_proj.weight, self.out_proj.bias)` 直接取权重，
    不走模块调用，包装成 LoRA 会因缺少 .weight 报 AttributeError。
    in_proj 是单个 Parameter，同样无法旁路。每层的容量主要在 MLP
    （192→768→192，15360 参数/层），只注入 fc1/fc2 是安全的折中。
    """
    n = 0
    for blk in encoder.blocks:
        blk.mlp[0] = LinearLoRA(blk.mlp[0], r, alpha)
        blk.mlp[2] = LinearLoRA(blk.mlp[2], r, alpha)
        n += 2
    return n


class JepaLoRA(nn.Module):
    """MicroJEPA encoder + ViT LoRA。forward 不包 no_grad，LoRA 才有梯度。"""

    def __init__(self, ckpt: str | Path, r: int = 8, alpha: int = 16):
        super().__init__()
        from diag_backbone import JepaBackbone
        base = JepaBackbone(Path(ckpt))
        self.enc = base.enc
        self.tok_h, self.tok_w = base.tok_h, base.tok_w
        self.n_lora = inject_vit_lora(self.enc, r, alpha)

    def train(self, mode: bool = True):
        super().train(mode)
        self.enc.eval()          # LN 统计一律冻结
        return self

    def forward(self, x):
        # MicroJEPA 的 Encoder 要 (B, T, C, H, W)；单帧训练传的是 4 维，补一个时间维
        if x.dim() == 4:
            x = x.unsqueeze(1)
        z = self.enc(x)          # [B, 1, N, D]
        return z[:, 0], self.tok_h, self.tok_w

    def trainable(self):
        return [p for p in self.parameters() if p.requires_grad]


# ---------------------------------------------------------------- 数据
def load_frame(image_path: str, depth_path: str):
    """与 diag_backbone.Frames 完全一致的前处理，保证与既有指标可比。"""
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(image_path)
    rgb = cv2.cvtColor(cv2.resize(bgr, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA),
                       cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
    dep = np.load(depth_path).astype(np.float32)
    dep = cv2.resize(dep, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
    val = np.isfinite(dep) & (dep >= MIN_DEPTH) & (dep <= MAX_DEPTH)
    disp = np.where(val, 1.0 / np.maximum(dep, 1e-3), 0.0).astype(np.float32)
    return (torch.from_numpy(rgb).permute(2, 0, 1),
            torch.from_numpy(disp)[None],
            torch.from_numpy(val.astype(np.float32))[None])


class LabelFrames(Dataset):
    """复核标签帧：返回图像、深度、掩码，以及帧级标签与权重。"""

    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        rgb, disp, mask = load_frame(r["image"], r["depth"])
        label = -1 if r["label"] is None else int(r["label"])
        return rgb, disp, mask, torch.tensor(float(r["weight"]), dtype=torch.float32), \
            torch.tensor(label, dtype=torch.long), r["item_id"]


class NearFrames(Dataset):
    """近场数据：只有深度监督，没有人的标签。"""

    def __init__(self, obs_dirs: list[tuple[Path, str]], limit: int = 0):
        self.items = obs_dirs[:limit] if limit else obs_dirs

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        obs, stem = self.items[i]
        rgb, disp, mask = load_frame(str(obs / ("%s_front_rgb.png" % stem)),
                                     str(obs / ("%s_front_depth.npy" % stem)))
        return rgb, disp, mask, torch.tensor(0.0), torch.tensor(-1), "%s/%s" % (obs, stem)


def scan_near(root: Path, limit: int = 0) -> list[tuple[Path, str]]:
    out = []
    for run in sorted(root.glob("run_*")):
        for ep in sorted(run.glob("episode_*")):
            obs = ep / "observations"
            if not obs.is_dir():
                continue
            for p in sorted(obs.glob("*_front_depth.npy")):
                out.append((obs, p.name.replace("_front_depth.npy", "")))
                if limit and len(out) >= limit:
                    return out
    return out


class MixedFrames(Dataset):
    """主训练集 + 近场重复采样。

    必须是模块级类：Windows 的 DataLoader 用 spawn 起 worker，内部类无法 pickle。
    """

    def __init__(self, base: Dataset, near: Dataset | None, near_weight: float = 1.0):
        self.base = base
        self.near = near
        self.n_near = int(len(near) * near_weight) if near is not None else 0

    def __len__(self):
        return len(self.base) + self.n_near

    def __getitem__(self, i):
        if i < len(self.base):
            rgb, disp, mask = self.base[i]
            # 补三个占位，与 NearFrames 的输出格式对齐，否则 collate 报长度不等
            return rgb, disp, mask, torch.tensor(0.0), torch.tensor(-1), ""
        return self.near[(i - len(self.base)) % max(1, len(self.near))]


# ---------------------------------------------------------------- 损失
def roi_front(depth: torch.Tensor, mask: torch.Tensor, q: float = 0.05) -> torch.Tensor:
    """ROI 内前方距离的 q 分位（与 task_metric.front_near 同口径，可求导）。"""
    b, h, w = depth.shape
    d = depth[:, h // 4: 3 * h // 4, w // 3: 2 * w // 3]
    m = mask[:, h // 4: 3 * h // 4, w // 3: 2 * w // 3]
    d = torch.where(m > 0.5, d, torch.full_like(d, MAX_DEPTH))
    flat = d.reshape(b, -1)
    return torch.quantile(flat, q, dim=1)


def hinge_loss(front: torch.Tensor, label: torch.Tensor, weight: torch.Tensor,
               T: float, margin: float) -> tuple[torch.Tensor, int]:
    """把帧级判定翻译成对前方距离的约束；只统计 weight>0 且有明确标签的帧。"""
    keep = (weight > 0) & (label >= 0)
    if int(keep.sum()) == 0:
        return torch.zeros((), device=front.device), 0
    f = front[keep]
    lab = label[keep]
    w = weight[keep]
    pos = F.relu(f - (T - margin))             # 人判该触发：别读远
    neg = F.relu((T + margin) - f)             # 人判不该触发：别读近
    per = torch.where(lab == 1, pos, neg)
    return (per * w).sum() / w.sum().clamp(min=1e-6), int(keep.sum())


# ---------------------------------------------------------------- 评估
@torch.no_grad()
def eval_external(bb, head, dl, dev, T: float) -> dict:
    """外部近场集：逐帧前方距离，看 δ1 与 9 米判据的召回/误触发。"""
    head.eval(); bb.eval()
    ps, gs = [], []
    for rgb, disp_gt, mask, *_ in dl:
        rgb, mask = rgb.to(dev), mask.to(dev)
        tok, th, tw = bb(rgb.unsqueeze(1))
        disp = head(tok, th, tw)
        depth_p = 1.0 / torch.clamp(disp.squeeze(1), min=1e-3)
        depth_g = 1.0 / torch.clamp(disp_gt.squeeze(1).to(dev), min=1e-3)
        ps.append(roi_front(depth_p, mask.squeeze(1)).cpu().numpy())
        gs.append(roi_front(depth_g, mask.squeeze(1)).cpu().numpy())
    p, g = np.concatenate(ps), np.concatenate(gs)
    ok = np.isfinite(p) & np.isfinite(g) & (g > 0)
    p, g = p[ok], g[ok]
    s = float(np.median(p / g)) if p.size else 1.0
    ratio = np.maximum(p / s / g, g / (p / s))
    # 判据必须用原始预测：部署时拿不到真值，没法做中位对齐。
    # 对齐后的 δ1 / AbsRel 只是报告口径，两者分开算。
    hit = (p < T) & (g < T)
    miss = (p >= T) & (g < T)
    false = (p < T) & (g >= T)
    out = {"n": int(p.size), "scale": round(s, 4),
           "absrel_med": round(float(np.mean(np.abs(p / s - g) / g)), 4),
           "d1": round(float((ratio < 1.25).mean()), 4),
           "front_median_pred_raw": round(float(np.median(p)), 2),
           "front_median_gt": round(float(np.median(g)), 2),
           "recall": round(float(hit.sum() / max(1, (g < T).sum())), 4),
           "miss": int(miss.sum()), "false_trigger": int(false.sum()),
           "n_near_gt": int((g < T).sum())}
    return out


@torch.no_grad()
def eval_held(bb, head, idx: dict, eps: list[str], dev, batch: int = 16) -> dict:
    """held 集评估：像素级与帧级都与 tab:backbone / tab:task 同口径。"""
    head.eval(); bb.eval()
    dl = DataLoader(Frames(idx, eps), batch_size=batch, num_workers=0)
    ps, gs, fp, fg = [], [], [], []
    for rgb, disp_gt, mask in dl:
        tok, th, tw = bb(rgb.to(dev))
        disp = head(tok, th, tw)
        pred_d = 1.0 / torch.clamp(disp.squeeze(1), min=1e-3)
        gt_d = 1.0 / torch.clamp(disp_gt.squeeze(1).to(dev), min=1e-3)
        m = mask.squeeze(1).to(dev) > 0.5
        for j in range(pred_d.shape[0]):
            g = gt_d[j].cpu().numpy(); p = pred_d[j].cpu().numpy(); mm = m[j].cpu().numpy()
            g = np.where(mm, g, np.nan); p = np.where(mm, p, np.nan)
            ps.append(p[np.isfinite(p) & (p > 0)]); gs.append(g[np.isfinite(g) & (g > 0)])
            fp.append(front_near(p)); fg.append(front_near(g))
    p = np.concatenate(ps); g = np.concatenate(gs)
    fm = np.array(fp); fg = np.array(fg)
    res = dict(d1=float((np.maximum(p / g, g / p) < 1.25).mean()),
               absrel=float(np.mean(np.abs(p - g) / g)),
               rmse=float(np.sqrt(np.mean((p - g) ** 2))),
               gt_median=float(np.median(fg)), pr_median=float(np.median(fm)),
               n_frame=int(fm.size))
    for t in (6.0, 9.0, 12.0):
        dg, dp = fg < t, fm < t
        res["t%d" % int(t)] = dict(
            hit=int((dg & dp).sum()), miss=int((dg & ~dp).sum()), false=int((~dg & dp).sum()),
            recall=float((dg & dp).sum() / max(1, (dg & dp).sum() + (dg & ~dp).sum())))
    return res


# ---------------------------------------------------------------- 主流程
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true", help="真正训练（默认 dry-run）")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--labels", default=str(LABELS))
    ap.add_argument("--near-root", default=None,
                    help="额外近场数据（只做深度监督）；不要与 --eval-root 相同")
    ap.add_argument("--near-limit", type=int, default=0)
    ap.add_argument("--near-weight", type=float, default=1.0,
                    help="近场帧在训练里的重复采样次数")
    ap.add_argument("--eval-root", default="D:/zzr/wmdata/near_out2")
    ap.add_argument("--eval-limit", type=int, default=256)
    ap.add_argument("--ckpt", default=None, help="本管线训练出的权重")
    ap.add_argument("--backbone-ckpt", default=str(ROOT / "runs/jepa_domain_v5/jepa.pt"),
                    help="被 LoRA 的 MicroJEPA 预训练权重")
    ap.add_argument("--threshold", type=float, default=9.0)
    ap.add_argument("--margin", type=float, default=1.5)
    ap.add_argument("--w-hinge", type=float, default=0.0,
                    help="帧级铰链权重；0 = 纯视差（表征问题）")
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=0, help="每轮最多几步（冒烟用）")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4, help="DataLoader 进程数；0=单进程（慢）")
    ap.add_argument("--lr-lora", type=float, default=2e-4)
    ap.add_argument("--lr-head", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)

    full = json.loads((ROOT / "cache/kept_all_v5.json").read_text(encoding="utf-8"))
    sp = json.loads((ROOT / "runs/v5_split.json").read_text(encoding="utf-8"))["split"]

    bb = JepaLoRA(a.backbone_ckpt, a.lora_r, a.lora_alpha).to(dev)
    head = DepthHead(192).to(dev)
    print("ViT LoRA 注入 %d 个线性层（r=%d, alpha=%d）| 骨干 %s"
          % (bb.n_lora, a.lora_r, a.lora_alpha, Path(a.backbone_ckpt).name))
    n_lora = sum(p.numel() for p in bb.parameters() if p.requires_grad)
    n_head = sum(p.numel() for p in head.parameters())
    n_base = sum(p.numel() for p in bb.parameters())
    print("可训练: LoRA %.2fM | head %.3fM | 冻结 encoder %.2fM（可训练占 %.1f%%）"
          % (n_lora / 1e6, n_head / 1e6, n_base / 1e6,
             100 * (n_lora + n_head) / (n_base + n_lora + n_head)))

    if a.ckpt:
        ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
        bb.load_state_dict(ck["backbone"], strict=False)
        head.load_state_dict(ck["head"])
        print("已载入本管线权重 %s" % a.ckpt)

    # 外部验收集（永远不参与训练）
    eval_root = Path(a.eval_root)
    if a.near_root and Path(a.near_root).resolve() == eval_root.resolve():
        raise SystemExit("--near-root 与 --eval-root 相同，训练集会污染验收集")
    dl_eval = None
    if eval_root.is_dir():
        items = scan_near(eval_root, a.eval_limit)
        dl_eval = DataLoader(NearFrames(items), batch_size=8, num_workers=0)
        print("外部近场集 %s：%d 帧" % (eval_root, len(items)))
    else:
        print("外部近场集不存在：%s" % eval_root)

    if a.eval_only:
        if not a.ckpt:
            raise SystemExit("--eval-only 需要 --ckpt")
        m = eval_held(bb, head, full, sp["val"] + sp["test"], dev)
        print("held %s" % json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                      for k, v in m.items()}, ensure_ascii=False))
        if dl_eval:
            print("外部近场 %s" % json.dumps(eval_external(bb, head, dl_eval, dev, a.threshold),
                                             ensure_ascii=False))
        return 0

    # 训练数据：主 split 全帧 + 可选近场重复采样
    ds_train = Frames(full, sp["train"])
    near = []
    if a.near_root:
        near = scan_near(Path(a.near_root), a.near_limit)
        print("近场补充（仅深度监督）：%d 帧 × %.0f 次采样" % (len(near), a.near_weight))
    near_ds = NearFrames(near) if near else None
    dl_train = DataLoader(MixedFrames(ds_train, near_ds, a.near_weight),
                          batch_size=a.batch, shuffle=True, num_workers=a.workers,
                          drop_last=True, pin_memory=True,
                          persistent_workers=a.workers > 0)
    print("训练帧 %d（近场 %d）| 每轮 %d 步" % (
        len(ds_train) + (int(len(near) * a.near_weight) if near else 0), len(near), len(dl_train)))

    # 复核标签（w_hinge>0 才加载）
    dl_label = None
    if a.w_hinge > 0:
        rows = [json.loads(l) for l in Path(a.labels).read_text(encoding="utf-8").splitlines()
                if l.strip()]
        usable = [r for r in rows if r["weight"] > 0]
        print("复核标签 %d 条（权重>0 的 %d 条）" % (len(rows), len(usable)))
        dl_label = DataLoader(LabelFrames(rows), batch_size=min(a.batch, max(1, len(rows))),
                              shuffle=True, num_workers=0, drop_last=True)

    opt = torch.optim.AdamW([
        {"params": [p for p in bb.parameters() if p.requires_grad], "lr": a.lr_lora},
        {"params": head.parameters(), "lr": a.lr_head},
    ], weight_decay=0.01)
    total = a.epochs * len(dl_train)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total))

    stamp = time.strftime("%Y%m%d_%H%M")
    outdir = Path(a.out) if a.out else ROOT / "runs" / ("jepa_lora_%s" % stamp)
    outdir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    step = 0
    lab_iter = iter(dl_label) if dl_label else None
    for ep in range(a.epochs):
        bb.train(); head.train()
        tot = n = 0
        for rgb, disp_gt, mask, *_ in dl_train:
            rgb, disp_gt, mask = rgb.to(dev), disp_gt.to(dev), mask.to(dev)
            tok, th, tw = bb(rgb)
            disp = head(tok, th, tw)
            l_depth = (torch.abs(disp - disp_gt) * mask).sum() / mask.sum().clamp(min=1.0)
            loss = l_depth
            if dl_label is not None and a.w_hinge > 0:
                try:
                    rgb2, disp2, mask2, w2, lab2, _ = next(lab_iter)
                except StopIteration:
                    lab_iter = iter(dl_label)
                    rgb2, disp2, mask2, w2, lab2, _ = next(lab_iter)
                rgb2, disp2, mask2 = rgb2.to(dev), disp2.to(dev), mask2.to(dev)
                w2, lab2 = w2.to(dev), lab2.to(dev)
                tok2, th2, tw2 = bb(rgb2)
                disp2 = head(tok2, th2, tw2)
                front = roi_front(1.0 / torch.clamp(disp2.squeeze(1), min=1e-3),
                                  mask2.squeeze(1))
                l_h, _ = hinge_loss(front, lab2, w2, a.threshold, a.margin)
                loss = loss + a.w_hinge * l_h
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
            tot += float(loss.detach()); n += 1; step += 1
            if a.max_steps and n >= a.max_steps:
                break
        msg = "epoch %2d/%d loss %.4f | %.0fs" % (ep + 1, a.epochs, tot / max(1, n), time.time() - t0)
        if dl_eval:
            m = eval_external(bb, head, dl_eval, dev, a.threshold)
            msg += " | 近场 recall %.2f miss %d false %d" % (m["recall"], m["miss"], m["false_trigger"])
        print(msg, flush=True)
        if a.max_steps and step >= a.max_steps:
            break

    ckpt = outdir / "ckpt.pt"
    torch.save({"backbone": bb.state_dict(), "head": head.state_dict(),
                "config": {"lora_r": a.lora_r, "alpha": a.lora_alpha,
                           "threshold": a.threshold, "margin": a.margin,
                           "w_hinge": a.w_hinge, "epochs": a.epochs,
                           "backbone_ckpt": a.backbone_ckpt}}, ckpt)
    print("已保存 %s" % ckpt)

    m = eval_held(bb, head, full, sp["val"] + sp["test"], dev)
    print("held: d1 %.3f | AbsRel %.3f | RMSE %.1f | 9 米识别 %.1f%% 误刹 %d"
          % (m["d1"], m["absrel"], m["rmse"], 100 * m["t9"]["recall"], m["t9"]["false"]))
    if dl_eval:
        m2 = eval_external(bb, head, dl_eval, dev, a.threshold)
        print("外部近场: %s" % json.dumps(m2, ensure_ascii=False))
    json.dump(dict(model="JEPA LoRA", **m), open(outdir / "metrics.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
