# -*- coding: utf-8 -*-
"""闭环第三环（二）：LoRA + 帧级铰链约束的微调管线。

只搭管线，不在这里跑训练。默认动作是 --dry-run：把数据、LoRA 注入、
前向反向、损失分量全部走一遍，确认接通后退出，不落权重。

结构
----
    复核记录 ──loop_labels.py──> cache/loop_labels.jsonl
                                        │
    近场新数据（可选，只做深度监督）─────┤
                                        ▼
              ResNet18（冻结） + LoRA(r, layer3/4) → DepthHead
                                        │
                 ┌──────────────────────┴──────────┐
            掩码视差 L1                   帧级铰链（来自人的判定）
                                        │
                                        ▼
                        外部近场集验收（从不参与训练）

为什么两种损失并存
------------------
掩码视差 L1 是逐像素的，它不知道"这一帧该不该触发"；人的复核只给到帧级，
没法逐像素用。铰链把帧级判定翻译成对 **ROI 内前方距离 p05** 的约束：
  人判"该触发" → 要求 f ≤ T - m（别让模型把近处读远）
  人判"不该触发" → 要求 f ≥ T + m
两类标签与真值矛盾时（人否掉了真值很近的帧），loop_labels.py 已标 conflict，
默认权重为 0，可用 --drop-conflict 强制丢弃。

用法
----
  # 只验证管线连通（默认）
  python scripts/loop_lora.py --dry-run

  # 标签统计看一眼
  python scripts/loop_lora.py --dry-run --verbose

  # 真正训练（现在不要跑，留给下一步）
  python scripts/loop_lora.py --train --epochs 10 --near-root D:/zzr/near_out2

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
from diag_backbone import IMG_H, IMG_W, MAX_DEPTH, MIN_DEPTH, DepthHead, ResnetBackbone  # noqa: E402

ROOT = Path("D:/zzr/ssl-depth")
LABELS = ROOT / "cache" / "loop_labels.jsonl"


# ---------------------------------------------------------------- LoRA
class ConvLoRA(nn.Module):
    """给冻结的卷积加一条低秩旁路：y = W x + scale * B(A x)。"""

    def __init__(self, base: nn.Conv2d, r: int = 8, alpha: int = 16):
        super().__init__()
        assert base.groups == 1, "分组卷积不做 LoRA"
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.scale = alpha / r
        self.A = nn.Conv2d(base.in_channels, r, base.kernel_size,
                           stride=base.stride, padding=base.padding, bias=False)
        self.B = nn.Conv2d(r, base.out_channels, 1, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.B.weight)          # 初始为恒等，不扰动预训练表征

    def forward(self, x):
        return self.base(x) + self.scale * self.B(self.A(x))


LAYER_INDEX = {"layer1": 4, "layer2": 5, "layer3": 6, "layer4": 7}


def inject_lora(body: nn.Module, targets=("layer3", "layer4"), r: int = 8,
                alpha: int = 16) -> int:
    """把 body 里指定 stage 的 Conv2d 换成 ConvLoRA，返回替换个数。

    body 是 nn.Sequential(*list(resnet18.children())[:-2])，子模块按顺序编号：
    0 conv1 / 1 bn1 / 2 relu / 3 maxpool / 4 layer1 / 5 layer2 / 6 layer3 / 7 layer4。
    所以 named_modules() 里看不到 "layer3" 这样的名字，得按索引取。
    """
    idxs = []
    for t in targets:
        t = str(t).strip()
        if t.isdigit():
            idxs.append(int(t))
        elif t in LAYER_INDEX:
            idxs.append(LAYER_INDEX[t])
        else:
            raise ValueError("未知 stage：%s（可用 layer1..layer4 或 0-7）" % t)
    n = 0
    for i in idxs:
        stage = body[i]
        for child_name, child in list(stage.named_children()):
            if isinstance(child, nn.Conv2d):
                setattr(stage, child_name, ConvLoRA(child, r, alpha))
                n += 1
            else:
                for sub_name, sub in list(child.named_children()):
                    if isinstance(sub, nn.Conv2d):
                        setattr(child, sub_name, ConvLoRA(sub, r, alpha))
                        n += 1
    return n


class ResnetLoRA(ResnetBackbone):
    """与 ResnetBackbone 同一套预处理与投影，只是不再用 no_grad 包住 body。"""

    def __init__(self, r: int = 8, alpha: int = 16, targets=("layer3", "layer4"),
                 pretrained: bool = True):
        super().__init__(pretrained)
        self.n_lora = inject_lora(self.body, targets, r, alpha)

    def forward(self, x):
        b = x.shape[0]
        x = x.reshape(b * x.shape[1], *x.shape[2:]) if x.dim() == 5 else x
        x = (x * 0.5 + 0.5 - self.mean) / self.std
        f = self.body(x)                       # 关键：不能包 no_grad，否则 A/B 拿不到梯度
        f = self.proj(f)
        return f.flatten(2).transpose(1, 2), f.shape[2], f.shape[3]

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


# ---------------------------------------------------------------- 主流程
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", default=True)
    ap.add_argument("--train", action="store_true", help="真正训练（默认不做）")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--labels", default=str(LABELS))
    ap.add_argument("--near-root", default=None,
                    help="额外近场数据（只做深度监督）；不要与 --eval-root 相同")
    ap.add_argument("--near-limit", type=int, default=0)
    ap.add_argument("--eval-root", default="D:/zzr/near_out2")
    ap.add_argument("--eval-limit", type=int, default=256)
    ap.add_argument("--ckpt", default=None,
                    help="本管线训练出的权重（backbone+head）")
    ap.add_argument("--head-ckpt", default="D:/zzr/ssl-depth/runs/v4_imagenet/head.pt",
                    help="既有 v4 解码头（含 proj），用来给出可信的基线")
    ap.add_argument("--threshold", type=float, default=9.0)
    ap.add_argument("--margin", type=float, default=1.5)
    ap.add_argument("--w-hinge", type=float, default=0.02)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--targets", default="layer3,layer4")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=0,
                    help="每轮最多走几步（0=不限），用于冒烟测试")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr-lora", type=float, default=1e-4)
    ap.add_argument("--lr-head", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    targets = tuple(t.strip() for t in a.targets.split(",") if t.strip())

    rows = [json.loads(l) for l in Path(a.labels).read_text(encoding="utf-8").splitlines()
            if l.strip()]
    usable = [r for r in rows if r["weight"] > 0]
    print("标签 %d 条（权重>0 的 %d 条；真值<%.0f m 的 %d 条）"
          % (len(rows), len(usable), a.threshold,
             sum(1 for r in rows if r["gt_front"] < a.threshold)))
    if a.verbose:
        for r in usable[:8]:
            print("  %-46s gt %5.2f m  label %s  w %.2f  votes %s"
                  % (r["item_id"][-46:], r["gt_front"], r["label"], r["weight"], r["votes"]))

    bb = ResnetLoRA(a.lora_r, a.lora_alpha, targets).to(dev)
    head = DepthHead(192).to(dev)
    print("LoRA 注入 %d 个卷积（r=%d, alpha=%d, targets=%s）"
          % (bb.n_lora, a.lora_r, a.lora_alpha, ",".join(targets)))
    n_lora = sum(p.numel() for p in bb.body.parameters() if p.requires_grad)
    n_proj = sum(p.numel() for p in bb.proj.parameters())
    n_head = sum(p.numel() for p in head.parameters())
    n_base = sum(p.numel() for p in bb.body.parameters())
    print("可训练参数: LoRA %.2fM | proj %.3fM | head %.3fM | 冻结主干 %.2fM（占全模型 %.1f%%）"
          % (n_lora / 1e6, n_proj / 1e6, n_head / 1e6, n_base / 1e6,
             100 * (n_lora + n_proj + n_head) / (n_base + n_lora + n_proj + n_head)))

    if a.ckpt:
        ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
        bb.load_state_dict(ck["backbone"], strict=False)
        head.load_state_dict(ck["head"])
        print("已载入本管线权重 %s" % a.ckpt)
    elif a.head_ckpt and Path(a.head_ckpt).exists():
        # v4 的权重是 {head, proj}：proj 是骨干里唯一的可训练层，必须一起载入，
        # 否则基线的 proj 仍是随机初始化，数字没意义
        ck = torch.load(a.head_ckpt, map_location=dev, weights_only=False)
        head.load_state_dict(ck["head"])
        if "proj" in ck:
            # head.pt 里的 proj 带 "proj." 前缀（存的是整个骨干的 state_dict 片段）
            pj = {k.split("proj.", 1)[-1]: v for k, v in ck["proj"].items()}
            bb.proj.load_state_dict(pj)
        print("基线权重 %s（head%s）" % (a.head_ckpt, " + proj" if "proj" in ck else ""))
    else:
        print("注意：没有载入任何权重，下面的基线是随机初始化，只能看管线是否接通")

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
        print(json.dumps(eval_external(bb, head, dl_eval, dev, a.threshold),
                         ensure_ascii=False, indent=1) if dl_eval else "无验收集")
        return 0

    ds_label = LabelFrames(rows)
    dl_label = DataLoader(ds_label, batch_size=a.batch, shuffle=True, num_workers=0)

    # ---------------- 试通：一个 batch 的前向反向
    if not a.train:
        batch = next(iter(dl_label))
        rgb, disp_gt, mask, w, lab, ids = batch
        rgb, disp_gt, mask = rgb.to(dev), disp_gt.to(dev), mask.to(dev)
        w, lab = w.to(dev), lab.to(dev)
        t0 = time.time()
        tok, th, tw = bb(rgb.unsqueeze(1))
        disp = head(tok, th, tw)
        l_depth = (torch.abs(disp - disp_gt) * mask).sum() / mask.sum().clamp(min=1.0)
        depth_p = 1.0 / torch.clamp(disp.squeeze(1), min=1e-3)
        front = roi_front(depth_p, mask.squeeze(1))
        l_hinge, n_hinge = hinge_loss(front, lab, w, a.threshold, a.margin)
        loss = l_depth + a.w_hinge * l_hinge
        loss.backward()
        grads = [p.grad for p in bb.body.parameters() if p.requires_grad]
        g_ok = sum(1 for g in grads if g is not None and torch.isfinite(g).all())
        print("\n[试通] batch=%d 前向反向 %.2fs" % (rgb.shape[0], time.time() - t0))
        print("  掩码视差 L1 %.4f | 铰链 %.4f（参与 %d 帧，权重和 %.2f）| 总损失 %.4f"
              % (float(l_depth.detach()), float(l_hinge.detach()), n_hinge, float(w.detach().sum()), float(loss.detach())))
        print("  LoRA 梯度: %d/%d 个参数块拿到有限梯度" % (g_ok, len(grads)))
        print("  ROI 前方距离预测(米): %s"
              % [round(float(v), 2) for v in front.detach().cpu()])
        print("  帧级标签: %s" % lab.cpu().numpy().tolist())
        if dl_eval:
            m0 = eval_external(bb, head, dl_eval, dev, a.threshold)
            print("\n[基线] 外部近场集 %d 帧：%s" % (dl_eval.dataset.__len__(),
                  json.dumps(m0, ensure_ascii=False)))
        print("\n管线已接通，未保存任何权重（要训练加 --train）。")
        return 0

    # ---------------- 真正训练
    near = []
    if a.near_root:
        near = scan_near(Path(a.near_root), a.near_limit)
        print("近场补充（仅深度监督）：%d 帧" % len(near))
    opt = torch.optim.AdamW([
        {"params": [p for p in bb.body.parameters() if p.requires_grad], "lr": a.lr_lora},
        {"params": list(bb.proj.parameters()), "lr": a.lr_head},
        {"params": list(head.parameters()), "lr": a.lr_head},
    ], weight_decay=0.01)

    stamp = time.strftime("%Y%m%d_%H%M")
    outdir = Path(a.out) if a.out else ROOT / "runs" / ("loop_%s" % stamp)
    outdir.mkdir(parents=True, exist_ok=True)
    for ep in range(a.epochs):
        bb.train(); head.train()
        tot = n = 0
        for rgb, disp_gt, mask, w, lab, _ in dl_label:
            rgb, disp_gt, mask = rgb.to(dev), disp_gt.to(dev), mask.to(dev)
            w, lab = w.to(dev), lab.to(dev)
            tok, th, tw = bb(rgb.unsqueeze(1))
            disp = head(tok, th, tw)
            l_depth = (torch.abs(disp - disp_gt) * mask).sum() / mask.sum().clamp(min=1.0)
            front = roi_front(1.0 / torch.clamp(disp.squeeze(1), min=1e-3), mask.squeeze(1))
            l_hinge, _ = hinge_loss(front, lab, w, a.threshold, a.margin)
            loss = l_depth + a.w_hinge * l_hinge
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()); n += 1
            if a.max_steps and n >= a.max_steps:
                break
        msg = "epoch %2d/%d loss %.4f" % (ep + 1, a.epochs, tot / max(1, n))
        if dl_eval:
            msg += " | 外部近场 %s" % json.dumps(
                eval_external(bb, head, dl_eval, dev, a.threshold), ensure_ascii=False)
        print(msg, flush=True)

    ckpt = outdir / "ckpt.pt"
    torch.save({"backbone": bb.state_dict(), "head": head.state_dict(),
                "config": {"lora_r": a.lora_r, "alpha": a.lora_alpha, "targets": targets,
                           "threshold": a.threshold, "margin": a.margin,
                           "w_hinge": a.w_hinge}}, ckpt)
    print("已保存 %s" % ckpt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
