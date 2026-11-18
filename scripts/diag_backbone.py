"""诊断：预训练权重对前视深度下游到底有没有用。

三个骨干，其余全部相同（同一划分、同一解码器、同一损失、同一超参）：
  scratch  MicroJEPA 随机初始化（形状与预训练一致，权重随机）
  jepa     outputs/base_s6/jepa.pt（VisDrone 俯视航拍，4000 步自监督）
  imagenet torchvision ResNet18 有监督预训练

编码器全部冻结，只训解码器。理由：要测的是「表征里有多少深度可读」，
不是「微调能补多少」。冻结后三组的可训练参数完全相同。

输入统一 192x108：这是预训练 pos embedding 的网格（patch 6 -> 32x18）。
数据原生 256x256，缩放到 192x108 会压扁宽高比，三组同等承受，不影响组间比较。

不改动 train_min.py / probe_data.py，输出落在 runs/diag_<backbone>/。
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler

sys.path.insert(0, "D:/zzr/microwm/src")

CACHE = Path("D:/zzr/ssl-depth/cache/kept_all.json")
ROOTS = [Path("D:/zzr/wmdata/runs"), Path("D:/zzr/wmdata/runs_batch2"),
         Path("D:/zzr/wmdata/runs_batch3"), Path("D:/zzr/wmdata/runs_batch4"),
         Path("D:/zzr/wmdata/runs_nearfield"), Path("D:/zzr/near_out")]


def find_obs(run: str, episode: str) -> Path:
    """两批数据分处不同根目录，按存在性定位。"""
    for r in ROOTS:
        p = r / run / episode / "observations"
        if p.is_dir():
            return p
    raise FileNotFoundError("%s/%s" % (run, episode))

IMG_H, IMG_W = 108, 192          # 预训练规格
MAX_DEPTH = 80.0
MIN_DEPTH = 0.5

# 两套自监督权重，模型结构相同（patch 6 / dim 192 / depth 6 / heads 3 / 576 token），
# 差别只在预训练数据：
#   base_s6  VisDrone，真实航拍俯视，4000 步     -> 内容与视角都不匹配
#   dl_act8k dronelab_train，AirSim 前视低空，8000 步 -> 视角匹配、场景简化
CKPTS = {
    "jepa": Path("D:/zzr/microwm/outputs/base_s6/jepa.pt"),
    "dronelab": Path("D:/zzr/microwm/outputs/dl_act8k/jepa.pt"),
}

# 与 jepa.pt 完全一致的模型配置（从 ckpt 里读出来后写死，避免依赖加载顺序）
JEPA_CFG = {
    "model": {"patch": 6, "dim": 192, "depth": 6, "heads": 3, "mlp_ratio": 4,
              "in_ch": 3, "pred_depth": 4, "ema_tau": 0.996,
              "vicreg_var": 0.15, "vicreg_cov": 0.01},
    "data": {"img_h": IMG_H, "img_w": IMG_W, "grid_h": 27, "grid_w": 48},
}


def split(idx: dict, seed: int = 0):
    """与 train_min.py 完全一致：同样排序、同一种子，保证结果可比。"""
    eps = sorted(idx["kept"])
    random.Random(seed).shuffle(eps)
    n = len(eps)
    a, b = int(round(0.7 * n)), int(round(0.85 * n))
    return eps[:a], eps[a:b], eps[b:]


class Frames(Dataset):
    def __init__(self, idx: dict, episodes: list[str], items=None):
        self.items = list(items) if items is not None else \
            [(e, s) for e in episodes for s in idx["kept"][e]]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ep, stem = self.items[i]
        run, episode = ep.split("/")
        obs = find_obs(run, episode)
        bgr = cv2.imread(str(obs / f"{stem}_front_rgb.png"), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(cv2.resize(bgr, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA),
                           cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
        dep = np.load(obs / f"{stem}_front_depth.npy").astype(np.float32)
        val = np.load(obs / f"{stem}_front_depth_valid.npy").astype(bool)
        dep = cv2.resize(dep, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
        val = cv2.resize(val.astype(np.uint8), (IMG_W, IMG_H),
                         interpolation=cv2.INTER_NEAREST).astype(bool)
        val &= np.isfinite(dep) & (dep >= MIN_DEPTH) & (dep <= MAX_DEPTH)
        disp = np.where(val, 1.0 / np.maximum(dep, 1e-3), 0.0).astype(np.float32)
        return (torch.from_numpy(rgb).permute(2, 0, 1),
                torch.from_numpy(disp)[None],
                torch.from_numpy(val.astype(np.float32))[None])


class DepthHead(nn.Module):
    """任意 token 网格 -> 108x192 视差图。三组共用，容量一致。"""

    def __init__(self, dim: int = 192):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(dim, 192, 3, padding=1), nn.GELU(),
            nn.Conv2d(192, 128, 3, padding=1), nn.GELU())
        self.up2 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.GELU())
        self.up4 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.GELU())
        self.upf = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.GELU())
        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, tokens, th, tw):
        b, n, d = tokens.shape
        x = tokens.transpose(1, 2).reshape(b, d, th, tw)
        x = self.stem(x)
        x = self.up2(F.interpolate(x, size=(th * 2, tw * 2), mode="bilinear", align_corners=False))
        x = self.up4(F.interpolate(x, size=(th * 4, tw * 4), mode="bilinear", align_corners=False))
        x = self.upf(F.interpolate(x, size=(IMG_H, IMG_W), mode="bilinear", align_corners=False))
        return self.out(x)


class JepaBackbone(nn.Module):
    def __init__(self, ckpt: Path | None):
        super().__init__()
        from microwm.models.microwm import MicroJEPA
        full = MicroJEPA(JEPA_CFG)
        if ckpt is not None:
            ck = torch.load(ckpt, map_location="cpu", weights_only=False)
            sd = ck["model"]
            own = full.state_dict()
            keep = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
            missing = [k for k in own if k not in keep]
            full.load_state_dict(keep, strict=False)
            print("  loaded %d/%d tensors (step %s); 未命中 %d 个，前 3: %s"
                  % (len(keep), len(own), ck.get("step"), len(missing), missing[:3]),
                  flush=True)
        self.enc = full.encoder
        self.tok_h, self.tok_w = full.tok_h, full.tok_w
        for p in self.enc.parameters():
            p.requires_grad_(False)
        self.enc.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.enc.eval()          # 冻结：BN/LN 一律不进训练态
        return self

    def trainable(self):
        return []

    def forward(self, x):
        with torch.no_grad():
            z = self.enc(x)
        return z[:, 0], self.tok_h, self.tok_w


class ResnetBackbone(nn.Module):
    """ImageNet ResNet18 特征 + 1x1 投影到 192 维。

    两处易错点已修：
    1. 输入是 [-1,1]，而预训练权重的统计假设是 ImageNet 归一化，必须先还原到 [0,1] 再标准化；
    2. proj 是可训练层，必须进优化器。之前只优化 head，proj 一直是随机权重，
       导致同一份 head.pt 在别的进程里复现不出训练时的结果。
    """

    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(self, pretrained: bool = True):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights
        w = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        r = resnet18(weights=w)
        self.body = nn.Sequential(*list(r.children())[:-2])   # -> [B,512,h,w]
        self.proj = nn.Conv2d(512, 192, 1)
        for p in self.body.parameters():
            p.requires_grad_(False)
        self.body.eval()
        self.register_buffer("mean", torch.tensor(self.MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(self.STD).view(1, 3, 1, 1))

    def train(self, mode: bool = True):
        super().train(mode)
        self.body.eval()
        return self

    def trainable(self):
        return list(self.proj.parameters())

    def forward(self, x):
        b = x.shape[0]
        x = x.reshape(b * x.shape[1], *x.shape[2:]) if x.dim() == 5 else x
        x = (x * 0.5 + 0.5 - self.mean) / self.std          # [-1,1] -> ImageNet 标准化
        with torch.no_grad():
            f = self.body(x)
        f = self.proj(f)
        return f.flatten(2).transpose(1, 2), f.shape[2], f.shape[3]


@torch.no_grad()
def evaluate(bb, head, dl, dev):
    head.eval()
    ps, gs, stds, rows = [], [], [], []
    for rgb, disp_gt, mask in dl:
        rgb, mask = rgb.to(dev), mask.to(dev)
        tok, th, tw = bb(rgb.unsqueeze(1))
        disp = head(tok, th, tw)
        pred_d = 1.0 / torch.clamp(disp.squeeze(1), min=1e-3)
        gt_d = 1.0 / torch.clamp(disp_gt.squeeze(1).to(dev), min=1e-3)
        m = mask.squeeze(1) > 0.5
        for j in range(pred_d.shape[0]):
            if int(m[j].sum()) > 100:
                pd = pred_d[j][m[j]]
                stds.append(float(pd.std()))
        ps.append(pred_d[m].cpu().numpy())
        gs.append(gt_d[m].cpu().numpy())
    p, g = np.concatenate(ps), np.concatenate(gs)

    def met(p, g):
        s = float(np.median(p / g))
        ratio = np.maximum(p / g, g / p)
        return {"n_px": int(p.size),
                "absrel_raw": float(np.mean(np.abs(p - g) / g)),
                "absrel_med": float(np.mean(np.abs(p / s - g) / g)),
                "scale": s,
                "d1": float((ratio < 1.25).mean()),
                "rmse": float(np.sqrt(np.mean((p - g) ** 2)))}

    out = met(p, g)
    near = g <= 30.0
    out["near30"] = met(p[near], g[near]) if near.sum() else None
    out["spatial_std_m"] = float(np.mean(stds)) if stds else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True,
                    choices=["scratch", "jepa", "dronelab", "imagenet", "resnet_scratch"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label-ratio", type=float, default=1.0,
                    help="训练集保留多少帧，用于标签效率扫描")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    dev = "cuda"
    torch.cuda.set_per_process_memory_fraction(0.45, 0)   # 有别的任务在跑，留余量
    idx = json.loads(CACHE.read_text(encoding="utf-8"))
    tr, va, te = split(idx, 0)
    n_tr = sum(len(idx["kept"][e]) for e in tr)
    print("[%s] train %d 趟 / %d 帧 | val %d / %d | test %d / %d"
          % (a.backbone, len(tr), n_tr, len(va),
             sum(len(idx["kept"][e]) for e in va), len(te),
             sum(len(idx["kept"][e]) for e in te)), flush=True)

    if a.backbone in ("imagenet", "resnet_scratch"):
        # resnet_scratch 与 imagenet 只差权重本身，用来把卷积归纳偏置与预训练分开
        bb = ResnetBackbone(a.backbone == "imagenet").to(dev)
    else:
        bb = JepaBackbone(None if a.backbone == "scratch" else CKPTS[a.backbone]).to(dev)
        print("  tokens %dx%d = %d" % (bb.tok_h, bb.tok_w, bb.tok_h * bb.tok_w), flush=True)
    head = DepthHead(192).to(dev)

    dl_tr = DataLoader(Frames(idx, tr), batch_size=a.batch, shuffle=True, num_workers=0, drop_last=True)
    dl_va = DataLoader(Frames(idx, va), batch_size=a.batch, num_workers=0)
    dl_te = DataLoader(Frames(idx, te), batch_size=a.batch, num_workers=0)

    # 标签效率：只抽训练帧，val/test 不动。所有比例看同样多的样本（固定步数），
    # 否则少标签时 epochs 变少会与「标签少」这个变量混在一起。
    full = [(e, s) for e in tr for s in idx["kept"][e]]
    n_full = len(full)
    pick = full if a.label_ratio >= 1.0 else random.Random(2000 + a.seed).sample(
        full, max(1, int(round(n_full * a.label_ratio))))
    spe = max(1, n_full // a.batch)                      # 每 epoch 步数，按满数据算
    nsamp = spe * a.batch
    ds_tr = Frames(idx, tr, pick)
    dl_tr = DataLoader(
        ds_tr, batch_size=a.batch, drop_last=True, num_workers=0,
        sampler=RandomSampler(ds_tr, replacement=len(pick) < nsamp, num_samples=nsamp))
    print("  标签比例 %.0f%%：训练帧 %d / %d，每轮 %d 步 × %d 轮"
          % (100 * a.label_ratio, len(pick), n_full, spe, a.epochs), flush=True)

    params = list(head.parameters()) + bb.trainable()
    print("  trainable: %.2fM (head %.2fM + 投影 %.2fM)"
          % (sum(p.numel() for p in params) / 1e6,
             sum(p.numel() for p in head.parameters()) / 1e6,
             sum(p.numel() for p in bb.trainable()) / 1e6), flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, a.epochs * len(dl_tr)))

    best = None
    for ep in range(a.epochs):
        head.train()
        t0, run = time.time(), 0.0
        for i, (rgb, disp_gt, mask) in enumerate(dl_tr):
            rgb, disp_gt, mask = rgb.to(dev), disp_gt.to(dev), mask.to(dev)
            tok, th, tw = bb(rgb.unsqueeze(1))
            disp = head(tok, th, tw)
            loss = (torch.abs(disp - disp_gt) * mask).sum() / mask.sum().clamp(min=1.0)
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
            run = 0.9 * run + 0.1 * float(loss.detach()) if i else float(loss.detach())
        vm = evaluate(bb, head, dl_va, dev)
        if best is None or vm["absrel_med"] < best["absrel_med"]:
            best = dict(vm, epoch=ep + 1)
        print("  ep %2d/%d loss %.4f | val AbsRel %.3f d1 %.3f | std %.1fm | %.1fs"
              % (ep + 1, a.epochs, run, vm["absrel_med"], vm["d1"],
                 vm["spatial_std_m"] or -1, time.time() - t0), flush=True)

    tm = evaluate(bb, head, dl_te, dev)
    print("  best-val ep%d | TEST AbsRel(med) %.3f d1 %.3f rmse %.2f std %.1fm"
          % (best["epoch"], tm["absrel_med"], tm["d1"], tm["rmse"],
             tm["spatial_std_m"] or -1), flush=True)

    out = Path("D:/zzr/ssl-depth/runs/v4_%s" % a.backbone)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(
        {"backbone": a.backbone, "args": vars(a), "best_val": best, "test": tm,
         "split": {"train": tr, "val": va, "test": te}},
        ensure_ascii=False, indent=1), encoding="utf-8")
    # 连投影层一起存，否则换进程重放推理时会得到另一组随机 proj
    torch.save({"head": head.state_dict(),
                "proj": {k: v for k, v in bb.state_dict().items() if k.startswith("proj")}},
               out / "head.pt")


if __name__ == "__main__":
    main()
