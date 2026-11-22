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

# --- 路径解析（不写死本机绝对路径）---
import os as _os
from pathlib import Path as _Path
REPO = str(_Path(__file__).resolve().parents[1])
DATA = str(_Path(_os.environ.get("WM_DATA_ROOT", _Path(__file__).resolve().parents[2] / "wmdata")))
MICROW = str(_Path(_os.environ.get("WM_CODE_ROOT", _Path(__file__).resolve().parents[2] / "microwm")))
# ---

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

sys.path.insert(0, f"{MICROW}/src")

CACHE = Path(f"{REPO}/cache/kept_all.json")
ROOTS = [Path(f"{DATA}/runs"), Path(f"{DATA}/runs_batch2"),
         Path(f"{DATA}/runs_batch3"), Path(f"{DATA}/runs_batch4"),
         Path(f"{DATA}/runs_nearfield"), Path(f"{DATA}/near_out"),
         Path(f"{DATA}/runs_batch5"), Path(f"{DATA}/runs_batch6"),
         Path(f"{DATA}/runs_batch7"), Path(f"{DATA}/runs_b8_newgoal"),
         Path(f"{DATA}/runs_other_ai"),
         Path(f"{DATA}/runs_b9_alt20"), Path(f"{DATA}/runs_b9_alt40"),
         Path(f"{DATA}/runs_b10_alt20"), Path(f"{DATA}/runs_b10_alt40"),
         Path(f"{DATA}/runs_b11_alt20"), Path(f"{DATA}/runs_b11_alt40"),
         Path(f"{DATA}/runs_b11_newgoal"), Path(f"{DATA}/runs_b12_explore"),
         Path(f"{DATA}/runs_b12_goal"), Path(f"{DATA}/runs_near2"),
         Path(f"{DATA}/near_out2"), Path(f"{DATA}/near_out3"),
         Path(f"{DATA}/near_out4")]


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

# 自监督权重（patch 6 / dim 192 / depth 6 / heads 3 / 576 token）
#   base_s6           VisDrone，真实航拍俯视，4000 步 -> 内容与视角都与本任务不匹配
#   jepa_domain       v4 划分下的 52 个纯训练 run / 58 航段 / 10014 帧对 / 4000 步
#                     （零泄漏，两道断言通过；论文唯一引用的域内权重）
#   已废弃：runs/_deprecated/ 下的 jepa_domain_leaky（14/16 测试航段被看过）
#           与 jepa_domain_clean（20/29 留出航段被看过）
CKPTS = {
    "jepa": Path(f"{MICROW}/outputs/base_s6/jepa.pt"),
    "jepa_domain": Path(f"{REPO}/runs/jepa_domain_v4/jepa.pt"),
    # v5：训练语料扩到全盘 152 航段 / 17,053 帧（v4 是 66 / 8,565），val/test 不变
    "jepa_domain_v5": Path(f"{REPO}/runs/jepa_domain_v5/jepa.pt"),
    # v5 + gap 4 + 12000 步：补齐上一版的欠训（4000 步只过 3 遍）
    "jepa_domain_v5_g4": Path(f"{REPO}/runs/jepa_domain_v5_g4/jepa.pt"),
    # v6：在 v5 的 153 航段基础上加入 2026-11-19/20 新采的 87 航段（近场两轮 + 合作方三批）
    "jepa_domain_v6": Path(f"{REPO}/runs/jepa_domain_v6/jepa.pt"),
    # v7：训练语料 375 航段 / 43,271 帧，val/test 与 v5 完全一致
    "jepa_domain_v7": Path(f"{REPO}/runs/jepa_domain_v7/jepa.pt"),
    # 2018 World Models 的视觉编码器：在 375 训练航段上重训，未看留出集
    "vae2018": Path("D:/zzr/wmdata/vae_runs/v7_vae/best_final.pt"),
}

# 与 jepa.pt 完全一致的模型配置（从 ckpt 里读出来后写死，避免依赖加载顺序）
JEPA_CFG = {
    "model": {"patch": 6, "dim": 192, "depth": 6, "heads": 3, "mlp_ratio": 4,
              "in_ch": 3, "pred_depth": 4, "ema_tau": 0.996,
              "vicreg_var": 0.15, "vicreg_cov": 0.01},
    "data": {"img_h": IMG_H, "img_w": IMG_W, "grid_h": 27, "grid_w": 48},
}


def split(idx: dict, seed: int = 0):
    """与 train_min.py 完全一致：同样排序、同一种子，保证结果可比。

    ⚠️ 这是「现算」划分，只在没有权威划分时用。索引重建过之后
    （补采 9 段近场、清理 3 段早期），现算会得到 104 航段的另一套，
    与 v4 表其余四行（98 航段 = 69/14/15）不可比。主流程请用 load_split()。
    """
    eps = sorted(idx["kept"])
    random.Random(seed).shuffle(eps)
    n = len(eps)
    a, b = int(round(0.7 * n)), int(round(0.85 * n))
    return eps[:a], eps[a:b], eps[b:]


AUTH_SPLIT = Path(f"{REPO}/runs/v4_imagenet/metrics.json")


def load_split(idx: dict, split_path: Path | None = None):
    """唯一权威划分：默认读 runs/v4_imagenet/metrics.json 的 split 字段（98 航段）。

    v4 表的四行结果都是在这套划分上跑的。索引后来删掉 3 个 train 航段，
    因此按现存航段过滤；val / test 完整，评测口径与其余四行一致。

    v5 用 `--split-file runs/v5_split.json`：训练集扩到 156 航段，val / test 沿用 v4。
    """
    p = Path(split_path) if split_path else AUTH_SPLIT
    sp = json.loads(p.read_text(encoding="utf-8"))["split"]
    cur = set(idx["kept"])
    tr = [e for e in sp["train"] if e in cur]
    va = [e for e in sp["val"] if e in cur]
    te = [e for e in sp["test"] if e in cur]
    if len(tr) != len(sp["train"]) or len(va) != len(sp["val"]) or len(te) != len(sp["test"]):
        print("  [split] 权威划分 %d/%d/%d，按现存索引过滤后 %d/%d/%d"
              % (len(sp["train"]), len(sp["val"]), len(sp["test"]),
                 len(tr), len(va), len(te)), flush=True)
    return tr, va, te


class Frames(Dataset):
    """监督目标固定 108x192；输入图像的分辨率可按骨干指定。

    VAE 在 96x96 上训练，喂 108x192 属于分布外，因此单独给它 96x96。
    真值始终在 108x192 上算，所以各行的 δ1 仍是同一批像素。
    """

    def __init__(self, idx: dict, episodes: list[str], items=None,
                 in_h: int = IMG_H, in_w: int = IMG_W):
        self.items = list(items) if items is not None else \
            [(e, s) for e in episodes for s in idx["kept"][e]]
        self.in_h, self.in_w = in_h, in_w

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ep, stem = self.items[i]
        run, episode = ep.split("/")
        obs = find_obs(run, episode)
        bgr = cv2.imread(str(obs / f"{stem}_front_rgb.png"), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(cv2.resize(bgr, (self.in_w, self.in_h), interpolation=cv2.INTER_AREA),
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
    """任意 token 网格 -> 108x192 视差图。所有骨干共用同一串卷积与容量。

    入口不做重采样：各骨干按自己的网格进入，上采样倍率随网格自适应。
    网格密度不是限制因素——ResNet 的 4x6 比 JEPA 的 18x32 粗十六倍，
    测试集 δ1 反而高 0.028（0.875 对 0.847）。
    """

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


class VaeBackbone(nn.Module):
    """2018 World Models 的视觉编码器（VAE）。

    只取编码器：latent 均值向量当作表征，投影到 192 维接同一个 DepthHead。
    VAE 的 latent 在 96x96 输入下是 6x6，比 JEPA 的 18x32 粗得多，
    这是该路线本身的性质，不是实现取舍。
    """

    def __init__(self, ckpt: Path, finetune: bool = False):
        super().__init__()
        import sys as _sys
        _sys.path.insert(0, str(Path(f"{MICROW}/../Airsim-WorldModelCollector/vae")))
        from train_vae import VAE
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        a = ck["args"]
        full = VAE(a["latent"], 3, a["size"], a.get("width", 1))
        full.load_state_dict(ck["state_dict"])
        self.enc = full.encoder
        self.proj = nn.Conv2d(a["latent"], 192, 1)
        self.finetune = finetune
        for p in self.enc.parameters():
            p.requires_grad_(finetune)
        if not finetune:
            self.enc.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.finetune:
            self.enc.eval()          # 冻结时 BN/IN 一律不进训练态
        return self

    def trainable(self):
        return list(self.proj.parameters())

    def finetune_params(self):
        return list(self.enc.parameters()) if self.finetune else []

    def forward(self, x):
        x = x[:, 0]
        if self.finetune:
            feat = self.enc(x)
        else:
            with torch.no_grad():
                feat = self.enc(x)      # (B, latent, h/16, w/16) 的卷积特征图
        z = self.proj(feat)
        return z.flatten(2).transpose(1, 2), z.shape[-2], z.shape[-1]


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

    def __init__(self, pretrained: bool = True, finetune: bool = False):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights
        w = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        r = resnet18(weights=w)
        self.body = nn.Sequential(*list(r.children())[:-2])   # -> [B,512,h,w]
        self.proj = nn.Conv2d(512, 192, 1)
        self.finetune = finetune
        for p in self.body.parameters():
            p.requires_grad_(finetune)
        if not finetune:
            self.body.eval()
        self.register_buffer("mean", torch.tensor(self.MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(self.STD).view(1, 3, 1, 1))

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.finetune:
            self.body.eval()
        return self

    def finetune_params(self):
        return list(self.body.parameters()) if self.finetune else []

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
                    choices=["scratch", "jepa", "jepa_domain", "jepa_domain_v5",
                             "jepa_domain_v5_g4", "jepa_domain_v6", "jepa_domain_v7",
                             "vae2018", "imagenet", "resnet_scratch"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4,
                    help="数据加载进程数。0 时每个样本要读 RGB+深度+掩码三个文件，GPU 会空转等磁盘")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                    help="cpu 时不碰显卡；训练速度约慢一个数量级")
    ap.add_argument("--finetune", action="store_true",
                    help="解冻编码器端到端训练（低一档学习率）；默认冻结，只训头")
    ap.add_argument("--label-ratio", type=float, default=1.0,
                    help="训练集保留多少帧，用于标签效率扫描")
    ap.add_argument("--out-name", default=None,
                    help="输出目录后缀，默认用骨干名；并行复跑时用来避免互相覆盖")
    ap.add_argument("--index", default=None,
                    help="帧索引 json，默认 cache/kept_all.json；v5 用 cache/kept_all_v5.json")
    ap.add_argument("--split-file", default=None,
                    help="划分来源，默认 runs/v4_imagenet/metrics.json；v5 用 runs/v5_split.json")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    dev = a.device
    if dev == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.45, 0)   # 有别的任务在跑，留余量
    idx = json.loads(Path(a.index or CACHE).read_text(encoding="utf-8"))
    tr, va, te = load_split(idx, Path(a.split_file) if a.split_file else None)
    n_tr = sum(len(idx["kept"][e]) for e in tr)
    print("[%s] train %d 趟 / %d 帧 | val %d / %d | test %d / %d"
          % (a.backbone, len(tr), n_tr, len(va),
             sum(len(idx["kept"][e]) for e in va), len(te),
             sum(len(idx["kept"][e]) for e in te)), flush=True)

    if a.backbone == "vae2018":
        bb = VaeBackbone(CKPTS["vae2018"], finetune=a.finetune).to(dev)
        print("  VAE latent 网格 %dx%d" % (bb.tok_h, bb.tok_w) if hasattr(bb, "tok_h") else "  VAE 编码器已加载", flush=True)
    elif a.backbone in ("imagenet", "resnet_scratch"):
        # resnet_scratch 与 imagenet 只差权重本身，用来把卷积归纳偏置与预训练分开
        bb = ResnetBackbone(a.backbone == "imagenet", finetune=a.finetune).to(dev)
    else:
        bb = JepaBackbone(None if a.backbone == "scratch" else CKPTS[a.backbone]).to(dev)
        print("  tokens %dx%d = %d" % (bb.tok_h, bb.tok_w, bb.tok_h * bb.tok_w), flush=True)
    head = DepthHead(192).to(dev)

    # 输入分辨率按骨干自己的训练分辨率给，不做统一；真值始终在 108x192 上算
    in_h, in_w = (96, 96) if a.backbone == "vae2018" else (IMG_H, IMG_W)
    print("  输入 %dx%d，监督目标 %dx%d" % (in_w, in_h, IMG_W, IMG_H), flush=True)

    dl_tr = DataLoader(Frames(idx, tr, in_h=in_h, in_w=in_w), batch_size=a.batch,
                       shuffle=True, num_workers=a.workers,
                       pin_memory=(dev == "cuda"), drop_last=True)
    dl_va = DataLoader(Frames(idx, va, in_h=in_h, in_w=in_w), batch_size=a.batch,
                       num_workers=a.workers, pin_memory=(dev == "cuda"))
    dl_te = DataLoader(Frames(idx, te, in_h=in_h, in_w=in_w), batch_size=a.batch,
                       num_workers=a.workers, pin_memory=(dev == "cuda"))

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
        ds_tr, batch_size=a.batch, drop_last=True, num_workers=a.workers,
        pin_memory=(dev == "cuda"),
        sampler=RandomSampler(ds_tr, replacement=len(pick) < nsamp, num_samples=nsamp))
    print("  标签比例 %.0f%%：训练帧 %d / %d，每轮 %d 步 × %d 轮"
          % (100 * a.label_ratio, len(pick), n_full, spe, a.epochs), flush=True)

    params = list(head.parameters()) + bb.trainable()
    ft = bb.finetune_params() if hasattr(bb, "finetune_params") else []
    print("  trainable: %.2fM (head %.2fM + 投影 %.2fM)"
          % (sum(p.numel() for p in params) / 1e6,
             sum(p.numel() for p in head.parameters()) / 1e6,
             sum(p.numel() for p in bb.trainable()) / 1e6), flush=True)
    if ft:
        print("  端到端：编码器 %d 个参数组，学习率 %.1e（头 %.1e）"
              % (len(ft), a.lr * 0.1, a.lr), flush=True)
        opt = torch.optim.AdamW(
            [{"params": params, "lr": a.lr},
             {"params": ft, "lr": a.lr * 0.1}], weight_decay=0.01)
    else:
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

    name = a.out_name or ("v4_" + a.backbone)
    out = Path(f"{REPO}/runs") / name
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
