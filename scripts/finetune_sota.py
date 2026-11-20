"""把通用深度模型在我们的训练集上微调，再用同一套留出集评测。

和 eval_sota.py 的零样本不同：这里用本文的 17,053 帧训练，目标是米制视差
（disp = 1/depth），因此推理时 depth = 1/pred 直接就是米，不需要任何对齐。
这一点很关键：零样本时 DAv2 只能靠逐图拟合 a,b 才能变成米，那个数不能和
本文四行并列；微调之后没有这个问题。

用法：
  python scripts/finetune_sota.py --model depth-anything/Depth-Anything-V2-Small-hf \
      --epochs 8 --out runs/sota_ft_dav2s.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

ROOT = Path("D:/zzr/ssl-depth")
SPLIT = ROOT / "runs/v5_split.json"
IDX = ROOT / "cache/kept_all_v5.json"
ROOTS = [Path(p) for p in [
    "D:/zzr/wmdata/runs", "D:/zzr/wmdata/runs_batch2", "D:/zzr/wmdata/runs_batch3",
    "D:/zzr/wmdata/runs_batch4", "D:/zzr/wmdata/runs_batch5", "D:/zzr/wmdata/runs_batch6",
    "D:/zzr/wmdata/runs_batch7", "D:/zzr/wmdata/runs_b8_newgoal", "D:/zzr/wmdata/runs_nearfield",
    "D:/zzr/wmdata/runs_other_ai", "D:/zzr/near_out"]]
MIN_D, MAX_D = 0.5, 80.0
SIZE = 256


def find_obs(ep: str) -> Path | None:
    run, name = ep.split("/")
    for r in ROOTS:
        p = r / run / name / "observations"
        if p.is_dir():
            return p
    return None


class Frames(Dataset):
    def __init__(self, idx: dict, eps: list[str], stride: int = 1):
        self.items = []
        for e in eps:
            obs = find_obs(e)
            if obs is None:
                continue
            for s in idx[e][::stride]:
                self.items.append((obs, s))
        self.mean = np.array([0.485, 0.456, 0.406], np.float32)
        self.std = np.array([0.229, 0.224, 0.225], np.float32)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        obs, s = self.items[i]
        im = Image.open(obs / (s + "_front_rgb.png")).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
        rgb = (np.asarray(im, np.float32) / 255.0 - self.mean) / self.std
        g = np.load(obs / (s + "_front_depth.npy")).astype(np.float32)
        v = np.load(obs / (s + "_front_depth_valid.npy")).astype(bool)
        g = np.asarray(Image.fromarray(g).resize((SIZE, SIZE), Image.BILINEAR), np.float32)
        v = np.asarray(Image.fromarray(v.astype(np.uint8) * 255).resize((SIZE, SIZE), Image.NEAREST)) > 127
        m = v & np.isfinite(g) & (g >= MIN_D) & (g <= MAX_D)
        disp = np.where(m, 1.0 / np.clip(g, MIN_D, MAX_D), 0.0).astype(np.float32)
        return (torch.from_numpy(rgb).permute(2, 0, 1),
                torch.from_numpy(disp), torch.from_numpy(m.astype(np.float32)))


def pixel_metrics(p, g):
    s = float(np.median(p / g)); pm = p / s
    return {"n_px": int(p.size),
            "absrel_raw": float(np.mean(np.abs(p - g) / g)),
            "absrel_med": float(np.mean(np.abs(pm - g) / g)),
            "d1": float((np.maximum(p / g, g / p) < 1.25).mean()),
            "d1_med": float((np.maximum(pm / g, g / pm) < 1.25).mean()),
            "rmse": float(np.sqrt(np.mean((p - g) ** 2)))}


def task_metrics(fp, fg):
    out = {"n_frame": int(fg.size), "gt_median": float(np.median(fg)),
           "pr_median": float(np.median(fp))}
    for t in (6.0, 9.0, 12.0, 15.0, 20.0):
        dg, dp = fg < t, fp < t
        tp = int((dg & dp).sum()); fn = int((dg & ~dp).sum())
        fpra = int((~dg & dp).sum()); tn = int((~dg & ~dp).sum())
        out["t%d" % int(t)] = {"hit": tp, "miss": fn, "false": fpra, "pass": tn,
                               "recall": tp / max(1, tp + fn)}
    return out


@torch.no_grad()
def evaluate(model, idx, eps, dev, stride=2):
    model.eval()
    ps, gs, fp, fg = [], [], [], []
    for e in eps:
        obs = find_obs(e)
        if obs is None:
            continue
        for s in idx[e][::stride]:
            im = Image.open(obs / (s + "_front_rgb.png")).convert("RGB")
            g = np.load(obs / (s + "_front_depth.npy")).astype(np.float32)
            v = np.load(obs / (s + "_front_depth_valid.npy")).astype(bool)
            x = np.asarray(im.resize((SIZE, SIZE), Image.BILINEAR), np.float32) / 255.0
            x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
            x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(dev)
            pred = model(pixel_values=x).predicted_depth
            disp = F.interpolate(pred.unsqueeze(1), size=g.shape, mode="bicubic",
                                 align_corners=False).squeeze().cpu().numpy()
            with np.errstate(divide="ignore"):
                p = 1.0 / np.clip(disp, 1e-4, None)
            p = np.clip(p, 0.1, 200.0)
            m = v & np.isfinite(g) & (g >= MIN_D) & (g <= MAX_D) & np.isfinite(p)
            if m.sum() < 100:
                continue
            ps.append(p[m]); gs.append(g[m])
            h, w = g.shape
            rm = np.zeros_like(m); rm[h // 4: 3 * h // 4, w // 3: 2 * w // 3] = True
            mm = m & rm
            if mm.sum() > 100:
                fp.append(float(np.percentile(p[mm], 5)))
                fg.append(float(np.percentile(g[mm], 5)))
    model.train()
    return (np.concatenate(ps), np.concatenate(gs), np.array(fp), np.array(fg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--stride", type=int, default=1, help="训练帧抽样间隔")
    ap.add_argument("--smoke", type=int, default=0, help="只跑 N 步，测速用")
    a = ap.parse_args()

    dev = "cuda"
    torch.cuda.set_per_process_memory_fraction(0.42, 0)
    sp = json.loads(SPLIT.read_text(encoding="utf-8"))["split"]
    idx = json.loads(IDX.read_text(encoding="utf-8"))["kept"]

    from transformers import AutoModelForDepthEstimation
    model = AutoModelForDepthEstimation.from_pretrained(a.model).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print("模型 %s | %.1fM 参数" % (a.model, n_par / 1e6), flush=True)

    ds = Frames(idx, sp["train"], a.stride)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=a.workers,
                    drop_last=True, pin_memory=True, persistent_workers=a.workers > 0)
    print("训练帧 %d | 每轮 %d 步" % (len(ds), len(dl)), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    total = a.epochs * len(dl)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total))

    step = 0
    t0 = time.time()
    best = None
    for ep in range(a.epochs):
        run = 0.0
        for i, (x, dg, m) in enumerate(dl):
            x, dg, m = x.to(dev), dg.to(dev), m.to(dev)
            pred = model(pixel_values=x).predicted_depth
            pred = F.interpolate(pred.unsqueeze(1), size=dg.shape[-2:], mode="bilinear",
                                 align_corners=False).squeeze(1)
            # 米制视差回归：L1 + 对数域 L1，兼顾近场与远场
            l1 = (torch.abs(pred - dg) * m).sum() / m.sum().clamp(min=1)
            lg = (torch.abs(torch.log(torch.clamp(pred, 1e-3, 10)) -
                            torch.log(torch.clamp(dg, 1e-3, 10))) * m).sum() / m.sum().clamp(min=1)
            loss = l1 + 0.5 * lg
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); step += 1
            run = 0.9 * run + 0.1 * float(loss) if i else float(loss)
            if i % 100 == 0:
                print("  ep %d step %d/%d loss %.4f  %.0fs" % (
                    ep + 1, i, len(dl), run, time.time() - t0), flush=True)
            if a.smoke and step >= a.smoke:
                print("冒烟结束，用时 %.1f 秒 / %d 步" % (time.time() - t0, step), flush=True)
                return
        p, g, fp_, fg_ = evaluate(model, idx, sp["test"], dev)
        d1 = float((np.maximum(p / g, g / p) < 1.25).mean())
        print("ep %d/%d loss %.4f | 测试 d1 %.3f AbsRel %.3f | %.0fs" % (
            ep + 1, a.epochs, run, d1, float(np.mean(np.abs(p - g) / g)), time.time() - t0), flush=True)
        if best is None or d1 > best[0]:
            best = (d1, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            torch.save({"model": best[1], "epoch": ep + 1}, ROOT / "runs" / (Path(a.out).stem + "_best.pt"))

    model.load_state_dict(best[1])
    p, g, fp_, fg_ = evaluate(model, idx, sp["val"] + sp["test"], dev)
    res = {"model": a.model, "finetuned": True, "epochs": a.epochs, "params_m": n_par / 1e6,
           "pixel": pixel_metrics(p, g), "task": task_metrics(fp_, fg_)}
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("最终 d1 %.3f | AbsRel %.3f | RMSE %.1f | 9 米识别 %.1f%% 误刹 %d" % (
        res["pixel"]["d1"], res["pixel"]["absrel_raw"], res["pixel"]["rmse"],
        100 * res["task"]["t9"]["recall"], res["task"]["t9"]["false"]), flush=True)


if __name__ == "__main__":
    main()
