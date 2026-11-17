"""用户测试素材：三版界面 × N 帧。

设计问题是：模型会漏判，而现有地面站只显示轨迹，调度员看不出模型已经错了。
三版方案对同一帧给出不同的信息量：
  A 基线   RGB + 一行距离数字（模仿现有地面站）
  B 深度叠加 RGB + 半透明深度热力图 + 色标（专业读法）
  C 风险语义 RGB + 三色边框 + 一句话结论（面向非专业者）

三版的信息来源完全相同，都是模型的预测距离，差别只在呈现方式。
这样测试出来的差异才可以归给设计，而不是归给信息多少。

输出 paper/stimuli/ 下的 png 与 manifest.json。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "D:/zzr/ssl-depth/scripts")
from diag_backbone import CACHE, DepthHead, Frames, ResnetBackbone, split  # noqa: E402
from task_metric import front_near  # noqa: E402

ROOT = Path("D:/zzr/ssl-depth")
OUT = ROOT / "paper" / "stimuli"
OUT.mkdir(parents=True, exist_ok=True)

W, H = 512, 512          # 画面区
BAR = 112                # 底部信息区
FONT = "C:/Windows/Fonts/msyh.ttc"


def font(size):
    return ImageFont.truetype(FONT, size)


RISK = [(9.0, (200, 45, 45)), (18.0, (225, 150, 40)), (1e9, (45, 110, 190))]


def colour_ramp(depth, vmax=60.0):
    """按风险区间分段着色，与 C 版的三档判据一致，避免整幅落进同一颜色。"""
    d = np.asarray(depth, dtype=np.float32)
    rgb = np.zeros(d.shape + (3,), dtype=np.uint8)
    prev = 0.0
    for hi, c in RISK:
        m = (d >= prev) & (d < hi)
        rgb[m] = c
        prev = hi
    rgb[~np.isfinite(d)] = 0
    alpha = np.where(np.isfinite(d), 110, 0).astype(np.uint8)
    return rgb, alpha


def render(rgb_img: Image.Image, depth: np.ndarray, pred_m: float, ver: str) -> Image.Image:
    base = rgb_img.resize((W, H), Image.LANCZOS).convert("RGB")

    if ver == "B":
        d = np.array(Image.fromarray(depth.astype(np.float32)).resize((W, H), Image.BILINEAR))
        col, alp = colour_ramp(d)
        overlay = Image.fromarray(col).convert("RGBA")
        overlay.putalpha(Image.fromarray(alp).resize((W, H), Image.NEAREST))
        base = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")

    canvas = Image.new("RGB", (W, H + BAR), (24, 26, 30))
    canvas.paste(base, (0, 0))
    dr = ImageDraw.Draw(canvas)

    if ver == "C":
        # 边框颜色由模型给出的距离等级决定，与 A/B 的信息源相同
        if pred_m < 9:
            edge, text, sub = (200, 40, 40), "危险：前方过近", "建议立即悬停"
        elif pred_m < 18:
            edge, text, sub = (220, 150, 30), "注意：前方有障碍", "建议减速通过"
        else:
            edge, text, sub = (60, 170, 90), "安全", "可维持当前速度"
        dr.rectangle([0, 0, W - 1, H - 1], outline=edge, width=10)
        dr.rectangle([10, 10, W - 11, 74], fill=(0, 0, 0))
        dr.text((24, 18), text, font=font(38), fill=edge)
        dr.text((24, H + 12), sub, font=font(30), fill=edge)
        dr.text((24, H + 56), "模型置信度中等" if abs(pred_m - 9) < 4 else "模型置信度高",
                font=font(22), fill=(150, 155, 165))

    elif ver == "A":
        # 基线版不提供任何模型信息，代表没有智能辅助时人的自然判断
        dr.text((20, H + 20), "请根据画面判断前方是否有危险", font=font(24),
                fill=(170, 176, 186))
        dr.text((20, H + 58), "无辅助信息", font=font(26), fill=(120, 126, 136))

    else:  # B
        dr.text((20, H + 20), "风险分区（红近蓝远）", font=font(24), fill=(170, 176, 186))
        dr.text((20, H + 54), "前方最近 %.1f m" % pred_m, font=font(34), fill=(240, 242, 246))
        legend = [(9.0, "0-9 m 危险", (200, 45, 45)),
                  (18.0, "9-18 m 注意", (225, 150, 40)),
                  (60.0, "18 m 以上安全", (45, 110, 190))]
        for i, (_, lab, c) in enumerate(legend):
            y0 = 22 + i * 40
            dr.rectangle([W - 210, y0, W - 186, y0 + 26], fill=c)
            dr.text((W - 178, y0 + 1), lab, font=font(20), fill=(225, 230, 238))
        dr.rectangle([W - 210, 22, W - 186, 22 + 3 * 40 - 14], outline=(120, 126, 136))

    return canvas


def main():
    dev = "cuda"
    torch.cuda.set_per_process_memory_fraction(0.35, 0)
    bb = ResnetBackbone(True).to(dev)
    head = DepthHead(192).to(dev)
    sd = torch.load(ROOT / "runs" / "v2_imagenet" / "head.pt", map_location="cpu", weights_only=False)
    head.load_state_dict(sd["head"])
    if "proj" in sd:
        bb.load_state_dict(sd["proj"], strict=False)
    bb.eval(); head.eval()

    idx = json.loads(CACHE.read_text(encoding="utf-8"))
    _, va, te = split(idx, 0)
    eps = va + te                                  # 只用留出集，被试没见过训练帧
    import torch.utils.data as tud
    dl = tud.DataLoader(Frames(idx, eps), batch_size=8, num_workers=0)

    rows = []
    with torch.no_grad():
        for rgb, disp_gt, mask in dl:
            tok, th, tw = bb(rgb.to(dev).unsqueeze(1))
            disp = head(tok, th, tw)
            pd = (1.0 / torch.clamp(disp.squeeze(1), min=1e-3)).cpu().numpy()
            gd = (1.0 / torch.clamp(disp_gt.squeeze(1), min=1e-3)).numpy()
            mm = (mask.squeeze(1) > 0.5).numpy()
            for j in range(pd.shape[0]):
                g = np.where(mm[j], gd[j], np.nan)
                p = np.where(mm[j], pd[j], np.nan)
                rows.append({"gt": front_near(g), "pred": front_near(p),
                             "rgb": rgb[j].clone(), "pd": pd[j], "gd": g})

    danger = sorted([r for r in rows if r["gt"] < 8.0], key=lambda r: r["gt"])[:10]
    safe = sorted([r for r in rows if r["gt"] > 15.0], key=lambda r: -r["gt"])[:10]
    picks = danger + safe
    print("危险 %d 帧、安全 %d 帧" % (len(danger), len(safe)), flush=True)

    manifest = []
    for i, r in enumerate(picks):
        arr = ((r["rgb"].permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        item = {"id": i, "gt_near": round(float(r["gt"]), 2),
                "pred_near": round(float(r["pred"]), 2),
                "truth": "danger" if r["gt"] < 9.0 else "safe", "files": {}}
        for ver in ("A", "B", "C"):
            p = OUT / ("f%02d_%s.png" % (i, ver))
            render(img, r["pd"], float(r["pred"]), ver).save(p)
            item["files"][ver] = p.name
        manifest.append(item)
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
    print("已生成", len(manifest) * 3, "张图 ->", OUT, flush=True)


if __name__ == "__main__":
    main()
