"""逐 episode 诊断：GT 深度分布 + 训练好的模型在每个 episode 上的 δ1。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
import train_min as T  # noqa: E402


def main():
    ck = torch.load("D:/zzr/ssl-depth/runs/min20_80m/min3.pt", map_location="cuda")
    model = T.MicroJEPA(ck["cfg"]).cuda()
    model.encoder.load_state_dict(ck["encoder"])
    head = T.DepthDecoder(ck["cfg"]["model"]["dim"]).cuda()
    head.load_state_dict(ck["head"])
    idx = T.build_index()
    meta = json.load(open("D:/zzr/ssl-depth/runs/min20_80m/metrics.json"))
    part = {e: k for k, v in meta["split"].items() for e in v}

    print("%-38s %-6s %5s %8s %8s %6s %6s" % ("episode", "split", "n", "gt_med", "gt_p90", "d1", "AbsRel"))
    for e in sorted(idx["kept"]):
        ds = T.DepthFrames(idx, [e])
        dl = DataLoader(ds, batch_size=16, num_workers=0)
        m = T.evaluate(model, head, dl, "cuda")
        g = []
        for _, dg, mask in dl:
            g.append((1.0 / torch.clamp(dg.squeeze(1), min=1e-3))[mask.squeeze(1) > .5].numpy())
        g = np.concatenate(g)
        print("%-38s %-6s %5d %8.1f %8.1f %6.3f %8.3f"
              % (e.split("/")[0][-8:] + "/" + e.split("/")[1][-4:], part.get(e, "?"),
                 len(ds), np.median(g), np.percentile(g, 90), m["d1"], m["absrel_med"]))


if __name__ == "__main__":
    main()
