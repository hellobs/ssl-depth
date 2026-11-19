# 交接文档（2026-11-19 09:30）

> 上一版交接（`archive/HANDOFF_20261117.md`）已过时：它把项目定性为「标签效率」研究、8–10 页、三张主图。
> 现在的实际形态是：**25 页课程论文**，核心是「四种视觉表征的对照 + 误差分层 + 设计学分析」，
> 并含一套人在回路复核原型。**以本文档为准。**

---

## 0. 一句话状态

论文已经写完并编译通过（25 页，0 个 overfull），**但里面关于「域内自监督」的四个数字来自一版有数据泄漏的权重，
必须重跑一次深度头再用新数替换。** 断点就在这一步。

具体断点：`scripts/diag_backbone.py --backbone jepa_domain --epochs 15 --device cpu`
跑到 `ep 1/15`（val AbsRel 0.334 / δ1 0.521，412 秒/轮）时被用户主动叫停，**没有写出任何权重**。

---

## 1. 项目是什么

一篇「设计 + AI」跨学科课程论文：《基于人工智能的无人机单目深度测距误差研究与设计学分析》。

- 平台：Unreal Engine 4 + AirSim，`CityEnviron` 城市峡谷场景（**场景固定，不允许换图**）
- 数据：多批采集，共 117 次飞行 / 19,773 帧 RGB + 稠密深度真值
- 任务：单目深度估计（视差空间），关键指标是**中心 ROI 内前方最近距离的 5% 分位**，用于 6/9/12/15/20 米避障判据
- 论文结构：引言 → 主题 → 背景 → 设计 → 算法 → 实施（含四表征对照）→ 设计学分析（含复核原型）→ 总结 → 结语 → 致谢 → 分工 → AI 说明 → 参考文献

---

## 2. 环境（先确认这个，否则会白跑）

**唯一的完整 Python 环境**：`D:/zzr/microwm/.venv/Scripts/python.exe`
（Python 3.13 + torch 2.9.1+cu128 + cv2 5.0.0 + torchvision）

其它环境都不完整，不要用：
- `drones` / `pyengineer`：有 torch，**无 cv2**
- `cosys-airsim`：有 cv2，**无 torch**
- `ue4-airsim`：torch 2.4.1 **纯 CPU**，无 cv2（这个只用于跑 AirSim 采集器）

**LaTeX**：`D:/texlive/2026/bin/windows/xelatex`，在 `paper/` 目录下编译两次（交叉引用需要跑两遍）。

**GPU**：RTX 5060 Laptop，8.5 GB。注意 `nvidia-smi` 里 3.2–3.4 GB 常驻占用是 `CityEnviron.exe`（AirSim 渲染），
**不是训练**。判断有没有训练占卡要看 `nvidia-smi --query-compute-apps` 里有没有 python 进程。

---

## 3. 数据与划分（这里有本项目最大的坑，务必读完）

### 3.1 数据落在 7 个根目录

```
D:/zzr/wmdata/runs            10 run / 20 航段 / 3167 帧
D:/zzr/wmdata/runs_batch2     13 run / 20 航段 / 3989 帧
D:/zzr/wmdata/runs_batch3     20 run / 20 航段 / 3151 帧
D:/zzr/wmdata/runs_batch4     19 run / 19 航段 / 4302 帧
D:/zzr/wmdata/runs_nearfield  15 run /  1 航段 /  152 帧   ← 14 个 run 是空跑，没有 observations
D:/zzr/near_out               30 run / 29 航段 / 3590 帧   ← 5 个航段缺 observations
D:/zzr/near_out2              10 run / 10 航段 / 1422 帧   ← 外部验收集，永不训练
```

全部 `run.json` 的 `schema` 都是 `airsim-delivery-m.v1`（M 包采集器）。

### 3.2 两套划分，只有一个是对的

- **权威划分**：`runs/v4_imagenet/metrics.json` 的 `split` 字段 → 训练 69 / 验证 14 / 测试 15 = **98 航段**。
  所有 v4 结果、论文表格、任务指标都基于它。
- **陷阱**：`cache/kept_all.json` 是 **104 航段**。因为索引在训练完成之后重建过（补采 9 个近场航段、删掉 3 个早期航段的记录）。
  98 + 9 − 3 = 104。**用 kept_all 现算 split() 会得到 73/15/16，与评测划分不一致。**

`scripts/task_metric.py` 里专门有一段注释写明这件事，别忽略。

### 3.3 数据缺口的硬事实

- 留出集 3326 帧里，真值 <9 m 只有 480 帧（14.4%），<6 m 只有 237 帧（7.1%）
- `near_out2` 那 10 趟专门采近场，实际全部落在 **7–12 m**（p05 7.3 / 中位 10.8）——3 m 高度下街道半宽约 11 m，飞不进去
- **0–7 m 是空的**，而 0–10 m 恰恰是 δ1 最低的一档（0.719）
- 结论：不是「总量不够」，是「从来没正面逼近过墙」。再采巡航里程边际近零。

---

## 4. 权重清单（哪些能用，哪些是毒药）

| 路径 | 语料 | 能否引用 |
|---|---|---|
| `runs/jepa_domain_v4/jepa.pt` | v4 划分下 52 个纯训练 run / 58 航段 / 10,014 帧对 / 4000 步 | **唯一可引用**。两道泄漏断言通过，`231/231` 张量可加载 |
| `runs/jepa_domain_clean/` | 56 run / 65 航段 / 11,752 帧对 | **不可引用**。20/29 个留出航段进过预训练（用错了 kept_all 划分） |
| `runs/_deprecated/jepa_domain_leaky/` | 100 航段 / 17,492 帧对 | **不可引用**。14/16 个测试航段进过预训练（按 run 名排除，但划分是航段级） |
| `D:/zzr/microwm/outputs/base_s6/jepa.pt` | VisDrone，4000 步 | 论文已删该臂，但代码里 `jepa` 臂仍指向它，留作可复现对照 |
| `D:/zzr/microwm/outputs/dl_act8k/` | dronelab_train，8000 步，`use_actions=True` | **已删除**（2026-11-19，707 MB）。下游 δ1 只有 0.238 |

三个域内权重的关系：`leaky` → `clean` → `v4`，每一版都在修上一版的泄漏，**而数字几乎不变（0.830 / 0.829）。
这就是最危险的地方——光看数字分辨不出哪版有污染，只能靠路径和断言。

`runs/held_out_only.txt` 登记「只评测、永不训练」的 run（目前是 `near_out2` 的 10 个）。
`pretrain_jepa.py` 的 `assert_no_leak()` 会读它，并在开跑前断言零重叠。

`runs/pure_train_runs_v4.txt` 是正确的 52 个纯训练 run 列表。
`runs/pure_train_runs.txt` 是**错的**那一版（56 个，基于 kept_all），保留只为对照。

---

## 5. 已完成的结果（可以直接用，除 JEPA 域内那一行）

### 5.1 像素级（v4 划分，测试 1849 帧）

| 骨干 | 预训练 | 验证 δ1 | 测试 δ1 | 测试 AbsRel | 测试 RMSE |
|---|---|---|---|---|---|
| ImageNet ResNet18 | ImageNet 有监督 | 0.819 | **0.866** | **0.127** | 6.79 |
| JEPA（域内，**旧版有泄漏**） | 本数据 | **0.849** | 0.829 | 0.163 | 6.94 |
| ResNet18 随机 | 无 | 0.810 | 0.825 | 0.252 | 8.10 |
| JEPA（VisDrone） | 航拍俯视 | 0.727 | 0.743 | 0.369 | 9.94 |
| ViT 随机 | 无 | 0.709 | 0.696 | 0.319 | 10.56 |

### 5.2 任务指标（留出集 3326 帧）

| 骨干 | 6 m 识别率 | 6 m 误刹 | 9 m 识别率 | 9 m 误刹 |
|---|---|---|---|---|
| JEPA 域内（旧） | 81.9% | 28 | 87.3% | 24 |
| ImageNet | 57.4% | **3** | 67.3% | **10** |
| JEPA VisDrone | 63.3% | 49 | 74.4% | 55 |
| ResNet18 随机 | 100% | 3089 | 100% | 2846 |
| ViT 随机 | 0% | 0 | 0% | 0 |

### 5.3 尺度（留出集，真值中位 17.08 m）

ImageNet 16.83（−1.5%）｜JEPA 域内 17.64（+3.3%）｜ViT 随机 19.76（+15.7%）｜ResNet18 随机 0.38（−97.8%）

### 5.4 两条待定稿的结论

1. **两个指标指向相反的失效模式**：δ1 上 ImageNet 最高（0.866 vs 0.829），但近场召回上域内自监督反超 24 个百分点，
   代价是误刹从 3 涨到 28（3326 帧的 0.84%）。对避障而言漏判比误刹危险，这个权衡方向是对的。
2. **语料贡献大于架构**：ViT 随机 0.696 → 域内自监督 0.829，+0.133；ResNet18 随机 0.825 → ImageNet 0.866 只有 +0.041；
   架构差（0.825 vs 0.696）= 0.129。**⚠️ 这一条必须等 v4 干净版重跑后才能定稿**，泄漏可能抬高了 +0.133。

---

## 6. 脚本清单

| 脚本 | 作用 | 关键参数 |
|---|---|---|
| `scripts/pretrain_jepa.py` | 域内自监督预训练 | `--runs <逗号后缀>` `--gap` `--steps` `--batch` `--workers` `--out` |
| `scripts/diag_backbone.py` | 冻结骨干 + 深度头训练评测 | `--backbone {scratch,jepa,jepa_domain,imagenet,resnet_scratch}` `--device cpu\|cuda` `--workers` `--epochs` |
| `scripts/task_metric.py` | 任务级指标（召回/误刹） | `--split all\|held` |
| `scripts/review_app.py` | 复核原型（FastAPI 单页） | 环境变量 `PORT`；配图用 `?peek=N` `?reasons=1` |
| `scripts/nearfield_report.py` | 任意目录的前方距离分布报告 | `--json` |
| `scripts/loop_labels.py` | 把 `review/reviews.jsonl` 转成训练标签 | `--drop-conflict` |
| `scripts/loop_lora.py` | LoRA 微调管线（**只搭好，没真跑过**） | 默认 `--dry-run`；`--train` 才训练 |
| `scripts/plot_paper_figs.py` | 论文配图 | ⚠️ 读的是 `runs/v2_*`，已过时；实际配图用 `plot_figs_v2.py` |

---

## 7. 踩过的坑（按踩到的次数排序）

1. **位置编码定长，不插值。** `MicroJEPA` 的 `Encoder.pos = nn.Parameter(zeros(1, n_tokens, dim))`，
   前向是 `z = z + self.pos`。预训练必须严格用 108×192 / patch 6 / dim 192 / heads 3 → 576 token。
   配置不对时加载阶段会按形状把 `pos` 静默过滤掉，冻结主干带着随机位置先验继续跑，**不报错**。
   `pretrain_jepa.py` 现在存盘后会自动 `verify_loadable()`（打印 `231/231 张量命中`）。
2. **航段级泄漏，犯过两次。** 第一次按 run 名排除（`--skip-runs`）——不行，同一 run 的不同航段会分到不同集合；
   第二次按 `kept_all`（104 航段）算纯训练 run——也不行，评测用的是 98 航段那套。
   **正确做法：只读 `runs/v4_imagenet/metrics.json` 的 `split`。** 现在有两道断言（v4 留出集 + `held_out_only.txt`）。
3. **`num_workers=0` 让 GPU 空转等磁盘。** 实测 nvidia-smi 在 100% 和 0% 之间跳。
   加 4 个加载进程后：预训练 4000 步从 46 分钟降到 17 分钟；深度头每轮从 300 秒降到 60 秒。
4. **后台任务不要用 `&` 挂起。** shell 一退出进程就被杀，已因此白跑两次。用 `run_in_background`。
5. **CPU 模式要限线程。** `OMP_NUM_THREADS=12`（24 核的一半）+ 把 `pin_memory` 关掉，否则互相抢。
6. **帧对数少于 batch 时**，`drop_last=True` 会让 DataLoader 长度为 0，接着除零崩溃。已加显式报错。
7. **`.reasons` 的 `display` 不能设成 `block`**，会盖掉 CSS 的 `grid-template-columns`（内联优先级更高）。
8. **论文旧表 3（尺度）四行口径不一致**：两行来自 held、一行来自 all、一行（ViT 25.2 m）对不上任何数据源。11-19 已统一为 held。

---

## 8. 未完成事项（按优先级）

### P0 — 论文数字必须替换

```bash
# 1. 重跑深度头（CPU 约 1h43m；GPU 约 12 分钟）
OMP_NUM_THREADS=12 D:/zzr/microwm/.venv/Scripts/python.exe -u scripts/diag_backbone.py \
  --backbone jepa_domain --epochs 15 --workers 3 --device cpu

# 2. 重算任务指标
D:/zzr/microwm/.venv/Scripts/python.exe -u scripts/task_metric.py --split held

# 3. 按新数改 paper/main.tex 四处：摘要（中英）、tab:backbone、tab:scale、6.x 决策段
```

现状：`runs/v4_jepa_domain/` 里仍是**旧结果**（`metrics.json` / `head.pt`，来自 `jepa_domain_clean`），
另有 `metrics_prev_clean.json` / `head_prev_clean.pt` 是同一份备份。
`CKPTS["jepa_domain"]` 已指向 `runs/jepa_domain_v4/jepa.pt`，直接重跑就会覆盖成干净结果。

**替换时必须同步的位置**（现在都写着旧数 0.829 / 0.163 / 87.3% / 24 / 3.3%）：
- `paper/main.tex` 中文摘要、英文摘要
- `tab:backbone` 的 JEPA 行、`tab:scale` 的 JEPA 行
- 6.x「如果选择换模型」那段（157→61、67.3%→87.3%、10→24）
- `README.md` 结果表
- `runs/DIAG_20261117.md` 若引用

### P1 — 近场采集（方案已定，未执行）

30 局：**进场 15 / explore 9 / goal 6**。采集器只有 `goal` 和 `explore` 两个 policy，
「进场」用 `goal` + `--via-points` 航点序列实现（不是独立策略）。

```bash
--policy goal --via-points approach_a.json --cruise-speed 1.2 --dt 0.2 --episode-seconds 40
```

验收：前方距离 5% 分位 ≤ 4.5 m（`near_out2` 现在是 7.3 m）；全批 <7 m 帧数 ≥ 1000；每局 `validate_dataset.py` 通过。
新数据放 `near_out3/` 并写进 `runs/held_out_only.txt`，**否则断言拦不住它**。

### P2 — 闭环第三环节（管线已搭好，未真跑）

`loop_labels.py`（复核记录 → 标签，含冲突检测与 κ）+ `loop_lora.py`（ConvLoRA 注入 ResNet18 的 layer3/4：
可训练 0.22M LoRA + 0.098M proj + 0.812M head，冻结主干 11.39M）。`--dry-run` 通过，`--train` 只做过 2 步冒烟。

⚠️ 复核数据（`review/reviews.jsonl`，46 条）是作者**随意测试**的，不是正式实验，用户说后续会给正式数据。
在此之前**不要**基于它写任何结论。实际形态：评审 1 做 38 条、评审 2 做 6 条、评审 3 做 2 条；理由填写率 4%；
单条中位 2.0 秒（均值 5.8 秒被一条 50 秒记录拉高）；4 个重复标注样本无一一致。

### P3 — 杂项

- `/summary` 页面有布局 bug：卡片只占左上角约一半，不居中也不撑满
- 大量改动**未提交 git**（论文、脚本、README、memory、figs）
- `paper/main-2.pdf` 与 `paper/【ykt主观题附件】…pdf` 有意不入库

---

## 9. 不要做的事

1. **不要改 `D:/zzr/tools/wm-collector-m/`**（M 包采集器）。用户明确要求：需要改就复制一份到工作区再改。
2. **不要在没有通过泄漏断言的情况下产出权重**。断言失败就修语料，不要调参数绕过。
3. **不要偷偷用 GPU**。机器上有别的任务在跑，GPU 状态要主动报告（见 §2 的判别方法）。
4. **不要为了让数字好看而换语料**。数字掉下来就改论文，不改数据。
5. **不要把 `dl_motion_v*` 当通用编码器**：它们是 6 通道（RGB + 帧间差分）且带 `use_actions`，
   与本项目用的 3 通道纯 RGB 权重不是一回事。
