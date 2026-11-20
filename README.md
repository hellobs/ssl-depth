# ssl-depth

低空物流配送无人机的视觉测距误差研究。课程论文项目，配套代码与数据索引。

论文：《基于人工智能的无人机单目深度测距误差研究与设计学分析》

作者：课程项目组（成员信息见论文首页）

## 这个项目在做什么

配送无人机的视觉分支需要从单张图像估计前方障碍的距离。这个估计会出错，而错误不是均匀分布的。项目做四件事：

1. 在仿真城市环境中采集飞行数据，训练单目深度估计模型
2. 固定解码器与训练预算，比较四种视觉表征，并对误差作分层分析
3. 把误差换算成避障决策，定位失效集中的距离区间与偏差方向
4. 从设计学角度分析信息传达的条件，并把人在回路的复核机制做成可运行的原型

## 目录

```
scripts/
  rebuild_index.py     扫描数据目录，按规则生成帧索引
  diag_backbone.py     四种表征的对照训练与评测
  analyze_error.py     误差分层分析（距离、架次、失效帧）
  task_metric.py       避障判据换算与混淆矩阵（--split held 仅算留出集）
  diag_bias.py         前方距离的带符号偏差（判断读远还是读近）
  plot_figs_v2.py      论文配图（图 1 至图 4）
  build_review_queue.py 生成复核队列（触发帧抽样）
  review_app.py        人在回路复核原型（FastAPI 单页应用）
paper/
  main.tex             论文源文件（XeLaTeX + ctexart）
  main.pdf             编译产物
  figs/                配图
review/
  review_queue.jsonl   复核队列（110 条）
  reviews.jsonl        评审记录
  imgs/                队列对应的触发帧
deploy_review/         复核原型的部署副本（含 requirements.txt）
cache/
  kept_all.json        帧索引（v4 口径：104 个航段、12647 帧）
  kept_all_v5.json     v5 口径：189 个航段、21129 帧（11 个数据根）
runs/
  v4_<骨干>/           上一轮训练产物（训练集 66 航段 / 8565 帧）
  v5_<骨干>/           论文主表口径（训练集 152 航段 / 17053 帧）
```

## 数据

原始数据不在本仓库。两个环境变量需要在运行时指定：`WM_DATA_ROOT` 指向数据根目录，`WM_PY` 指向训练用的 Python 解释器。原始数据位于本地 `$WM_DATA_ROOT` 下 10 个目录与 `$WM_DATA_ROOT/near_out`、`$WM_DATA_ROOT/near_out2`，共 202 个可用航段。

每帧包含前视 RGB、稠密深度图、深度有效掩码。深度由 Unreal Engine 4 的 AirSim 插件直接渲染输出，单位为米，与图像严格对齐。

清洗规则：有效范围 0.5 至 80 米，帧有效率不低于 50%。v5 索引在 36757 帧候选上保留 21129 帧（57%），被剔除的帧集中在 40 米以外的远景与大面积天空。

划分按航段进行：训练 152、验证 14（1641 帧）、测试 15（1685 帧）。`near_out2` 的 10 个航段登记在 `runs/held_out_only.txt`，只评测、永不训练。

v5 的索引与划分用下面三条命令生成：

```bash
python scripts/index_roots.py --roots runs,runs_batch2,runs_batch3 --out cache/kept_all_v5.json
python scripts/index_roots.py --roots runs_batch4,runs_batch5,runs_batch6 --out cache/kept_all_v5.json
python scripts/make_v5_split.py          # -> runs/v5_split.json + runs/pure_train_runs_v5.txt
```

训练与评测都要显式指定 `--index cache/kept_all_v5.json --split-file runs/v5_split.json`；不指定则回落到 v4 口径。

## 运行

环境：`$WM_PY`（Python 3.13 + PyTorch 2.9.1 + CUDA 12.8）

```bash
# 重建索引
python scripts/rebuild_index.py

# 训练并评测某一组骨干
python scripts/diag_backbone.py --backbone imagenet --epochs 15

# 误差分层分析
python scripts/analyze_error.py --backbone imagenet

# 避障判据换算（全量 / 仅留出集）
python scripts/task_metric.py
python scripts/task_metric.py --split held

# 带符号偏差
python scripts/diag_bias.py

# 配图
python scripts/plot_figs_v2.py

# 复核原型（本地 8000 端口）
python scripts/review_app.py
```

可用的骨干：`imagenet`、`resnet_scratch`、`jepa`、`jepa_domain`、`scratch`。
其中 `jepa_domain` 指向 `runs/jepa_domain_v4/jepa.pt`（域内自监督，零泄漏，论文唯一引用的域内权重）。
`runs/jepa_domain_clean/`（20/29 个留出航段进过预训练）与 `runs/_deprecated/jepa_domain_leaky/`
（14/16 个测试航段进过预训练）都不可引用，保留只为对照。

域内预训练怎么来的：

```bash
# 52 个纯训练 run / 58 航段 / 10014 帧对，4000 步，约 25 分钟（RTX 5060）
python scripts/pretrain_jepa.py --runs <逗号分隔的 52 个后缀> --steps 4000 \
  --batch 16 --workers 4 --out runs/jepa_domain_v4
```

预训练的超参默认值就是下游规格（108×192 / patch 6 / dim 192 / heads 3 / 576 token）。
位置编码是定长相加、不插值，token 数差一个就会在加载时被形状过滤掉且不报错，
所以脚本存盘后会用下游配置做一次 `verify_loadable` 自检。

## 论文编译

```bash
cd paper
xelatex main.tex
xelatex main.tex
```

需要 TeXLive（含 ctex）与 SimSun、SimHei 字体。

## 结果

v5 口径：训练 152 航段 / 17053 帧，留出集 3326 帧（29 航段，与 v4 完全一致）。

| 骨干 | 预训练数据 | 验证集 δ1 | 测试集 δ1 | 测试集 AbsRel |
|---|---|---|---|---|
| ImageNet ResNet18 | ImageNet 有监督 | **0.836** | **0.867** | **0.133** |
| ResNet18 随机初始化 | 无 | 0.829 | 0.846 | 0.208 |
| JEPA 自监督（域内） | 本数据 143 run / 153 episode / 30.1k 帧对 | 0.813 | 0.812 | 0.263 |
| ViT 随机初始化 | 无 | 0.762 | 0.721 | 0.311 |

任务指标（留出集 3326 帧）：

| 骨干 | 6 米识别率 | 6 米误触发 | 9 米识别率 | 9 米误触发 | 预测中位距离 |
|---|---|---|---|---|---|
| ImageNet ResNet18 | 67.5% | **8** | 78.1% | **11** | 17.00 m |
| ResNet18 随机初始化 | 61.6% | 9 | 74.0% | **7** | 16.97 m |
| JEPA 自监督（域内） | **75.5%** | 45 | **84.0%** | 100 | 15.38 m |
| ViT 随机初始化 | 70.5% | 43 | 77.1% | 96 | 14.91 m |

四个骨干的 9 米识别率只差十个百分点（74.0% 到 84.0%），误触发却差十四倍（7 到 100）。JEPA 与 ViT 随机偏向多报，尺度分别偏近 10.0% 与 12.7%；ImageNet 与随机卷积偏向少报。换 JEPA 这笔账不划算：9 米判据上每多守 28 帧要多付 89 帧误刹，6 米判据上是 19 帧换 37 帧。

两轮对照（唯一变量是训练集，8.5k → 17k 帧）：

| 骨干 | v4 test δ1 | v5 test δ1 | 变化 |
|---|---|---|---|
| ImageNet | 0.866 | 0.867 | +0.001 |
| ResNet18 随机 | 0.825 | 0.846 | +0.021 |
| ViT 随机 | 0.696 | 0.721 | +0.025 |
| JEPA 域内 | 0.835 | 0.812 | −0.023 |

受益最多的是两个没有预训练的骨干，受益最小的是已经有强先验的 ImageNet。JEPA 的下降要打折扣看：它的预训练语料同时扩到三倍而步数没变（4000 步，约 3 遍对 6 遍），属于欠训，不能单独用来判断语料多寡。

**欠训对照（2026-11-19 18:30，`runs/v5_jepa_g4/`）**：同一份语料，把帧对间隔从 1 帧改成 4 帧、步数从 4000 提到 12000。验证集 δ1 从 0.813 升到 0.840，测试集 AbsRel 从 0.263 降到 0.220，最差那一趟的误差从 2.20 降到 1.40，**但测试集 δ1 仍是 0.813**。把预训练做足改善的是误差量级，不是模型排序，所以“域内自监督在测试集上低于随机卷积”不是训练不足的假象。预训练权重在 `runs/jepa_domain_v5_g4/jepa.pt`（231/231 张量命中，12000 步，gap 4）。

## 两条已修正的测量错误

**一、旧 JEPA 行是另一套划分。** `runs/v4_jepa_domain` 早期版本跑在 `kept_all` 的 104 航段上（73/15/16），与其余三行的 98 航段划分只重叠 4/15 个测试航段，两套数从来不可比。改用 98 航段重跑后 δ1 从 0.829 变 0.835，几乎没动。论文里对应的是 `runs/v4_jepa_domain`（v4 口径）。

**二、随机初始化骨干的任务指标曾经全错。** `task_metric.py` 重建随机骨干时不设种子，每次都是另一份权重，于是 v4 曾出现「ResNet18 随机识别率 100%/误刹 2846」「ViT 随机 0%/0」这类退化数字。修正后（从 metrics.json 读回 seed）v4 的真实值是：ResNet18 随机 9 米 69.2%/21，ViT 随机 81.7%/99。**论文 `tab:scale` 与第七章那两段已按修正值重写，旧数字不可引用。** 原有结果备份在 `runs/_bak_taskmetric_20261119/`。

## 分层与偏差（v5，ImageNet 预训练）

| 区间 | 帧数 | AbsRel | δ1 | δ1<0.5 占比 |
|---|---|---|---|---|
| 0-10 米 | 408 | 0.355 | 0.732 | 16% |
| 10-20 米 | 1157 | 0.131 | 0.870 | 0.0% |
| 20-40 米 | 1055 | 0.105 | 0.878 | 3% |
| 40-81 米 | 706 | 0.111 | 0.878 | 0.0% |

9 米判据：该触发 480 帧，识别 375 帧（78.1%），漏判 105，误触发 11。放宽到 12 米，识别率 97.0%。

近场偏差仍是读远，但比 v4 小了一半：6 至 9 米区间中位 +0.72 米（v4 是 +1.58），75% 的帧高估，15% 高估超过 2 米。40 米以外不再转为读近，中位 +0.51 米。

## 复核原型

`scripts/review_app.py` 是一个单页应用，评审者逐帧判断“可信 / 有保留 / 不可信 / 无法判断”，后三项展开按机理分类的理由清单。队列 110 条，由 `build_review_queue.py` 从触发帧中按距离分档抽样生成。

正式复核收集到 157 次提交，覆盖 106 个不同样本（队列 110），平均 8.1 秒/条。判定分布：采信 54、有保留 30、不采信 72、无法判断 1。结构化理由 67 条，约占需理由判定的 65%。

**不要引用旧的 46 条/4% 结论。** 那是作者随意试填，已在 2026-11-19 删除。A/B 两模式提交 82/75 次，判定准确率 0.468/0.484，差异在噪声内，不支持任何界面优劣结论。论文口径不区分提交者。

## 闭环的第三个环节

前两个环节（可溯源、可评判）已经在跑；第三个环节（判定转成训练数据）搭好了管线，但**还没有真正训练过**。

```
复核记录 ──loop_labels.py──> cache/loop_labels.jsonl
                                    │
   近场新数据（可选，只做深度监督）──┤
                                    ▼
             MicroJEPA encoder（冻结）+ ViT LoRA → DepthHead
                                    │
              ┌─────────────────────┴──────────────────┐
         掩码视差 L1                        帧级铰链（来自人的判定）
                                    │
                                    ▼
                    外部近场集验收（从不参与训练）
```

```bash
# 标签：正式复核 157 次提交 / 106 个去重样本 -> cache/loop_labels.jsonl
python scripts/loop_labels.py

# 试通：走一遍前向反向，确认接通，不保存权重
python scripts/loop_lora.py --dry-run

# 真正训练（尚未执行）
python scripts/loop_lora.py --train --epochs 10 --near-root $WM_DATA_ROOT/near_out2
```

两个设计决定值得说明：

**标签不绑队列条目。** `item_id` 形如 `run/episode/stem`，直接按存在性定位图像与深度，真值用 GT 深度现算。队列中途重建过，按队列查表会丢失记录；按文件定位则全部解析成功。旧版 46 条试填数据的冲突率与一致率数字（16% / 82% / 54%）已作废，不要引用。

旧版的 ResNet 卷积 LoRA（layer3/4）已删除：论文主线转向世界模型，微调对象换成 MicroJEPA 本身。LoRA 注入每个 Block 的 `mlp.fc1` / `mlp.fc2`（r=8, α=16），共 12 层、0.09M 参数，占可训练量的 10%。`attn.out_proj` 不能注入——`nn.MultiheadAttention.forward` 内部直接用 `out_proj.weight` 做投影，不走模块调用，包装会报 AttributeError；`in_proj` 是单个 Parameter，同样无法旁路。每层容量主要在 MLP（192→768→192），只注入 fc1/fc2 是安全折中。

**纯视差 LoRA 只验证机制，不作结论（2026-11-20，`runs/v5_jepa_lora/`）**：v5 JEPA 权重 + 12 层 LoRA，8 轮。跑出来的数字是 test δ1 0.812→0.832、AbsRel 0.263→0.205、9 米误刹 100→33、召回 84.0%→69.1%，但这些数字**不进论文**。理由：训练集 17,053 帧对适配偏小，复核标签 157 次提交也不够做帧级监督，微调收益落在噪声范围内。论文只在第 7 章描述管线结构，不给效果结论。

铰链把帧级判定翻译成对 ROI 前方距离 p05 的约束：判"该触发"要求 `f ≤ 9−1.5`，判"不该触发"要求 `f ≥ 9+1.5`。`--w-hinge 0` 即纯视差（回答表征问题），`--w-hinge 0.5` 起为决策级目标。

**外部验收集**是 2026-11-18 新采的 10 次近场飞行（`$WM_DATA_ROOT/near_out2`，1422 帧，3 米高度贴楼，前方距离中位 10.8 米），从未参与训练。现有 v4 模型在这个集上的基线：中位误差 −0.02 米、δ1 0.871，但真值 <9 米的 218 帧只召回 31 帧（**14%**）——预测几乎恒定在 11 米附近，失去了近距离的分辨能力。这是闭环需要改进的起点。

## 已知限制

- 数据全部来自仿真，缺少真实相机的噪声与光照突变
- 近场样本偏薄（留出集仅 408 帧占 12%），而它恰是失效最集中的区间
- 9 个后补近场航段未进入本轮训练
- 人在回路的闭环搭好了第三个环节的管线（MicroJEPA + ViT LoRA + 帧级铰链），纯视差 LoRA 已跑，铰链项尚未启用
- 复核记录 157 次提交但只有 106 个去重样本，且一位参与者占 72%，不足以支撑界面效果的统计推断
