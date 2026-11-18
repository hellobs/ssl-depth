# ssl-depth

低空物流配送无人机的视觉测距误差研究。课程论文项目，配套代码与数据索引。

论文：《基于人工智能的无人机单目深度测距误差研究与设计学分析》

作者：张子睿（华南理工大学设计学院）、盛祺（自动化科学与工程学院）、刘子瑄（电子商务系）、马婧怡（材料科学与工程学院）

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
  kept_all.json        帧索引（104 个航段、12647 帧）
runs/
  v4_<骨干>/           训练产物与指标
```

## 数据

原始数据不在本仓库，位于本地 `D:/zzr/wmdata/runs`、`runs_batch2`、`runs_batch3`、`runs_batch4`、`runs_nearfield` 与 `D:/zzr/near_out`，共六批。

每帧包含前视 RGB、稠密深度图、深度有效掩码。深度由 Unreal Engine 4 的 AirSim 插件直接渲染输出，单位为米，与图像严格对齐。

清洗规则：有效范围 0.5 至 80 米，帧有效率不低于 50%。18351 帧候选中保留 12647 帧（69%），对应 87 次飞行、104 个航段。被剔除的帧 95% 在 40 米以外的远景。

划分按航段进行：训练 69、验证 14（1641 帧）、测试 15（1685 帧）。

索引在训练完成后重建过一次：补入 9 个近场航段（756 帧），另有 3 个早期航段的原始记录被清理删除。因此索引的 104 段与实验用的 98 段不完全重合，`scripts/task_metric.py --split held` 从 `runs/v4_imagenet/metrics.json` 读取训练时的划分，避免训练帧漏进留出集。

## 运行

环境：`D:/zzr/microwm/.venv`（Python 3.13 + PyTorch 2.9.1 + CUDA 12.8）

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

可用的骨干：`imagenet`、`resnet_scratch`、`jepa`、`scratch`。

## 论文编译

```bash
cd paper
xelatex main.tex
xelatex main.tex
```

需要 TeXLive（含 ctex）与 SimSun、SimHei 字体。

## 结果

四组表征的对照（留出集 3326 帧）：

| 骨干 | 预训练数据 | 验证集 δ1 | 测试集 δ1 | 测试集 AbsRel |
|---|---|---|---|---|
| ImageNet ResNet18 | ImageNet 有监督 | 0.819 | 0.866 | 0.127 |
| ResNet18 随机初始化 | 无 | 0.810 | 0.825 | 0.252 |
| JEPA 自监督 | 航拍图像 | 0.727 | 0.743 | 0.369 |
| ViT 随机初始化 | 无 | 0.709 | 0.696 | 0.319 |

按真值中位距离分档（ImageNet 预训练）：

| 区间 | 帧数 | AbsRel | δ1 | δ1<0.5 占比 |
|---|---|---|---|---|
| 0-10 米 | 408 | 0.354 | 0.719 | 17.6% |
| 10-20 米 | 1157 | 0.131 | 0.868 | 0.0% |
| 20-40 米 | 1055 | 0.110 | 0.859 | 5.8% |
| 40-81 米 | 706 | 0.110 | 0.874 | 0.4% |

9 米判据（留出集）：该触发 480 帧，识别 323 帧（67.3%），漏判 157，误触发 10。放宽到 12 米，识别率 94.2%。

近场偏差是读远：6 至 9 米区间中位 +1.58 米，85% 的帧高估，三分之一高估超过 2 米。

## 复核原型

`scripts/review_app.py` 是一个单页应用，评审者逐帧判断“可信 / 有保留 / 不可信 / 无法判断”，后三项展开按机理分类的理由清单。队列 110 条，由 `build_review_queue.py` 从触发帧中按距离分档抽样生成。

目前已收集 3 名评审的 46 条记录（覆盖 38 个样本，平均 5.8 秒/条），规模仅够试点。结构化理由的填写率只有 4%，说明理由环节在秒级节奏下会被跳过——这是原型暴露出的主要设计问题。

## 已知限制

- 数据全部来自仿真，缺少真实相机的噪声与光照突变
- 近场样本偏薄（留出集仅 408 帧占 12%），而它恰是失效最集中的区间
- 9 个后补近场航段未进入本轮训练
- 人在回路的闭环实现了记录与评判两环，尚未把复核记录用于训练下一版模型
- 界面效果没有做过用户实验
