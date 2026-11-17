# ssl-depth

低空物流配送无人机的视觉测距误差研究。课程论文项目，配套代码与数据索引。

论文：《基于人工智能的无人机单目深度测距误差研究与设计学分析》

作者：张子睿（华南理工大学设计学院）、盛祺（自动化科学与工程学院）、刘子瑄（电子商务系）、马婧怡（材料科学与工程学院）

## 这个项目在做什么

配送无人机的视觉分支需要从单张图像估计前方障碍的距离。这个估计会出错，而错误不是均匀分布的。项目做三件事：

1. 在仿真城市环境中采集飞行数据，训练单目深度估计模型
2. 固定解码器与训练预算，比较四种视觉表征，并对误差作分层分析
3. 从设计学角度分析现有地面站为什么无法传达这种不确定性，提出人在回路的改进机制

## 目录

```
scripts/
  rebuild_index.py     扫描数据目录，按规则生成帧索引
  diag_backbone.py     四种表征的对照训练与评测
  analyze_error.py     误差分层分析（距离、架次、失效帧）
  task_metric.py       避障判据换算与混淆矩阵
  plot_figs.py         论文配图
  make_stimuli.py      界面素材生成
paper/
  main.tex             论文源文件（XeLaTeX + ctexart）
  main.pdf             编译产物
  figs/                配图
cache/
  kept_all.json        帧索引（42 个架次、4581 帧）
runs/
  v3_<骨干>/           训练产物与指标
```

## 数据

原始数据不在本仓库，位于本地 `D:/zzr/wmdata/runs` 与 `D:/zzr/wmdata/runs_batch2`。

每帧包含前视 RGB、稠密深度图、深度有效掩码。深度由 Unreal Engine 4 的 AirSim 插件直接渲染输出，单位为米，与图像严格对齐。

清洗规则：有效范围 0.5 至 80 米，帧有效率不低于 50%。原始 47 个架次中保留 42 个、4581 帧。

## 运行

环境：`D:/zzr/microwm/.venv`（Python 3.13 + PyTorch 2.9.1 + CUDA 12.8）

```bash
# 重建索引
python scripts/rebuild_index.py

# 训练并评测某一组骨干
python scripts/diag_backbone.py --backbone imagenet --epochs 15

# 误差分层分析
python scripts/analyze_error.py --backbone imagenet

# 避障判据换算
python scripts/task_metric.py
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

四组表征的对照，验证集：

| 骨干 | 预训练数据 | δ1 | 绝对相对误差 |
|---|---|---|---|
| ImageNet ResNet18 | ImageNet 有监督 | 0.895 | 0.103 |
| ResNet18 随机初始化 | 无 | 0.852 | 0.135 |
| ViT 随机初始化 | 无 | 0.676 | 0.227 |
| JEPA 自监督 | 航拍图像 | 0.652 | 0.367 |

注：上表为 2512 帧数据集的结果。扩充到 4581 帧后正在重跑，数字会更新。

## 已知限制

- 数据全部来自仿真，缺少真实相机的噪声与光照突变
- 测试集规模有限，改造前只有 3 个架次
- 人在回路的闭环只提出机制，未实现
