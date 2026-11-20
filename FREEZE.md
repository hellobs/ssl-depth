# 技术数据冻结记录

冻结时间：2026-11-20 16:05
冻结范围：全部模型训练、数据采集、结果评测。此后不再新增实验，论文只做文字与排版修改。

采集输出目录已于 2026-11-20 归拢到 `D:/zzr/wmdata/near_out*`，与 `runs_*` 系列并列；脚本中的硬编码路径同步更新。

## 一、论文引用的最终数字与来源

| 论文位置 | 数字 | 唯一来源 |
|---|---|---|
| tab:backbone 前四行 | 0.829/0.846/0.208 等 | `runs/v5_{imagenet,resnet_scratch,jepa_domain,scratch}/metrics.json` |
| tab:backbone JEPA 153 行 | 0.813 / 0.812 / 0.263 | `runs/v5_jepa_domain/metrics.json` |
| tab:backbone JEPA 240 行 | 0.817 / 0.831 / 0.189 | `runs/v5_jepa_domain_v6/metrics.json`（见注 1） |
| tab:backbone DAv2 行 | 0.862 / 0.886 / 0.198 | `runs/sota_ft_dav2s_{val,test,task_test}.json` |
| tab:scale 四行 | −0.5 / −10.0 / −12.7 / −0.7 % | `runs/v5_*/task_metric_held.json` 的 pr/gt_median |
| tab:distance 四行 | 408/1157/1055/706 | `runs/v5_imagenet/error_analysis.json` 的 by_bin |
| tab:bias 六行 | 237/0.71/78%/11% 等 | 重跑 `scripts/diag_bias.py --run runs/v5_imagenet` 可复现 |
| tab:task 五行 | 375/105/11 / 78.1% 等 | `runs/v5_imagenet/task_metric_held.json` 的 by_threshold |
| 延迟数据 | 5.76 / 22.67 ms | 实测，脚本见本文件注 2 |
| 动作分布 | 贴障 7.4%/0.1% 对空旷 22.8%/13.8% | 由 near_out3/4 与 b11 三批的 transitions.jsonl 统计 |

注 1：v6 的下游训练被中断过一次。重跑完成后脚本把 metrics.json 写进了 `runs/runs/v5_jepa_domain_v6/` 这个嵌套路径（路径 bug），已归位到 `runs/v5_jepa_domain_v6/metrics.json`。同目录下的 `metrics_from_log.json` 是当时从训练日志提取的交叉验证记录，与 metrics.json 的 best_val 0.817、test 0.831、AbsRel 0.189 完全一致。
注 2：延迟测量为 JepaBackbone + DepthHead 各 200 次前向的均值，输入 1×1×3×108×192。

## 二、论文引用的权重

| 用途 | 路径 |
|---|---|
| MicroJEPA 240 航段（论文 v6 行） | `runs/jepa_domain_v6/jepa.pt` |
| MicroJEPA 153 航段（论文 v5 行） | `runs/jepa_domain_v5/jepa.pt` |
| DAv2-S 整模型微调 | `runs/sota_ft_dav2s_best.pt` |

## 三、数据划分（不得再改）

- `runs/v5_split.json`：训练 152 航段 / 验证 14 / 测试 15，按航段划分
- `cache/kept_all_v5.json`：21129 帧索引，与上面的划分一一对应
- `runs/jepa_v6_runs.txt`：240 航段白名单（153 旧 + 87 新，与留出集零重叠）

## 四、不再进入论文的结果

| 内容 | 数值 | 原因 |
|---|---|---|
| LoRA 适配（人机复核数据） | 0.832 / 0.205 / 69.1% | 复核样本量不足，按作者决定撤出正文，仅保留机制描述 |
| 近场补采 17 局 | 见 near_out3/near_out4 | 只用于 v6 预训练语料，不作独立结果 |
| 三次中断的采集尝试 | — | 数据为空目录，已清理 |

## 五、复现方式

论文的所有表格与图都由脚本从上述 JSON 生成，命令记录在 `HANDOFF.md` 与 `paper/figs/dsl/*.json`、`paper/figs/plot/*.json` 的 spec 里。图的唯一真源是 spec，改 spec 重跑渲染器即可重出，不需要重新训练。

冻结之后如需恢复某次训练，权重仍在，但不要覆盖上表中的 JSON 与 spec。
