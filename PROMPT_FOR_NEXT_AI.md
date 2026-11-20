# 交接用 Prompt（复制下面整段给下一个 AI）

---

你接手一个已经做到后期的课程论文项目，工作目录 `D:/zzr/ssl-depth`。

**第一步：完整读 `D:/zzr/ssl-depth/HANDOFF.md`，不要跳过。** 里面写了环境路径、数据划分陷阱、权重清单、
已完成的全部数字、踩过的八个坑。不读就动手一定会重复已经犯过的错。

## 现在的断点

论文 `paper/main.tex` 已写完并编译通过（25 页）。但**里面关于「域内自监督」那一臂的四个数字是脏的**——
来自一版数据泄漏的预训练权重。我正在重跑深度头，跑到 `ep 1/15` 时被叫停，没有写出权重。

干净权重已经就绪：`runs/jepa_domain_v4/jepa.pt`（零泄漏，52 个纯训练 run / 10,014 帧对 / 4000 步，
加载自检 231/231 张量命中）。`diag_backbone.py` 的 `CKPTS["jepa_domain"]` 已指向它。

## 你现在要做的三件事（按顺序）

1. **重跑深度头**（CPU，约 1 小时 43 分；如果 GPU 空着且用户同意，12 分钟）：

```bash
cd D:/zzr/ssl-depth
OMP_NUM_THREADS=12 D:/zzr/microwm/.venv/Scripts/python.exe -u scripts/diag_backbone.py \
  --backbone jepa_domain --epochs 15 --workers 3 --device cpu
```

2. **重算任务指标**：

```bash
D:/zzr/microwm/.venv/Scripts/python.exe -u scripts/task_metric.py --split held
```

3. **按新数改 `paper/main.tex` 四处**（中英摘要、`tab:backbone`、`tab:scale`、6.x 决策段），
   然后再改 `README.md` 的结果表，最后在 `paper/` 下用 xelatex 编译两遍，确认 0 个 Overfull。

做完后向用户报告：**旧数 vs 新数并列**，说明泄漏抬了多少，以及「语料贡献大于架构」这条结论是否还成立。

## 硬约束（违反任何一条都会造成返工）

- **Python 只有一个能用的**：`D:/zzr/microwm/.venv/Scripts/python.exe`。其它 conda 环境要么没 cv2、要么没 torch。
- **绝对不要修改 `D:/zzr/tools/wm-collector-m/`**（采集器，另一个项目在用）。需要改就复制到工作区再改。
- **不要把 GPU 当自己的**。先 `nvidia-smi --query-compute-apps` 看有没有别人的进程；
  注意常驻的 3.2–3.4 GB 是 AirSim（`CityEnviron.exe`），不是训练。用不用 GPU 要先问。
- **数据划分只有一个权威来源**：`runs/v4_imagenet/metrics.json` 的 `split`（98 航段）。
  不要用 `cache/kept_all.json`（104 航段），也不要用 `split()` 现算——两套不一样，我已经因为这个返工两次。
- **预训练开跑前必须过泄漏断言**。`pretrain_jepa.py` 会打印两道自检，
  只要出现「冲突航段」就是拒绝开跑，**不许调参数绕过**，要修语料。
- **数字掉下来就改论文，不改语料。** 不许为了让表格好看去动划分或换数据。

## 几条判断，供你参考（不要推翻，除非有新证据）

- JEPA 这条线在 δ1 上打不过 ImageNet ResNet18（0.866 vs 0.829 左右），但**近场召回明显更好**
  （9 米判据 87% vs 67%），代价是误刹从 10 次涨到 24 次。两个指标指向相反的失效模式，
  论文的论点是「不能只用单一指标排序模型」，这个论点不依赖具体数字，重跑后仍然成立。
- 数据真正缺的是 **0–7 米的正面逼近样本**，不是总量。留出集里 <6 m 只有 237 帧，
  而 `near_out2` 那 10 趟近场飞行实际全在 7–12 米。再采巡航里程没用。
- 复核数据已换正式版（2026-11-19 导出）：157 次提交 / 106 个去重样本 / 理由 67 条。
  旧的 46 条试填及「4% 填写率」结论已从论文与 README 删除，不要 resurrect。
  论文口径不区分提交者，不讨论评审人数分布。

## 沟通方式

用户是这篇论文的作者，技术底子好，讨厌空话和奉承。报告时直接给数字和判断，
不确定就说不确定。做了什么、改了什么文件、哪一处数字变了，都要说清楚。
长任务不要反复问，但涉及删数据、动别人的项目、抢 GPU，必须先确认。
