# 本轮消融执行记录与入口

以 [PLAN.md](PLAN.md) 为方案，以 `frozen/baseline_manifest.json` 为历史版本依据。完整模型沿用 AUC 0.911744 版本；新训练 Base/A/B/C 各三个种子，单次 60 epoch，只报告 ROC-AUC。

本轮输出目录：`runs/ablation_auc_09117_v1/`。源码、登记规则及环境保存在该目录，完成后生成 `RESULTS.md`、`auc_runs.csv`、`auc_summary.csv`、`paired_auc_differences.csv` 和 `roc_curves.png`。

**本轮已完成。**Base/A/B/C 各三个种子、共 720 epoch；Full 复用原模型。新模型全部 127104 次 CUDA 重载评分差异均为 0。结果与解释见 [完整报告](../runs/ablation_auc_09117_v1/RESULTS.md)，最终核验见 `runs/ablation_auc_09117_v1/final_verification.json`。

## 已通过的检查

- 小样本 Full 参数与归档旧源码逐张量一致。
- 四组消融的开启组件实际更新，教师及关闭组件不更新。
- 无 AQT 时全部训练帧保留；无 Critic 时评分严格等于重构误差。
- B/Base 的无标签 BN 校准不更新参数；阶段二冻结 BN 统计量。
- 用历史模型作为测试夹具，新的 AUC 系数校准流程重现 0.911744，Critic 关闭路径的导出与重载也通过；这些夹具不是新的消融结果。
- 在真实 20000 帧训练数据上重跑 Full 前两个 epoch，全部共有数值列与历史训练一致。

有效预检查为 `runs/ablation_auc_09117_precheck_v2/precheck_report.json`。v1 记录了较早入口的检查；v2 另补充了校准导出流程检查并修正无 Critic 时的候选名称。

## 本轮实际执行命令

在项目目录执行：

```powershell
python ablation_auc_09117/run_ablation.py precheck --output runs/ablation_auc_09117_precheck_v2
python ablation_auc_09117/run_ablation.py train --output runs/ablation_auc_09117_v1 --precheck runs/ablation_auc_09117_precheck_v2
```

上述目录已被本轮使用。复跑时应指定新输出目录；训练代码发生变化时，应先在新目录重新通过预检查。

中断后的继续命令：

```powershell
python ablation_auc_09117/run_ablation.py train --resume --output runs/ablation_auc_09117_v1 --precheck runs/ablation_auc_09117_precheck_v2
```

继续时会检查代码与基准，复用已完整完成的种子及其未校准模型。未完成的种子会将原目录保留为 `interrupted_seed_*`，从该种子起点重新训练；不声称恢复了中间 epoch 的优化器状态。所有四组都会执行，不按中途成绩挑组。

## 新模型预测

每个变体的最终 `model.pt` 保存归一化、组件开关、固定评分系数和模型权重。使用本轮入口输出连续评分，输入是原量纲的 N×840 频谱 `.npy`：

```powershell
python ablation_auc_09117/run_ablation.py predict --checkpoint runs/ablation_auc_09117_v1/A/seed_2025/model.pt --data runs/station_background_v2/inputs/development.npy --output runs/ablation_auc_09117_v1/A_reloaded_scores.csv
```

输出列为 `score`、`reconstruction_sum`、`critic`；无 Critic 的组最后一列为 0。这里不计算 F1/Recall，不另设最终二元判决阈值。每组完成后，入口会自动核对三个导出模型全部开发帧的重载评分。

图表整理命令只读取已经完成的数值表，不更换模型或重算选择规则：

```powershell
python ablation_auc_09117/plot_auc_report.py --run runs/ablation_auc_09117_v1
```

输出 ROC 及配对 AUC 差值的 PNG/PDF，并记录输入表哈希与图形处理方法。已完成的实验若需重新训练，请使用新的输出目录；原目录与冻结基准保留。

`source/historical_source_changes.diff` 展示相对归档旧源码的有限改动：组件开关和只计算 AUC 的评价部分。旧基准、旧训练源码及原文档均保持不变。
