# B 低于 Base 的补充诊断

这是原五组消融完成后，针对 B 的 AUC 低于 Base 追加的诊断实验。原表和原模型保留；暂时只新增两个组，不自动重跑或替换 Base/B。评价仍只使用 ROC-AUC。

**本轮已完成。**六次新增训练共 360 epoch；结果与解释见 [诊断报告](../runs/ablation_auc_09117_diagnostic_v1/RESULTS.md)。AE＋AQT 的 AUC 为 0.856068 ± 0.047525，AE＋WGAN-GP 为 0.843579 ± 0.053575。当前无特征引导固定配方下，对抗分支出现负收益，AQT 有小幅正收益。已检查路径可复现，未发现需要立即修正的实现不一致，原 Base/B 正式结果未重跑或替换。

## 四组对照

四组均无专家特征预训练、无教师/学生蒸馏；使用相同训练参考背景进行一次不更新参数的 BN 统计校准，然后冻结编码器 BN。

| 组别 | WGAN-GP | AQT | 处理 |
| --- | --- | --- | --- |
| Base | 无 | 无 | 复用原三个种子结果，AUC 0.854330 ± 0.049008 |
| GAN_only | 有 | 无 | 已完成，AUC 0.843579 ± 0.053575 |
| AQT_only | 无 | 有 | 已完成，AUC 0.856068 ± 0.047525 |
| B | 有 | 有 | 复用原三个种子结果，AUC 0.846050 ± 0.051586 |

同一 20000 帧训练池、10592 帧开发集，种子 2025/2026/2027。编码器 LR 2e-6、解码器和 Critic LR 2e-5、对抗权重 0.03、GP 权重 10、每批 Critic 更新 5 次等配置均不额外调整。新增两组使用原训练入口的相同函数，旧源码文件保持不变，仅在新进程内登记两个新组合。

## 先检查什么

1. 旧 Base/B 的模型、数值表和训练源码校验值仍与完成时一致。
2. 新 GAN_only 与旧 B 的全部初始张量一致，新 AQT_only 与旧 Base 的全部初始张量一致，配置仅在 AQT 开关和输出目录上不同。
3. 在真实数据上，旧 Base/B 种子 2026 的首轮训练精确重放。
4. 两个新组的小样本训练检查组件更新、BN 冻结、AQT 开关、无 Critic 时的纯重构评分和 AUC-only 训练历史。

## 如何分析

按每个相同种子分别计算下列 AUC 差值，再报告三种子的均值、样本标准差及方向：

- 无 AQT 时，对抗模块效应：`GAN_only − Base`。
- 有 AQT 时，对抗模块效应：`B − AQT_only`。
- 无对抗时，AQT 效应：`AQT_only − Base`。
- 有对抗时，AQT 效应：`B − GAN_only`。
- AUC 差值尺度的交互项：`B − GAN_only − AQT_only + Base`。

主 AUC 与统一重构 AUC 都计算，使用相同已选检查点，不额外挑模型。对抗组的主 AUC 同时包含训练与 Critic 评分影响，因此这些差值不是单一损失项的纯因果效应。交互项只是 AUC 尺度的描述量，不能直接解释成已经证实网络内部机制。

有对抗的组沿用历史六候选正系数校准流程，三个种子共享候选；无对抗组只用重构平方误差和。训练 AQT 不使用后处理选中的系数。所有八个开发文件历史上已用于模型开发，其中较晚文件也参与是否采用校准的判定；本轮不作为新的独立测试。

另外保存 AQT_only/B 的训练池人工注入比例、移除的未注入背景数量及保留集合重合度。这些是训练数据诊断，不是测试集 Recall。相同 60 epoch 下，有 AQT 组 G 更新 4200 次，无 AQT 组 4740 次；过滤质量与更新预算不能由当前对照完全分开。

## 是否重跑 Base/B

先根据两个新增组定位固定配置下的条件效应。如果没有发现实现不一致，仅观察到单模块或组合的负收益，就保留结果，不以 B 必须高于 Base 为验收条件。

若之后有具体证据需要修改无特征引导的训练配方，则 Base/B 使用同一修正规则重跑，原固定配置结果仍保留并分开报告。影响通用实现的问题也要检查相应其他组，不能只更新 B。此次不自动追加调参。

## 本轮入口

在项目目录执行，本轮使用的目录复跑时需更换：

```powershell
python ablation_auc_09117/diagnose_no_features.py precheck --output runs/ablation_auc_09117_diagnostic_precheck_v1
python ablation_auc_09117/diagnose_no_features.py train --output runs/ablation_auc_09117_diagnostic_v1 --precheck runs/ablation_auc_09117_diagnostic_precheck_v1
```

输出目录保存登记规则、源码、旧结果引用哈希、每个新模型及连续评分。完成后生成 `RESULTS.md`、四组 `auc_summary.csv`、`conditional_effects_by_seed.csv`、`conditional_effects_summary.csv`、训练池诊断和配对效应图。

原 Base/B 的真实首轮训练重放一致；新增组与对应旧组的全部初始张量一致。63552 次 CUDA 模型重载评分差异为 0。独立核验及额外条件分析入口如下，只读取已完成结果，不重新训练：

```powershell
python ablation_auc_09117/analyse_diagnostic.py --run runs/ablation_auc_09117_diagnostic_v1
```

扩展分析固定 AQT 开启，组合原 A/Full/B 与新 AQT_only，发现对抗模块主 AUC 的条件效应在有、无完整特征引导时方向相反。这属于结果出来后的补充分析，不作为新的独立验证或网络内部机制证明。

中断时可对上述训练命令添加 `--resume`，复用已完整完成的种子；未完成种子的原目录保留，从该种子起点重新训练，不声称恢复中间 epoch 的优化器状态。
