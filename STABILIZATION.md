# 完整 FGAAE 的性能提升试验

基线为 `runs/full_fgaae_v4`：三个种子的最佳开发检查点 AUC 均值 0.877556，样本标准差 0.040129。训练末期教师特征误差明显漂移。本轮保持完整模型，按优先级检查训练稳定性、损失平衡和评分系数，所有结果仍属于合成异常开发调参。

本轮已完成，见 [结果报告](runs/full_fgaae_improvement_round1/RESULTS.md)。训练调整后为 0.891949 ± 0.023722；评分校准后为 **0.911744 ± 0.035992**。校准提高了均值，但其种子波动大于未校准的训练调整版，且种子 2027 没有获得校准收益。不能将均值达到约 0.91 理解为每个种子都稳定达到该数值。

## 第一、二项：编码器与 AQT

入口：`stabilize_full_model.py`，默认输出 `runs/full_fgaae_v5_stabilized`。

- 编码器学习率为 2e-6，解码器和学生头仍为 2e-5，Critic 仍为 2e-5。
- 固定编码器 BatchNorm 的运行均值、方差和计数；仿射参数继续训练，Dropout 仍按训练模式工作。
- AQT 从 95% 逐步保留到约 85%，保留 3 次筛选、每次 20 轮联合训练和每批 5 次 Critic 更新。
- 主评分仍为平方误差求和 + 0.1×Critic(重构)，损失权重暂与 v4 相同。
- 训练仍为相同 20000 帧，开发集仍为相同 10592 帧，种子仍为 2025、2026、2027。
- 复用每个种子已完成的 100 轮预训练检查点，只载入编码器和教师头。解码器、学生和 Critic 按原种子初始化。参考成员、特征标准化、预训练配置和训练成员均校验，预训练来源有哈希记录。

每轮记录特征参考留出集上的教师 MSE，以及首个训练批次三项加权损失对编码器的梯度范数与方向。这些仅用于诊断，不参与筛选训练样本；监测过程恢复随机状态，避免改变训练随机序列。

针对性自检覆盖预训练复用、实际编码器学习率、最佳和末轮 BatchNorm 统计量冻结、教师参数冻结以及所有训练组件更新。新控制默认关闭，旧入口的小样本结果仍保持一致。

## 第三项：有限的损失权重搜索

入口：`improve_full_model.py`，在 v5 训练和汇总完成后运行，默认输出 `runs/full_fgaae_improvement_round1`。

若 v5 三种子的开发 AUC 均值仍低于 0.90，固定中间位置的种子 2026，试验以下联合损失权重：

| 对抗权重 | 蒸馏权重 | 重构权重 |
| --- | --- | --- |
| 0.003 | 0.1 | 1 |
| 0.01 | 0.1 | 1 |
| 0.03 | 0.1 | 1 |

各试验运行 12 轮、1 次 AQT，保留完整组件。只有最优短程方案比 v5 对应种子的完整训练最佳 AUC 高至少 0.002，才用该方案执行三个种子、每种子 60 轮的完整验证。这是控制计算预算的筛选门槛，不能保证找到全局最优超参数；单种子试验不会被当成稳定提升。

三个权重来自已观察到的蒸馏梯度较强这一诊断，是自适应开发调参。完整模型组件均保持，当前不开展组件消融。最终在具有同一数据和三个种子的完整训练方案之间选择，失败的候选也保留。

## 第四项：Critic 评分系数

`calibrate_full_model.py` 保持公式正号和完整模型，比较原始系数 0.1，以及相对于训练参考分数标准差的 0.01、0.03、0.1、0.3、1.0 五个固定候选。尺度只由 6620 帧特征训练参考样本估计。

按时间将八个开发源文件划为较早四个和较晚四个。较早文件用于选择所有种子共享的相对系数及各自 F1 阈值；较晚文件用于系数效果检查。较晚文件以前已参与模型检查点选择，不能称为独立或未查看过的测试集。只有较晚文件和全开发集平均 AUC 都提高时，流程才接受校准结果。

校准模型和全部候选成绩单独保存于选中训练目录的 `score_calibration/`。主实验的公式系数、检查点和报告保持原有定义。校准 F1 的阈值只来自较早文件，与原始主报告使用全部开发标签选阈值的协议不同，须结合这一差异解读。

## 执行

在本目录使用已有依赖与 CUDA 环境：

```powershell
$env:PYTHONIOENCODING = 'utf-8'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
python stabilize_full_model.py smoke-test
python stabilize_full_model.py train --device cuda
python summarize_full_model.py --run runs/full_fgaae_v5_stabilized
python improve_full_model.py
```

这些正式输出目录必须不存在；重跑应指定新的输出路径。单独执行评分校准前可先运行 `python calibrate_full_model.py --smoke-test`，该测试检查较晚文件的标签不会影响系数选择。

已选中的联合损失权重为重构 1、对抗 0.03、蒸馏 0.1。若只想复跑选中配置，可跳过搜索：

```powershell
python stabilize_full_model.py train --device cuda --adversarial-weight 0.03 --distillation-weight 0.1 --output runs/full_fgaae_selected_repeat
python summarize_full_model.py --run runs/full_fgaae_selected_repeat
python calibrate_full_model.py --run runs/full_fgaae_selected_repeat --device cuda
```

已导出的校准模型位于 `runs/full_fgaae_improvement_round1/full_loss_verification/score_calibration/seed_2025_model.pt`，另有 2026、2027 两个模型。共同相对系数为 1，实际系数由各模型在训练参考集上的分数标准差比计算；使用已保存模型即可保留其标准化参数、系数及阈值。

```powershell
python fgaae.py predict --checkpoint runs/full_fgaae_improvement_round1/full_loss_verification/score_calibration/seed_2025_model.pt --data runs/station_background_v2/inputs/development.npy --output selected_model_predictions.csv --device cuda --batch-size 256
```

结果包括三种子完整训练对比、有限权重试验、正号评分校准、弱注入分组曲线和最终模型位置。原始训练源码按实验保存于 `source/`，历史 v4 的源码也已按登记哈希归档，后续代码扩展不会破坏历史来源核验。

本轮未评估确认集、测试集或外站集，也未将这些集合用于训练或调参。即使开发 AUC 接近 0.91，原始 25 项特征定义和异常生成/标注协议仍不完全一致，仍不能宣称论文原始实验性能已严格复现。
