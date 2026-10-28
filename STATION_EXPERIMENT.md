# 单站背景归一化实验 v2

目标：检验上轮跨站混合背景是否掩盖了模型效果；检验模型是否优于不需要训练的能量和背景统计，而不是通过改变异常强度追逐 AUC。

入口：`station_experiment.py`。默认输出：`runs/station_background_v2/`。

## 数据与隔离

- 只使用三合海关站点，原始数据仍来自 `datasets/fm_87_108_v1/`。
- 直接复用 `runs/grouped_pipeline_v1/benchmark/` 中已经冻结的训练/验证数值和标签，不重新生成异常，不改变异常比例、位置、宽度或功率比。
- 使用原训练来源中最早的一个完整文件作为独立背景参考，共 1324 帧：前 993 帧拟合背景和专家特征标准化，后 331 帧用于内部特征验证。此文件从阶段二训练池排除。
- 阶段二候选池为 69461 帧。默认固定其前 20000 帧，三种方法、三个模型种子共享相同的数值和来源行。
- 原验证池中三合海关有 12 个文件；按时间排序，前 8 个文件的 10592 帧为开发集，后 4 个文件的 5296 帧为确认子集。文件不交叉。
- 所有归一化参数只从背景参考的前 993 帧拟合。开发/确认数据不参与背景拟合。
- 本轮不读取或计算旧 test/external_test 的检测结果。

**确认子集不是全新的独立测试集。**这些文件历史上已经属于 v1 的验证池；重新切出一个确认子集可以约束本轮模型选择，但不能抹去历史上的数据使用。最终论文性能结论仍需要尚未参与开发的新文件确认。

背景参考也没有人工“正常”真值；它是未加入本协议合成干扰的实测背景，不代表已经确认不存在真实异常。

## 两种输入变换

先把原始 int16 电平乘以 0.1 转为既有单位假设下的 dB。

1. 全局标准化：用参考训练子集的一个总体均值/标准差变换所有频点，作为对照。
2. 逐频点背景归一化：各频点减去参考中位数，再除以 `max(IQR/1.3489795, 1 dB)`。尺度下限预先固定，避免低方差频点把数值放大到无穷；不根据验证结果修改。

变换的中心、尺度、拟合来源和单位随模型一起保存。归一化在推理时固定，不用待测文件重新估计。

## 五个无训练基线

- `mean_level`：平均电平。
- `total_power_db`：线性功率求和后转换回 dB。
- `background_mean_z`：背景归一化后各频点均值。
- `background_zero_mse`：背景归一化后的均方值，等价于始终预测背景中心的零预测器；这是判断网络重构是否真有额外价值的重要对照。
- `background_top5_positive_z`：正向背景偏离最大的 5% 频点的均值。

这些基线均只使用训练背景参数；检测阈值统一在开发集以 F1 选取。确认子集只应用固定阈值，不重新标定。

## 三个学习模型

- `global_zscore_ae`：全局标准化的编码器/解码器。
- `background_ae`：逐频点背景归一化的编码器/解码器。
- `background_featureguided`：相同背景输入，加 25 维专家特征回归预训练及教师/学生蒸馏。

共享 `fgaae.py` 中的三层编码器、512 维潜空间及对称解码器。这里没有训练 GAN，也没有使用 AQT；Critic 等未使用模块虽然随共享模型结构实例化，但不加入优化器或评分。

默认种子为 2025、2026、2027；同一种子下网络初始参数相同，阶段二开始前重置随机数流，使批次顺序和 Dropout 随机流可比。特征引导版本额外进行 100 epoch 预训练，因此不能把它与纯 AE 的比较当成计算预算完全匹配的证明。

阶段二默认上限 40 epoch，AdamW，lr=2e-5，weight_decay=1e-3，batch=256。按开发集 AUC 保存最佳快照；最少 10 epoch，6 epoch 无超过 1e-4 的改善后早停。

## 预先固定的扩容规则

先完成九个学习模型和五个基线的开发结果，锁定配置，再访问确认指标。

只有最优学习方法的三种子平均开发 AUC 超过最优无训练基线 **0.005**，才将该学习方法扩展到全部 69461 帧，仍运行相同三个种子。0.005 是预先设定的实用门槛，不是显著性检验。

若未通过，就停止无收益扩容，不因为某个确认指标好看而反过来改变这个决定。也不自动加入 Critic/AQT，或对旧测试集继续调参。

多种子均值/标准差只反映这几个初始化下的结果差异，不是独立采样数据的置信区间。

## 命令

自检只使用临时合成夹具，不使用正式数据：

```powershell
python "代码/FGAAE_论文复现/station_experiment.py" smoke-test
```

完整单站流程，输出目录必须不存在：

```powershell
python "代码/FGAAE_论文复现/station_experiment.py" run --device cuda --output "代码/FGAAE_论文复现/runs/station_background_new"
```

可用 `--seeds`、`--train-size` 等设置新实验，但应在查看新结果之前固定这些参数，不根据确认分数反复追分。

推理必须使用本脚本，因为变换包含逐频点向量，而不是旧入口的标量归一化：

```powershell
python "代码/FGAAE_论文复现/station_experiment.py" predict --checkpoint "代码/FGAAE_论文复现/runs/station_background_v2/background_ae/seed_2025/model.pt" --data "D:\数据\same_station_spectra.npy" --output "代码/FGAAE_论文复现/runs/station_predictions.csv"
```

模型的背景和阈值针对已校准的站点。不要不经独立校准就套到新站点。

## 产物

- `registered_plan.json`、`environment.json`：运行前固定的计划、环境和脚本哈希。
- `inputs/manifest.json`：来源文件、背景变换、数据大小与输入哈希。
- `inputs/`：本轮使用的参考、训练、开发、确认数据及来源元数据。
- `baseline_development_scores.csv`、`baseline_thresholds.json`：无训练基线开发分数与固定阈值。
- 各模型/种子目录：`history.csv`、`development_scores.csv`、`model.pt`、`report.json`、锁定模型后生成的 `confirmation_scores.csv`。
- `development_comparison.csv`、`confirmation_comparison.csv`：均值/标准差比较。
- `locked_development_selection.json`：确认阶段之前锁定的基线、学习方法和扩容决定。
- 若通过门槛：`selected_full/` 和 `full_scale_confirmation.csv`。
- `station_comparison.png`、`final_report.json`：图和机器可读总报告。

本轮仍是原有频谱域功率注入代理实验，不是具有真实异常标签的论文原始实验。即使某个分数超过 0.91，也要区分：无训练背景统计、学习模型、开发结果、确认结果以及真正独立测试结果。
