# 冻结 FGAAE 的局部残差诊断与评分增强

本轮复用已完成训练的三个完整 FGAAE 模型，不修改检查点或原始基准成绩。入口为 `local_residual.py`，导出核验和方向压力检查入口为 `verify_local_residual.py`。

## 结论

模型并没有大量复制当前合成注入：在 −14～−8 dB 注入/背景功率比的各组中，沿注入方向的重构响应投影增益中位数约为 4%～6%，残差响应范数比约为 96%。这说明注入变化大部分仍存在于残差中。本轮先改评分，没有直接增加降噪训练或修改网络结构。

对连续频段中的正向残差求平均，可以显著提高当前功率抬升异常的检出；但该规则带有方向先验，不适用于泛化到所有异常类型。

| 三种子开发均值 | 原校准全局评分 | 正向局部增强 | 双向局部增强 |
| --- | --- | --- | --- |
| ROC-AUC | 0.911744 | 0.982408 | 0.924782 |
| F1 | 0.739680 | 0.915502 | 0.795939 |
| Precision | 0.890689 | 0.964850 | 0.920771 |
| Recall | 0.637600 | 0.871058 | 0.701917 |

正向版开发 AUC 样本标准差为 0.010092，F1 样本标准差为 0.027119。三个种子的 F1 分别为 0.936746、0.924805、0.884956，并非只展示最好一个种子。

## 尝试过程与规则

1. `local_residual_v1`：对逐频点标准化残差先平方，再进行局部均值或最大 5% 频点聚合。最优候选仅将开发 F1 提到 0.748737、召回率提到 0.647186，没有通过预先登记的采用门槛，保留原基准。
2. `local_residual_v2_signed`：先对带符号残差做连续频段平均，再扫描最大的正向偏移。选中 21 个频点的窗口，局部分数权重为 4，开发 F1 达到 0.915502。
3. `local_residual_v3_coherent`：对连续频段平均后再取绝对值，同时响应正负偏移。选中 21 个频点的窗口，局部分数权重为 1，开发 F1 达到 0.795939。

频点残差中心与尺度仅由 6620 帧特征训练参考拟合。全局与局部分数的中心和尺度也只用训练参考拟合。新分数始终保留原校准全局分数，因此原重构项和 Critic 项均参与评分。

窗口候选为 5、11、21、41 个频点，局部权重候选为 0.5、1、2、4。正向版另比较最大 5% 频点均值。三个种子共用同一窗口和权重，各自的正常尺度由相同训练参考成员拟合。

较早四个开发源文件用于选择平均 F1 最好的共同规则及每个模型的阈值；较晚四个文件用于开发检查。采用门槛要求后半文件 F1 至少提高 0.01、召回至少提高 0.03、精确率至少 0.80，并要求全开发集 F1 和召回也达到相同增幅。没有用已知异常位置进行预测。

## 较晚历史文件与弱注入表现

候选和阈值锁定后，才评估历史上已查看过的 5296 帧确认子集。

| 历史确认均值 | 原校准全局评分 | 正向局部增强 | 双向局部增强 |
| --- | --- | --- | --- |
| ROC-AUC | 0.902725 | 0.983508 | 0.919313 |
| F1 | 0.710912 | 0.916227 | 0.772148 |
| Recall | 0.604126 | 0.867182 | 0.669246 |

这说明正向版收益也出现在同站较晚文件，不能将其称为新的独立测试或跨站泛化证据。开发后半文件曾参与过旧检查点选择，本轮也属于自适应开发调参。

正向版在 −10 dB 注入组的开发召回率从 20.97% 提高到 81.65%，分组 AUC 从 0.822404 提高到 0.973952；−14 dB 组召回率从 4.08% 提高到 23.13%，AUC 从 0.614709 提高到 0.882378。最弱一档仍然存在明显漏检。

以上 dB 是注入/背景功率比，不是论文信道 SNR。

## 方向压力检查

评分规则和阈值冻结后，将历史确认样本的正向注入扰动按原电平单位镜像为负向：`负向样本 = 原始背景 - (原正向样本 - 原始背景)`。未改变背景和异常成员位置，没有据此重新调参。

| 镜像负向扰动均值 | 原校准全局评分 | 正向局部增强 | 双向局部增强 |
| --- | --- | --- | --- |
| ROC-AUC | 0.910019 | 0.808294 | 0.961930 |
| F1 | 0.689379 | 0.425143 | 0.849831 |
| Recall | 0.571889 | 0.290135 | 0.785945 |

这直接说明正向版的高分不等同于通用异常能力。双向版更能兼顾负向扰动，但原正向任务上的表现有所下降。该检查只是一种合成方向压力检查，不是物理 SNR 等价实验，也不是真实异常或新站点测试。

## 使用

```powershell
$env:PYTHONIOENCODING = 'utf-8'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
python local_residual.py smoke-test
```

已保存的正向版预测示例：

```powershell
python local_residual.py predict --artifact runs/local_residual_v2_signed/scorers/seed_2025_recommended.json --data runs/station_background_v2/inputs/development.npy --output local_positive_predictions.csv --device cuda
```

兼顾正负偏移的版本：

```powershell
python local_residual.py predict --artifact runs/local_residual_v3_coherent/scorers/seed_2025_recommended.json --data runs/station_background_v2/inputs/development.npy --output local_bidirectional_predictions.csv --device cuda
```

输入仍要求 840 个频点、87–108 MHz、与原检查点相同的电平单位和频率顺序。评分 JSON 保存原检查点路径及哈希、参考尺度和阈值；移动检查点后可用 `--checkpoint` 指定同哈希文件。旧 `fgaae.py predict` 仍只计算原全局评分，不能用它替代局部评分入口。

重跑实验时指定新输出目录：

```powershell
python local_residual.py run --mode signed --output runs/local_positive_repeat --device cuda
python local_residual.py run --mode coherent --output runs/local_bidirectional_repeat --device cuda
```

## 核验与结果

小样本自检覆盖局部聚合、常量频点尺度下限、后半文件标签不能影响候选选择、原全局分数复现、三种局部规则的参数保存与预测。

两个采用版本通过预测入口重新加载，共核对 95328 次开发及历史确认预测，分数最大差异为 0，全部判定一致。模型和评分参数哈希保持不变。

- [第一轮双侧能量试验](runs/local_residual_v1/RESULTS.md)：未通过采用门槛。
- [正向增强结果](runs/local_residual_v2_signed/RESULTS.md)、[PR 与弱注入曲线](runs/local_residual_v2_signed/comparison.png)。
- [双向增强结果](runs/local_residual_v3_coherent/RESULTS.md)。
- [导出核验](runs/local_residual_checks/deployment_verification.json)、[负向压力检查](runs/local_residual_checks/mirror_negative_summary.csv)。

这些结果应与原论文式评分基线分开保存和报告。当前改善来自学习模型残差上的评分增强，网络没有重训；仍需真实标注、新日期或新站点的数据验证更广泛的泛化性能。
