# FGAAE 论文模型复现

对应论文：Research on Anomaly Signal Detection for Radio Monitoring Based on Deep Learning。

本目录独立于原项目；不导入、不修改原有训练实现。`fgaae.py` 集成数据处理、25 维专家特征、模型、两阶段训练、WGAN-GP、AQT、评估和保存/加载。

**完整 FGAAE 复现入口：**使用 [full_model.py](full_model.py)，操作说明与论文对齐边界见 [FULL_MODEL.md](FULL_MODEL.md)。该入口训练特征预训练、教师/学生蒸馏、WGAN-GP 和 AQT 的完整模型；模块消融使用下述独立入口。保持文件分组隔离，只在既有开发集评估，不把统计基线或普通自编码器结果当作 FGAAE 成绩。

**AUC 消融已完成，Full 基准锁定 0.9117 版：**见 [五组结果报告](runs/ablation_auc_09117_v1/RESULTS.md)、[方案](ablation_auc_09117/PLAN.md) 和 [执行说明](ablation_auc_09117/EXECUTION.md)。三个种子平均主 AUC：Base **0.854330**、A（无对抗）**0.873912**、B（无特征引导）**0.846050**、C（无 AQT）**0.902456**、Full **0.911744**。Full 对 A/B 三个种子均更高；AQT 平均收益约 0.93 个百分点，但一个种子反向，且 Full 最佳模型均在第二次 AQT 前，不据此宣称多轮动态过滤已验证。12 次新训练共 720 epoch、127104 次 CUDA 模型重载评分差异均为 0，历史 Full 保持不变。只报告 ROC-AUC，未使用局部残差增强。

**B 与 Base 的补充诊断已完成：**见 [诊断结果](runs/ablation_auc_09117_diagnostic_v1/RESULTS.md) 和 [设计与入口](ablation_auc_09117/DIAGNOSTIC_PLAN.md)。复用原 Base/B，新增 AE＋WGAN-GP 的 AUC 为 **0.843579**，AE＋AQT 为 **0.856068**（各三个种子、60 epoch）。无特征引导的固定配方下，从 Base 单独加入对抗模块后 AUC 平均降低约 1.08 个百分点，AQT 小幅改善，不能完全抵消该负项。旧 Base/B 首轮重放及新增组初始状态检查一致，63552 次 CUDA 重载评分差异为 0；本轮未重跑或替换原正式结果。固定 AQT 开启的扩展分析中，GAN 在有完整特征引导时平均提高 3.78 个百分点，无特征引导时降低 1.00 个百分点，结论限于此配方下的条件效应。

**复现改进版本（非本次消融基准）：**见 [结果报告](runs/original_fidelity_round1/RESULTS.md) 和 [执行说明](ORIGINAL_FIDELITY.md)。保持原模型、损失项和正号 Critic 评分形式，统一 D/G 假样本的前向状态，按 F1 选择检查点与正评分系数。两个 20 轮短程组及三个种子各 60 轮完整训练已完成：开发集 F1 **0.784422**、Recall **0.691713**、AUC **0.932156 ± 0.022427**；相对上一轮原模型，F1/Recall 提升约 **4.47/5.41 个百分点**。31776 次 CUDA 预测重载分数差异为 0。局部评分增强版未作为该轮基准；F1、召回率与论文读图约 0.845/0.835 仍有差距。

**上一轮性能提升试验：**见 [完整对比报告](runs/full_fgaae_improvement_round1/RESULTS.md) 和 [执行说明](STABILIZATION.md)。相同 20000 训练帧、10592 开发帧及三个种子下，训练调整后的开发 AUC 为 **0.891949 ± 0.023722**；进一步校准正号 Critic 评分系数后为 **0.911744 ± 0.035992**，F1 均值 **0.739680**。三个导出模型的全部 31776 次 CUDA 预测重载核验通过。该轮 AUC 数值接近论文约 0.91，但仍是合成开发集调参结果，不能视为原始实验的严格复现。

**与论文逐项对照：**论文图 10 标注 AUC 0.915，图 9 的 F1/召回率读图约为 0.845/0.835。本轮 F1/召回率为 0.784422/0.691713，仍分别相差约 6.06/14.33 个百分点；异常协议与评估集合不同，不能据 AUC 判断整体能力已对齐。论文原图和上一轮分析见 [性能对照分析](runs/full_fgaae_improvement_round1/PAPER_COMPARISON.md)，最新差距见 [本轮报告](runs/original_fidelity_round1/RESULTS.md)。

**局部残差增强已完成：**见 [LOCAL_RESIDUAL.md](LOCAL_RESIDUAL.md)。冻结原 FGAAE 模型后，正向局部评分在同一合成开发协议上达到 AUC **0.982408**、F1 **0.915502**、Recall **0.871058**；历史较晚文件 F1 为 **0.916227**。这是独立于原论文评分的增强方案，针对局部功率增加；镜像负向扰动的 F1 只有 **0.425143**。双向版本在正向开发集/负向压力检查上的 F1 分别为 **0.795939/0.849831**。原基准成绩保留，未宣称跨站泛化或严格复现原始实验。

**完整模型 v4 已完成：**见 [三种子结果报告](runs/full_fgaae_v4/RESULTS.md)。每个种子均完成 100 轮特征预训练和 60 轮联合训练，阶段二使用 20000 帧、开发集 10592 帧。最佳检查点的开发 AUC 均值为 **0.877556 ± 0.040129**，F1 均值为 **0.674249**；完整组件与保存/加载核验通过。共享编码器更新后，教师特征误差明显漂移，训练后期检测性能回落；当前尚未稳定复现论文性能。


**历史数据与分组流程：**十二万帧重建、文件分组划分及旧实验见 [PIPELINE.md](PIPELINE.md)。不要对整个 `spectra.npy` 直接运行旧的随机逐帧划分入口。

**已完成的受控实验：**见 [grouped_pipeline_v1 结果报告](runs/grouped_pipeline_v1/RESULTS.md)。120186 帧重建、四组对照及全量训练已完成；当前协议下全量模型测试 AUC 为 0.771591，外站 AUC 为 0.651097，尚未复现论文性能。合成协议和旧实验不同，不能将两次 AUC 当成同一任务的提升。

**单站背景实验 v2：**见 [方案](STATION_EXPERIMENT.md) 和 [结果](runs/station_background_v2/RESULTS.md)。五个无训练基线与三种模型、三个种子已比较；局部背景统计在确认子集 AUC 为 0.967042，而最优学习模型的三种子均值为 0.845767。确认子集不是新独立测试，高分基线不是 FGAAE 复现成功；该轮学习模型未通过扩容门槛，当时未继续加 GAN/AQT 或全量训练。

这是按论文重建的可运行实验实现，不是原作者最终实验代码，也不保证达到论文报告的 AUC 0.91。

## 1. 运行

建议在独立虚拟环境中安装依赖，PyTorch 的 CPU/CUDA 版本按本机环境选择。

```powershell
cd "D:\荣珍代码与数据\代码\FGAAE_论文复现"
python -m pip install -r requirements.txt
python fgaae.py --help
python fgaae.py train --help
```

首先做小样本 CPU 自检，不运行完整实验，临时文件自动清理：

```powershell
python fgaae.py smoke-test
```

使用现有 `all_md_files_cutted` 做短流程检查：

```powershell
python fgaae.py train --device cpu --num-threads 2 --stage1-epochs 2 --iterations 2 --epochs-per-iteration 1 --critic-steps 1
```

按默认轮数训练（100 个阶段一 epoch；阶段二 10 次筛选，每次 20 个 epoch）：

```powershell
python fgaae.py train
```

默认自动选择可用 CUDA，否则 CPU。完整训练可能耗时较长。

默认数据路径相对于脚本确定，与终端当前目录无关。输入是每行一帧的数值矩阵，支持逗号或空白分隔，可以没有扩展名。要求至少 40 帧，每帧至少 16 点；训练和推理的采样点数必须一致。

### 无标签数据

现有默认输入没有异常标签，程序会明确警告并执行**合成异常演示**：

- 将原始帧视为候选正常参考，不代表这些帧已被人工确认正常。
- 先划分互不重叠的训练/验证/测试来源行，再分别合成异常。
- 阶段二训练池默认注入 5% 异常，用于演示 AQT 对污染训练池的筛选。
- 验证/测试分别默认注入 10% 异常，包含窄峰、局部能量抬升、漂移、谐波、缺失、饱和、阶跃。
- 比例、强度、随机种子均可设置；此类结果不能当成论文真实数据实验结果。

### 有真实标签的数据

有表头的 CSV 可包含 `label` 列，0 为正常、1 为异常，其余列为频谱采样点。允许额外的 `original_index`、`index` 或 `Unnamed:` 索引列，它们不会进入网络。其他说明性字符串列请预先移除。

```powershell
python fgaae.py train --data "D:\数据\labeled_spectra.csv" --evaluation real --output ".\runs\real_experiment"
```

各类别至少需要三帧；完整两阶段训练还需要足够的正常训练帧。阶段一只使用训练划分中的正常样本；阶段二不使用标签进行梯度优化或 AQT 筛选。验证标签用于阈值校准和报告指标，测试标签仅在最终评估时使用。

`--level-scale 0.1` 表示现有文本数值为 0.1 dB 单位；若输入已经是 dB，改成 `--level-scale 1`。若输入为非负线性功率，使用 `--units linear --level-scale 1`，并按数据尺度设置 `--peak-prominence`。默认频率轴为左闭右开的 87–108 MHz，840 点对应 25 kHz 间隔；其他数据应修改 `--start-mhz` 和 `--end-mhz`。

## 2. 对论文的实现

| 模块 | 实现 |
| --- | --- |
| 编码器 | 三层 Conv1d，通道 8/16/32，核 4，padding 1，BatchNorm + LeakyReLU；第二层后 Dropout 0.1 |
| 潜空间 | 默认 512 维 |
| 解码器 | 对称三层 ConvTranspose1d；根据编码器尺寸计算 output_padding，精确恢复长度 |
| Critic | 通道 64/128/256，输出不经过 sigmoid 的实数分数 |
| 特征头 | 教师/学生均为 512→256→128→64→25 |
| 阶段一 | 编码器与教师头回归专家特征，MSE，AdamW，lr=1e-5，weight_decay=1e-3，100 epoch；恢复阶段一验证损失最低的参数并冻结教师头 |
| 阶段二 | 重构 MSE + WGAN 生成器损失 + 教师/学生蒸馏 MSE；AdamW，lr=2e-5 |
| WGAN-GP | Critic 损失 mean(C(fake))-mean(C(real))+λGP；真实/生成样本随机插值，对输入梯度施加惩罚 |
| AQT | 每次外层迭代计算整个无标签池的异常分数，按经验分位数保留低分样本，然后训练多个 epoch；每次都重新审视完整池 |
| 评估 | 验证集校准最终检测阈值；测试集报告 accuracy/precision/recall/F1/ROC-AUC/AP/混淆矩阵 |

频谱全局均值/标准差及特征逐维均值/标准差**仅拟合阶段一训练子集**，同一变换应用于其他所有划分。阶段一训练、阶段一验证、阶段二池、最终验证、测试的来源行互不重叠。

## 3. 论文未明确或相互矛盾之处

以下是实现选择，不声称是论文原始配置：

1. **卷积步长**：正文说全部 stride=2，但表 1 给出 840→210→105→52。默认 `--strides 4 2 2` 匹配表 1；用 `--strides 2 2 2` 匹配正文，此时长度为 840→420→210→105。
2. **GP 权重**：正文分别出现 5 和 10，默认 10，可用 `--gp-weight 5` 切换。
3. **Critic 归一化**：论文写 BatchNorm，但它使样本间相互依赖，不符合逐样本 GP 的常见假设。默认 LayerNorm；`--critic-normalization batch` 可切换为论文文字版本，也可选择 `none`。
4. **异常评分符号**：论文对 Critic 规定“越大越真实”，式 11 却对其正向加权。默认采用 `MSE - 0.1*C(reconstruction)`，保证低真实度增加异常分数；`--critic-score-sign 1` 可按式 11 的正号计算。AQT 与最终检测使用同一个评分定义。
5. **AQT 衰减**：公式 13 是指数衰减，表 2 写线性衰减。默认指数 `r_end+(r_start-r_end)*exp(-alpha*progress)`，progress 是归一化外层迭代进度；默认 start=0.95、end=0.5、alpha=5。也支持 `--retention-schedule linear`。指数式在有限迭代末尾趋近而不严格等于 end。
6. **25 维专家特征**：论文未列出具体定义。本实现提供明确、可检查的替代向量，名称保存在 `FEATURE_NAMES` 和运行目录的 `normalization.json`；不是简单截取旧项目的前 25 列。包括电平统计、相对功率重心/展宽/熵/平坦度、总功率、峰数、主峰位置/幅值/显著度/半功率宽度、峰间距、占用率和三段功率比例。寻峰使用 SciPy，并未将原项目 AMPD 假定为论文指定的方法。
7. **损失权重与循环次数**：论文未给出完整取值。默认重构 1、对抗 0.01、蒸馏 1；每批生成器更新对应 5 次 Critic 更新；阶段二 10 次外层迭代，每次 20 epoch。均可通过 CLI 修改。
8. **最终检测阈值**：训练池 AQT 筛选阈值与最终检测阈值不同。论文没有充分给出最终校准方案；默认使用正常验证样本分数的 99% 分位数。`--threshold-method validation-f1` 可在验证集选择 F1 最优阈值。绝不使用测试标签选阈值。
9. **实验规模**：程序读取实际输入行数，不复制或伪造论文的约十万帧数据。默认数据仅有约 1514 帧。

## 4. 输出与推理

每次训练默认新建 `runs/日期时间/`，显式 `--output` 必须是尚不存在的目录，防止覆盖旧实验。

- `config.json`、`environment.json`：实际参数和运行环境。
- `normalization.json`、`splits.csv`：变换参数、特征名称、来源行和合成异常类型。
- `stage1_history.csv`、`stage2_history.csv`：逐 epoch 损失和验证指标。
- `aqt_iteration_*.csv`：各轮训练池分数和保留状态；不使用训练异常标签筛选。
- `stage1_best.pt`、`stage2_best.pt`、`stage2_latest.pt`、`model.pt`：阶段一最佳参数、阶段二验证最优参数、最近迭代参数、含归一化和校准阈值的完整推理包。阶段二结束时恢复验证最优参数；`selection.json` 记录选中 epoch，默认 10 轮无改善且至少训练 10 轮后早停。
- `metrics.json`、`validation_scores.csv`、`test_scores.csv`：评估来源、指标及逐样本分数。
- `stage1_history.png`、`stage2_history.png`、`test_roc.png`、`reconstructions.png`：非交互图像；`--no-plots` 可关闭。

`stage2_latest.pt` 是阶段快照，不是包含优化器状态和随机状态的断点续训包。`predict` 应使用最终 `model.pt`。仅加载可信来源的模型文件。

```powershell
python fgaae.py predict --checkpoint ".\runs\你的实验目录\model.pt" --data "D:\数据\new_spectra.csv" --output ".\new_scores.csv"
```

本目录暂不包含论文中 CIFAR-10/WideResNet、FixMatch/Dash/FreeMatch、完整 SNR 扫描及多模型对比实验，也不会把一次单模型训练当成这些实验的复现。

## 5. 数据目录排查

2026-10-27 的逐文件排查见 [数据可用性报告](data_audit_20261027/README.md)。现有原始 FScan 中找到 99 组、120186 帧可提取为 87–108 MHz 原生 840 点输入的数据；它们尚无正常/异常真值标签。另发现部分已转换 TXT 误分帧，因此扩充训练集应优先从原始 IX/MD 重建，而不是直接拼接所有 TXT。此前训练使用的 1514 帧切片已经与原始数据逐点核对，一致。
