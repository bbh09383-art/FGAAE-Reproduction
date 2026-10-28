# 完整 FGAAE 复现

当前目标是先跑通并验证完整模型，再考虑消融。入口为 `full_model.py`，共享 `fgaae.py` 的模型和两阶段训练实现。

默认三种子实验已完成，见 [结果报告](runs/full_fgaae_v4/RESULTS.md)。开发 AUC 为 0.877556 ± 0.040129，尚未稳定复现论文性能；已确认预训练有效，同时发现共享编码器在联合训练后出现明显的特征映射漂移。

随后完成的性能提升试验见 [STABILIZATION.md](STABILIZATION.md) 和 [新结果](runs/full_fgaae_improvement_round1/RESULTS.md)：训练调整后开发 AUC 为 0.891949 ± 0.023722，正号 Critic 系数校准后为 0.911744 ± 0.035992。该校准成绩属于开发调参，不能替代独立测试或原始协议验证。

## 运行

在本目录执行：

```powershell
$env:PYTHONIOENCODING = 'utf-8'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
python full_model.py smoke-test
python full_model.py train --device cuda
python summarize_full_model.py
```

默认输出 `runs/full_fgaae_v4`。输出目录必须不存在，重跑需用 `--output` 指定新目录，避免覆盖已登记的实验。

默认使用种子 2025、2026、2027，每个种子均独立完成：

1. 编码器和教师头训练 100 个 epoch，批大小 32，AdamW 学习率 1e-5。
2. 冻结教师头，继续训练共享编码器、解码器、学生头和 Critic。学生头独立初始化，蒸馏目标来自冻结教师头。
3. 运行 3 次 AQT，每次训练 20 个 epoch，批大小 256，Critic 每个生成器步骤更新 5 次；不提前停止。
4. 以开发集 AUC 选择保存的完整模型，以开发集标签确定 F1 阈值。该指标属于开发成绩，不能用作独立测试成绩。

预训练默认读取 5 个原始训练文件的 6620 帧，另留 1 个训练文件的 1324 帧做特征验证。两个集合按源文件隔离，也全部排除在阶段二之外。相比旧流程每个 epoch 只有几个批次，本轮每个种子的预训练约有 20700 次优化更新。

阶段二默认使用同一站点的 20000 帧，按已冻结成员顺序取样，不根据标签挑选正常帧。正式扩大时可用 `--train-size 0` 使用该站点所有符合隔离要求的训练帧。开发集固定为 10592 帧。三种子共享相同数据与异常注入，区别只在模型随机性。

## 与论文的对齐及未明确部分

| 项目 | 当前实现与边界 |
| --- | --- |
| 输入和编码器 | 840 点；8、16、32 通道，512 维潜变量。按论文表格长度采用步长 4、2、2；正文全部步长 2 与表格不一致。 |
| 特征引导 | 25 维显式专家特征，教师/学生头为 512→256→128→64→25。论文没有完整列出特征定义，因此本项目特征为可追溯替代定义。 |
| 教师冻结 | 冻结教师头参数，共享编码器继续更新；教师的输出会随共享编码器变化。教师输出按蒸馏目标停止梯度，论文未详述该梯度路径，是当前实现的一个假设。 |
| WGAN-GP | 保留完整 Critic、真实/重构对抗项与梯度惩罚；GP 权重 10。Critic 使用 LayerNorm，避免 BatchNorm 的跨样本耦合影响逐样本梯度惩罚。 |
| 联合损失 | 重构 MSE 权重 1，对抗项 0.01，特征蒸馏 MSE 权重 1；未从论文恢复的权重作为明确假设保存。 |
| AQT | 每次重算整个训练池分数，以经验分位数筛选；保留率为 0.5 + 0.45×exp(-5×进度)，3 次外循环的进度取 0、0.5、1，保留率约为 0.95、0.537、0.503。论文未充分明确这些数值和进度缩放方式。 |
| 检测评分 | 预先登记公式 11 的字面解释：平方误差之和 + 0.1×C(重构)。重构训练损失仍用 MSE。 |
| 公式歧义 | 论文公式中 Critic 为正号，与“分数越高越真实”的描述存在冲突。主结果固定采用正号，负号和纯重构只保存诊断，不事后选择较高分方案。平方范数按求和实现；840 点输入时是逐点 MSE 的 840 倍。 |

WGAN-GP 的逐样本梯度惩罚及归一化选择参考[原论文](https://papers.neurips.cc/paper/7159-improved-training-of-wasserstein-gans.pdf)。`fgaae.py` 原有入口仍默认按 MSE 评分，保证历史实验设置不被静默更改；本入口显式设置 `reconstruction_score_reduction=sum`。

## 数据与评估边界

沿用 `datasets/fm_87_108_v1` 的文件分组划分、`grouped_pipeline_v1/benchmark` 的冻结训练注入、`station_background_v2/inputs` 的冻结开发集。源文件、数据指纹、训练成员与完整配置均记录在输出目录。

当前异常仍是局部线性功率注入，背景未经过人工正常性确认；注入/背景功率比不等同于论文的波形信噪比。不能将当前 AUC 与论文 AUC 直接判作同一实验复现。先前查看过的确认集、测试集和外站集不在本入口评估范围内。

## 结果文件

- `registered_plan.json`：训练前登记的配置、公式解释、代码哈希和已知差异。
- `inputs/manifest.json`、成员 CSV：参考文件、训练/开发成员、标准化参数及数据哈希。
- `seed_*/teacher_health.json`：教师在留出文件上的误差，和仅预测训练特征均值的误差比较，确认预训练是否有效。
- `seed_*/component_audit.json`：编码器、解码器、学生头和 Critic 的实际参数变化，以及教师头冻结检查。
- `seed_*/stage1_history.csv`、`stage2_history.csv`、`full_model_training.png`：训练历史与曲线。
- `seed_*/aqt_iteration_*.csv`：逐帧筛选记录；`aqt_purity_diagnostics.csv` 仅在训练完成后统计注入比例，标签不用于筛选。
- `seed_*/model.pt`：包含完整模型、标准化参数及阈值，可由 `fgaae.py predict` 加载。
- `development_runs.csv`、`full_model_report.json`：全部种子的开发成绩与均值，不只报最好种子。
- `RESULTS.md`、`verification.json`、`development_curves.png`：完成训练后由 `summarize_full_model.py` 生成，含数据/代码指纹、训练预算、CPU 加载分数核验和各注入强度的开发曲线。

推理示例：

```powershell
python fgaae.py predict --checkpoint runs/full_fgaae_v4/seed_2025/model.pt --data runs/station_background_v2/inputs/development.npy --output runs/full_fgaae_v4/prediction_example.csv --device cpu
```

小样本自检只证明接口、训练、参数更新、冻结和保存/加载链路可用，不能证明模型性能或论文结果已经复现。
