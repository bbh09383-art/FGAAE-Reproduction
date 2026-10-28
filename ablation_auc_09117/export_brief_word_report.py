"""Export only the five requested primary-AUC results as a brief Word report."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from xml.etree import ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd

from export_word_report import Report, ROOT, MAIN, W


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build():
    destination = ROOT / '汇报材料/五组消融结果_简洁版'
    destination.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(MAIN / 'auc_summary.csv').set_index('arm')
    seeds = pd.read_csv(MAIN / 'auc_runs.csv')
    arms = ['Base', 'A', 'B', 'C', 'Full']
    labels = ['Base：纯 AE', 'A：无对抗模块', 'B：无特征引导流程', 'C：无 AQT', 'Full：完整 FGAAE']
    expected = ['0.8543 ± 0.0490', '0.8739 ± 0.0454', '0.8460 ± 0.0516', '0.9025 ± 0.0458', '0.9117 ± 0.0360']
    formatted = []
    for arm in arms:
        values = seeds.loc[seeds.arm == arm, 'auc_primary'].to_numpy()
        assert len(values) == 3
        np.testing.assert_allclose([values.mean(), values.std(ddof=1)], summary.loc[arm, ['auc_mean', 'auc_sample_std']].to_numpy(dtype=float), rtol=1e-12)
        formatted.append(f"{values.mean():.4f} ± {values.std(ddof=1):.4f}")
    assert formatted == expected

    font = FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family': font.get_name(), 'axes.unicode_minus': False, 'font.size': 11, 'pdf.fonttype': 42})
    colors = ['#8b8b8b', '#0072b2', '#e69f00', '#009e73', '#d55e00']
    means = summary.loc[arms, 'auc_mean'].to_numpy()
    deviations = summary.loc[arms, 'auc_sample_std'].to_numpy()
    fig, ax = plt.subplots(figsize=(9.2, 4.4), layout='constrained')
    x = np.arange(5)
    ax.bar(x, means, width=.62, color=colors, alpha=.83, yerr=deviations,
           error_kw={'ecolor': '#243746', 'capsize': 5, 'elinewidth': 1.4})
    for xi, mean, deviation in zip(x, means, deviations):
        ax.text(xi, mean + deviation + .012, f'{mean:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set(xticks=x, xticklabels=arms, ylim=(0, 1.03), ylabel='ROC-AUC', title='五组消融实验的平均 ROC-AUC')
    ax.set_yticks(np.arange(0, 1.01, .2))
    ax.set_axisbelow(True)
    ax.grid(axis='y', alpha=.18)
    ax.spines[['top', 'right']].set_visible(False)
    chart = destination / '五组AUC对比.png'
    fig.savefig(chart, dpi=220)
    fig.savefig(chart.with_suffix('.pdf'))
    plt.close(fig)

    report = Report()
    report.p('FGAAE 五组消融实验结果', 'Title')
    report.p('ROC-AUC 结果汇总与图表', 'Subtitle')
    report.table(['组别', 'ROC-AUC（均值 ± 标准差）'], list(zip(labels, formatted)), [5000, 4866], font=23)
    report.p('注：各组均在相同数据划分下运行三个随机种子（2025、2026、2027）。表中为主评分 ROC-AUC 的均值 ± 样本标准差，保留四位小数。', 'Small')
    report.h2('为什么报告“均值 ± 标准差”？')
    report.p('均值反映三次实验的平均性能，标准差反映不同随机初始化带来的成绩波动。例如 Full 的 0.9117 是平均 AUC，0.0360 是样本标准差；“±”并不是把两者相加，也不是置信区间。重复实验并同时报告波动，比只呈现最好一次更便于判断结果是否稳定。')
    report.image(chart, 16.5, '图 1　五组平均 ROC-AUC 对比。误差线为三个种子的样本标准差；柱形图纵轴从 0 开始。')
    report.p('评价范围：本地既有开发协议下的结果。开发数据曾用于模型选择与校准，未作为新的独立测试集。', 'Small')

    report.h1('五组结果的曲线与配对比较', newpage=True)
    report.image(MAIN / 'roc_curves.png', 15.8,
                 '图 2　五组主评分 ROC 曲线。曲线为三个种子在固定 FPR 网格上插值后的平均，阴影为逐点样本标准差；AUC 由各次原始评分计算。')
    report.image(MAIN / 'paired_auc_differences.png', 15.8,
                 '图 3　相同种子下 Full−消融组的 AUC 差值。柱为均值，点为三个种子，误差线为样本标准差。C 组存在一个反向种子。')

    output = destination / 'FGAAE_五组消融结果_图表简版.docx'
    report.save(output)
    # Customize only this new package; keep the prior report and exporter intact.
    with ZipFile(output) as archive:
        parts = {name: archive.read(name) for name in archive.namelist()}
    parts['word/header1.xml'] = parts['word/header1.xml'].replace(
        'FGAAE 本地复现｜模块消融与补充诊断'.encode(), 'FGAAE 五组消融实验｜ROC-AUC 结果'.encode())
    parts['word/footer1.xml'] = parts['word/footer1.xml'].replace(b'>6</w:t>', b'>2</w:t>')
    parts['docProps/core.xml'] = parts['docProps/core.xml'].replace(
        'FGAAE 本地复现与模块消融实验结果汇报'.encode(), 'FGAAE 五组消融实验结果与图表'.encode()).replace(
        '主消融与事后补充诊断；ROC-AUC；冻结 Full 0.9117 版本。'.encode(), '仅五组主消融；ROC-AUC；含均值与标准差说明。'.encode())
    with ZipFile(output, 'w', ZIP_DEFLATED) as archive:
        for name, content in parts.items():
            archive.writestr(name, content)
    with ZipFile(output) as archive:
        tree = ET.fromstring(archive.read('word/document.xml'))
        tables = list(tree.iter(f'{{{W}}}tbl'))
        assert len(tables) == 1
        rows = list(tables[0].iter(f'{{{W}}}tr'))
        assert len(rows) == 6
        actual = []
        for row in rows[1:]:
            cells = list(row.iter(f'{{{W}}}tc'))
            actual.append(''.join(n.text or '' for n in cells[1].iter(f'{{{W}}}t')))
        assert actual == expected
        text = ''.join(n.text or '' for n in tree.iter(f'{{{W}}}t'))
        assert all(banned not in text for banned in ['GAN_only', 'AQT_only', 'F1', '统一重构'])
        assert len([n for n in archive.namelist() if n.startswith('word/media/')]) == 3
    inputs = [MAIN / n for n in ['auc_summary.csv', 'auc_runs.csv', 'roc_curves.png', 'paired_auc_differences.png']]
    manifest = {'scope': 'Only five primary-AUC groups requested in the screenshot', 'planned_pages': 2,
                'random_seeds': [2025, 2026, 2027], 'standard_deviation': 'sample (ddof=1)',
                'table_values_verified_against_screenshot': expected, 'input_sha256': {str(p.relative_to(ROOT)): sha(p) for p in inputs},
                'docx_initial_sha256': sha(output), 'tables': 1, 'embedded_figures': 3}
    output.with_suffix('.sources.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'docx': str(output), 'figures': 3, 'tables': 1, 'planned_pages': 2}, ensure_ascii=False))


if __name__ == '__main__':
    build()
