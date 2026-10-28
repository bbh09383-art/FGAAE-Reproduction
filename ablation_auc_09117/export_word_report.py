"""Build an editable, six-page Chinese Word report from verified result tables."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MAIN = ROOT / "runs/ablation_auc_09117_v1"
DIAG = ROOT / "runs/ablation_auc_09117_diagnostic_v1"
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_run(text, bold=False, color=None, size=None):
    props = ('<w:b/>' if bold else '') + (f'<w:color w:val="{color}"/>' if color else '') + (f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>' if size else '')
    content = '<w:br/>'.join(f'<w:t xml:space="preserve">{escape(line)}</w:t>' for line in str(text).split('\n'))
    return f'<w:r><w:rPr>{props}</w:rPr>{content}</w:r>'


class Report:
    def __init__(self):
        self.body = []
        self.images = []

    def p(self, text, style="Normal", bold=False, color=None, size=None, center=False, newpage=False, keep=False, callout=False):
        properties = f'<w:pStyle w:val="{style}"/>'
        if center:
            properties += '<w:jc w:val="center"/>'
        if newpage:
            properties += '<w:pageBreakBefore/>'
        if keep:
            properties += '<w:keepNext/>'
        if callout:
            properties += '<w:shd w:val="clear" w:fill="EDF3F8"/><w:pBdr><w:left w:val="single" w:sz="18" w:space="9" w:color="1F4E79"/></w:pBdr><w:ind w:left="150" w:right="110"/>'
        self.body.append(f'<w:p><w:pPr>{properties}</w:pPr>{text_run(text, bold, color, size)}</w:p>')

    def h1(self, text, newpage=False):
        self.p(text, "Heading1", newpage=newpage, keep=True)

    def h2(self, text):
        self.p(text, "Heading2", keep=True)

    def bullet(self, text):
        self.p("• " + text, "Bullet")

    def table(self, headers, rows, widths=None, font=19):
        widths = widths or [9866 // len(headers)] * len(headers)
        widths[-1] += 9866 - sum(widths)
        xml = '<w:tbl><w:tblPr><w:tblW w:w="9866" w:type="dxa"/><w:tblLayout w:type="fixed"/><w:tblBorders>'
        for edge in ["top", "left", "bottom", "right", "insideH", "insideV"]:
            xml += f'<w:{edge} w:val="single" w:sz="4" w:color="D9E2F3"/>'
        xml += '</w:tblBorders><w:tblCellMar><w:top w:w="65" w:type="dxa"/><w:left w:w="85" w:type="dxa"/><w:bottom w:w="65" w:type="dxa"/><w:right w:w="85" w:type="dxa"/></w:tblCellMar></w:tblPr>'
        xml += '<w:tblGrid>' + ''.join(f'<w:gridCol w:w="{width}"/>' for width in widths) + '</w:tblGrid>'
        for index, values in enumerate([headers, *rows]):
            xml += '<w:tr><w:trPr><w:cantSplit/>' + ('<w:tblHeader/>' if index == 0 else '') + '</w:trPr>'
            for column, value in enumerate(values):
                fill = "1F4E79" if index == 0 else ("F2F6FA" if index % 2 else "FFFFFF")
                alignment = "left" if column == 0 and index else "center"
                xml += f'<w:tc><w:tcPr><w:tcW w:w="{widths[column]}" w:type="dxa"/><w:vAlign w:val="center"/><w:shd w:fill="{fill}"/></w:tcPr>'
                xml += f'<w:p><w:pPr><w:pStyle w:val="TableText"/><w:jc w:val="{alignment}"/></w:pPr>'
                xml += text_run(value, index == 0, "FFFFFF" if index == 0 else "172B4D", font)
                xml += '</w:p></w:tc>'
            xml += '</w:tr>'
        self.body.append(xml + '</w:tbl>')
        self.p("", "Spacer")

    def image(self, path, width_cm, caption):
        index = len(self.images) + 1
        with Image.open(path) as source:
            width, height = source.size
        cx, cy = int(width_cm * 360000), int(width_cm * 360000 * height / width)
        self.images.append(path)
        self.body.append(f'''<w:p><w:pPr><w:jc w:val="center"/><w:keepNext/><w:spacing w:before="0" w:after="0"/></w:pPr><w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="{cx}" cy="{cy}"/><wp:docPr id="{index}" name="Figure {index}" descr="{escape(caption)}"/><wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr><a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:pic><pic:nvPicPr><pic:cNvPr id="0" name="figure_{index}.png"/><pic:cNvPicPr/></pic:nvPicPr><pic:blipFill><a:blip r:embed="rIdImage{index}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill><pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>''')
        self.p(caption, "Caption", center=True)

    def save(self, path):
        section = '<w:sectPr><w:headerReference w:type="default" r:id="rIdHeader"/><w:footerReference w:type="default" r:id="rIdFooter"/><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1020" w:right="1020" w:bottom="900" w:left="1020" w:header="400" w:footer="400"/><w:cols w:space="720"/></w:sectPr>'
        document = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="{W}" xmlns:r="{R}" xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"><w:body>{''.join(self.body)}{section}</w:body></w:document>'''
        relationships = f'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        for name, target, kind in [("Styles", "styles.xml", "styles"), ("Settings", "settings.xml", "settings"), ("Header", "header1.xml", "header"), ("Footer", "footer1.xml", "footer")]:
            relationships += f'<Relationship Id="rId{name}" Type="{R}/{kind}" Target="{target}"/>'
        for index in range(1, len(self.images) + 1):
            relationships += f'<Relationship Id="rIdImage{index}" Type="{R}/image" Target="media/figure_{index}.png"/>'
        relationships += '</Relationships>'
        styles = f'''<w:styles xmlns:w="{W}"><w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:eastAsia="宋体" w:cs="Calibri"/><w:sz w:val="22"/><w:szCs w:val="22"/><w:lang w:val="en-US" w:eastAsia="zh-CN"/></w:rPr></w:rPrDefault><w:pPrDefault><w:pPr><w:spacing w:after="100" w:line="300" w:lineRule="auto"/><w:widowControl/></w:pPr></w:pPrDefault></w:docDefaults>
<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style>
<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:pPr><w:keepNext/><w:spacing w:before="70" w:after="170"/></w:pPr><w:rPr><w:rFonts w:eastAsia="微软雅黑"/><w:b/><w:color w:val="1F4E79"/><w:sz w:val="42"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Subtitle"><w:name w:val="Subtitle"/><w:basedOn w:val="Normal"/><w:pPr><w:spacing w:after="140"/></w:pPr><w:rPr><w:color w:val="61758A"/><w:sz w:val="22"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:pPr><w:keepNext/><w:keepLines/><w:spacing w:before="80" w:after="120"/><w:outlineLvl w:val="0"/></w:pPr><w:rPr><w:rFonts w:eastAsia="微软雅黑"/><w:b/><w:color w:val="1F4E79"/><w:sz w:val="30"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:pPr><w:keepNext/><w:spacing w:before="110" w:after="65"/><w:outlineLvl w:val="1"/></w:pPr><w:rPr><w:rFonts w:eastAsia="微软雅黑"/><w:b/><w:color w:val="1F4E79"/><w:sz w:val="25"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="TableText"><w:name w:val="Table Text"/><w:basedOn w:val="Normal"/><w:pPr><w:spacing w:before="0" w:after="0" w:line="250" w:lineRule="auto"/><w:keepLines/></w:pPr></w:style>
<w:style w:type="paragraph" w:styleId="Caption"><w:name w:val="Caption"/><w:basedOn w:val="Normal"/><w:pPr><w:spacing w:before="45" w:after="90" w:line="240" w:lineRule="auto"/><w:keepLines/></w:pPr><w:rPr><w:color w:val="61758A"/><w:sz w:val="18"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Bullet"><w:name w:val="Report Bullet"/><w:basedOn w:val="Normal"/><w:pPr><w:ind w:left="220" w:hanging="220"/><w:spacing w:after="85"/><w:keepLines/></w:pPr></w:style>
<w:style w:type="paragraph" w:styleId="Small"><w:name w:val="Small"/><w:basedOn w:val="Normal"/><w:pPr><w:spacing w:after="60" w:line="240" w:lineRule="auto"/></w:pPr><w:rPr><w:sz w:val="18"/><w:color w:val="61758A"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Spacer"><w:name w:val="Spacer"/><w:pPr><w:spacing w:after="30" w:line="60" w:lineRule="exact"/></w:pPr><w:rPr><w:sz w:val="2"/></w:rPr></w:style></w:styles>'''
        types = '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Default Extension="png" ContentType="image/png"/>'
        for target, kind in [("document", "document.main"), ("styles", "styles"), ("settings", "settings"), ("header1", "header"), ("footer1", "footer")]:
            types += f'<Override PartName="/word/{target}.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.{kind}+xml"/>'
        types += '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/><Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/></Types>'
        header = f'<w:hdr xmlns:w="{W}"><w:p><w:pPr><w:spacing w:after="45"/><w:pBdr><w:bottom w:val="single" w:sz="4" w:color="D9E2F3"/></w:pBdr></w:pPr>{text_run("FGAAE 本地复现｜模块消融与补充诊断", color="61758A", size=17)}</w:p></w:hdr>'
        footer = f'<w:ftr xmlns:w="{W}"><w:p><w:pPr><w:jc w:val="center"/></w:pPr>{text_run("汇报整理稿  ·  第 ", color="61758A", size=17)}<w:fldSimple w:instr="PAGE">{text_run("1", size=17)}</w:fldSimple>{text_run(" 页 / 共 ", size=17)}<w:fldSimple w:instr="NUMPAGES">{text_run("6", size=17)}</w:fldSimple>{text_run(" 页", size=17)}</w:p></w:ftr>'
        created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        core = f'<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"><dc:title>FGAAE 本地复现与模块消融实验结果汇报</dc:title><dc:creator>实验结果整理</dc:creator><dc:description>主消融与事后补充诊断；ROC-AUC；冻结 Full 0.9117 版本。</dc:description><dcterms:created xsi:type="dcterms:W3CDTF">{created}</dcterms:created></cp:coreProperties>'
        with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
            archive.writestr("[Content_Types].xml", types)
            archive.writestr("_rels/.rels", f'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="{R}/officeDocument" Target="word/document.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/><Relationship Id="rId3" Type="{R}/extended-properties" Target="docProps/app.xml"/></Relationships>')
            archive.writestr("word/document.xml", document)
            archive.writestr("word/_rels/document.xml.rels", relationships)
            archive.writestr("word/styles.xml", styles)
            archive.writestr("word/settings.xml", f'<w:settings xmlns:w="{W}"><w:updateFields w:val="true"/><w:defaultTabStop w:val="420"/></w:settings>')
            archive.writestr("word/header1.xml", header)
            archive.writestr("word/footer1.xml", footer)
            archive.writestr("docProps/core.xml", core)
            archive.writestr("docProps/app.xml", '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"><Application>FGAAE report export</Application></Properties>')
            for index, image in enumerate(self.images, 1):
                archive.write(image, f"word/media/figure_{index}.png")


def build(output):
    main = pd.read_csv(MAIN / "auc_summary.csv").set_index("arm")
    diagnostic = pd.read_csv(DIAG / "auc_summary.csv").set_index("arm")
    runs = pd.read_csv(MAIN / "auc_runs.csv")
    druns = pd.read_csv(DIAG / "auc_runs.csv")
    paired = pd.read_csv(MAIN / "paired_auc_differences.csv")
    effects = pd.read_csv(DIAG / "conditional_effects_summary.csv")
    effects = effects[effects.score == "auc_primary"].set_index("effect")
    assert abs(main.loc["Full", "auc_mean"] - 0.9117440964193119) < 1e-12
    for arm in ["Base", "B"]:
        assert abs(main.loc[arm, "auc_mean"] - diagnostic.loc[arm, "auc_mean"]) < 1e-12
    fmt = lambda row, prefix="auc": f"{row[prefix+'_mean']:.4f} ± {row[prefix+'_sample_std']:.4f}"
    report = Report()

    # Page 1: executive summary and experimental scope.
    report.p("FGAAE 本地复现与模块消融\n实验结果汇报", "Title")
    report.p("冻结 Full AUC 0.9117 版本｜主指标：ROC-AUC｜含 B 与 Base 补充诊断", "Subtitle")
    report.p(f"整理日期：{datetime.now():%Y年%m月%d日}　　统计口径：三个初始化种子的均值 ± 样本标准差", "Small")
    report.p("汇报要点：完整模型平均 AUC 最高；对抗与完整特征引导流程的主 AUC 收益方向较一致；AQT 收益较小。B 低于 Base 的补充诊断显示，无特征引导配方下的对抗分支出现负收益，AQT 有小幅正收益。", bold=True, callout=True)
    report.h2("1. 实验目标与版本")
    report.p("围绕已保留的 FGAAE 本地复现版本开展逐模块移除实验，并对 B 组低于 Base 的现象追加诊断。Full 使用历史三个校准模型，平均 AUC 为 0.911744；本次不使用后续 AUC 0.9322 版本或局部残差增强作为基准。")
    report.h2("2. 统一实验设置")
    report.table(["项目", "设置"], [
        ["阶段二训练 / 开发", "20000 帧 / 10592 帧；相同源文件划分与冻结合成异常"],
        ["训练参考背景", "6620 帧训练参考、1324 帧特征验证；背景非人工确认正常"],
        ["随机种子", "2025、2026、2027；各组配对比较"],
        ["联合训练预算", "3 个训练段 × 20 epoch；batch size 256"],
        ["特征预训练", "保留特征流程的组复用相同 100 epoch 预训练；无特征组仅做无标签 BN 校准"],
        ["学习率与损失", "编码器 2e-6；其余生成器与 Critic 2e-5；对抗 0.03、蒸馏 0.1、GP 10"],
        ["新训练工作量", "主消融 12 次 / 720 epoch；补充诊断 6 次 / 360 epoch；Full 历史结果复用"],
    ], [2200, 7666], font=20)
    report.h2("3. 评价口径")
    report.bullet("仅以 ROC-AUC 评价；同时列出模型合法评分的主 AUC，以及同一检查点统一使用重构误差的 AUC。")
    report.bullet("主 AUC 的检查点与正评分系数选择沿用历史流程。开发文件已被用于模型开发和校准，当前成绩不属于新的独立测试。")

    # Page 2: main design and results, all editable tables.
    report.h1("一、主消融实验：五组结果", newpage=True)
    report.table(["组别", "对抗模块", "特征引导流程", "AQT"], [
        ["Base：纯 AE", "无", "无", "无"], ["A：无对抗", "无", "有", "有"],
        ["B：无特征引导", "有", "无", "有"], ["C：无 AQT", "有", "有", "无"],
        ["Full：完整模型", "有", "有", "有"],
    ], [3100, 2100, 2566, 2100])
    report.p("特征引导流程包含专家特征预训练及教师—学生蒸馏；B 同时删除二者。A 删除对抗训练及最终/AQT 评分中的 Critic 项；C 始终使用全部 20000 帧。", "Small")
    report.table(["组别", "主 ROC-AUC", "统一重构 ROC-AUC", "Full − 本组"], [
        [arm, fmt(main.loc[arm]), fmt(main.loc[arm], "reconstruction_auc"), f"{main.loc[arm, 'delta_full_minus_variant']:+.4f}"]
        for arm in ["Base", "A", "B", "C", "Full"]
    ], [1300, 3000, 3266, 2300])
    report.p("表 1　均值 ± 三个种子的样本标准差。差值为原始 AUC 单位，乘 100 为百分点；两列使用同一已选检查点，不为补充评分另挑模型。", "Caption")
    report.h2("主要观察")
    report.bullet("对抗模块：移除后平均主 AUC 下降 3.78 个百分点，三个种子均为 Full 更高。该差异同时包含训练、Critic 评分及其对 AQT 的影响。")
    report.bullet("完整特征引导流程：移除后平均下降 6.57 个百分点，三个种子方向一致；不能将全部差异只归因于蒸馏或物理知识。")
    report.bullet("AQT：平均收益约 0.93 个百分点，但仅 2/3 种子支持 Full 更高，证据弱于前两项。")
    report.bullet("B 的 AUC 低于 Base，表明模块数量增加并不保证性能单调提高；该现象触发了后续补充诊断。")
    report.h2("评分解释")
    report.p("Full 主 AUC 为 0.9117，统一重构 AUC 为 0.8919。含对抗模块的组保留原式 R＋λc·C(G(x))，其中 R 为重构残差平方和；无对抗模块的组只用 R。因此主表检验的是完整流程的局部贡献，而非某一种内部机制的单独因果效应。")

    # Page 3: original scientific figures, with editable captions.
    report.h1("二、主消融图示与种子一致性", newpage=True)
    report.image(MAIN / "roc_curves.png", 15.8,
                 "图 1　主评分 ROC：各种子在 1001 个固定 FPR 点插值后平均，阴影为逐点样本标准差；表中 AUC 直接由原始评分计算。")
    report.image(MAIN / "paired_auc_differences.png", 15.8,
                 "图 2　配对差值 Full−消融组：点为三个种子，柱为均值，误差线为样本标准差，均不是置信区间。C 组有一个反向种子。")

    # Page 4: additional diagnostic factorial slice.
    report.h1("三、补充诊断：B 为什么低于 Base", newpage=True)
    report.p("针对已观察到的现象事后追加两个组。四组均无完整特征引导，旧 Base/B 原样复用；新增组仍使用相同数据、三个种子、60 epoch、BN 与校准规则。")
    report.table(["组别", "对抗 / AQT", "主 ROC-AUC", "统一重构 ROC-AUC"], [
        [label, flags, fmt(diagnostic.loc[arm]), fmt(diagnostic.loc[arm], "reconstruction_auc")]
        for arm, label, flags in [("Base", "Base（复用）", "无 / 无"), ("GAN_only", "AE＋WGAN-GP", "有 / 无"),
                                  ("AQT_only", "AE＋AQT", "无 / 有"), ("B", "B（复用）", "有 / 有")]
    ], [2350, 1700, 2900, 2916], font=19)
    report.table(["条件比较", "平均 AUC 变化", "方向一致性"], [
        ["无 AQT 时加入对抗模块", f"{effects.loc['GAN_without_AQT', 'mean']*100:+.2f} 个百分点", "3/3 下降"],
        ["有 AQT 时加入对抗模块", f"{effects.loc['GAN_with_AQT', 'mean']*100:+.2f} 个百分点", "3/3 下降"],
        ["无对抗时加入 AQT", f"{effects.loc['AQT_without_GAN', 'mean']*100:+.2f} 个百分点", "3/3 提高"],
        ["有对抗时加入 AQT", f"{effects.loc['AQT_with_GAN', 'mean']*100:+.2f} 个百分点", "3/3 提高"],
    ], [4600, 2800, 2466])
    report.p("结果更指向无特征引导配方下的对抗分支负收益。沿 Base→AE＋GAN→B 的路径，平均变化约为 −1.08＋0.25＝−0.83 个百分点。AQT 的小幅增益不足以抵消前一项。", bold=True, callout=True)
    report.image(DIAG / "conditional_effects.png", 16.5,
                 "图 3　主 AUC 与统一重构 AUC 的条件差值。两个评分视角下方向相近；交互项仅为 AUC 差值尺度上的描述量，不直接证明内部损失冲突。")
    report.p("旧 Base/B 首轮真实训练已精确重放；新增组与对应旧组的初始张量完全一致。检查未发现组件开关或初始化不一致。该结果尚未区分对抗权重、学习率、前向状态等更细的优化原因。", "Small")

    # Page 5: interpretation, reproducibility and reporting boundaries.
    report.h1("四、讨论、核验与结论边界", newpage=True)
    report.h2("1. 对抗模块存在条件依赖")
    report.table(["固定 AQT 开启", "加入对抗模块后的主 AUC 变化", "三个种子"], [
        ["无完整特征引导：B−AE＋AQT", "−1.00 个百分点", "均下降"],
        ["有完整特征引导：Full−A", "+3.78 个百分点", "均提高"],
    ], [4100, 3566, 2200])
    report.p("这项事后扩展分析复用原 A/Full/B 与新 AE＋AQT，提示在当前数据和配方下，对抗模块的收益依赖完整特征流程。该流程同时包含预训练和蒸馏，不能据此只证明专家知识、单独蒸馏或网络内部协同机制。")
    report.h2("2. AQT 结果需谨慎表述")
    c = paired[paired.arm == "C"].sort_values("seed")
    report.table(["种子", "Full AUC", "无 AQT AUC", "Full−无 AQT"], [
        [str(int(row.seed)), f"{row.full_auc:.6f}", f"{row.variant_auc:.6f}", f"{row.delta_full_minus_variant:+.6f}"]
        for row in c.itertuples()
    ], [1500, 2700, 2700, 2966])
    report.bullet("Full 的最佳 epoch 为 2/9/10，均在第二次筛选前；当前主表不能直接验证后续多轮动态过滤的额外收益。")
    report.bullet("相同 60 epoch 不等于相同更新次数：有 AQT 每种子 4200 次 G 更新，无 AQT 4740 次。过滤质量与优化预算尚未完全分离。")
    report.h2("3. 可复查性与目前决定")
    report.p("18 个新增模型共 1080 个正式训练 epoch；190656 次 CUDA 全量重载评分差异为 0。历史 Full 的 31776 次预测另已核验。AUC 从保存的连续评分独立重算，旧 Base/B 与原五组结果保持不变。")
    report.p("本轮未发现需要立即修正的已检查实现不一致，未重跑或替换 Base/B 正式结果。后续若调整配方，应另立配对实验并保留原表；通用对抗实现发生变化时，应检查所有受影响组。")
    report.h2("4. 汇报时需要说明")
    report.bullet("所有开发文件历史上已被查看，并参与检查点/校准选择；结果应称为本地既有开发协议下的实验，不是新独立测试或原论文原始指标。")
    report.bullet("异常为冻结的合成功率注入；名义注入功率比不等于论文信道 SNR。专家特征和部分实现仍含本地重建假设。")
    report.bullet("AUC 支持评分排序区分能力的讨论，不直接证明固定阈值检出率、潜在流形完全分离或跨站泛化改善。")

    # Page 6: exact per-seed primary AUC and source index.
    report.h1("附录：逐种子数值与汇报口径", newpage=True)
    report.p("下表为主评分 ROC-AUC，保留六位小数，便于复制到汇报表。三个种子使用相同数据；± 为初始化种子的样本标准差。", "Small")
    all_runs = pd.concat([runs, druns[druns.arm.isin(["GAN_only", "AQT_only"])]], ignore_index=True)
    appendix = []
    for arm, label in [("Base", "Base"), ("A", "A：无对抗"), ("B", "B：无特征"), ("C", "C：无 AQT"),
                       ("Full", "Full"), ("GAN_only", "AE＋GAN"), ("AQT_only", "AE＋AQT")]:
        values = all_runs[all_runs.arm == arm].set_index("seed").auc_primary
        assert len(values) == 3
        appendix.append([label, *[f"{values.loc[seed]:.6f}" for seed in [2025, 2026, 2027]], f"{values.mean():.6f} ± {values.std(ddof=1):.6f}"])
    report.table(["组别", "2025", "2026", "2027", "均值 ± 标准差"], appendix, [1800, 1500, 1500, 1500, 3566], font=18)
    report.h2("可直接用于汇报的结论")
    report.p("在相同本地数据与评价协议下，完整 FGAAE 的平均 ROC-AUC 最高。移除对抗模块和完整特征引导流程后，主 AUC 在三个配对种子上均下降；AQT 的平均收益较小，存在一个反向种子。针对 B 低于 Base 的补充诊断发现，无特征引导时，对抗模块出现负收益，而 AQT 带来小幅正收益。结合原实验，对抗模块的条件效应随完整特征流程而改变，提示模块增加并不保证性能单调提高。具体优化机制和多轮动态过滤的额外贡献仍需进一步对照验证。", callout=True)
    report.h2("后续研究建议（本轮未执行）")
    report.bullet("若要论证多轮动态 AQT，补充固定比例、等更新次数或随机过滤对照。")
    report.bullet("若继续优化无特征引导模型，围绕一个明确因素设计敏感性实验，不将优化后的个别组直接混入原固定配置主表。")
    report.h2("结果来源索引")
    report.p("项目目录：D:\\荣珍代码与数据\\代码\\FGAAE_论文复现", "Small")
    for text in [
        "主消融：runs/ablation_auc_09117_v1/RESULTS.md、auc_runs.csv、auc_summary.csv",
        "补充诊断：runs/ablation_auc_09117_diagnostic_v1/RESULTS.md、conditional_effects_summary.csv",
        "条件扩展：runs/ablation_auc_09117_diagnostic_v1/feature_conditioning_summary.csv",
        "核验记录：主目录 final_verification.json；诊断目录 analysis_verification.json",
    ]:
        report.p(text, "Small")
    report.p("本文件仅整理已完成结果，不修改模型、数值表或训练规则；图表沿用已核验版本，表格与正文均可在 Word 中编辑。", "Small")

    output.parent.mkdir(parents=True, exist_ok=True)
    report.save(output)
    from xml.etree import ElementTree as ET
    with ZipFile(output) as archive:
        for name in archive.namelist():
            if name.endswith(".xml") or name.endswith(".rels"):
                ET.fromstring(archive.read(name))
        assert len([name for name in archive.namelist() if name.startswith("word/media/")]) == 3
        xml = ET.fromstring(archive.read("word/document.xml"))
        texts = ''.join(node.text or '' for node in xml.iter(f'{{{W}}}t'))
        for number in ["0.9117", "0.8543", "0.8436", "0.8561", "190656", "63552"]:
            # Diagnostic-only replay count appears in source records rather than summary text.
            if number != "63552":
                assert number in texts, number
    inputs = [MAIN / name for name in ["auc_summary.csv", "auc_runs.csv", "paired_auc_differences.csv", "roc_curves.png", "paired_auc_differences.png"]]
    inputs += [DIAG / name for name in ["auc_summary.csv", "auc_runs.csv", "conditional_effects_summary.csv", "conditional_effects.png", "analysis_verification.json"]]
    manifest = {"word_file": str(output.resolve()), "planned_pages": 6, "images_embedded": 3,
                "tables_editable": True, "only_performance_metric": "ROC-AUC", "input_sha256": {str(path.relative_to(ROOT)): sha(path) for path in inputs},
                "export_script_sha256": sha(Path(__file__)), "docx_initial_sha256": sha(output)}
    output.with_suffix(".sources.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"docx": str(output.resolve()), "bytes": output.stat().st_size, "planned_pages": 6, "embedded_figures": 3}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "汇报材料/FGAAE_消融实验与补充诊断_汇报.docx")
    build(parser.parse_args().output.resolve())
