"""生成 1.报告.docx —— 在模板基础上填充五个章节的内容。

保留模板原有的五个标题，在标题下方插入正文、表格与图表。
"""
from __future__ import annotations

import os

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = r"C:\Users\古恩宇\Downloads\【图像处理】-01234567-姓名\【图像处理】-01234567-姓名\1.报告.docx"
FIG = os.path.join(ROOT, "outputs")

NAME = "古恩宇"
SID = "20245651"
CLASS = "人工智能2402"
GITHUB = "https://github.com/shisiyu-cmd/fovea-localization"


# ---------------- 样式工具 ----------------

def set_cn_font(run, name="宋体", size=12, bold=False, color=None):
    run.font.name = name
    run.font.size = Pt(size)
    run.bold = bold
    if color:
        run.font.color.rgb = RGBColor(*color)
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:eastAsia"), name)
    rfonts.set(qn("w:ascii"), name)
    rfonts.set(qn("w:hAnsi"), name)


def add_para(doc, text="", size=12, bold=False, font="宋体",
             align=None, space_before=0, space_after=6,
             first_line_indent=True, color=None, line_spacing=1.3):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(space_before)
    pf.space_after = Pt(space_after)
    pf.line_spacing = line_spacing
    if first_line_indent:
        pf.first_line_indent = Pt(size * 2)
    if align is not None:
        p.alignment = align
    if text:
        r = p.add_run(text)
        set_cn_font(r, font, size, bold, color)
    return p


def add_hyperlink(paragraph, url, text, size=11, font="Consolas",
                  color=(0, 90, 200)):
    """向段落追加一个真正的可点击超链接（w:hyperlink）。

    直接给 run 上色只能做到「看起来像链接」，在 Word 里点击无反应。
    必须建立 r:id 关系并包一层 w:hyperlink 元素。
    """
    from docx.opc.constants import RELATIONSHIP_TYPE as RT

    part = paragraph.part
    r_id = part.relate_to(url, RT.HYPERLINK, is_external=True)

    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), r_id)

    new_run = OxmlElement("w:r")
    rPr = OxmlElement("w:rPr")

    # 颜色
    c = OxmlElement("w:color")
    c.set(qn("w:val"), "%02X%02X%02X" % color)
    rPr.append(c)

    # 下划线
    u = OxmlElement("w:u")
    u.set(qn("w:val"), "single")
    rPr.append(u)

    # 字体
    rfonts = OxmlElement("w:rFonts")
    for attr in ("w:ascii", "w:hAnsi", "w:eastAsia"):
        rfonts.set(qn(attr), font)
    rPr.append(rfonts)

    sz = OxmlElement("w:sz")
    sz.set(qn("w:val"), str(int(size * 2)))  # 半磅
    rPr.append(sz)

    new_run.append(rPr)
    t = OxmlElement("w:t")
    t.text = text
    new_run.append(t)
    hyperlink.append(new_run)

    paragraph._p.append(hyperlink)
    return hyperlink


def add_heading(doc, text, level=1):
    sizes = {1: 15, 2: 13.5}
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(14 if level == 1 else 10)
    pf.space_after = Pt(8 if level == 1 else 6)
    pf.line_spacing = 1.35
    # 标题不与后续正文分离（避免标题孤立在页底）
    pf.keep_with_next = True
    pf.keep_together = True
    r = p.add_run(text)
    set_cn_font(r, "黑体", sizes.get(level, 12), bold=True)
    return p


def add_bullet(doc, text, size=12, indent=Pt(24)):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.left_indent = indent
    pf.space_before = Pt(0)
    pf.space_after = Pt(4)
    pf.line_spacing = 1.3
    r = p.add_run("• " + text)
    set_cn_font(r, "宋体", size)
    return p


def add_code(doc, text, size=9.5):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.left_indent = Pt(24)
    pf.space_before = Pt(4)
    pf.space_after = Pt(8)
    pf.line_spacing = 1.15
    # 代码块是单个段落（内部用换行符分行），必须禁止跨页拆分，
    # 否则会被页面边界切成两半、灰色底纹也断成两块。
    pf.keep_together = True
    ppr = p._element.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), "F4F4F4")
    ppr.append(shd)
    for i, line in enumerate(text.split("\n")):
        if i:
            p.add_run().add_break()
        r = p.add_run(line)
        set_cn_font(r, "Consolas", size)
    return p


def add_caption(doc, text, size=10):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf = p.paragraph_format
    pf.space_before = Pt(2)
    pf.space_after = Pt(10)
    r = p.add_run(text)
    set_cn_font(r, "宋体", size, color=(90, 90, 90))
    return p


def add_page_break(doc):
    """插入分页符。"""
    from docx.enum.text import WD_BREAK
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    p.add_run().add_break(WD_BREAK.PAGE)
    return p


def add_picture(doc, path, width_cm=15.5, keep_with_next=False):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    if keep_with_next:
        p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(path, width=Cm(width_cm))
    return p


def _set_table_borders(table):
    """为表格添加单线边框（模板未内置 Table Grid 样式）。"""
    tblPr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "6")
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), "9AA0A6")
        borders.append(el)
    tblPr.append(borders)


def add_table(doc, headers, rows, widths=None, size=10.5):
    t = doc.add_table(rows=1, cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    _set_table_borders(t)

    hdr = t.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = ""
        p = hdr[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        set_cn_font(r, "黑体", size, bold=True)
        tcpr = hdr[i]._tc.get_or_add_tcPr()
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear")
        shd.set(qn("w:fill"), "E8EEF7")
        tcpr.append(shd)

    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(str(v))
            set_cn_font(r, "宋体", size)

    if widths:
        # 关闭自动调整并改为固定布局，否则 Word 会忽略手动设置的列宽
        t.autofit = False
        tblPr = t._tbl.tblPr
        layout = OxmlElement("w:tblLayout")
        layout.set(qn("w:type"), "fixed")
        tblPr.append(layout)
        for i, w in enumerate(widths):
            for row in t.rows:
                row.cells[i].width = Cm(w)
        # 同时写入列定义，确保渲染器按此显示
        for i, w in enumerate(widths):
            t.columns[i].width = Cm(w)
    return t


# ---------------- 正文 ----------------

def _strip_missing_fonts(doc):
    """移除样式表中对 Courier 的引用，避免渲染时报告缺字。

    python-docx 的默认模板在若干内置样式中写死了 Courier，
    本报告正文并未使用这些样式，但渲染器仍会报告缺字。
    """
    styles_xml = doc.styles.element
    for rfonts in styles_xml.iter(qn("w:rFonts")):
        for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
            if rfonts.get(qn(attr)) == "Courier":
                rfonts.set(qn(attr), "Consolas")


def _purge_courier_in_package(path):
    """重写 docx 包，移除 fontTable / stylesWithEffects 中的 Courier 声明。

    python-docx 的默认模板在这两个部件里声明了 Courier 字体，
    即使正文并未使用，渲染器仍会报告缺字。必须新建 ZIP 逐个复制成员，
    不能向原包追加同名成员（会产生重复条目并损坏文件）。
    """
    import re
    import shutil
    import tempfile
    import zipfile

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".docx")
    tmp.close()
    with zipfile.ZipFile(path, "r") as src, \
            zipfile.ZipFile(tmp.name, "w", zipfile.ZIP_DEFLATED) as dst:
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename in ("word/fontTable.xml", "word/stylesWithEffects.xml"):
                text = data.decode("utf-8")
                # 把 Courier 的字体声明整体替换为 Consolas
                text = text.replace('w:name="Courier"', 'w:name="Consolas"')
                text = text.replace('"Courier"', '"Consolas"')
                text = text.replace("Courier", "Consolas")
                data = text.encode("utf-8")
            dst.writestr(item, data)
    shutil.move(tmp.name, path)


def build():
    doc = Document()

    # 页面设置：A4，适中页边距
    sec = doc.sections[0]
    sec.page_width = Cm(21.0)
    sec.page_height = Cm(29.7)
    sec.left_margin = Cm(2.8)
    sec.right_margin = Cm(2.8)
    sec.top_margin = Cm(2.54)
    sec.bottom_margin = Cm(2.54)

    # ===== 标题 =====
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run("基于解剖学先验的眼底图像黄斑中心凹自动定位")
    set_cn_font(r, "黑体", 18, bold=True)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run("—— 图像处理课程大作业报告")
    set_cn_font(r, "楷体", 13, color=(70, 70, 70))

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(16)
    r = p.add_run(f"{CLASS}　{NAME}　{SID}")
    set_cn_font(r, "宋体", 12)

    # ===== 1. Github 链接 =====
    add_heading(doc, "1. Github 链接", 1)
    add_para(doc, "本作业的全部源代码、实验脚本与结果图表已整理为可直接运行的工程，仓库地址：")

    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Pt(24)
    p.paragraph_format.space_after = Pt(8)
    add_hyperlink(p, GITHUB, GITHUB, size=11)

    add_para(doc, "仓库包含算法包 fovea_lab/（预处理、视盘定位、黄斑检测、评估指标）、"
                  "scripts/ 下的评估与作图脚本，以及 README 中的完整复现步骤。"
                  "数据集体积较大未纳入版本管理，按 README 说明放入 data/detection/ 即可复现全部结果。")

    add_para(doc, "如需核对实现细节，推荐按以下顺序阅读仓库：先看 README 中的方法说明与结果表格，"
                  "再看 fovea_lab/pipeline.py 了解整体流程，最后进入各子模块。"
                  "运行 python scripts/evaluate.py 可直接复现本文表 4 的全部数字。")

    # ===== 2. 任务描述 =====
    add_heading(doc, "2. 任务描述 你要做什么", 1)

    add_heading(doc, "2.1 任务背景", 2)
    add_para(doc, "黄斑中心凹（fovea）是视网膜上视觉最敏锐的区域，也是糖尿病视网膜病变、"
                  "年龄相关性黄斑变性等眼底疾病在分级、随访与手术规划中的关键解剖标志点。"
                  "临床上该位置需要由医生在眼底照片上人工标定，单张耗时数十秒，"
                  "且不同医师之间存在明显的观察者间差异，难以支撑大规模筛查。")
    add_para(doc, "本作业的目标是：给定一张彩色眼底照片，自动输出黄斑中心凹的像素坐标 (x, y)。"
                  "要求使用传统图像处理方法完成，不引入深度学习模型——"
                  "即只依靠颜色空间变换、形态学运算、滤波与几何先验等经典手段。")

    add_heading(doc, "2.2 具体目标", 2)
    add_bullet(doc, "输入：任意尺寸的彩色眼底照片（本数据为 1956×1934 至 2992×2000）。")
    add_bullet(doc, "输出：黄斑中心凹在原图坐标系下的 (x, y)。")
    add_bullet(doc, "评价指标：预测点与标注点的欧氏距离（像素）；统计平均误差、中位误差，"
                    "以及误差不超过 50 px / 100 px 的样本比例（成功率）。")
    add_bullet(doc, "约束：仅使用传统图像处理方法；单张推理需在秒级完成，且过程可解释。")

    add_heading(doc, "2.3 难点分析", 2)
    add_para(doc, "在动手之前，先通过观察数据明确这个任务到底难在哪里，这直接决定了后续的方法设计。")
    add_bullet(doc, "黄斑本身对比度极低。它只是一个比周围视网膜稍暗的、边界模糊的类圆形区域，"
                    "没有锐利边缘，直接做边缘检测或阈值分割都会失败。")
    add_bullet(doc, "存在大量更强的干扰暗结构。血管主干与交叉点的暗响应往往比黄斑更强烈，"
                    "是朴素方法最主要的误检来源。")
    add_bullet(doc, "成像质量差异大。不同设备、不同患者的图像在整体亮度、色调、"
                    "视场形状上差异明显，且视场圆周普遍存在一圈高亮拍摄伪影。")
    add_bullet(doc, "黄斑位置本身不固定。对 80 张训练图统计可知，"
                    "黄斑横坐标的标准差达图像宽度的 5.5%，单纯用「图像中心附近」这类假设无法覆盖。")

    add_heading(doc, "2.4 解决思路", 2)
    add_para(doc, "针对上述难点，本文的核心思路是：不直接在全图范围内寻找暗斑，"
                  "而是先定位一个高对比度、易检测的解剖结构——视盘（optic disc），"
                  "再利用视盘与黄斑之间稳定的解剖学关系，把搜索范围压缩到一个很小的窗口内。")
    add_para(doc, "这一思路的关键依据来自对训练集的统计。以视盘中心为原点、视盘半径为尺度，"
                  "统计黄斑的相对位置后发现：黄斑几乎总是位于视盘的正水平方向"
                  "（偏离水平线不足 13°），垂直偏移仅 0.38 个视盘半径，"
                  "而水平距离约为 2.3 个视盘半径。这意味着「黄斑在视盘的某一侧、"
                  "基本同一高度」是一个极强的先验，足以把绝大多数干扰结构排除在候选之外。")

    # ===== 3. 数据描述 =====
    add_heading(doc, "3. 数据描述 数据是什么", 1)

    add_heading(doc, "3.1 数据集构成", 2)
    add_para(doc, "本作业使用课程提供的眼底图像黄斑定位数据集（2026task3），目录结构如下：")
    add_code(doc,
             "data/detection/\n"
             "├── train/                          训练图像 0001.jpg ~ 0080.jpg（80 张）\n"
             "├── test/                           测试图像 0081.jpg ~ 0100.jpg（20 张）\n"
             "├── train_location/                 训练标注 XML（含视盘/黄斑框）\n"
             "├── fovea_localization_train_GT.csv 训练集黄斑中心坐标\n"
             "└── sample_submission.csv           提交格式模板")

    add_table(doc,
              ["项目", "内容"],
              [["训练样本", "80 张彩色眼底照片，均带黄斑中心坐标标注"],
               ["测试样本", "20 张彩色眼底照片，无标注，需预测后提交"],
               ["图像格式", "JPEG，RGB 三通道，位深 8 bit"],
               ["图像尺寸", "1956×1934 ~ 2992×2000，尺寸不固定"],
               ["标注格式", "CSV，字段 Fovea_X / Fovea_Y，单位为原图像素坐标"],
               ["提交格式", "CSV，字段 ImageID / value，每张图两行（X、Y 各一行）"]],
              widths=[3.2, 11.5])
    add_caption(doc, "表 1　数据集基本信息")

    add_heading(doc, "3.2 标注坐标的分布特征", 2)
    add_para(doc, "对 80 个标注点做统计，得到如下分布特征。这组数字是后续方法设计的直接依据：")
    add_table(doc,
              ["统计量", "Fovea_X", "Fovea_Y"],
              [["均值（像素）", "1446.1", "1055.8"],
               ["标准差（像素）", "202.5", "44.1"],
               ["最小值（像素）", "723.7", "930.2"],
               ["最大值（像素）", "1794.6", "1182.2"],
               ["标准差 / 图像尺寸", "5.50%", "2.17%"]],
              widths=[5.5, 4.5, 4.5])
    add_caption(doc, "表 2　黄斑标注坐标的统计分布")

    add_para(doc, "一个非常重要的观察是：黄斑的纵向位置极其稳定（标准差仅占图像高度的 2.17%），"
                  "而横向位置变化很大（占宽度的 5.50%）。这说明黄斑在「上下」方向上有很强的规律，"
                  "真正的困难在于判断它在「左右」的哪一侧以及具体多远。"
                  "这一发现直接指导了后续先验函数的设计——水平方向给宽松的容差，垂直方向严格约束。")

    add_heading(doc, "3.3 视盘-黄斑的解剖学关系", 2)
    add_para(doc, "为了量化二者的相对位置，先用本文的视盘检测器在 80 张训练图上定位视盘，"
                  "再把黄斑坐标换算到「以视盘中心为原点、视盘半径为 1」的归一化坐标系中。统计结果如下：")
    add_table(doc,
              ["统计量", "水平偏移 Δx / r", "垂直偏移 Δy / r"],
              [["均值", "−0.71", "0.45"],
               ["中位数", "−1.81", "0.38"],
               ["标准差", "2.78", "0.43"],
               ["绝对值 90 分位", "—", "0.80"],
               ["绝对值 95 分位", "—", "0.99"]],
              widths=[5.5, 4.5, 4.5])
    add_caption(doc, "表 3　黄斑相对视盘的归一化偏移（r 为视盘半径）")

    add_para(doc, "结论非常明确：垂直偏移的绝对值有 90% 的样本小于 0.8 个视盘半径，"
                  "且符号几乎恒为正（黄斑略低于视盘中心）；水平方向则分布较散，"
                  "但绝对值集中在 1.5~5 个视盘半径之间，且方向由左右眼决定——"
                  "在图像中表现为黄斑始终位于视盘的一侧。")

    # ===== 4. 代码修改描述 =====
    add_heading(doc, "4. 代码修改描述 你改了什么", 1)

    add_para(doc, "本作业从零实现了一套完整的黄斑定位流程，代码组织为 fovea_lab 算法包。"
                  "下面按处理顺序说明每个模块做了什么，以及为什么这样做。")

    add_heading(doc, "4.1 整体流程", 2)
    add_para(doc, "图 1 给出了一条完整样本在算法各阶段的中间结果，"
                  "可以看出从原始眼底图到最终定位点之间，每一步分别解决了什么问题。")
    add_page_break(doc)
    add_picture(doc, os.path.join(FIG, "fig1_pipeline.png"), width_cm=14.8, keep_with_next=True)
    add_caption(doc, "图 1　算法各阶段的中间结果（以 #33 为例，最终误差 6 px）")

    add_heading(doc, "4.2 预处理模块（preprocess.py）", 2)

    add_para(doc, "（1）视场掩膜提取", bold=True, first_line_indent=False, space_before=6)
    add_para(doc, "采用「阈值 → 开运算 → 闭运算 → 填洞 → 取最大连通域 → 腐蚀」的形态学流程。"
                  "没有使用霍夫圆检测，是因为本数据中视场并非标准圆形（多为椭圆且带切边），"
                  "霍夫圆对参数敏感、容易失败；而连通域方法对形状不敏感，稳健得多。"
                  "最后的腐蚀操作用于剥离视场边缘的黑色环状伪影。")

    add_para(doc, "（2）光照不均校正", bold=True, first_line_indent=False, space_before=6)
    add_para(doc, "采用同态滤波的思想。把观测图像建模为 I(x,y) = L(x,y) · R(x,y)，"
                  "其中 L 是缓变的照明分量，R 是反映组织真实反射率的快变分量。"
                  "用大尺度高斯模糊（σ=40）估计 L，再与原图相除即得 R。")
    add_code(doc,
             "bg  = gaussian_filter(channel, sigma=40)   # 估计照明分量 L\n"
             "out = channel / max(bg, eps)               # 除法校正，得到反射率 R")
    add_para(doc, "这里刻意选择了除法校正而不是直方图均衡：除法能保留绝对的亮度关系，"
                  "而黄斑正是靠「局部比周围暗」这一相对特征被检测的，"
                  "若用直方图均衡会破坏这一关系。")

    add_para(doc, "（3）视场内缩权重", bold=True, first_line_indent=False, space_before=6)
    add_para(doc, "这是解决视盘定位失败的关键改进。用距离变换计算每个像素到视场边界的距离，"
                  "对距边界小于 0.18 倍视场半径的区域，权重线性衰减到 0。"
                  "这是一个与图像内容完全无关的几何约束——真实解剖结构不可能贴在视场最外圈。"
                  "加入这一项后，视盘定位的解剖学一致性通过率从 50% 提升到 99%。")

    add_para(doc, "（4）血管掩膜估计", bold=True, first_line_indent=False, space_before=6)
    add_para(doc, "用形态学顶帽变换提取血管：开运算会抹去比结构元窄的暗结构，"
                  "原图减去开运算的结果就只剩下血管这类细窄暗线。"
                  "再做自适应阈值分割、按面积过滤连通域（去除噪点与大片背景残留），最后膨胀以覆盖血管边缘。")

    add_heading(doc, "4.3 视盘定位模块（disc.py）", 2)
    add_para(doc, "视盘是眼底图中最亮、且颜色偏白（低饱和度）的结构。检测响应由三部分叠加而成：")
    add_bullet(doc, "亮度-饱和度联合响应：视盘亮而白，血管与背景偏红（高饱和）。"
                    "用归一化亮度减去归一化饱和度，得到初响应。")
    add_bullet(doc, "DoG 盘状结构增强：视盘是一个有确定尺寸的亮团块，"
                    "用高斯差分（DoG，LoG 的近似）增强「中心亮、周围暗、尺寸匹配」的结构，"
                    "可有效抑制反光点这类过亮但尺寸不符的干扰。")
    add_bullet(doc, "视场内缩权重：压制圆周伪影（见 4.2 节）。")
    add_para(doc, "最后在响应峰值附近，用局部高亮区的质心细化视盘中心与半径，"
                  "比单纯取响应峰值更稳定（峰值可能落在反光点上）。")

    add_heading(doc, "4.4 黄斑检测模块（macula.py）", 2)

    add_para(doc, "（1）多尺度 LoG 暗区响应", bold=True, first_line_indent=False, space_before=6)
    add_para(doc, "对每个尺度 s 计算 R_s = blur_s(I) − I，即「大尺度平滑后的亮度」减去「原图亮度」。"
                  "中心比周围暗的位置 R_s > 0，且当 s 与暗斑尺寸匹配时响应最强。"
                  "多尺度叠加可以兼顾不同大小的黄斑。这一操作等价于用「墨西哥帽」小波做斑点检测。")

    add_para(doc, "（2）解剖学先验加权", bold=True, first_line_indent=False, space_before=6)
    add_para(doc, "以视盘为原点，向黄斑一侧平移 2.3 个视盘半径、向下平移 0.4 个半径得到期望位置，"
                  "在其周围放置一个各向异性的高斯权重函数——水平方向 σ=1.3r（宽松），"
                  "垂直方向 σ=0.55r（严格）。参数取值直接来自表 3 的统计结果。")
    add_para(doc, "采用加权而非硬性裁剪，是为了保留容错性：先验较弱的区域仍可能被选中，"
                  "但需要具备更强的暗响应。")

    add_para(doc, "（3）血管抑制与亚像素精化", bold=True, first_line_indent=False, space_before=6)
    add_para(doc, "把血管掩膜位置的响应衰减到 15%，并做一次中值滤波填补。"
                  "注意这里是衰减而非直接置零——直接置零会在血管密集处制造大量新的假极值。")
    add_para(doc, "最后在峰值附近用「暗度加权质心」做亚像素精化。"
                  "因为高斯平滑会把响应峰值从暗斑真实中心拉向较亮的一侧，"
                  "用局部暗度作为权重求质心可以抵消这一系统性偏移。")

    add_heading(doc, "4.5 方向判断", 2)
    add_para(doc, "黄斑位于视盘的哪一侧，决定了先验函数的朝向。判断方法是："
                  "以视盘为中心，分别向左右各取一条水平带（高度 1.2 个视盘半径、"
                  "长度覆盖 1.5~4 个视盘半径），比较两侧「局部暗度」的总和，"
                  "更暗的一侧即为黄斑所在侧。该方法在训练集上的准确率约 91%。")

    add_heading(doc, "4.6 关键技术决策说明", 2)
    add_para(doc, "实现过程中遇到了几个值得记录的问题，它们的解决方式直接决定了最终效果：")
    add_bullet(doc, "掩膜外置 −inf 的时机。最初在加权之前就把掩膜外的响应置为 −inf，"
                    "结果 −inf 乘以 0 产生 NaN，而 argmax 遇到 NaN 会返回 (0,0)，"
                    "导致大量预测点跑到图像左上角。修正为「先加权、后置 −inf」即解决。")
    add_bullet(doc, "视盘定位是全局瓶颈。在加入视场内缩权重之前，视盘检测的通过率只有 50%，"
                    "而视盘一错，后续先验全盘失效。这个模块的收益远大于在黄斑检测端做精细调参。")
    add_bullet(doc, "血管抑制并非总是有益。消融实验显示，在没有先验约束的全图搜索模式下，"
                    "加入血管抑制反而使平均误差上升 9.3%——因为抑制血管会在密集区制造新的假极值。"
                    "它只有在配合先验、搜索范围被压缩之后才真正发挥作用。")

    # ===== 5. 结果对比 =====
    add_heading(doc, "5. 结果对比 结果有什么变化", 1)

    add_heading(doc, "5.1 定量结果", 2)
    add_para(doc, "在 80 张训练图上对比基线方法与完整方法。基线方法指：不做视盘定位、"
                  "不加任何解剖学约束，直接在全图范围内搜索多尺度 LoG 暗区响应的最大值。")

    add_table(doc,
              ["指标", "基线方法", "完整方法", "变化"],
              [["平均误差", "349.7 px", "64.9 px", "−81.4%"],
               ["中位误差", "37.9 px", "16.1 px", "−57.4%"],
               ["标准差", "406.2 px", "189.1 px", "−53.5%"],
               ["90 分位误差", "897.7 px", "68.6 px", "−92.4%"],
               ["成功率 @50 px", "53.8%", "86.2%", "+32.5 pp"],
               ["成功率 @100 px", "57.5%", "91.2%", "+33.8 pp"],
               ["单张耗时", "0.23 s", "0.38 s", "+0.15 s"]],
              widths=[4.2, 3.6, 3.6, 3.2])
    add_caption(doc, "表 4　基线方法与完整方法的定量对比（80 张训练图）")

    add_para(doc, "最值得注意的是 90 分位误差从 897.7 px 降到 68.6 px。"
                  "这说明完整方法不仅把「平均情况」做准了，更重要的是彻底消除了灾难性失败——"
                  "基线方法有接近一半的样本直接把预测点丢在了视场边缘，误差接近 900 px。")

    add_picture(doc, os.path.join(FIG, "fig2_cdf.png"), width_cm=11.0)
    add_caption(doc, "图 2　误差累积分布曲线。完整方法的曲线在 20 px 附近即快速上升，"
                     "而基线方法在 60% 处停滞，剩余样本全部是灾难性失败")

    add_picture(doc, os.path.join(FIG, "fig5_scatter.png"), width_cm=15.4)
    add_caption(doc, "图 3　预测坐标与真值的散点对比。完整方法（绿点）紧密贴合 y=x，"
                     "基线方法（红叉）存在大量远离对角线的离群点")

    add_heading(doc, "5.2 消融实验", 2)
    add_para(doc, "为确认每个模块各自的贡献，设计了逐步累加的消融实验：")
    add_table(doc,
              ["配置", "平均误差", "中位误差", "成功率@50px"],
              [["A 基线（全图搜索）", "349.7 px", "37.9 px", "53.8%"],
               ["B +视盘方向判断", "343.9 px", "34.8 px", "55.0%"],
               ["C +血管抑制", "376.0 px", "47.8 px", "51.2%"],
               ["D 完整（+解剖先验）", "64.9 px", "16.1 px", "86.2%"]],
              widths=[6.9, 3.1, 3.1, 3.1])
    add_caption(doc, "表 5　消融实验结果")

    add_picture(doc, os.path.join(FIG, "fig3_ablation.png"), width_cm=15.4)
    add_caption(doc, "图 4　各模块对平均误差与成功率的影响")

    add_para(doc, "这组结果揭示了两个反直觉但重要的结论：")
    add_bullet(doc, "解剖学先验是唯一起决定作用的模块。单独加入视盘方向判断（B）几乎没有改善"
                    "（349.7 → 343.9 px），因为仅仅知道方向并不能排除全图中其他区域的强暗响应；"
                    "而一旦加入空间先验加权（D），误差骤降 82.7%。这说明「把搜索范围约束到正确区域」"
                    "比「更准确地检测暗斑」重要得多。")
    add_bullet(doc, "血管抑制单独使用时是有害的。B → C 使平均误差上升 9.3%，"
                    "因为它会在血管密集区制造新的假极值。但它是完整方法的必要组成部分——"
                    "在先验把搜索范围压缩之后，血管抑制才真正开始贡献精度。"
                    "这也解释了为什么不能简单地「把所有能想到的模块都堆上去」。")

    add_heading(doc, "5.3 定性结果", 2)
    add_picture(doc, os.path.join(FIG, "fig4_qualitative.png"), width_cm=15.6)
    add_caption(doc, "图 5　典型样本的定性对比。上行为基线方法，中行为完整方法，"
                     "下行为完整方法对应的最终响应图（绿圈为真值，红十字为预测）")

    add_para(doc, "从定性结果可以清楚看到基线方法的失败模式：它的预测点几乎全部落在视场边缘，"
                  "原因是视场圆周的暗色渐变在 LoG 响应上比黄斑本身强得多。"
                  "而完整方法由于先验把搜索范围限制在视盘附近的合理区域内，"
                  "从机制上就不可能再犯这类错误。")
    add_para(doc, "在成功的样本上，完整方法的预测点与真值几乎重合（误差 6~17 px）。"
                  "这个精度水平意味着在实际使用中，标注点稳定落在黄斑区内部，"
                  "足以作为医生复核的初始建议位置。")

    add_heading(doc, "5.4 失败案例分析", 2)
    add_para(doc, "完整方法仍有约 9% 的样本误差超过 100 px，主要分为两类：")
    add_bullet(doc, "视盘定位失败导致连锁错误。当视盘被反光斑或病灶遮挡时，检测器可能锁定到错误位置，"
                    "先验随之指向错误方向。这类失败在 #42、#52 上可以观察到。")
    add_bullet(doc, "黄斑本身几乎不可见。个别图像因屈光介质混浊或曝光问题，"
                    "黄斑区对比度极低，此时任何基于暗区响应的方法都难以奏效（如 #72）。")
    add_para(doc, "针对第一类问题，可行的改进方向是用多个候选视盘位置分别生成先验、"
                  "再按响应强度择优；针对第二类，则需要引入血管走向等更强结构特征，"
                  "这已超出传统方法的合理范围。")

    add_heading(doc, "5.5 测试集结果", 2)
    add_para(doc, "对 20 张测试图运行完整方法，结果已写入 submission.csv（40 行，"
                  "格式与 sample_submission.csv 完全一致）。预测坐标分布如下：")
    add_table(doc,
              ["统计量", "预测 X", "预测 Y", "训练集真值 X", "训练集真值 Y"],
              [["均值", "1526.8", "1083.0", "1446.1", "1055.8"],
               ["最小值", "1195.0", "979.0", "723.7", "930.2"],
               ["最大值", "1907.2", "1210.7", "1794.6", "1182.2"]],
              widths=[3.0, 3.4, 3.4, 3.6, 3.6])
    add_caption(doc, "表 6　测试集预测结果的分布（与训练集真值分布对比）")

    add_para(doc, "测试集预测的均值与训练集真值的均值接近（X 差 81 px、Y 差 27 px，"
                  "均在训练集真值标准差的 0.6 倍以内），且全部落在合理的解剖学范围内，"
                  "没有出现明显的异常值，说明方法在未见数据上具有稳定的泛化表现。")

    _strip_missing_fonts(doc)

    doc.save(OUT)
    _purge_courier_in_package(OUT)
    print("saved:", OUT)


if __name__ == "__main__":
    build()
