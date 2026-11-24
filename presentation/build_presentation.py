#!/usr/bin/env python3
"""Build the ENSAE Paris 16:9 presentation (reference typography, compiled math, muted upcoming placeholders)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml import parse_xml
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

# ---------------------------------------------------------------------------
# Paths & Design Tokens
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
PRES_DIR = Path(__file__).resolve().parent
ASSETS_DIR = PRES_DIR / "assets"
PLOTS_DIR = ROOT_DIR / "plots"
OUTPUT_PPTX = PRES_DIR / "smc_chmsv_ensae_presentation.pptx"

SLIDE_W = 13.333
SLIDE_H = 7.5

# Page & Uniform Card Padding Constants
SLIDE_MARGIN_X = 0.80
USABLE_W = SLIDE_W - 2 * SLIDE_MARGIN_X  # 11.733 in
CONTENT_TOP = 1.54
CONTENT_BOTTOM = 6.58

CARD_PAD_X = 0.24
CARD_PAD_TOP = 0.20
CARD_PAD_BOTTOM = 0.20
CARD_TITLE_H = 0.36
CARD_TITLE_GAP = 0.10

PILL_PAD_X = 0.20
PILL_PAD_Y = 0.14

NAVY = RGBColor(0x00, 0x27, 0x52)
CYAN = RGBColor(0x00, 0x8B, 0xD2)
SUBTITLE_BLUE = RGBColor(0x1B, 0x44, 0x6B)
STEEL = RGBColor(0x4E, 0x74, 0x96)
SLATE_MUTED = RGBColor(0x7A, 0x8E, 0xA3)
TEXT_DARK = RGBColor(0x1E, 0x29, 0x3B)
TEXT_MUTED = RGBColor(0x64, 0x74, 0x8B)
CARD_BG = RGBColor(0xF4, 0xF7, 0xFB)
CARD_BG_ALT = RGBColor(0xEB, 0xF1, 0xF7)
CARD_BG_TBD = RGBColor(0xF8, 0xFA, 0xFC)
CARD_BORDER = RGBColor(0xC8, 0xD4, 0xE3)
BORDER_TBD = RGBColor(0xB8, 0xC5, 0xD4)
ROW_ALT = RGBColor(0xEB, 0xF0, 0xF6)
DIVIDER = RGBColor(0x7B, 0x92, 0xAA)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
SOFT_WHITE = RGBColor(0xE2, 0xE8, 0xF0)

FONT_FAMILY = "Calibri"
PRES_DATE = "17.12.2026"

BulletItem = Union[str, Tuple[str, str]]
_MATH_SCRIPT_RE = re.compile(r"(_\{[^}]+\}|\^\{[^}]+\})")


# ---------------------------------------------------------------------------
# Low-level XML, Inline Formula Runs & Styling Helpers
# ---------------------------------------------------------------------------
def _set_cell_margins(cell, left=0.07, right=0.07, top=0.04, bottom=0.04) -> None:
    cell.margin_left = Inches(left)
    cell.margin_right = Inches(right)
    cell.margin_top = Inches(top)
    cell.margin_bottom = Inches(bottom)


def _set_cell_border(cell, color_hex: str = "D6DFE8", width_pt: float = 0.65) -> None:
    w_emu = int(width_pt * 12700)
    tcPr = cell._tc.get_or_add_tcPr()
    for tag in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
        existing = tcPr.find(qn(tag))
        if existing is not None:
            tcPr.remove(existing)
        ln = parse_xml(
            f'<{tag} xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
            f'w="{w_emu}" cap="flat" cmpd="sng" algn="ctr">'
            f'<a:solidFill><a:srgbClr val="{color_hex}"/></a:solidFill>'
            f'<a:prstDash val="solid"/>'
            f'</{tag}>'
        )
        tcPr.append(ln)


def _enable_bullet_xml(
    paragraph, char: str = "•", left_in: float = 0.22, hang_in: float = 0.16, color_hex: str = "002752"
) -> None:
    pPr = paragraph._p.get_or_add_pPr()
    pPr.set("marL", str(int(Inches(left_in))))
    pPr.set("indent", str(-int(Inches(hang_in))))
    for old_tag in ("a:buNone", "a:buChar", "a:buAutoNum", "a:buFont", "a:buClr"):
        old = pPr.find(qn(old_tag))
        if old is not None:
            pPr.remove(old)
    buClr = parse_xml(
        f'<a:buClr xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
        f'<a:srgbClr val="{color_hex}"/></a:buClr>'
    )
    buFont = parse_xml(
        f'<a:buFont xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" typeface="{FONT_FAMILY}"/>'
    )
    buChar = parse_xml(
        f'<a:buChar xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" char="{char}"/>'
    )
    pPr.append(buClr)
    pPr.append(buFont)
    pPr.append(buChar)


def _add_rich_runs(
    paragraph,
    text: str,
    font_size: float,
    bold: bool = False,
    italic: bool = False,
    color: RGBColor = TEXT_DARK,
) -> None:
    """Split explicit _{sub} and ^{sup} markup into native PowerPoint sub/superscript runs."""
    parts = _MATH_SCRIPT_RE.split(text)
    for part in parts:
        if not part:
            continue
        baseline = None
        clean_txt = part
        if part.startswith("_{") and part.endswith("}"):
            clean_txt = part[2:-1]
            baseline = "-25000"
        elif part.startswith("^{") and part.endswith("}"):
            clean_txt = part[2:-1]
            baseline = "30000"
        r = paragraph.add_run()
        r.text = clean_txt
        r.font.name = FONT_FAMILY
        r.font.size = Pt(font_size)
        r.font.bold = bold
        r.font.italic = italic
        r.font.color.rgb = color
        if baseline is not None:
            rPr = r._r.get_or_add_rPr()
            rPr.set("baseline", baseline)


def add_textbox(
    slide,
    left: float,
    top: float,
    width: float,
    height: float,
    text: str = "",
    font_size: float = 14.0,
    bold: bool = False,
    italic: bool = False,
    color: RGBColor = TEXT_DARK,
    align: PP_ALIGN = PP_ALIGN.LEFT,
    valign: MSO_ANCHOR = MSO_ANCHOR.TOP,
    line_spacing: float = 1.15,
):
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = valign
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = Inches(0.0)
    if text:
        for li, line_txt in enumerate(text.split("\n")):
            p = tf.paragraphs[0] if li == 0 else tf.add_paragraph()
            p.alignment = align
            p.line_spacing = line_spacing
            _add_rich_runs(p, line_txt, font_size=font_size, bold=bold, italic=italic, color=color)
    return tb


def add_rounded_card(
    slide,
    left: float,
    top: float,
    width: float,
    height: float,
    fill_rgb: RGBColor = CARD_BG,
    border_rgb: Optional[RGBColor] = CARD_BORDER,
    border_pt: float = 1.0,
    adj: float = 0.035,
    dashed: bool = False,
):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(left),
        Inches(top),
        Inches(width),
        Inches(height),
    )
    shape.shadow.inherit = False
    if len(shape.adjustments) > 0:
        shape.adjustments[0] = adj
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_rgb
    if border_rgb is not None:
        shape.line.color.rgb = border_rgb
        shape.line.width = Pt(border_pt)
        if dashed:
            ln = shape.line._get_or_add_ln()
            for old in ln.findall(qn("a:prstDash")):
                ln.remove(old)
            ln.append(
                parse_xml(
                    '<a:prstDash xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" val="dash"/>'
                )
            )
    else:
        shape.line.fill.background()
    return shape


def add_picture_fit(
    slide,
    img_path: Path,
    left: float,
    top: float,
    max_w: float,
    max_h: float,
    align_right: bool = False,
):
    with Image.open(img_path) as im:
        pw, ph = im.size
    aspect = pw / ph
    box_aspect = max_w / max_h
    if aspect >= box_aspect:
        w = max_w
        h = w / aspect
    else:
        h = max_h
        w = h * aspect
    x = (left + max_w - w) if align_right else (left + (max_w - w) / 2.0)
    y = top + (max_h - h) / 2.0
    return slide.shapes.add_picture(str(img_path), Inches(x), Inches(y), width=Inches(w), height=Inches(h))


def add_header_footer(
    slide,
    title: str,
    subtitle: Optional[str],
    page_num: int,
) -> None:
    add_textbox(
        slide,
        left=SLIDE_MARGIN_X,
        top=0.34,
        width=10.65,
        height=0.48,
        text=title,
        font_size=27.5,
        bold=True,
        color=NAVY,
    )
    if subtitle:
        add_textbox(
            slide,
            left=SLIDE_MARGIN_X,
            top=0.88,
            width=10.65,
            height=0.36,
            text=subtitle,
            font_size=16.5,
            color=SUBTITLE_BLUE,
        )

    add_picture_fit(
        slide,
        ASSETS_DIR / "ensae_logo.png",
        left=11.73,
        top=0.26,
        max_w=0.80,
        max_h=0.98,
        align_right=True,
    )

    add_textbox(
        slide,
        left=SLIDE_MARGIN_X,
        top=6.96,
        width=2.00,
        height=0.28,
        text=PRES_DATE,
        font_size=11.0,
        color=NAVY,
    )
    add_textbox(
        slide,
        left=11.53,
        top=6.96,
        width=1.00,
        height=0.28,
        text=str(page_num),
        font_size=11.0,
        color=NAVY,
        align=PP_ALIGN.RIGHT,
    )


def add_bullets(
    slide,
    left: float,
    top: float,
    width: float,
    height: float,
    items: Sequence[BulletItem],
    font_size: float = 13.5,
    space_after: float = 9.0,
    line_spacing: float = 1.15,
    bullet_char: str = "•",
    lead_color: RGBColor = NAVY,
    body_color: RGBColor = TEXT_DARK,
    bullet_hex: str = "002752",
):
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = Inches(0.0)

    for idx, item in enumerate(items):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.space_after = Pt(space_after if idx < len(items) - 1 else 0.0)
        p.line_spacing = line_spacing
        _enable_bullet_xml(p, char=bullet_char, color_hex=bullet_hex)

        if isinstance(item, tuple):
            lead, body = item
            _add_rich_runs(p, lead + " ", font_size=font_size, bold=True, color=lead_color)
            _add_rich_runs(p, body, font_size=font_size, bold=False, color=body_color)
        else:
            _add_rich_runs(p, item, font_size=font_size, bold=False, color=body_color)
    return tb


def add_titled_bullet_card(
    slide,
    left: float,
    top: float,
    width: float,
    height: float,
    title: str,
    items: Sequence[BulletItem],
    title_size: float = 14.5,
    font_size: float = 13.0,
    space_after: float = 9.0,
    muted: bool = False,
) -> None:
    bg = CARD_BG_TBD if muted else CARD_BG
    bd = BORDER_TBD if muted else CARD_BORDER
    t_col = SLATE_MUTED if muted else NAVY
    b_col = TEXT_MUTED if muted else TEXT_DARK
    bu_hex = "7A8EA3" if muted else "002752"

    add_rounded_card(slide, left, top, width, height, fill_rgb=bg, border_rgb=bd, dashed=muted)

    inner_x = left + CARD_PAD_X
    inner_w = width - 2 * CARD_PAD_X
    title_y = top + CARD_PAD_TOP
    body_y = title_y + CARD_TITLE_H + CARD_TITLE_GAP
    body_h = max(0.40, (top + height - CARD_PAD_BOTTOM) - body_y)

    add_textbox(
        slide,
        left=inner_x,
        top=title_y,
        width=inner_w,
        height=CARD_TITLE_H,
        text=title,
        font_size=title_size,
        bold=True,
        color=t_col,
        valign=MSO_ANCHOR.MIDDLE,
    )
    add_bullets(
        slide,
        left=inner_x,
        top=body_y,
        width=inner_w,
        height=body_h,
        items=items,
        font_size=font_size,
        space_after=space_after,
        lead_color=t_col,
        body_color=b_col,
        bullet_hex=bu_hex,
    )


def add_styled_table(
    slide,
    left: float,
    top: float,
    col_widths: Sequence[float],
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    font_size: float = 10.5,
    row_height: float = 0.36,
    header_height: Optional[float] = None,
    col_aligns: Optional[Sequence[PP_ALIGN]] = None,
    bold_first_col: bool = True,
    bold_last_row: bool = False,
    muted: bool = False,
):
    n_rows = 1 + len(rows)
    n_cols = len(headers)
    total_w = sum(col_widths)
    hh = header_height if header_height is not None else row_height
    total_h = hh + len(rows) * row_height

    graphic_frame = slide.shapes.add_table(
        n_rows, n_cols, Inches(left), Inches(top), Inches(total_w), Inches(total_h)
    )
    table = graphic_frame.table
    tblPr = table._tbl.tblPr
    for attr in ("firstRow", "bandRow", "bandCol", "firstCol", "lastCol", "lastRow"):
        tblPr.set(attr, "0")

    for j, cw in enumerate(col_widths):
        table.columns[j].width = Inches(cw)

    table.rows[0].height = Inches(hh)
    for i in range(1, n_rows):
        table.rows[i].height = Inches(row_height)

    hdr_fill = SLATE_MUTED if muted else NAVY
    fc_color = TEXT_MUTED if muted else NAVY
    body_color = TEXT_MUTED if muted else TEXT_DARK

    for j, htxt in enumerate(headers):
        cell = table.cell(0, j)
        cell.fill.solid()
        cell.fill.fore_color.rgb = hdr_fill
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        _set_cell_margins(cell, left=0.07, right=0.07, top=0.03, bottom=0.03)
        _set_cell_border(cell, "B8C7D8", 0.6)
        p = cell.text_frame.paragraphs[0]
        p.alignment = col_aligns[j] if col_aligns else (PP_ALIGN.LEFT if j == 0 else PP_ALIGN.RIGHT)
        _add_rich_runs(p, htxt, font_size=font_size, bold=True, color=WHITE)

    for i, row_vals in enumerate(rows):
        r_idx = i + 1
        bg = ROW_ALT if i % 2 == 0 else WHITE
        is_last = bold_last_row and (i == len(rows) - 1)
        if is_last:
            bg = RGBColor(0xDC, 0xE6, 0xF2)
        for j, val in enumerate(row_vals):
            cell = table.cell(r_idx, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = bg
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            _set_cell_margins(cell, left=0.07, right=0.07, top=0.03, bottom=0.03)
            _set_cell_border(cell, "D6DFE8", 0.6)
            p = cell.text_frame.paragraphs[0]
            p.alignment = col_aligns[j] if col_aligns else (PP_ALIGN.LEFT if j == 0 else PP_ALIGN.RIGHT)
            is_bold = is_last or (bold_first_col and j == 0)
            col = fc_color if is_bold else body_color
            _add_rich_runs(p, val, font_size=font_size, bold=is_bold, color=col)

    return graphic_frame


def add_hline(slide, left: float, top: float, width: float, color: RGBColor = DIVIDER, weight_pt: float = 0.85):
    line = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Inches(left),
        Inches(top),
        Inches(width),
        Pt(weight_pt),
    )
    line.shadow.inherit = False
    line.fill.solid()
    line.fill.fore_color.rgb = color
    line.line.fill.background()
    return line


def add_pipeline_box(
    slide,
    left: float,
    top: float,
    width: float,
    height: float,
    title: str,
    subtitle: str,
    dark: bool = False,
    muted: bool = False,
) -> None:
    if dark:
        fill_c = SLATE_MUTED if muted else NAVY
        add_rounded_card(slide, left, top, width, height, fill_rgb=fill_c, border_rgb=fill_c, adj=0.08)
        t_color, s_color = WHITE, SOFT_WHITE
    else:
        border = BORDER_TBD if muted else NAVY
        add_rounded_card(
            slide, left, top, width, height, fill_rgb=WHITE, border_rgb=border, border_pt=1.0, adj=0.08, dashed=muted
        )
        t_color = SLATE_MUTED if muted else NAVY
        s_color = TEXT_MUTED if muted else TEXT_DARK

    tb = slide.shapes.add_textbox(
        Inches(left + PILL_PAD_X),
        Inches(top + PILL_PAD_Y),
        Inches(width - 2 * PILL_PAD_X),
        Inches(height - 2 * PILL_PAD_Y),
    )
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = Inches(0.0)

    p1 = tf.paragraphs[0]
    p1.alignment = PP_ALIGN.CENTER if dark else PP_ALIGN.LEFT
    p1.space_after = Pt(3.0)
    _add_rich_runs(p1, title, font_size=12.2, bold=True, color=t_color)

    p2 = tf.add_paragraph()
    p2.alignment = PP_ALIGN.CENTER if dark else PP_ALIGN.LEFT
    p2.line_spacing = 1.12
    _add_rich_runs(p2, subtitle, font_size=10.8, color=s_color)


def add_muted_plot_frame(
    slide,
    left: float,
    top: float,
    width: float,
    height: float,
    title: str,
    details: str,
) -> None:
    """Clean, understated muted figure frame without explicit '[INSERT PLOT]' clutter."""
    add_rounded_card(
        slide,
        left=left,
        top=top,
        width=width,
        height=height,
        fill_rgb=CARD_BG_TBD,
        border_rgb=BORDER_TBD,
        border_pt=1.2,
        adj=0.03,
        dashed=True,
    )
    tb = slide.shapes.add_textbox(
        Inches(left + 0.42),
        Inches(top + 0.30),
        Inches(width - 0.84),
        Inches(height - 0.60),
    )
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = Inches(0.0)

    p1 = tf.paragraphs[0]
    p1.alignment = PP_ALIGN.CENTER
    p1.space_after = Pt(8.0)
    _add_rich_runs(p1, title, font_size=15.0, bold=True, color=SLATE_MUTED)

    p2 = tf.add_paragraph()
    p2.alignment = PP_ALIGN.CENTER
    p2.line_spacing = 1.22
    _add_rich_runs(p2, details, font_size=12.2, italic=False, color=TEXT_MUTED)


# ---------------------------------------------------------------------------
# Slide Builders (1 to 20)
# ---------------------------------------------------------------------------
def build_slide_01_title(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_textbox(
        slide,
        left=SLIDE_MARGIN_X,
        top=0.38,
        width=10.65,
        height=0.52,
        text="Sequential Monte Carlo: Project Presentation",
        font_size=28.0,
        bold=True,
        color=NAVY,
    )
    add_textbox(
        slide,
        left=SLIDE_MARGIN_X,
        top=0.96,
        width=10.65,
        height=0.38,
        text="Cholesky Multivariate Stochastic Volatility (ChMSV): Model Setup, Filtering & Estimation",
        font_size=16.5,
        color=SUBTITLE_BLUE,
    )
    add_picture_fit(
        slide,
        ASSETS_DIR / "ensae_logo.png",
        left=11.68,
        top=0.28,
        max_w=0.85,
        max_h=1.06,
        align_right=True,
    )

    slide.shapes.add_picture(
        str(ASSETS_DIR / "cover_banner.png"),
        Inches(0.0),
        Inches(1.88),
        width=Inches(SLIDE_W),
        height=Inches(SLIDE_W / 3.75),
    )

    add_textbox(
        slide,
        left=SLIDE_MARGIN_X,
        top=5.74,
        width=USABLE_W,
        height=0.32,
        text="ENSAE Paris — Sequential Monte Carlo (particles)   |   Prof. Nicolas Chopin",
        font_size=12.5,
        color=TEXT_MUTED,
    )
    add_textbox(
        slide,
        left=SLIDE_MARGIN_X,
        top=6.12,
        width=USABLE_W,
        height=0.38,
        text="Gabriel Himmelein, Maria-Giulia Salvarani, Tomàs Planelles Alonso, Matteo Mola",
        font_size=13.5,
        bold=True,
        color=NAVY,
    )
    add_textbox(slide, left=SLIDE_MARGIN_X, top=6.96, width=2.00, height=0.28, text=PRES_DATE, font_size=11.0, color=NAVY)
    add_textbox(
        slide, left=11.53, top=6.96, width=1.00, height=0.28, text="1", font_size=11.0, color=NAVY, align=PP_ALIGN.RIGHT
    )


def build_slide_02_agenda(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header_footer(slide, "Agenda", None, 2)

    sections = [
        ("Use Case, Motivation & Empirical Data", "01", False),
        ("Phase I: ChMSV State-Space Model & Synthetic Benchmarks", "02", False),
        ("Phase II: Bootstrap Particle Filter & Safe-Haven Results", "03", False),
        ("Phase III: Iterated Filtering (IF2)", "04", True),
        ("Phase IV: Bayesian Estimation (PMCMC / SMC^{2})", "05", True),
        ("Evaluation & Conclusion", "06", False),
    ]
    start_y = 1.62
    step_y = 0.86
    for i, (label, num, is_muted) in enumerate(sections):
        y = start_y + i * step_y
        col = SLATE_MUTED if is_muted else NAVY
        bu_hex = "7A8EA3" if is_muted else "002752"

        tb = slide.shapes.add_textbox(Inches(1.10), Inches(y), Inches(8.85), Inches(0.44))
        tf = tb.text_frame
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = Inches(0.0)
        p = tf.paragraphs[0]
        _enable_bullet_xml(p, char="•", left_in=0.34, hang_in=0.24, color_hex=bu_hex)
        _add_rich_runs(p, label, font_size=19.0, color=col)

        add_textbox(
            slide,
            left=10.05,
            top=y,
            width=0.75,
            height=0.44,
            text=num,
            font_size=19.0,
            bold=True,
            color=col,
            align=PP_ALIGN.RIGHT,
        )
        if i < len(sections) - 1:
            add_hline(slide, left=1.40, top=y + 0.58, width=9.40, color=DIVIDER, weight_pt=0.85)


def build_slide_03_use_case(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header_footer(
        slide,
        "Use Case & Motivation",
        "Dynamic Volatility and Safe-Haven Assessment Across Four Asset Classes",
        3,
    )

    bullets = [
        (
            "Beyond Static Correlation:",
            "Full-sample Pearson correlations hide crisis regimes when asset co-movements jump simultaneously.",
        ),
        (
            "Safe Haven vs. Hedge (Baur & Lucey, 2009):",
            "A strong safe haven stays negatively correlated with stocks during turmoil; a weak safe haven is uncorrelated.",
        ),
        (
            "Guaranteed Covariance Validity:",
            "ChMSV factors the covariance matrix via LDL^{T} Cholesky decomposition so Σ_{t} stays positive definite.",
        ),
        (
            "Joint 4D Scope in particles:",
            "Models S&P 500, Gold, Bitcoin, and Ethereum together (2018–2023) to uncover cross-asset dynamics.",
        ),
    ]
    add_bullets(slide, left=SLIDE_MARGIN_X, top=1.64, width=6.20, height=4.90, items=bullets, font_size=14.2, space_after=15.0)

    cx, cy, cw, ch = 7.30, CONTENT_TOP, 5.23, 5.04
    add_rounded_card(slide, cx, cy, cw, ch, fill_rgb=CARD_BG, border_rgb=CARD_BORDER)
    ix = cx + CARD_PAD_X
    iw = cw - 2 * CARD_PAD_X
    add_textbox(
        slide,
        left=ix,
        top=cy + CARD_PAD_TOP,
        width=iw,
        height=CARD_TITLE_H,
        text="Four-Asset Empirical Universe (2018–2023)",
        font_size=14.5,
        bold=True,
        color=NAVY,
        align=PP_ALIGN.CENTER,
        valign=MSO_ANCHOR.MIDDLE,
    )

    boxes = [
        ("S&P 500 Index (SPX)", "1,258 returns  |  Equity benchmark (Kurtosis 16.22)"),
        ("Spot Gold (XAU/USD)", "1,258 returns  |  Traditional candidate (St. Dev. 0.93)"),
        ("Bitcoin & Ethereum (BTC/ETH)", "1,258 returns each  |  High-volatility crypto assets"),
    ]
    box_y = cy + CARD_PAD_TOP + CARD_TITLE_H + CARD_TITLE_GAP
    for bt, bs in boxes:
        add_pipeline_box(slide, left=ix, top=box_y, width=inner_w if False else iw, height=0.78, title=bt, subtitle=bs, dark=False)
        box_y += 0.90

    add_textbox(
        slide,
        left=inner_x if False else ix,
        top=box_y - 0.08,
        width=iw,
        height=0.34,
        text="↓",
        font_size=18.0,
        bold=True,
        color=NAVY,
        align=PP_ALIGN.CENTER,
        valign=MSO_ANCHOR.MIDDLE,
    )
    add_pipeline_box(
        slide,
        left=ix,
        top=box_y + 0.30,
        width=iw,
        height=(cy + ch - CARD_PAD_BOTTOM) - (box_y + 0.30),
        title="ChMSV State-Space Model (p = 4)",
        subtitle="T = 1,258 daily returns  |  10 latent states  |  30 parameters",
        dark=True,
    )


def build_slide_04_architecture(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header_footer(
        slide,
        "Project Scope & Architecture",
        "Four Modular Phases in particles — Empirical Filtering & Parameter Inference",
        4,
    )

    cards = [
        (
            "Phase I:\nData & ChMSV Model",
            [
                "Align Stooq closes (2018–2023, T = 1,258) & Table 1 stats",
                "Build vectorized ChMSV models (p = 2, p = 4) in particles",
                "Generate synthetic ground-truth benchmarks",
            ],
            ASSETS_DIR / "icon_data.png",
            False,
        ),
        (
            "Phase II:\nBootstrap Filtering",
            [
                "Run Bootstrap SMC with adaptive & step resampling",
                "Verify CLT scaling & Table 2 log-likelihood",
                "Extract Figs. 2–4 volatilities & correlations",
            ],
            ASSETS_DIR / "icon_filter.png",
            False,
        ),
        (
            "Phase III:\nIterated Filtering (IF2)",
            [
                "Implement IF2 with cooling schedule (α = 0.5)",
                "Run 10+ Monte Carlo recovery checks (Table 3)",
                "Estimate Table 2 MLE parameters on real data",
            ],
            ASSETS_DIR / "icon_if2_muted.png",
            True,
        ),
        (
            "Phase IV:\nPMCMC / SMC^{2}",
            [
                "Specify StructDist priors on (μ, φ, σ)",
                "Run multi-chain PMCMC & SMC^{2} on synthetic data",
                "Compare Bayesian posteriors vs. IF2 MLE",
            ],
            ASSETS_DIR / "icon_bayes_muted.png",
            True,
        ),
    ]

    w = 2.71
    gap = (USABLE_W - 4 * w) / 3.0
    for i, (ctitle, citems, icon_path, is_muted) in enumerate(cards):
        x = SLIDE_MARGIN_X + i * (w + gap)
        y, h = CONTENT_TOP, 5.04
        add_rounded_card(
            slide,
            left=x,
            top=y,
            width=w,
            height=h,
            fill_rgb=CARD_BG_TBD if is_muted else CARD_BG,
            border_rgb=BORDER_TBD if is_muted else CARD_BORDER,
            dashed=is_muted,
        )

        ix = x + CARD_PAD_X
        iw = w - 2 * CARD_PAD_X
        add_textbox(
            slide,
            left=ix,
            top=y + CARD_PAD_TOP,
            width=iw,
            height=0.62,
            text=ctitle,
            font_size=14.8,
            bold=True,
            color=SLATE_MUTED if is_muted else NAVY,
        )
        add_bullets(
            slide,
            left=ix,
            top=y + CARD_PAD_TOP + 0.72,
            width=iw,
            height=2.66,
            items=citems,
            font_size=12.4,
            space_after=8.0,
            lead_color=SLATE_MUTED if is_muted else NAVY,
            body_color=TEXT_MUTED if is_muted else TEXT_DARK,
            bullet_hex="7A8EA3" if is_muted else "002752",
        )
        add_picture_fit(slide, icon_path, left=x + (w - 1.22) / 2.0, top=y + h - CARD_PAD_BOTTOM - 1.22, max_w=1.22, max_h=1.22)


def build_slide_05_datasets_profiling(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header_footer(
        slide,
        "Input Datasets & Profiling",
        "Stooq Alignment, Summary Statistics, and Close-Time Verification",
        5,
    )

    t1_headers = ["Asset", "Mean", "St.Dev.", "Skew.", "Kurt.", "Min", "Max"]
    t1_rows = [
        ["SPX", "0.035", "1.374", "-0.770", "16.221", "-12.765", "8.968"],
        ["Gold", "0.033", "0.926", "-0.413", "6.528", "-5.868", "4.605"],
        ["Bitcoin", "0.102", "4.302", "-0.262", "8.485", "-32.360", "23.230"],
        ["Ethereum", "0.092", "5.693", "-0.445", "8.553", "-42.446", "31.151"],
    ]
    add_styled_table(
        slide,
        left=SLIDE_MARGIN_X,
        top=CONTENT_TOP,
        col_widths=[1.05, 0.72, 0.80, 0.78, 0.80, 0.85, 0.85],
        headers=t1_headers,
        rows=t1_rows,
        font_size=10.3,
        row_height=0.36,
        header_height=0.36,
    )

    add_textbox(
        slide,
        left=SLIDE_MARGIN_X,
        top=3.52,
        width=5.85,
        height=0.34,
        text="Paper (Table 1) vs. Replicated Dataset",
        font_size=14.0,
        bold=True,
        color=NAVY,
    )
    cmp_headers = ["Asset", "Paper vs. Repo", "Alignment Note"]
    cmp_rows = [
        ["SPX", "Exact Match", "Fixed 16:00 EST closing bell across all metrics"],
        ["Gold", "Spot xauusd", "Mean 0.033 & St.Dev. 0.926 (not GC=F futures)"],
        ["Bitcoin", "4.305 vs. 4.302", "Early Stooq reciprocal FX rounding (~0.005%)"],
        ["Ethereum", "-42.86 vs. -42.45", "Warsaw midnight (00:00 CET) aligns COVID crash"],
    ]
    add_styled_table(
        slide,
        left=SLIDE_MARGIN_X,
        top=3.92,
        col_widths=[0.95, 1.55, 3.35],
        headers=cmp_headers,
        rows=cmp_rows,
        font_size=10.0,
        row_height=0.56,
        header_height=0.36,
        col_aligns=[PP_ALIGN.LEFT, PP_ALIGN.LEFT, PP_ALIGN.LEFT],
    )

    add_picture_fit(slide, ASSETS_DIR / "chart_slide5_stats.png", left=6.88, top=1.48, max_w=5.65, max_h=2.48)

    add_titled_bullet_card(
        slide,
        left=6.88,
        top=4.12,
        width=5.65,
        height=2.46,
        title="Key Data Alignment Findings",
        items=[
            (
                "ETH Timestamp Discovery:",
                "00:00 UTC feeds distort 12 Mar 2020 to -55.07%; Warsaw midnight matches Table 1 (-42.45%, kurtosis 8.55).",
            ),
            (
                "Common Trading Calendar:",
                "Retains 1,259 aligned trading days across all 4 Stooq feeds (T = 1,258 returns).",
            ),
            (
                "Dual Outputs:",
                "Exports real_{4,2}assets_log_returns.csv and synthetic ground-truth CSVs.",
            ),
        ],
        title_size=14.0,
        font_size=12.0,
        space_after=7.0,
    )


def build_slide_06_fig1_reproduction(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header_footer(
        slide,
        "Empirical Data: Prices & Log-Returns (Fig. 1)",
        "Daily Closing Prices and Percentage Log-Returns (4 June 2018 – 4 June 2023)",
        6,
    )

    add_picture_fit(
        slide,
        PLOTS_DIR / "fig1_real_4assets_prices_and_returns.png",
        left=SLIDE_MARGIN_X,
        top=1.50,
        max_w=6.05,
        max_h=5.08,
    )

    add_titled_bullet_card(
        slide,
        left=7.10,
        top=CONTENT_TOP,
        width=5.43,
        height=3.74,
        title="Key Return Characteristics (T = 1,258)",
        items=[
            (
                "S&P 500 Tail Shock:",
                "Low typical spread (1.374%), but highest kurtosis (16.221) from March 2020 (-12.77%).",
            ),
            (
                "Gold Stability:",
                "Lowest daily volatility (0.926%), lowest kurtosis (6.528), and tightest range.",
            ),
            (
                "Crypto Volatility:",
                "Bitcoin (4.302%) and Ethereum (5.693%) show persistent bursts (-42.45% / +31.15%).",
            ),
            (
                "Common Calendar:",
                "1,259 aligned Stooq closing prices yield 1,258 daily log-returns × 100.",
            ),
        ],
        title_size=14.5,
        font_size=12.8,
        space_after=9.0,
    )

    add_pipeline_box(
        slide,
        left=7.10,
        top=5.44,
        width=5.43,
        height=1.10,
        title="Cleaned Empirical Datasets Ready for Filtering",
        subtitle="data/processed/real_4assets_log_returns.csv  |  real_2assets_log_returns.csv",
        dark=True,
    )


def build_slide_07_chmsv_model(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header_footer(
        slide,
        "ChMSV State-Space Specification",
        "LDL^{T} Cholesky Decomposition and Vectorized particles Architecture",
        7,
    )

    # Left: Compiled math cards + concise takeaways
    cx_l, cw_l = SLIDE_MARGIN_X, 6.08
    add_rounded_card(slide, cx_l, CONTENT_TOP, cw_l, 3.64, fill_rgb=CARD_BG, border_rgb=CARD_BORDER)
    ix_l = cx_l + CARD_PAD_X
    iw_l = cw_l - 2 * CARD_PAD_X

    add_textbox(
        slide, left=ix_l, top=CONTENT_TOP + CARD_PAD_TOP, width=iw_l, height=0.28,
        text="1. Observation & Covariance Factorization (Eqs. 1, 3)", font_size=12.5, bold=True, color=NAVY
    )
    add_picture_fit(slide, ASSETS_DIR / "eq_chmsv_obs.png", left=ix_l, top=CONTENT_TOP + 0.52, max_w=iw_l, max_h=0.52)

    add_textbox(
        slide, left=ix_l, top=CONTENT_TOP + 1.24, width=iw_l, height=0.28,
        text="2. Independent Mean-Reverting AR(1) Latent States (Eqs. 8–11)", font_size=12.5, bold=True, color=NAVY
    )
    add_picture_fit(slide, ASSETS_DIR / "eq_chmsv_ar1.png", left=ix_l, top=CONTENT_TOP + 1.56, max_w=iw_l, max_h=0.52)

    add_textbox(
        slide, left=ix_l, top=CONTENT_TOP + 2.28, width=iw_l, height=0.28,
        text="3. Conditional Volatilities & Dynamic Correlations (Eqs. 6–7)", font_size=12.5, bold=True, color=NAVY
    )
    add_picture_fit(slide, ASSETS_DIR / "eq_chmsv_corr.png", left=ix_l, top=CONTENT_TOP + 2.60, max_w=iw_l, max_h=0.78)

    add_bullets(
        slide,
        left=SLIDE_MARGIN_X,
        top=5.32,
        width=cw_l,
        height=1.28,
        items=[
            (
                "Guaranteed Positive Definiteness:",
                "V_{t} = diag(exp(h_{i,t})) and unit lower-triangular L_{t} yield d_{x} = p(p+1)/2 unconstrained states.",
            ),
            (
                "Dimensionality:",
                "ChMSV2Asset (p=2): 3 states, 9 params; ChMSV4Asset (p=4): 10 states, 30 params.",
            ),
        ],
        font_size=12.5,
        space_after=6.5,
    )

    # Right vertical architecture pipeline card
    cx, cy, cw, ch = 7.15, CONTENT_TOP, 5.38, 5.04
    add_rounded_card(slide, cx, cy, cw, ch, fill_rgb=CARD_BG, border_rgb=CARD_BORDER)
    ix = cx + CARD_PAD_X
    iw = cw - 2 * CARD_PAD_X
    add_textbox(
        slide,
        left=ix,
        top=cy + CARD_PAD_TOP,
        width=iw,
        height=CARD_TITLE_H,
        text="ChMSV Architecture (src/base_model.py)",
        font_size=14.5,
        bold=True,
        color=NAVY,
        align=PP_ALIGN.CENTER,
        valign=MSO_ANCHOR.MIDDLE,
    )

    steps = [
        ("1. ChMSVInitDist (PX0)", "Dirac point-mass at unconditional mean μ (t = 0)"),
        ("2. ChMSVTransDist (PX)", "Diagonal Gaussian AR(1) transitions across d_{x} states"),
        ("3. ChMSVObsDist (PY)", "Emit y_{t} | X_{t} ~ N_{p}(0, L_{t}V_{t}L_{t}^{T}) via Cholesky solve"),
    ]
    y_curr = cy + CARD_PAD_TOP + CARD_TITLE_H + CARD_TITLE_GAP
    for st_title, st_sub in steps:
        add_pipeline_box(slide, left=ix, top=y_curr, width=iw, height=0.74, title=st_title, subtitle=st_sub, dark=False)
        add_textbox(
            slide,
            left=ix,
            top=y_curr + 0.74,
            width=iw,
            height=0.26,
            text="↓",
            font_size=15.0,
            bold=True,
            color=NAVY,
            align=PP_ALIGN.CENTER,
            valign=MSO_ANCHOR.MIDDLE,
        )
        y_curr += 1.00

    add_pipeline_box(
        slide,
        left=ix,
        top=y_curr,
        width=iw,
        height=(cy + ch - CARD_PAD_BOTTOM) - y_curr,
        title="4. State-to-Finance Transforms (Eqs. 4–7)",
        subtitle="states_to_covariance()  →  stdev & dynamic corr_pairs",
        dark=True,
    )


def build_slide_08_simulated_benchmarks(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header_footer(
        slide,
        "Simulated Ground-Truth Benchmarks",
        "Controlled Synthetic Testbeds Using Szczepocki (2023) Table 2 Parameters",
        8,
    )

    cx, cy, cw, ch = SLIDE_MARGIN_X, CONTENT_TOP, 4.45, 5.04
    add_rounded_card(slide, cx, cy, cw, ch, fill_rgb=CARD_BG, border_rgb=CARD_BORDER)
    ix = cx + CARD_PAD_X
    iw = cw - 2 * CARD_PAD_X
    add_textbox(
        slide,
        left=ix,
        top=cy + CARD_PAD_TOP,
        width=iw,
        height=CARD_TITLE_H,
        text="Synthetic Testbed Configurations",
        font_size=14.5,
        bold=True,
        color=NAVY,
        valign=MSO_ANCHOR.MIDDLE,
    )

    sim_headers = ["Config", "2-Asset (p=2)", "4-Asset (p=4)"]
    sim_rows = [
        ["Assets", "SPX, Gold", "SPX, Gold, BTC, ETH"],
        ["States (d_{x})", "3 (h_{1}, h_{2}, q_{21})", "10 (4 h_{i}, 6 q_{ij})"],
        ["Parameters", "9 (Table 2 sub)", "30 (Table 2 full)"],
        ["Length (T)", "500", "1,258"],
        ["Seed / Init", "42 / Dirac(μ)", "42 / Dirac(μ)"],
    ]
    tbl_y = cy + CARD_PAD_TOP + CARD_TITLE_H + CARD_TITLE_GAP
    add_styled_table(
        slide,
        left=ix,
        top=tbl_y,
        col_widths=[1.12, 1.32, 1.53],
        headers=sim_headers,
        rows=sim_rows,
        font_size=10.2,
        row_height=0.38,
        header_height=0.36,
        col_aligns=[PP_ALIGN.LEFT, PP_ALIGN.LEFT, PP_ALIGN.LEFT],
    )

    add_bullets(
        slide,
        left=ix,
        top=4.68,
        width=iw,
        height=1.70,
        items=[
            (
                "Exact Ground Truth:",
                "Stores latent x_true, stdev_true, and corr_true to evaluate filtering accuracy.",
            ),
            (
                "Fast Prototyping:",
                "2-asset subset (T = 500) accelerates debugging before full 10D runs.",
            ),
        ],
        font_size=12.0,
        space_after=8.0,
    )

    add_picture_fit(
        slide,
        PLOTS_DIR / "simulated_chmsv_datasets_overview.png",
        left=5.46,
        top=CONTENT_TOP,
        max_w=7.07,
        max_h=4.32,
    )
    add_textbox(
        slide,
        left=5.46,
        top=5.94,
        width=7.07,
        height=0.58,
        text="Ground-truth simulated volatilities (Eq. 6) and dynamic correlations (Eq. 7) for 4-asset (top, T = 1,258) and 2-asset (bottom, T = 500) datasets.",
        font_size=11.5,
        italic=True,
        color=TEXT_MUTED,
        align=PP_ALIGN.CENTER,
    )


def build_slide_09_bpf_sim_tracking(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header_footer(
        slide,
        "Bootstrap Particle Filter & Simulated Tracking",
        "Feynman–Kac Implementation and Ground-Truth State Recovery",
        9,
    )

    cx, cy, cw, ch = SLIDE_MARGIN_X, CONTENT_TOP, 4.35, 2.58
    add_rounded_card(slide, cx, cy, cw, ch, fill_rgb=CARD_BG, border_rgb=CARD_BORDER)
    ix = cx + CARD_PAD_X
    iw = cw - 2 * CARD_PAD_X
    add_textbox(
        slide,
        left=ix,
        top=cy + CARD_PAD_TOP,
        width=iw,
        height=CARD_TITLE_H,
        text="Feynman–Kac Setup (src/filtering.py)",
        font_size=14.0,
        bold=True,
        color=NAVY,
        valign=MSO_ANCHOR.MIDDLE,
    )
    add_bullets(
        slide,
        left=ix,
        top=cy + CARD_PAD_TOP + CARD_TITLE_H + CARD_TITLE_GAP,
        width=iw,
        height=1.20,
        items=[
            ("Bootstrap SMC:", "N = 1,000 particles in particles.SMC (seed 42)."),
            ("Resampling & Bands:", "Systematic resampling + 5th–95th percentiles."),
        ],
        font_size=12.2,
        space_after=6.5,
    )
    add_picture_fit(slide, ASSETS_DIR / "eq_bpf_ess.png", left=ix, top=cy + 1.88, max_w=iw, max_h=0.52)

    trk_headers = ["Simulated Metric (T = 1,258)", "Value"]
    trk_rows = [
        ["SPX Volatility RMSE", "0.180"],
        ["Gold Volatility RMSE", "0.379"],
        ["SPX–Gold Correlation RMSE", "0.026"],
        ["Simulated Log-Likelihood", "-9885.50"],
        ["Mean Daily ESS", "401.3 / 1000"],
    ]
    add_styled_table(
        slide,
        left=SLIDE_MARGIN_X,
        top=4.30,
        col_widths=[3.00, 1.35],
        headers=trk_headers,
        rows=trk_rows,
        font_size=10.4,
        row_height=0.38,
        header_height=0.36,
    )

    add_picture_fit(
        slide,
        PLOTS_DIR / "simulated_tracking_benchmark.png",
        left=5.36,
        top=CONTENT_TOP,
        max_w=7.17,
        max_h=4.44,
    )
    add_textbox(
        slide,
        left=5.36,
        top=6.04,
        width=7.17,
        height=0.52,
        text="True hidden volatilities and correlations (dashed black) track inside 90% particle credibility bands; sharp shocks drop ESS below 500 and trigger resampling.",
        font_size=11.2,
        italic=True,
        color=TEXT_MUTED,
        align=PP_ALIGN.CENTER,
    )


def build_slide_10_sensitivity_table2(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header_footer(
        slide,
        "Monte Carlo Sensitivity & Table 2 Verification",
        "CLT Scaling Across N and Real 4-Asset Log-Likelihood Replication",
        10,
    )

    add_picture_fit(slide, ASSETS_DIR / "chart_slide10_clt.png", left=SLIDE_MARGIN_X, top=1.48, max_w=5.65, max_h=2.72)

    n_headers = ["Particles (N)", "Mean log L", "Std log L", "SE log L", "Mean ESS"]
    n_rows = [
        ["250", "-9904.59", "7.56", "2.39", "103.2"],
        ["500", "-9889.84", "5.28", "1.67", "203.0"],
        ["1000", "-9885.20", "3.66", "1.16", "402.6"],
        ["2000", "-9880.92", "2.89", "0.91", "798.7"],
    ]
    add_styled_table(
        slide,
        left=SLIDE_MARGIN_X,
        top=4.36,
        col_widths=[1.25, 1.25, 1.05, 1.00, 1.10],
        headers=n_headers,
        rows=n_rows,
        font_size=10.2,
        row_height=0.41,
        header_height=0.38,
    )

    add_textbox(
        slide,
        left=6.78,
        top=1.50,
        width=5.75,
        height=0.34,
        text="Real 4-Asset Log-Likelihood vs. Table 2 (N = 1,000)",
        font_size=13.5,
        bold=True,
        color=NAVY,
    )

    t2_headers = ["Metric", "Adap. (γ=0.5)", "Std. (γ=1.0)", "Paper Table 2"]
    t2_rows = [
        ["Mean Log-Lik.", "-9979.09", "-9979.42", "-9972.73"],
        ["logmeanexp (pomp)", "-9962.88", "-9975.17", "-9972.73*"],
        ["Single-Run Std", "9.11", "4.16", "—"],
        ["SE / Jackknife SE", "2.88 / 6.24", "1.32 / 1.12", "0.65 (12 runs)"],
        ["Mean / Min ESS", "404.2 / 2.1", "476.3 / 1.8", "—"],
    ]
    add_styled_table(
        slide,
        left=6.78,
        top=1.88,
        col_widths=[1.75, 1.32, 1.32, 1.36],
        headers=t2_headers,
        rows=t2_rows,
        font_size=9.8,
        row_height=0.35,
        header_height=0.35,
    )

    add_titled_bullet_card(
        slide,
        left=6.78,
        top=4.18,
        width=5.75,
        height=2.22,
        title="Key Diagnostic Takeaways",
        items=[
            ("CLT O(N^{-1/2}) Scaling:", "Empirical Std halves from N=250 to 1,000 (7.56 → 3.66); ESS stays ~40% of N."),
            ("Std vs. SE:", "Paper's 0.65 is a multi-run jackknife SE (12 runs), matching our γ=1.0 SE (1.32 / 1.12)."),
            ("Step Resampling (γ=1.0):", "Halves single-run Std (9.11 → 4.16) under 10D COVID crash shocks."),
        ],
        title_size=13.8,
        font_size=11.8,
        space_after=7.5,
    )


def build_slide_11_fig2_volatilities(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header_footer(
        slide,
        "Empirical Results: Return Volatilities (Fig. 2)",
        "Daily Conditional Standard Deviations with 90% Particle Credibility Bands",
        11,
    )

    add_picture_fit(
        slide,
        PLOTS_DIR / "fig2_reproduction_volatilities.png",
        left=SLIDE_MARGIN_X,
        top=1.50,
        max_w=6.15,
        max_h=5.08,
    )

    add_titled_bullet_card(
        slide,
        left=7.18,
        top=CONTENT_TOP,
        width=5.35,
        height=3.74,
        title="Volatility Findings (Eq. 6, N = 1,000)",
        items=[
            (
                "S&P 500 (SPX):",
                "Calm baseline (~0.7%–1.5%) spiking sharply near ~3.0% during March 2020.",
            ),
            (
                "Spot Gold:",
                "Most stable asset (~0.7%–1.1% baseline), peaking milder near ~2.6%.",
            ),
            (
                "Bitcoin & Ethereum:",
                "High-volatility regimes (BTC ~2%–12%, ETH ~3%–16%) with recurrent bursts.",
            ),
            (
                "Calibrated Uncertainty:",
                "90% credibility bands widen during market stress and reproduce Fig. 2 cleanly.",
            ),
        ],
        title_size=14.5,
        font_size=12.6,
        space_after=9.0,
    )

    add_pipeline_box(
        slide,
        left=7.18,
        top=5.44,
        width=5.35,
        height=1.10,
        title="Empirical Volatility Hierarchy (2018–2023)",
        subtitle="Gold (0.93% std) < SPX (1.37%, extreme COVID tail)  <<  Bitcoin (4.30%) < Ethereum (5.69%)",
        dark=True,
    )


def build_slide_12_fig3_spx_corr(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header_footer(
        slide,
        "Empirical Results: SPX Correlations (Fig. 3)",
        "Dynamic Correlations Between S&P 500 and Safe-Haven Candidates",
        12,
    )

    add_picture_fit(
        slide,
        PLOTS_DIR / "fig3_reproduction_spx_correlations.png",
        left=SLIDE_MARGIN_X,
        top=1.52,
        max_w=6.80,
        max_h=4.92,
    )

    add_titled_bullet_card(
        slide,
        left=7.82,
        top=CONTENT_TOP,
        width=4.71,
        height=3.30,
        title="SPX Pairwise Correlations (Eq. 7)",
        items=[
            (
                "Strictly Positive:",
                "SPX–Gold (~0.03–0.13), SPX–BTC (~0.02–0.21), and SPX–ETH (~0.01–0.14) never turn negative.",
            ),
            (
                "COVID-19 Stress Peak:",
                "All three correlations peak during the March 2020 equity sell-off.",
            ),
            (
                "Dynamic Scale:",
                "Y-axes adapt cleanly around zero to expose regime shifts.",
            ),
        ],
        title_size=14.2,
        font_size=12.4,
        space_after=9.5,
    )

    add_pipeline_box(
        slide,
        left=7.82,
        top=5.02,
        width=4.71,
        height=1.42,
        title="Safe-Haven Verdict (Baur & McDermott, 2010)",
        subtitle="None of Gold, BTC, or ETH is a strong safe haven. Gold offers the best diversifier profile (lowest vol & crisis co-movement).",
        dark=True,
    )


def build_slide_13_fig4_candidate_corr(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header_footer(
        slide,
        "Empirical Results: Cross-Correlations (Fig. 4)",
        "Joint 4D Dynamics Among Gold, Bitcoin, and Ethereum",
        13,
    )

    add_picture_fit(
        slide,
        PLOTS_DIR / "fig4_reproduction_candidate_correlations.png",
        left=SLIDE_MARGIN_X,
        top=1.52,
        max_w=6.80,
        max_h=4.92,
    )

    add_titled_bullet_card(
        slide,
        left=7.82,
        top=CONTENT_TOP,
        width=4.71,
        height=3.44,
        title="Joint 4D Candidate Dynamics",
        items=[
            (
                "4D Joint Advantage:",
                "Captures cross-candidate co-movement missed by prior 2D studies.",
            ),
            (
                "Gold vs. Crypto:",
                "Low-to-moderate positive correlation, briefly dipping negative in 2018, 2019, and 2022.",
            ),
            (
                "BTC vs. ETH:",
                "Strong, noisy co-movement (ρ_{34,t} ≈ 0.40–0.95) with limited mutual diversification.",
            ),
        ],
        title_size=14.2,
        font_size=12.4,
        space_after=9.5,
    )

    add_rounded_card(slide, left=7.82, top=5.16, width=4.71, height=1.28, fill_rgb=CARD_BG_ALT, border_rgb=CYAN, border_pt=1.2)
    add_textbox(
        slide,
        left=7.82 + PILL_PAD_X,
        top=5.16 + PILL_PAD_Y,
        width=4.71 - 2 * PILL_PAD_X,
        height=1.28 - 2 * PILL_PAD_Y,
        text="Portfolio Insight: Gold provides cross-asset hedging stability, whereas BTC and ETH largely reflect one shared crypto risk factor.",
        font_size=12.2,
        color=NAVY,
        valign=MSO_ANCHOR.MIDDLE,
    )


def build_slide_14_if2_setup(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header_footer(
        slide,
        "Iterated Filtering (IF2): Algorithm Setup",
        "Frequentist Maximum Likelihood Estimation via Perturbed Particle Filtering",
        14,
    )

    bullets = [
        (
            "Plug-and-Play MLE:",
            "Estimates all 30 ChMSV parameters via forward simulation and density evaluation—no MCMC or priors required.",
        ),
        (
            "Random-Walk Perturbations:",
            "Augments parameters with time-varying perturbations (initial σ_{0} = 0.01) to smooth the particle likelihood.",
        ),
        (
            "Geometric Cooling Schedule:",
            "Iterates M = 200 filter passes (N = 1,000) with decay α = 0.5 until parameter variance collapses to the MLE.",
        ),
    ]
    add_bullets(
        slide,
        left=SLIDE_MARGIN_X,
        top=1.64,
        width=6.05,
        height=3.15,
        items=bullets,
        font_size=14.0,
        space_after=14.0,
        lead_color=SLATE_MUTED,
        body_color=TEXT_MUTED,
        bullet_hex="7A8EA3",
    )

    # Muted compiled IF2 perturbation/cooling equation box
    add_rounded_card(slide, left=SLIDE_MARGIN_X, top=4.98, width=6.05, height=1.60, fill_rgb=CARD_BG_TBD, border_rgb=BORDER_TBD, dashed=True)
    add_textbox(
        slide,
        left=SLIDE_MARGIN_X + CARD_PAD_X,
        top=4.98 + CARD_PAD_TOP,
        width=6.05 - 2 * CARD_PAD_X,
        height=0.30,
        text="Parameter Perturbation & Cooling Rule (Ionides et al., 2015)",
        font_size=12.5,
        bold=True,
        color=SLATE_MUTED,
        align=PP_ALIGN.CENTER,
    )
    add_picture_fit(
        slide,
        ASSETS_DIR / "eq_if2_cool.png",
        left=SLIDE_MARGIN_X + CARD_PAD_X,
        top=5.56,
        max_w=6.05 - 2 * CARD_PAD_X,
        max_h=0.80,
    )

    # Right muted pipeline card
    cx, cy, cw, ch = 7.15, CONTENT_TOP, 5.38, 5.04
    add_rounded_card(slide, cx, cy, cw, ch, fill_rgb=CARD_BG_TBD, border_rgb=BORDER_TBD, dashed=True)
    ix = cx + CARD_PAD_X
    iw = cw - 2 * CARD_PAD_X
    add_textbox(
        slide,
        left=ix,
        top=cy + CARD_PAD_TOP,
        width=iw,
        height=CARD_TITLE_H,
        text="IF2 Pipeline (src/estimation.py)",
        font_size=14.5,
        bold=True,
        color=SLATE_MUTED,
        align=PP_ALIGN.CENTER,
        valign=MSO_ANCHOR.MIDDLE,
    )

    steps = [
        ("1. Perturbed Parameter Cloud", "Initialize θ_{0}^{n} with random-walk scale σ_{0} = 0.01"),
        ("2. Bootstrap Particle Filter Pass", "Propagate (X_{t}^{n}, θ_{t}^{n}) across t = 1..T and resample"),
        ("3. Geometric Cooling (m = 1..200)", "Shrink perturbation std via σ_{m} = σ_{0} α^{(m-1)/M}"),
    ]
    y_curr = cy + CARD_PAD_TOP + CARD_TITLE_H + CARD_TITLE_GAP
    for st_title, st_sub in steps:
        add_pipeline_box(slide, left=ix, top=y_curr, width=iw, height=0.74, title=st_title, subtitle=st_sub, muted=True)
        add_textbox(
            slide,
            left=ix,
            top=y_curr + 0.74,
            width=iw,
            height=0.26,
            text="↓",
            font_size=15.0,
            bold=True,
            color=SLATE_MUTED,
            align=PP_ALIGN.CENTER,
            valign=MSO_ANCHOR.MIDDLE,
        )
        y_curr += 1.00

    add_pipeline_box(
        slide,
        left=ix,
        top=y_curr,
        width=iw,
        height=(cy + ch - CARD_PAD_BOTTOM) - y_curr,
        title="4. Converged MLE & Numerical Hessian SEs",
        subtitle="Table 3 synthetic recovery  |  Table 2 real 4-asset fit",
        dark=True,
        muted=True,
    )


def build_slide_15_if2_results(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header_footer(
        slide,
        "Iterated Filtering (IF2): Recovery & Real Fit",
        "Synthetic Parameter Recovery (Table 3) and Empirical 4-Asset Fit (Table 2)",
        15,
    )

    cx, cy, cw, ch = SLIDE_MARGIN_X, CONTENT_TOP, 5.55, 3.16
    add_rounded_card(slide, cx, cy, cw, ch, fill_rgb=CARD_BG_TBD, border_rgb=BORDER_TBD, dashed=True)
    ix = cx + CARD_PAD_X
    iw = cw - 2 * CARD_PAD_X
    add_textbox(
        slide,
        left=ix,
        top=cy + CARD_PAD_TOP,
        width=iw,
        height=CARD_TITLE_H,
        text="Table 3: Simulated Parameter Recovery (10+ Runs)",
        font_size=13.8,
        bold=True,
        color=SLATE_MUTED,
        valign=MSO_ANCHOR.MIDDLE,
    )
    t3_headers = ["Param Group", "Mean Error (ME)", "RMSE", "Paper ME Range"]
    t3_rows = [
        ["Log-Vol (μ_{i})", "—", "—", "-0.23 .. 0.17"],
        ["Log-Vol (φ_{i})", "—", "—", "-0.02 .. 0.08"],
        ["Log-Vol (σ_{ηi})", "—", "—", "-0.05 .. 0.10"],
        ["Cholesky (μ, φ, σ)", "—", "—", "-0.19 .. 0.16"],
    ]
    add_styled_table(
        slide,
        left=ix,
        top=cy + CARD_PAD_TOP + CARD_TITLE_H + CARD_TITLE_GAP,
        col_widths=[1.58, 1.15, 0.90, 1.44],
        headers=t3_headers,
        rows=t3_rows,
        font_size=10.2,
        row_height=0.42,
        header_height=0.38,
        col_aligns=[PP_ALIGN.LEFT, PP_ALIGN.CENTER, PP_ALIGN.CENTER, PP_ALIGN.RIGHT],
        muted=True,
    )

    add_titled_bullet_card(
        slide,
        left=SLIDE_MARGIN_X,
        top=4.88,
        width=5.55,
        height=1.70,
        title="Real 4-Asset IF2 Estimates vs. Table 2",
        items=[
            ("Estimated Log-Likelihood:", "— (Paper Table 2: -9972.73 ± 0.65)."),
            ("Parameter Concordance:", "Numerical Hessian SEs and comparison across 30 parameters."),
        ],
        title_size=13.5,
        font_size=12.0,
        space_after=7.0,
        muted=True,
    )

    add_muted_plot_frame(
        slide,
        left=6.58,
        top=CONTENT_TOP,
        width=5.95,
        height=3.16,
        title="IF2 Convergence Trajectories & Log-Likelihood Trace",
        details="200-iteration cooling trajectory curves across parameter families and log-likelihood progression toward the MLE.",
    )

    add_titled_bullet_card(
        slide,
        left=6.58,
        top=4.88,
        width=5.95,
        height=1.70,
        title="Bias & Cooling Sensitivity",
        items=[
            ("Long-Term Mean Support:", "Assess ME symmetry and whether unbounded μ_{i} shows largest RMSE."),
            ("Hyperparameter Stability:", "Sensitivity to cooling rate α and initial perturbation scale σ_{0}."),
        ],
        title_size=13.5,
        font_size=12.0,
        space_after=7.0,
        muted=True,
    )


def build_slide_16_bayesian_pmcmc(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header_footer(
        slide,
        "Bayesian Estimation: PMCMC / SMC^{2}",
        "Prior Specification, Multi-Chain Posteriors, and Dimensional Scaling",
        16,
    )

    cx, cy, cw, ch = SLIDE_MARGIN_X, CONTENT_TOP, 5.45, 2.90
    add_rounded_card(slide, cx, cy, cw, ch, fill_rgb=CARD_BG_TBD, border_rgb=BORDER_TBD, dashed=True)
    ix = cx + CARD_PAD_X
    iw = cw - 2 * CARD_PAD_X
    add_textbox(
        slide,
        left=ix,
        top=cy + CARD_PAD_TOP,
        width=iw,
        height=CARD_TITLE_H,
        text="Bayesian Setup (src/estimation.py)",
        font_size=14.0,
        bold=True,
        color=SLATE_MUTED,
        valign=MSO_ANCHOR.MIDDLE,
    )
    add_bullets(
        slide,
        left=ix,
        top=cy + CARD_PAD_TOP + CARD_TITLE_H + CARD_TITLE_GAP,
        width=iw,
        height=1.18,
        items=[
            ("StructDist Priors:", "Weakly informative priors enforcing |φ| < 1, σ > 0."),
            ("PMMH / SMC^{2}:", "Scale from 2-asset (9 params) to 4-asset (30 params) with R-hat checks."),
        ],
        font_size=12.0,
        space_after=6.0,
        lead_color=SLATE_MUTED,
        body_color=TEXT_MUTED,
        bullet_hex="7A8EA3",
    )
    add_picture_fit(slide, ASSETS_DIR / "eq_bayes_post.png", left=ix, top=cy + 2.02, max_w=iw, max_h=0.68)

    cx2, cy2, cw2, ch2 = SLIDE_MARGIN_X, 4.60, 5.45, 1.98
    add_rounded_card(slide, cx2, cy2, cw2, ch2, fill_rgb=CARD_BG_TBD, border_rgb=BORDER_TBD, dashed=True)
    add_textbox(
        slide,
        left=cx2 + CARD_PAD_X,
        top=cy2 + CARD_PAD_TOP,
        width=cw2 - 2 * CARD_PAD_X,
        height=CARD_TITLE_H,
        text="Posterior Parameter Summary",
        font_size=13.8,
        bold=True,
        color=SLATE_MUTED,
        valign=MSO_ANCHOR.MIDDLE,
    )
    p4_headers = ["Dataset", "Sampler", "95% Cred. Int.", "R-hat / ESS"]
    p4_rows = [
        ["Simulated 2-Asset (p=2)", "PMMH / SMC^{2}", "—", "—"],
        ["Real 4-Asset (p=4)", "PMMH / SMC^{2}", "—", "—"],
    ]
    add_styled_table(
        slide,
        left=cx2 + CARD_PAD_X,
        top=cy2 + CARD_PAD_TOP + CARD_TITLE_H + CARD_TITLE_GAP,
        col_widths=[1.72, 1.25, 1.05, 0.95],
        headers=p4_headers,
        rows=p4_rows,
        font_size=10.0,
        row_height=0.36,
        header_height=0.36,
        col_aligns=[PP_ALIGN.LEFT, PP_ALIGN.CENTER, PP_ALIGN.CENTER, PP_ALIGN.CENTER],
        muted=True,
    )

    add_muted_plot_frame(
        slide,
        left=6.48,
        top=CONTENT_TOP,
        width=6.05,
        height=5.04,
        title="PMCMC / SMC^{2} Posterior Marginals & Multi-Chain Traces",
        details="Simulated parameter recovery posteriors (overlaid with Table 2 ground-truth markers) and multi-chain mixing diagnostics across 2-asset and 4-asset runs.",
    )


def build_slide_17_head_to_head(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header_footer(
        slide,
        "Evaluation: IF2 vs. PMCMC / SMC^{2} Comparison",
        "Head-to-Head Frequentist vs. Bayesian Assessment",
        17,
    )

    add_muted_plot_frame(
        slide,
        left=SLIDE_MARGIN_X,
        top=CONTENT_TOP,
        width=6.15,
        height=5.04,
        title="Parameter RMSE, Log-Likelihood & Runtime Comparison",
        details="Side-by-side comparison of IF2 MLE vs. PMCMC / SMC^{2} posterior means across (μ, φ, σ) and wall-clock computational cost.",
    )

    comp_cards = [
        (
            "1. Estimation Accuracy",
            "Compare IF2 MLE (ME / RMSE) against PMCMC / SMC^{2} posterior means on simulated truth and Szczepocki (2023) Table 2.",
        ),
        (
            "2. Uncertainty Quantification",
            "Contrast IF2 numerical Hessian standard errors with finite-sample Bayesian 95% credible intervals across 30 parameters.",
        ),
        (
            "3. Computational Cost & Scaling",
            "Benchmark runtime, particle count N sensitivity, and mixing efficiency from 2-asset (9 params) to 4-asset (30 params).",
        ),
    ]
    for idx, (ctitle, cbody) in enumerate(comp_cards):
        cy = CONTENT_TOP + idx * 1.74
        ch = 1.56
        cx, cw = 7.18, 5.35
        add_rounded_card(slide, cx, cy, cw, ch, fill_rgb=CARD_BG_TBD, border_rgb=BORDER_TBD, dashed=True)
        ix = cx + CARD_PAD_X
        iw = cw - 2 * CARD_PAD_X
        add_textbox(
            slide,
            left=ix,
            top=cy + CARD_PAD_TOP,
            width=iw,
            height=CARD_TITLE_H,
            text=ctitle,
            font_size=14.2,
            bold=True,
            color=SLATE_MUTED,
            valign=MSO_ANCHOR.MIDDLE,
        )
        add_textbox(
            slide,
            left=ix,
            top=cy + CARD_PAD_TOP + CARD_TITLE_H + CARD_TITLE_GAP,
            width=iw,
            height=ch - CARD_PAD_TOP - CARD_TITLE_H - CARD_TITLE_GAP - CARD_PAD_BOTTOM,
            text=cbody,
            font_size=12.2,
            color=TEXT_MUTED,
        )


def build_slide_18_discussion(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header_footer(slide, "Evaluation", "Discussion, Limitations & Next Steps", 18)

    items = [
        (
            "1. Interpretability of Cholesky Latent States",
            "Raw AR(1) parameters (μ_{ij}, φ_{ij}, σ_{ij}) of q_{ij,t} mix variances and covariances; interpretable volatilities and correlations require reconstructing Σ_{t} = L_{t}V_{t}L_{t}^{T} via Eqs. 4–7.",
            False,
        ),
        (
            "2. Weight Degeneracy & Resampling in 10D",
            "During extreme COVID shocks, adaptive resampling (γ = 0.5) drops ESS to ~2.1/1000 and increases log-likelihood Std (9.11); every-step resampling (γ = 1.0) halves variance (Std 4.16).",
            False,
        ),
        (
            "3. Model Specification Boundaries",
            "Current ChMSV assumes symmetric Gaussian innovations without leverage effects (volatility rising faster after negative returns) or heavy-tailed Student-t errors.",
            False,
        ),
        (
            "4. Frequentist IF2 vs. Bayesian PMCMC / SMC^{2} Trade-offs",
            "Comparative synthesis of convergence stability, cooling sensitivity, runtime, and posterior uncertainty across the 30-parameter ChMSV space.",
            True,
        ),
    ]

    y = 1.58
    for title_txt, desc_txt, is_muted in items:
        add_textbox(
            slide,
            left=SLIDE_MARGIN_X,
            top=y,
            width=USABLE_W,
            height=0.36,
            text=title_txt,
            font_size=15.8,
            bold=True,
            color=SLATE_MUTED if is_muted else NAVY,
        )
        add_textbox(
            slide,
            left=1.18,
            top=y + 0.38,
            width=USABLE_W - 0.38,
            height=0.72,
            text=desc_txt,
            font_size=13.8,
            italic=is_muted,
            color=TEXT_MUTED if is_muted else TEXT_DARK,
        )
        y += 1.28


def build_slide_19_conclusion(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header_footer(slide, "Conclusion", None, 19)

    cx, cy, cw, ch = SLIDE_MARGIN_X, 1.62, USABLE_W, 4.82
    add_rounded_card(slide, cx, cy, cw, ch, fill_rgb=CARD_BG, border_rgb=CARD_BORDER)

    pad_x, pad_top = 0.36, 0.28
    ix = cx + pad_x
    iw = cw - 2 * pad_x

    # Render first 2 active bullets + 2 muted upcoming bullets in one coherent list
    tb = slide.shapes.add_textbox(Inches(ix), Inches(cy + pad_top), Inches(iw), Inches(3.22))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = Inches(0.0)

    concl_items = [
        (
            "Phase I (Data & ChMSV Setup):",
            "Aligned 1,258 Stooq trading days matching Table 1 and built vectorized 2D/4D ChMSV StateSpaceModel classes plus synthetic benchmarks.",
            False,
        ),
        (
            "Phase II (Bootstrap PF & Safe-Haven Findings):",
            "Validated tracking RMSE (0.180 SPX vol, 0.026 corr), verified O(N^{-1/2}) CLT scaling, reproduced Table 2 log-likelihood (-9975.17), and confirmed no strong safe haven in Figs. 2–4.",
            False,
        ),
        (
            "Phase III (Iterated Filtering — IF2):",
            "Simulated Table 3 ME/RMSE parameter recovery benchmark and real 4-asset MLE estimates.",
            True,
        ),
        (
            "Phase IV (Bayesian PMCMC / SMC^{2}):",
            "Multi-chain posterior inference with StructDist priors and head-to-head comparison against IF2.",
            True,
        ),
    ]

    for idx, (lead, body, is_muted) in enumerate(concl_items):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.space_after = Pt(11.5 if idx < len(concl_items) - 1 else 0.0)
        p.line_spacing = 1.15
        _enable_bullet_xml(p, char="•", color_hex="7A8EA3" if is_muted else "002752")
        l_col = SLATE_MUTED if is_muted else NAVY
        b_col = TEXT_MUTED if is_muted else TEXT_DARK
        _add_rich_runs(p, lead + " ", font_size=13.8, bold=True, color=l_col)
        _add_rich_runs(p, body, font_size=13.8, bold=False, color=b_col)

    add_hline(slide, left=ix, top=5.14, width=iw, color=CARD_BORDER, weight_pt=1.0)

    tb2 = slide.shapes.add_textbox(Inches(ix), Inches(5.34), Inches(iw), Inches(0.88))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    tf2.margin_left = tf2.margin_right = tf2.margin_top = tf2.margin_bottom = Inches(0.0)
    p2 = tf2.paragraphs[0]
    p2.line_spacing = 1.2
    _add_rich_runs(p2, "Takeaway: ", font_size=14.2, bold=True, color=NAVY)
    _add_rich_runs(
        p2,
        "The ChMSV StateSpaceModel and Bootstrap Particle Filter pipeline are fully validated in Python (particles), providing a solid foundation for IF2 and Bayesian estimation.",
        font_size=14.2,
        color=TEXT_DARK,
    )


def build_slide_20_questions(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header_footer(slide, "Questions?", None, 20)

    members = [
        (
            "GH",
            "Gabriel\nHimmelein",
            "Data & Base ChMSV Model\nsrc/base_model.py\nStooq Alignment & Table 1\nSynthetic Benchmarks",
            False,
        ),
        (
            "MS",
            "Maria-Giulia\nSalvarani",
            "Bootstrap Particle Filter\nsrc/filtering.py\nCLT & Table 2 Verification\nFigures 2–4 & ESS",
            False,
        ),
        (
            "TP",
            "Tomàs Planelles\nAlonso",
            "Iterated Filtering (IF2)\nsrc/estimation.py\nTable 3 ME/RMSE Recovery\nTable 2 Real 4-Asset MLE",
            True,
        ),
        (
            "MM",
            "Matteo\nMola",
            "Bayesian PMCMC / SMC^{2}\nsrc/estimation.py\nStructDist Priors\nPosteriors vs. IF2",
            True,
        ),
    ]

    w = 2.71
    gap = (USABLE_W - 4 * w) / 3.0
    for i, (initials, name, desc, is_muted) in enumerate(members):
        x = SLIDE_MARGIN_X + i * (w + gap)
        y, h = 1.58, 5.00
        add_rounded_card(
            slide,
            left=x,
            top=y,
            width=w,
            height=h,
            fill_rgb=CARD_BG_TBD if is_muted else CARD_BG,
            border_rgb=BORDER_TBD if is_muted else CARD_BORDER,
            dashed=is_muted,
        )

        circ = slide.shapes.add_shape(
            MSO_SHAPE.OVAL,
            Inches(x + (w - 1.18) / 2.0),
            Inches(y + 0.30),
            Inches(1.18),
            Inches(1.18),
        )
        circ.shadow.inherit = False
        circ.fill.solid()
        circ.fill.fore_color.rgb = SLATE_MUTED if is_muted else NAVY
        circ.line.fill.background()
        ctf = circ.text_frame
        ctf.vertical_anchor = MSO_ANCHOR.MIDDLE
        cp = ctf.paragraphs[0]
        cp.alignment = PP_ALIGN.CENTER
        _add_rich_runs(cp, initials, font_size=22.0, bold=True, color=WHITE)

        ix = x + CARD_PAD_X
        iw = w - 2 * CARD_PAD_X
        add_textbox(
            slide,
            left=ix,
            top=y + 1.66,
            width=iw,
            height=0.68,
            text=name,
            font_size=14.8,
            bold=True,
            color=SLATE_MUTED if is_muted else NAVY,
            align=PP_ALIGN.CENTER,
            valign=MSO_ANCHOR.MIDDLE,
        )
        add_hline(slide, left=x + 0.48, top=y + 2.48, width=w - 0.96, color=CARD_BORDER, weight_pt=1.0)
        add_textbox(
            slide,
            left=ix,
            top=y + 2.70,
            width=iw,
            height=1.96,
            text=desc,
            font_size=12.0,
            color=TEXT_MUTED if is_muted else TEXT_DARK,
            align=PP_ALIGN.CENTER,
            line_spacing=1.22,
        )


def main() -> None:
    prs = Presentation()
    prs.slide_width = Inches(SLIDE_W)
    prs.slide_height = Inches(SLIDE_H)

    build_slide_01_title(prs)
    build_slide_02_agenda(prs)
    build_slide_03_use_case(prs)
    build_slide_04_architecture(prs)
    build_slide_05_datasets_profiling(prs)
    build_slide_06_fig1_reproduction(prs)
    build_slide_07_chmsv_model(prs)
    build_slide_08_simulated_benchmarks(prs)
    build_slide_09_bpf_sim_tracking(prs)
    build_slide_10_sensitivity_table2(prs)
    build_slide_11_fig2_volatilities(prs)
    build_slide_12_fig3_spx_corr(prs)
    build_slide_13_fig4_candidate_corr(prs)
    build_slide_14_if2_setup(prs)
    build_slide_15_if2_results(prs)
    build_slide_16_bayesian_pmcmc(prs)
    build_slide_17_head_to_head(prs)
    build_slide_18_discussion(prs)
    build_slide_19_conclusion(prs)
    build_slide_20_questions(prs)

    prs.save(str(OUTPUT_PPTX))
    print(f"Saved {len(prs.slides)}-slide ENSAE presentation to: {OUTPUT_PPTX}")


if __name__ == "__main__":
    main()
