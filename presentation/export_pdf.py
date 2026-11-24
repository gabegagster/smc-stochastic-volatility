#!/usr/bin/env python3
"""Export `smc_chmsv_ensae_presentation.pptx` to a multi-page vector PDF in authentic Calibri."""

from __future__ import annotations

import base64
import ctypes
import html
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import cairosvg
from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn

PRES_DIR = Path(__file__).resolve().parent
DEFAULT_PPTX = PRES_DIR / "smc_chmsv_ensae_presentation.pptx"
DEFAULT_PDF = PRES_DIR / "smc_chmsv_ensae_presentation.pdf"

EMU_PER_PT = 12700.0

CALIBRI_DIR = Path("/Applications/Microsoft PowerPoint.app/Contents/Resources/DFonts")
FONT_REG = str(CALIBRI_DIR / "Calibri.ttf")
FONT_BOLD = str(CALIBRI_DIR / "Calibrib.ttf")
FONT_ITAL = str(CALIBRI_DIR / "Calibrii.ttf")
FONT_BI = str(CALIBRI_DIR / "Calibriz.ttf")

_DUMMY_DRAW = ImageDraw.Draw(Image.new("RGB", (10, 10)))
_FONT_CACHE = {}


def register_calibri_fonts() -> None:
    """Register Microsoft Calibri TTFs at process scope via CoreText for CairoSVG."""
    cf = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
    ct = ctypes.CDLL("/System/Library/Frameworks/CoreText.framework/CoreText")
    cf.CFURLCreateFromFileSystemRepresentation.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_long,
        ctypes.c_bool,
    ]
    cf.CFURLCreateFromFileSystemRepresentation.restype = ctypes.c_void_p
    ct.CTFontManagerRegisterFontsForURL.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p]
    ct.CTFontManagerRegisterFontsForURL.restype = ctypes.c_bool

    for fpath in (FONT_REG, FONT_BOLD, FONT_ITAL, FONT_BI):
        bpath = fpath.encode("utf-8")
        url = cf.CFURLCreateFromFileSystemRepresentation(None, bpath, len(bpath), False)
        ct.CTFontManagerRegisterFontsForURL(url, 1, None)


def _get_pil_font(pt: float, bold: bool = False, italic: bool = False) -> ImageFont.FreeTypeFont:
    key = (round(pt, 2), bold, italic)
    if key not in _FONT_CACHE:
        path = FONT_BI if (bold and italic) else FONT_BOLD if bold else FONT_ITAL if italic else FONT_REG
        _FONT_CACHE[key] = ImageFont.truetype(path, max(8, int(round(pt * 4.0))))
    return _FONT_CACHE[key]


def _text_width_pt(txt: str, pt: float, bold: bool = False, italic: bool = False) -> float:
    font = _get_pil_font(pt, bold=bold, italic=italic)
    return _DUMMY_DRAW.textlength(txt, font=font) / 4.0


def _rgb_hex(rgb_color, default: str = "#1E293B") -> str:
    if rgb_color is None:
        return default
    return f"#{rgb_color[0]:02X}{rgb_color[1]:02X}{rgb_color[2]:02X}"


def _fill_hex(shape) -> Optional[str]:
    try:
        if shape.fill.type == 1:
            return _rgb_hex(shape.fill.fore_color.rgb, None)
    except Exception:
        pass
    return None


def _line_spec(shape) -> Tuple[Optional[str], float, bool]:
    try:
        if shape.line.fill.type == 1:
            col = _rgb_hex(shape.line.color.rgb, None)
            w_pt = max(0.5, (shape.line.width or 12700) / EMU_PER_PT)
            ln = shape.line._get_or_add_ln()
            dash_el = ln.find(qn("a:prstDash"))
            is_dashed = dash_el is not None and dash_el.get("val") == "dash"
            return col, w_pt, is_dashed
    except Exception:
        pass
    return None, 0.0, False


def _bullet_color_hex(pPr, default: str = "#002752") -> str:
    if pPr is None:
        return default
    buClr = pPr.find(qn("a:buClr"))
    if buClr is not None:
        srgb = buClr.find(qn("a:srgbClr"))
        if srgb is not None and srgb.get("val"):
            return "#" + srgb.get("val")
    return default


def _run_baseline(r) -> int:
    try:
        rPr = r._r.rPr
        if rPr is not None and rPr.get("baseline") is not None:
            return int(rPr.get("baseline"))
    except Exception:
        pass
    return 0


def _render_text_frame_svg(
    tf,
    x0: float,
    y0: float,
    w: float,
    h: float,
    default_color: str = "#1E293B",
) -> str:
    ml = (tf.margin_left or 0) / EMU_PER_PT
    mr = (tf.margin_right or 0) / EMU_PER_PT
    mt = (tf.margin_top or 0) / EMU_PER_PT
    mb = (tf.margin_bottom or 0) / EMU_PER_PT

    inner_x0 = x0 + ml
    inner_x1 = max(inner_x0 + 8.0, x0 + w - mr)
    inner_w = inner_x1 - inner_x0

    blocks = []
    total_h = 0.0
    for p in tf.paragraphs:
        pPr = p._p.pPr
        has_bu = pPr is not None and pPr.find(qn("a:buChar")) is not None
        bu_col = _bullet_color_hex(pPr)
        indent_pt = 12.0 if has_bu else 0.0

        runs = p.runs
        if not runs:
            blocks.append((p.alignment, has_bu, bu_col, [], 4.0, 1.15))
            total_h += 6.0
            continue

        # Tokenize into space-attached atoms so sub/superscripts stick to their base word
        styled_atoms = []  # (text, eff_pt, base_pt, bold, italic, col, dy_pt, glue_prev)
        prev_was_script = False
        for r in runs:
            base_pt = (r.font.size / EMU_PER_PT) if r.font.size else 12.0
            bl = _run_baseline(r)
            eff_pt = base_pt * 0.70 if bl != 0 else base_pt
            dy_pt = -(bl / 100000.0) * base_pt
            bold = bool(r.font.bold)
            italic = bool(r.font.italic)
            col = default_color
            try:
                if r.font.color and r.font.color.rgb is not None:
                    col = _rgb_hex(r.font.color.rgb, default_color)
            except Exception:
                pass

            parts = r.text.split("\n")
            for pi, part in enumerate(parts):
                if pi > 0:
                    styled_atoms.append(("\n", eff_pt, base_pt, bold, italic, col, 0.0, False))
                    prev_was_script = False
                    continue
                if bl != 0:
                    # Script token stays glued to preceding word
                    styled_atoms.append((part, eff_pt, base_pt, bold, italic, col, dy_pt, True))
                    prev_was_script = True
                else:
                    # Split on spaces while preserving exact leading/trailing spaces when adjacent to scripts
                    words = part.split(" ")
                    for wi, word in enumerate(words):
                        glue = (wi == 0 and prev_was_script and word != "")
                        if wi > 0:
                            styled_atoms.append((" ", eff_pt, base_pt, bold, italic, col, 0.0, False))
                        if word:
                            styled_atoms.append((word, eff_pt, base_pt, bold, italic, col, 0.0, glue))
                    prev_was_script = False

        # Group glued atoms into wrap units (each wrap unit won't break mid-token, e.g. Σ_t)
        units = []  # list of list of atoms
        for atom in styled_atoms:
            if atom[0] == "\n":
                units.append([atom])
            elif atom[0] == " ":
                units.append([atom])
            elif atom[7] and units and units[-1][0][0] not in ("\n", " "):
                units[-1].append(atom)
            else:
                units.append([atom])

        lines = []
        curr_line = []
        curr_w = 0.0
        avail_w = max(12.0, inner_w - indent_pt)
        for u in units:
            if u[0][0] == "\n":
                lines.append(curr_line)
                curr_line = []
                curr_w = 0.0
                continue
            if u[0][0] == " ":
                if not curr_line:
                    continue
                sw = _text_width_pt(" ", u[0][1], bold=u[0][3], italic=u[0][4])
                curr_line.append(u[0])
                curr_w += sw
                continue

            uw = sum(_text_width_pt(a[0], a[1], bold=a[3], italic=a[4]) for a in u)
            if curr_w + uw <= avail_w or not curr_line:
                curr_line.extend(u)
                curr_w += uw
            else:
                # Strip trailing space on wrapped line
                while curr_line and curr_line[-1][0] == " ":
                    curr_line.pop()
                lines.append(curr_line)
                curr_line = list(u)
                curr_w = uw
        while curr_line and curr_line[-1][0] == " ":
            curr_line.pop()
        if curr_line:
            lines.append(curr_line)

        space_after = (p.space_after or 0) / EMU_PER_PT
        line_mult = p.line_spacing if isinstance(p.line_spacing, float) else 1.15
        block_h = 0.0
        for line in lines:
            max_base_pt = max((seg[2] for seg in line), default=12.0)
            block_h += max_base_pt * line_mult
        total_h += block_h + space_after
        blocks.append((p.alignment, has_bu, bu_col, lines, space_after, line_mult))

    avail_h = max(1.0, (y0 + h - mb) - (y0 + mt))
    if tf.vertical_anchor == MSO_ANCHOR.MIDDLE:
        cy = y0 + mt + max(0.0, (avail_h - total_h) / 2.0)
    elif tf.vertical_anchor == MSO_ANCHOR.BOTTOM:
        cy = max(y0 + mt, y0 + h - mb - total_h)
    else:
        cy = y0 + mt

    out: List[str] = []
    for align, has_bu, bu_col, lines, space_after, line_mult in blocks:
        if not lines:
            cy += space_after
            continue
        indent_pt = 12.0 if has_bu else 0.0
        for li, line in enumerate(lines):
            max_base_pt = max((seg[2] for seg in line), default=12.0)
            lh = max_base_pt * line_mult
            baseline_y = cy + max_base_pt * 0.86

            # Merge adjacent atoms sharing identical style (eff_pt, bold, italic, col, dy_pt)
            merged_runs = []  # (text, eff_pt, bold, italic, col, dy_pt)
            for txt_piece, eff_pt, _, bold, italic, col, dy_pt, _ in line:
                style = (round(eff_pt, 2), bold, italic, col, round(dy_pt, 2))
                if merged_runs and merged_runs[-1][1] == style:
                    merged_runs[-1][0].append(txt_piece)
                else:
                    merged_runs.append(([txt_piece], style))

            line_w = sum(
                _text_width_pt("".join(chunks), st[0], bold=st[1], italic=st[2])
                for chunks, st in merged_runs
            )
            if align == PP_ALIGN.CENTER:
                cx = inner_x0 + indent_pt + max(0.0, (inner_w - indent_pt - line_w) / 2.0)
            elif align == PP_ALIGN.RIGHT:
                cx = inner_x1 - line_w
            else:
                cx = inner_x0 + indent_pt

            if has_bu and li == 0 and line:
                bpt = max_base_pt
                out.append(
                    f'<text x="{inner_x0 + 1.5:.2f}" y="{baseline_y:.2f}" '
                    f'font-family="Calibri, sans-serif" font-size="{bpt:.2f}" '
                    f'fill="{bu_col}">&#8226;</text>'
                )

            # Render runs; keep exact x advancement via merged <tspan> runs
            tspans = []
            for chunks, (eff_pt, bold, italic, col, dy_pt) in merged_runs:
                seg_txt = "".join(chunks)
                fw = "700" if bold else "400"
                fs = "italic" if italic else "normal"
                y_attr = f' y="{baseline_y + dy_pt:.2f}"' if abs(dy_pt) > 0.05 else f' y="{baseline_y:.2f}"'
                tspans.append(
                    f'<tspan{y_attr} font-size="{eff_pt:.2f}" font-weight="{fw}" font-style="{fs}" fill="{col}">'
                    f'{html.escape(seg_txt)}</tspan>'
                )
            out.append(
                f'<text x="{cx:.2f}" y="{baseline_y:.2f}" font-family="Calibri, sans-serif" '
                f'xml:space="preserve">{"".join(tspans)}</text>'
            )
            cy += lh
        cy += space_after

    return "\n".join(out)


def slide_to_svg(slide, width_pt: float, height_pt: float) -> str:
    parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{width_pt:.3f}pt" height="{height_pt:.3f}pt" viewBox="0 0 {width_pt:.3f} {height_pt:.3f}">',
        f'<rect x="0" y="0" width="{width_pt:.3f}" height="{height_pt:.3f}" fill="#FFFFFF"/>',
    ]

    for sh in slide.shapes:
        x = sh.left / EMU_PER_PT
        y = sh.top / EMU_PER_PT
        w = sh.width / EMU_PER_PT
        h = sh.height / EMU_PER_PT

        if sh.shape_type == MSO_SHAPE_TYPE.PICTURE:
            mime = sh.image.content_type or "image/png"
            b64 = base64.b64encode(sh.image.blob).decode("ascii")
            parts.append(
                f'<image x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
                f'preserveAspectRatio="none" href="data:{mime};base64,{b64}"/>'
            )

        elif sh.shape_type == MSO_SHAPE_TYPE.TABLE:
            tbl = sh.table
            cy = y
            for r_i, row in enumerate(tbl.rows):
                rh = row.height / EMU_PER_PT
                cx = x
                for c_i, col in enumerate(tbl.columns):
                    cw = col.width / EMU_PER_PT
                    cell = tbl.cell(r_i, c_i)
                    bg = "#FFFFFF"
                    try:
                        if cell.fill.type == 1:
                            bg = _rgb_hex(cell.fill.fore_color.rgb, bg)
                    except Exception:
                        pass
                    parts.append(
                        f'<rect x="{cx:.2f}" y="{cy:.2f}" width="{cw:.2f}" height="{rh:.2f}" '
                        f'fill="{bg}" stroke="#D6DFE8" stroke-width="0.6"/>'
                    )
                    if cell.text_frame:
                        parts.append(_render_text_frame_svg(cell.text_frame, cx, cy, cw, rh))
                    cx += cw
                cy += rh

        elif sh.shape_type == MSO_SHAPE_TYPE.AUTO_SHAPE:
            fill_col = _fill_hex(sh) or "none"
            line_col, line_w, is_dashed = _line_spec(sh)
            stroke_attr = f'stroke="{line_col}" stroke-width="{line_w:.2f}"' if line_col else 'stroke="none"'
            dash_attr = ' stroke-dasharray="4,3"' if is_dashed else ""
            st_name = str(sh.auto_shape_type)

            if "OVAL" in st_name:
                rx = w / 2.0
                ry = h / 2.0
                parts.append(
                    f'<ellipse cx="{x + rx:.2f}" cy="{y + ry:.2f}" rx="{rx:.2f}" ry="{ry:.2f}" '
                    f'fill="{fill_col}" {stroke_attr}{dash_attr}/>'
                )
            elif "ROUNDED_RECTANGLE" in st_name:
                adj = sh.adjustments[0] if len(sh.adjustments) > 0 else 0.04
                rad = min(w, h) * adj
                parts.append(
                    f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
                    f'rx="{rad:.2f}" ry="{rad:.2f}" fill="{fill_col}" {stroke_attr}{dash_attr}/>'
                )
            else:
                parts.append(
                    f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
                    f'fill="{fill_col}" {stroke_attr}{dash_attr}/>'
                )

            if sh.has_text_frame and sh.text_frame.text.strip():
                parts.append(_render_text_frame_svg(sh.text_frame, x, y, w, h))

        elif sh.has_text_frame:
            parts.append(_render_text_frame_svg(sh.text_frame, x, y, w, h))

    parts.append("</svg>")
    return "\n".join(parts)


def export_pptx_to_pdf(pptx_path: Path = DEFAULT_PPTX, pdf_path: Path = DEFAULT_PDF) -> None:
    register_calibri_fonts()
    prs = Presentation(str(pptx_path))
    width_pt = prs.slide_width / EMU_PER_PT
    height_pt = prs.slide_height / EMU_PER_PT

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        page_pdfs: List[str] = []
        for idx, slide in enumerate(prs.slides, start=1):
            svg_str = slide_to_svg(slide, width_pt, height_pt)
            page_pdf = tmp / f"slide_{idx:02d}.pdf"
            cairosvg.svg2pdf(bytestring=svg_str.encode("utf-8"), write_to=str(page_pdf))
            page_pdfs.append(str(page_pdf))

        cmd = [
            "gs",
            "-dBATCH",
            "-dNOPAUSE",
            "-q",
            "-sDEVICE=pdfwrite",
            f"-sOutputFile={pdf_path}",
            *page_pdfs,
        ]
        subprocess.run(cmd, check=True)

    print(f"Exported {len(prs.slides)}-page Calibri vector PDF to: {pdf_path}")


if __name__ == "__main__":
    export_pptx_to_pdf()
