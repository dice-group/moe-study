from __future__ import annotations

from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


HEADER_FILL = PatternFill(fill_type="solid", fgColor="1F4E78")
HEADER_FONT = Font(color="FFFFFF", bold=True)
SECTION_FILL = PatternFill(fill_type="solid", fgColor="305496")
SECTION_FONT = Font(color="FFFFFF", bold=True, size=11)
AVERAGE_FILL = PatternFill(fill_type="solid", fgColor="E2ECF9")
AVERAGE_FONT = Font(color="1F2937", bold=True)


def style_header(ws, header_row: int = 1) -> None:
    for cell in ws[header_row]:
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center")


def apply_row_heatmap(ws, row_idx: int, better: str, start_col: int, end_col: int) -> None:
    has_num = any(isinstance(ws.cell(row_idx, c).value, (int, float)) for c in range(start_col, end_col + 1))
    if not has_num:
        return
    rng = f"{get_column_letter(start_col)}{row_idx}:{get_column_letter(end_col)}{row_idx}"
    if better == "higher":
        rule = ColorScaleRule(
            start_type="min",
            start_color="F8696B",
            mid_type="percentile",
            mid_value=50,
            mid_color="FFEB84",
            end_type="max",
            end_color="63BE7B",
        )
    else:
        rule = ColorScaleRule(
            start_type="min",
            start_color="63BE7B",
            mid_type="percentile",
            mid_value=50,
            mid_color="FFEB84",
            end_type="max",
            end_color="F8696B",
        )
    ws.conditional_formatting.add(rng, rule)
