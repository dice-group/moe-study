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
SUCCESS_FILL = PatternFill(fill_type="solid", fgColor="C6EFCE")
SUCCESS_FONT = Font(color="006100", bold=True)
MUTED_FILL = PatternFill(fill_type="solid", fgColor="F3F4F6")
BEST_FILL = PatternFill(fill_type="solid", fgColor="E2F0D9")
BEST_FONT = Font(color="1F2937", bold=True)


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


def apply_pvalue_heatmap(ws, start_row: int, end_row: int, col_idx: int) -> None:
    if end_row < start_row:
        return
    rng = f"{get_column_letter(col_idx)}{start_row}:{get_column_letter(col_idx)}{end_row}"
    rule = ColorScaleRule(
        start_type="num",
        start_value=0,
        start_color="63BE7B",
        mid_type="num",
        mid_value=0.05,
        mid_color="FFEB84",
        end_type="num",
        end_value=0.25,
        end_color="F8696B",
    )
    ws.conditional_formatting.add(rng, rule)


def apply_signed_effect_heatmap(ws, start_row: int, end_row: int, col_idx: int) -> None:
    if end_row < start_row:
        return
    rng = f"{get_column_letter(col_idx)}{start_row}:{get_column_letter(col_idx)}{end_row}"
    rule = ColorScaleRule(
        start_type="min",
        start_color="F8696B",
        mid_type="num",
        mid_value=0,
        mid_color="FFFFFF",
        end_type="max",
        end_color="63BE7B",
    )
    ws.conditional_formatting.add(rng, rule)


def apply_positive_magnitude_heatmap(ws, start_row: int, end_row: int, col_idx: int) -> None:
    if end_row < start_row:
        return
    rng = f"{get_column_letter(col_idx)}{start_row}:{get_column_letter(col_idx)}{end_row}"
    rule = ColorScaleRule(
        start_type="num",
        start_value=0,
        start_color="FFFFFF",
        mid_type="percentile",
        mid_value=50,
        mid_color="D9EAD3",
        end_type="max",
        end_color="63BE7B",
    )
    ws.conditional_formatting.add(rng, rule)


def style_significance_cell(cell) -> None:
    value = str(cell.value or "").strip().lower()
    if value == "yes":
        cell.fill = SUCCESS_FILL
        cell.font = SUCCESS_FONT
    elif value == "no":
        cell.fill = MUTED_FILL


def style_best_cell(cell) -> None:
    cell.fill = BEST_FILL
    cell.font = BEST_FONT
