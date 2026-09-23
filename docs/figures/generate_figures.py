"""Generate the publication figures from frozen D1-D5 CSV artifacts only.

This script uses only the Python standard library.  It does not train a model,
run inference, regenerate predictions, or read any F1/2025 artifact.  Before
rendering, it checks the content-addressed source filenames, required schemas,
row counts, and the headline counts used in the figures.

Run from the repository root:

    python docs/figures/generate_figures.py
"""

from __future__ import annotations

import csv
import hashlib
import html
import math
import re
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

MACRO = ROOT / "results/combined/development_macro_fold_comparison_042b08dfe55c6449331f1ee5b9e8437a8540505c5061f61648d66c8d588203f6.csv"
ASSET = ROOT / "results/combined/development_asset_fold_comparison_61ad31f5e745338ccf679be9614188bab869964b350ab4f32a26067f85a2f3a1.csv"
REGIME = ROOT / "results/regime/development/asset_fold_regime_degradation_278dc64486ce255773a578bf6c285d7831e4f846c156ffab2a08ca99d45495eb.csv"
REPORTABILITY = ROOT / "results/regime/development/regime_failure_pattern_summary_68ba67ebb6f5f5ec2e801dee99d0345e9756d5b02c15613800e9c85d1d50391f.csv"

FOLDS = ["D1", "D2", "D3", "D4", "D5"]
YEARS = {"D1": 2020, "D2": 2021, "D3": 2022, "D4": 2023, "D5": 2024}
ASSETS = ["AAPL", "MSFT", "GOOGL", "NVDA"]
DIMENSIONS = ["trend", "volatility", "transition"]
MODELS = ["lightgbm", "lstm_seed_mean"]
MODEL_LABELS = {"lightgbm": "LightGBM", "lstm_seed_mean": "LSTM (3-seed mean)"}

INK = "#172033"
MUTED = "#5D6678"
GRID = "#D9DEE8"
LIGHT_GRID = "#EDF0F5"
BLUE = "#0072B2"
VERMILION = "#D55E00"
GREEN = "#009E73"
GRAY = "#AEB6C4"
WHITE = "#FFFFFF"


def _load(path: Path, expected_rows: int, required: set[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    match = re.search(r"_([0-9a-f]{64})\.csv$", path.name)
    if not match:
        raise ValueError(f"Source is not content-addressed: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != match.group(1):
        raise ValueError(f"Content hash mismatch for {path.name}: {digest}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows in {path.name}; found {len(rows)}")
    columns = set(rows[0]) if rows else set()
    missing = required - columns
    if missing:
        raise ValueError(f"Missing columns in {path.name}: {sorted(missing)}")
    return rows


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


class SVG:
    def __init__(self, width: int, height: int, title: str, desc: str):
        self.width = width
        self.height = height
        self.parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="title desc" viewBox="0 0 {width} {height}">',
            f"<title id=\"title\">{_esc(title)}</title>",
            f"<desc id=\"desc\">{_esc(desc)}</desc>",
            "<style>",
            "text{font-family:Inter,Segoe UI,Arial,sans-serif;fill:#172033}",
            ".title{font-size:27px;font-weight:700}.subtitle{font-size:15px;fill:#5D6678}",
            ".axis{font-size:13px;fill:#5D6678}.label{font-size:14px}.small{font-size:12px;fill:#5D6678}",
            ".value{font-size:12px;font-weight:650}.panel{font-size:17px;font-weight:700}",
            "</style>",
            f'<rect width="{width}" height="{height}" fill="#FFFFFF"/>',
        ]

    def add(self, value: str) -> None:
        self.parts.append(value)

    def text(
        self,
        x: float,
        y: float,
        value: object,
        cls: str = "label",
        anchor: str = "start",
        fill: str | None = None,
        weight: int | None = None,
        rotate: float | None = None,
    ) -> None:
        attrs = [f'x="{x:.2f}"', f'y="{y:.2f}"', f'class="{cls}"', f'text-anchor="{anchor}"']
        if fill:
            attrs.append(f'fill="{fill}"')
            attrs.append(f'style="fill:{fill}"')
        if weight:
            attrs.append(f'font-weight="{weight}"')
        if rotate is not None:
            attrs.append(f'transform="rotate({rotate:.2f} {x:.2f} {y:.2f})"')
        self.add(f"<text {' '.join(attrs)}>{_esc(value)}</text>")

    def line(self, x1: float, y1: float, x2: float, y2: float, stroke: str, width: float = 1, dash: str | None = None) -> None:
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{stroke}" stroke-width="{width}"{extra}/>' )

    def rect(self, x: float, y: float, w: float, h: float, fill: str, stroke: str | None = None, rx: float = 0, opacity: float = 1) -> None:
        border = f' stroke="{stroke}"' if stroke else ""
        self.add(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" rx="{rx:.2f}" fill="{fill}" opacity="{opacity:.3f}"{border}/>' )

    def circle(self, cx: float, cy: float, r: float, fill: str, opacity: float = 1, stroke: str = WHITE) -> None:
        self.add(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" fill="{fill}" opacity="{opacity:.3f}" stroke="{stroke}" stroke-width="1"/>')

    def diamond(self, cx: float, cy: float, size: float, fill: str) -> None:
        points = f"{cx:.2f},{cy-size:.2f} {cx+size:.2f},{cy:.2f} {cx:.2f},{cy+size:.2f} {cx-size:.2f},{cy:.2f}"
        self.add(f'<polygon points="{points}" fill="{fill}" stroke="#FFFFFF" stroke-width="1.5"/>')

    def save(self, path: Path) -> None:
        self.parts.append("</svg>")
        path.write_text("\n".join(self.parts) + "\n", encoding="utf-8", newline="\n")


def _mix_with_white(color: str, strength: float) -> str:
    strength = max(0.0, min(1.0, strength))
    rgb = tuple(int(color[i : i + 2], 16) for i in (1, 3, 5))
    mixed = tuple(round(255 + strength * (channel - 255)) for channel in rgb)
    return "#" + "".join(f"{channel:02X}" for channel in mixed)


def _pct(value: float, digits: int = 2) -> str:
    return f"{value * 100:+.{digits}f}%"


def _base_title(svg: SVG, title: str, subtitle: str) -> None:
    svg.text(56, 48, title, "title")
    svg.text(56, 76, subtitle, "subtitle")


def macro_figure(rows: list[dict[str, str]]) -> Path:
    ordered = sorted(rows, key=lambda row: FOLDS.index(row["fold"]))
    svg = SVG(1120, 690, "Macro MAE Skill by development fold", "Grouped bars compare LightGBM and the three-seed LSTM mean for folds D1 through D5. Positive skill favors the learned model over zero-return persistence.")
    _base_title(svg, "Macro MAE Skill by development fold", "Positive values favor the learned model over the zero-return persistence baseline")

    left, right, top, bottom = 118, 1070, 125, 520
    ymin, ymax = -0.08, 0.02
    ymap = lambda value: top + (ymax - value) / (ymax - ymin) * (bottom - top)
    for tick in [-0.08, -0.06, -0.04, -0.02, 0.00, 0.02]:
        y = ymap(tick)
        svg.line(left, y, right, y, INK if tick == 0 else GRID, 2 if tick == 0 else 1)
        svg.text(left - 12, y + 4, f"{tick * 100:.0f}%", "axis", "end")
    svg.text(31, (top + bottom) / 2, "MAE Skill", "axis", "middle", rotate=-90)

    group_w = (right - left) / len(ordered)
    bar_w = 54
    for index, row in enumerate(ordered):
        center = left + group_w * (index + 0.5)
        values = [float(row["lightgbm_macro_mae_skill"]), float(row["lstm_seed_mean_macro_mae_skill"])]
        for offset, (value, color) in zip((-bar_w / 2 - 4, bar_w / 2 + 4), zip(values, (BLUE, VERMILION))):
            x = center + offset - bar_w / 2
            y0, yv = ymap(0), ymap(value)
            svg.rect(x, min(y0, yv), bar_w, abs(yv - y0), color, rx=2)
            label_y = yv - 8 if value >= 0 else yv + 18
            svg.text(x + bar_w / 2, label_y, _pct(value, 2), "value", "middle", color)
        fold = row["fold"]
        svg.text(center, 553, f"{fold} ({YEARS[fold]})", "label", "middle", weight=650)
        svg.text(center, 575, f"n = {int(row['n_observations']):,}", "small", "middle")

    svg.rect(758, 101, 14, 14, BLUE, rx=2)
    svg.text(780, 113, "LightGBM", "small")
    svg.rect(875, 101, 14, 14, VERMILION, rx=2)
    svg.text(897, 113, "LSTM (3-seed metric mean)", "small")
    svg.text(56, 625, "Equal-weight macro across AAPL, MSFT, GOOGL, and NVDA; exactly matched prediction keys.", "small")
    svg.text(56, 647, "Descriptive estimates only: no confidence intervals or significance tests were performed.", "small")
    svg.text(56, 671, f"Source: {MACRO.relative_to(ROOT).as_posix()}", "small")
    target = OUT / "macro_mae_skill_by_fold.svg"
    svg.save(target)
    return target


def heatmap_figure(rows: list[dict[str, str]]) -> Path:
    keyed = {(row["fold"], row["asset"]): row for row in rows}
    if set(keyed) != {(fold, asset) for fold in FOLDS for asset in ASSETS}:
        raise ValueError("Asset/fold table does not contain exactly the expected 20 keys")
    fields = [("LightGBM", "lightgbm_mae_skill"), ("LSTM (3-seed metric mean)", "lstm_seed_mean_mae_skill")]
    maximum = max(abs(float(row[field])) for row in rows for _, field in fields)

    svg = SVG(1120, 735, "Asset/fold MAE Skill", "Two zero-centered heat maps show MAE Skill for four assets and five folds for LightGBM and the three-seed LSTM metric mean.")
    _base_title(svg, "Asset/fold MAE Skill", "Cell labels are percentage points; blue is positive and orange is negative")
    panel_x = [104, 590]
    cell_w, cell_h = 82, 72
    grid_top = 165
    for panel, ((label, field), x0) in enumerate(zip(fields, panel_x)):
        svg.text(x0, 121, label, "panel")
        for column, fold in enumerate(FOLDS):
            svg.text(x0 + column * cell_w + cell_w / 2, 151, f"{fold}", "axis", "middle", weight=650)
        for row_index, asset in enumerate(ASSETS):
            y = grid_top + row_index * cell_h
            svg.text(x0 - 14, y + cell_h / 2 + 5, asset, "axis", "end", weight=650)
            for column, fold in enumerate(FOLDS):
                value = float(keyed[(fold, asset)][field])
                color = BLUE if value >= 0 else VERMILION
                fill = _mix_with_white(color, 0.12 + 0.78 * abs(value) / maximum)
                x = x0 + column * cell_w
                svg.rect(x, y, cell_w - 3, cell_h - 3, fill, WHITE, rx=3)
                text_color = WHITE if abs(value) / maximum > 0.58 else INK
                svg.text(x + (cell_w - 3) / 2, y + cell_h / 2 + 5, _pct(value, 2), "value", "middle", text_color)

    legend_x, legend_y, legend_w = 362, 507, 395
    segments = 100
    for i in range(segments):
        value = -maximum + (2 * maximum) * i / (segments - 1)
        color = VERMILION if value < 0 else BLUE
        fill = _mix_with_white(color, 0.12 + 0.78 * abs(value) / maximum)
        svg.rect(legend_x + legend_w * i / segments, legend_y, legend_w / segments + 0.5, 15, fill)
    svg.line(legend_x + legend_w / 2, legend_y - 2, legend_x + legend_w / 2, legend_y + 18, INK, 1)
    svg.text(legend_x, legend_y + 35, _pct(-maximum, 1), "small", "start")
    svg.text(legend_x + legend_w / 2, legend_y + 35, "0%", "small", "middle")
    svg.text(legend_x + legend_w, legend_y + 35, _pct(maximum, 1), "small", "end")

    lgbm_positive = sum(float(row["lightgbm_mae_skill"]) > 0 for row in rows)
    lstm_negative = sum(float(row["lstm_seed_mean_mae_skill"]) < 0 for row in rows)
    if (lgbm_positive, lstm_negative) != (12, 16):
        raise ValueError(f"Unexpected headline counts: {(lgbm_positive, lstm_negative)}")
    svg.text(56, 592, "LightGBM: positive skill in 12/20 cells.  LSTM: negative skill in 16/20 cells.", "small")
    svg.text(56, 615, "LSTM metrics were computed by seed and then averaged; predictions were not ensembled and no best seed was chosen.", "small")
    svg.text(56, 638, "A shared color scale preserves cross-model comparability; exact labels remain visible for small effects.", "small")
    svg.text(56, 661, "Descriptive results for this frozen protocol; not a universal model ranking.", "small")
    svg.text(56, 697, f"Source: {ASSET.relative_to(ROOT).as_posix()}", "small")
    target = OUT / "asset_fold_mae_skill.svg"
    svg.save(target)
    return target


def regime_figure(rows: list[dict[str, str]]) -> Path:
    reportable = [row for row in rows if row["headline_reportable"] == "True"]
    if len(reportable) != 88:
        raise ValueError(f"Expected 88 reportable asset/fold regime contrasts; found {len(reportable)}")
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in reportable:
        grouped.setdefault((row["model"], row["regime_dimension"]), []).append(float(row["mae_skill_degradation"]))
    expected_counts = {("lightgbm", "trend"): 12, ("lightgbm", "volatility"): 16, ("lightgbm", "transition"): 16,
                       ("lstm_seed_mean", "trend"): 12, ("lstm_seed_mean", "volatility"): 16, ("lstm_seed_mean", "transition"): 16}
    if {key: len(value) for key, value in grouped.items()} != expected_counts:
        raise ValueError("Unexpected reportable regime group sizes")

    svg = SVG(1120, 790, "Regime degradation in development folds", "Reportable stressed-minus-reference changes in MAE Skill are plotted for LightGBM and the LSTM seed mean. Negative values indicate worse skill under stress.")
    _base_title(svg, "Regime degradation in development folds", "Stressed − reference MAE Skill; negative values indicate worse baseline-relative skill under stress")
    panels = [("lightgbm", 92, 528, BLUE), ("lstm_seed_mean", 592, 1028, VERMILION)]
    top, bottom = 145, 585
    ymin, ymax = -0.12, 0.20
    ymap = lambda value: top + (ymax - value) / (ymax - ymin) * (bottom - top)
    ticks = [-0.10, -0.05, 0.00, 0.05, 0.10, 0.15, 0.20]
    for model, left, right, color in panels:
        svg.text(left, 118, MODEL_LABELS[model], "panel")
        for tick in ticks:
            y = ymap(tick)
            svg.line(left, y, right, y, INK if tick == 0 else LIGHT_GRID, 2 if tick == 0 else 1)
            if model == "lightgbm":
                svg.text(left - 12, y + 4, f"{tick * 100:+.0f}%", "axis", "end")
        centers = [left + (right - left) * fraction for fraction in (0.18, 0.50, 0.82)]
        for dimension, center in zip(DIMENSIONS, centers):
            values = sorted(grouped[(model, dimension)])
            count = len(values)
            for idx, value in enumerate(values):
                jitter = (idx - (count - 1) / 2) * min(7.2, 95 / max(1, count - 1))
                svg.circle(center + jitter, ymap(value), 5.3, color, 0.68)
            median = statistics.median(values)
            svg.diamond(center, ymap(median), 8, INK)
            negative = sum(value < -1e-12 for value in values)
            svg.text(center, 614, dimension.title(), "label", "middle", weight=650)
            svg.text(center, 636, f"n={count}; worse {negative}/{count}", "small", "middle")
            svg.text(center, 656, f"median {_pct(median, 2)}", "small", "middle")
    svg.text(30, (top + bottom) / 2, "MAE-Skill degradation", "axis", "middle", rotate=-90)
    svg.diamond(73, 696, 7, INK)
    svg.text(89, 701, "Median", "small")
    svg.circle(176, 696, 5, BLUE, 0.68)
    svg.text(188, 701, "Reportable asset/fold contrast", "small")
    svg.text(56, 729, "Regimes overlap and daily observations are serially dependent; these are descriptive associations, not causal effects.", "small")
    svg.text(56, 751, "Only contrasts meeting the frozen n ≥ 30 rule in both cells are shown; coverage is incomplete.", "small")
    svg.text(56, 775, f"Source: {REGIME.relative_to(ROOT).as_posix()}", "small")
    target = OUT / "regime_mae_skill_degradation.svg"
    svg.save(target)
    return target


def reportability_figure(rows: list[dict[str, str]]) -> Path:
    keyed = {(row["reporting_level"], row["model"], row["regime_dimension"]): row for row in rows}
    expected = {(level, model, dim) for level in ("asset_fold", "macro_fold") for model in MODELS for dim in DIMENSIONS}
    if set(keyed) != expected:
        raise ValueError("Reportability table is missing expected level/model/dimension rows")
    primary_reportable = sum(int(row["reportable_contrasts"]) for row in rows if row["reporting_level"] == "asset_fold")
    macro_reportable = sum(int(row["reportable_contrasts"]) for row in rows if row["reporting_level"] == "macro_fold")
    if (primary_reportable, macro_reportable) != (88, 22):
        raise ValueError(f"Unexpected reportability totals: {(primary_reportable, macro_reportable)}")

    svg = SVG(1120, 900, "Regime-analysis reportability", "Stacked bars show reportable and suppressed asset/fold and macro/fold regime contrasts for two models and three dimensions.")
    _base_title(svg, "Regime-analysis reportability", "Frozen minimum-cell rule: both compared cells must contain at least 30 observations")
    bar_left, bar_width = 355, 650
    panels = [("asset_fold", "Asset/fold contrasts", 126, 20), ("macro_fold", "Macro/fold contrasts", 462, 5)]
    for level, panel_label, y0, total in panels:
        svg.text(56, y0, panel_label, "panel")
        svg.text(56, y0 + 22, f"{total} possible per model × regime dimension", "small")
        row_y = y0 + 51
        for model in MODELS:
            svg.text(74, row_y + 12, MODEL_LABELS[model], "label", weight=650)
            row_y += 25
            for dimension in DIMENSIONS:
                row = keyed[(level, model, dimension)]
                reportable = int(row["reportable_contrasts"])
                suppressed = int(row["descriptive_or_unavailable_contrasts"])
                if reportable + suppressed != total:
                    raise ValueError(f"Counts do not sum for {(level, model, dimension)}")
                svg.text(bar_left - 18, row_y + 15, dimension.title(), "axis", "end")
                reportable_w = bar_width * reportable / total
                suppressed_w = bar_width * suppressed / total
                svg.rect(bar_left, row_y, reportable_w, 20, GREEN, rx=2)
                svg.rect(bar_left + reportable_w, row_y, suppressed_w, 20, GRAY, rx=2)
                svg.text(bar_left + reportable_w / 2, row_y + 15, str(reportable), "value", "middle", WHITE)
                if suppressed:
                    svg.text(bar_left + reportable_w + suppressed_w / 2, row_y + 15, str(suppressed), "value", "middle", INK)
                svg.text(bar_left + bar_width + 12, row_y + 15, f"{reportable}/{total}", "small")
                row_y += 31
            row_y += 14

    svg.rect(355, 794, 15, 15, GREEN, rx=2)
    svg.text(379, 807, "Reportable", "small")
    svg.rect(471, 794, 15, 15, GRAY, rx=2)
    svg.text(495, 807, "Suppressed / descriptive only", "small")
    svg.text(56, 835, "Totals: 88/120 asset/fold contrasts and 22/30 macro/fold contrasts were reportable.", "small")
    svg.text(56, 857, "No model-by-dimension summary has complete five-fold macro support; suppressed cells are missing evidence, not zero effects.", "small")
    svg.text(56, 884, f"Source: {REPORTABILITY.relative_to(ROOT).as_posix()}", "small")
    target = OUT / "regime_reportability.svg"
    svg.save(target)
    return target


def main() -> None:
    macro = _load(MACRO, 5, {"fold", "n_observations", "lightgbm_macro_mae_skill", "lstm_seed_mean_macro_mae_skill"})
    asset = _load(ASSET, 20, {"fold", "asset", "lightgbm_mae_skill", "lstm_seed_mean_mae_skill"})
    regime = _load(REGIME, 120, {"model", "fold", "asset", "regime_dimension", "headline_reportable", "mae_skill_degradation"})
    reportability = _load(REPORTABILITY, 12, {"reporting_level", "model", "regime_dimension", "total_contrasts", "reportable_contrasts", "descriptive_or_unavailable_contrasts"})
    outputs = [macro_figure(macro), heatmap_figure(asset), regime_figure(regime), reportability_figure(reportability)]
    print("Validated four content-addressed D1-D5 source artifacts.")
    for path in outputs:
        print(f"Wrote {path.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
