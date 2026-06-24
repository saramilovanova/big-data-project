
from __future__ import annotations

import json
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
PROJECT_ROOT = Path.cwd().parent  # notebook is in notebooks/
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_FILE: Path = Path(PROJECT_ROOT) / "big-data-project" / "results" / "t2_quality_results.json"
FIGURES_DIR:  Path = Path(PROJECT_ROOT) / "big-data-project" / "results" / "t2_figures"
FIGURES_DIR.mkdir(exist_ok=True)

ISSUE_COLS: list[str] = [
    "invalid_year", "null_pickup", "null_dropoff", "placeholder_dropoff",
    "same_timestamps", "negative_duration", "zero_distance", "negative_distance",
    "zero_passengers", "negative_fare", "excessive_duration", "high_fare",
]

ISSUE_LABELS: dict[str, str] = {
    "invalid_year":        "Invalid year",
    "null_pickup":         "Null pickup",
    "null_dropoff":        "Null dropoff",
    "placeholder_dropoff": "Placeholder dropoff",
    "same_timestamps":     "Same timestamps",
    "negative_duration":   "Negative duration",
    "zero_distance":       "Zero distance",
    "negative_distance":   "Negative distance",
    "zero_passengers":     "Zero passengers",
    "negative_fare":       "Negative fare",
    "excessive_duration":  "Excessive duration (>24 h)",
    "high_fare":           "High fare (>$500)",
}

# 12 perceptually distinct, print-safe colours — one per issue type
ISSUE_COLORS: dict[str, str] = {
    "invalid_year":        "#DC2626",   # red
    "null_pickup":         "#EA580C",   # orange
    "null_dropoff":        "#D97706",   # amber
    "placeholder_dropoff": "#92400E",   # dark brown
    "same_timestamps":     "#7C3AED",   # violet
    "negative_duration":   "#DB2777",   # pink
    "zero_distance":       "#0891B2",   # cyan
    "negative_distance":   "#0F766E",   # dark teal
    "zero_passengers":     "#2563EB",   # blue
    "negative_fare":       "#E11D48",   # rose
    "excessive_duration":  "#C2410C",   # dark orange
    "high_fare":           "#9333EA",   # purple
}

# Dataset display label, expected year range, and accent colour
DS_CONFIG: dict[str, dict] = {
    "yellow_tripdata": {"label": "Yellow Taxi", "valid_range": (2009, 2026), "color": "#CA8A04"},
    "green_tripdata":  {"label": "Green Taxi",  "valid_range": (2014, 2026), "color": "#16A34A"},
    "fhv_tripdata":    {"label": "FHV",          "valid_range": (2015, 2026), "color": "#4338CA"},
    "fhvhv_tripdata":  {"label": "FHVHV",        "valid_range": (2019, 2026), "color": "#0284C7"},
}

# Abbreviated column headers used in wide tables
COL_ABBREV: dict[str, str] = {
    "invalid_year":        "Inv. Year",
    "null_pickup":         "Null PU",
    "null_dropoff":        "Null DO",
    "placeholder_dropoff": "Plchldr DO",
    "same_timestamps":     "Same TS",
    "negative_duration":   "Neg. Dur.",
    "zero_distance":       "Zero Dist.",
    "negative_distance":   "Neg. Dist.",
    "zero_passengers":     "Zero Pax",
    "negative_fare":       "Neg. Fare",
    "excessive_duration":  "Exc. Dur.",
    "high_fare":           "High Fare",
}

# ── Matplotlib RC (applies globally for this module) ─────────────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          9,
    "axes.titlesize":     10,
    "axes.titleweight":   "bold",
    "axes.labelsize":     9,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "axes.grid.axis":     "y",
    "axes.axisbelow":     True,
    "grid.color":         "#E5E7EB",
    "grid.linewidth":     0.6,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.1,
})


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING & SHARED HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_data(path: Path | str = RESULTS_FILE) -> pd.DataFrame:
    """Load *t2_quality_results.json* and add derived columns."""
    with open(path) as f:
        raw = json.load(f)
    df = pd.DataFrame(raw)
    df["total_issues"] = df[ISSUE_COLS].sum(axis=1)
    df["issue_rate_%"] = (
        df["total_issues"] / df["total_rows"].replace(0, pd.NA) * 100
    ).round(4)
    return df


def _get_ds(df: pd.DataFrame, ds_name: str, valid_only: bool = True) -> pd.DataFrame:
    """Subset *df* to one dataset, sort by year, optionally filter to valid range."""
    lo, hi = DS_CONFIG[ds_name]["valid_range"]
    sub = df[df["dataset"] == ds_name].sort_values("year")
    if valid_only:
        sub = sub[(sub["year"] >= lo) & (sub["year"] <= hi)]
    return sub.set_index("year")


def _active_cols(sub: pd.DataFrame, exclude: list[str] | None = None) -> list[str]:
    """Return ISSUE_COLS that have at least one non-zero value in *sub*."""
    cols = [c for c in ISSUE_COLS if sub[c].sum() > 0]
    return [c for c in cols if c not in (exclude or [])]


def _yfmt(v, _) -> str:
    """Y-axis tick label: '1.2 M', '450 K', or plain integer."""
    if v >= 1_000_000:
        return f"{v / 1_000_000:.1f} M"
    if v >= 1_000:
        return f"{int(v / 1_000):,} K"
    return f"{int(v):,}"


def _save_fig(fig: plt.Figure, stem: str) -> None:
    """Save *fig* as PNG and PDF inside FIGURES_DIR."""
    for ext in ("png", "pdf"):
        p = FIGURES_DIR / f"{stem}.{ext}"
        fig.savefig(p)
        print(f"  Saved {p}")


# ══════════════════════════════════════════════════════════════════════════════
# STACKED BAR CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def _draw_bars(ax: plt.Axes, sub: pd.DataFrame, cols: list[str]) -> None:
    """
    Draw stacked issue bars on *ax*.
    *sub* must be indexed by year; *cols* is the ordered list of issue columns to stack.
    """
    x = np.arange(len(sub))
    bottom = np.zeros(len(sub))
    for col in cols:
        vals = sub[col].fillna(0).values.astype(float)
        ax.bar(
            x, vals, bottom=bottom,
            color=ISSUE_COLORS[col], width=0.72, linewidth=0,
            zorder=2, label=ISSUE_LABELS[col],
        )
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(sub.index.astype(str), rotation=45, ha="right")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_yfmt))
    ax.set_ylabel("Issue count")
    ax.set_xlabel("Year")


def _add_legend(fig: plt.Figure, cols: list[str], n_col: int = 1) -> None:
    """Attach a legend outside the rightmost axes."""
    handles = [
        Patch(facecolor=ISSUE_COLORS[c], label=ISSUE_LABELS[c])
        for c in cols
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=7.5,
        frameon=False,
        title="Issue type",
        title_fontsize=8,
        ncol=n_col,
    )


def plot_quality_chart(
    df: pd.DataFrame,
    ds_name: str,
    save: bool = True,
) -> None:
    """
    Generate and display the quality-issues stacked bar chart for one dataset.

    Only the *valid year range* is plotted (see DS_CONFIG).
    FHV gets a two-panel layout:
      (A) all issues — shows the 1989-sentinel placeholder dominance in 2015-2017;
      (B) excluding placeholder_dropoff — reveals the remaining issues at a
          readable scale.

    Parameters
    ----------
    df      : output of load_data()
    ds_name : one of DS_CONFIG.keys()
    save    : if True, writes PNG + PDF to FIGURES_DIR
    """
    cfg = DS_CONFIG[ds_name]
    sub = _get_ds(df, ds_name, valid_only=True)

    if ds_name == "fhv_tripdata":
        _fhv_chart(sub, cfg, save)
    else:
        _standard_chart(sub, ds_name, cfg, save)


# ── Standard single-panel chart ───────────────────────────────────────────────

def _standard_chart(
    sub: pd.DataFrame,
    ds_name: str,
    cfg: dict,
    save: bool,
) -> None:
    cols = _active_cols(sub)
    fig, ax = plt.subplots(figsize=(10, 4))

    _draw_bars(ax, sub, cols)
    ax.set_title(f"{cfg['label']} — data quality issues by year", loc="left", pad=6)
    # Thin accent line along x-axis using the dataset colour
    ax.axhline(0, color=cfg["color"], lw=1.5, alpha=0.5, zorder=1)

    _add_legend(fig, cols)
    fig.tight_layout()

    if save:
        _save_fig(fig, f"t2_{ds_name}_chart")
    plt.show()
    plt.close(fig)


# ── FHV two-panel chart ───────────────────────────────────────────────────────

def _fhv_chart(sub: pd.DataFrame, cfg: dict, save: bool) -> None:
    """
    Two-panel chart for FHV.

    Panel A: full picture — placeholder_dropoff dominates 2015-2017 and
             communicates the scale of the data-quality problem.
    Panel B: the same years with placeholder_dropoff excluded so that
             excessive_duration, negative_duration, and same_timestamps
             are visible at a sensible scale.
    """
    cols_all   = _active_cols(sub)
    cols_no_ph = _active_cols(sub, exclude=["placeholder_dropoff"])

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13, 4),
        gridspec_kw={"wspace": 0.38},
    )

    _draw_bars(ax1, sub, cols_all)
    ax1.set_title(
        "(A) All issues\n(incl. 1989-sentinel placeholder dropoff)",
        loc="left", fontsize=9, pad=4,
    )

    _draw_bars(ax2, sub, cols_no_ph)
    ax2.set_title(
        "(B) Excluding placeholder dropoff\n(remaining issues at readable scale)",
        loc="left", fontsize=9, pad=4,
    )

    fig.suptitle(
        f"{cfg['label']} — data quality issues by year",
        fontsize=10, fontweight="bold", x=0.02, ha="left", y=1.03,
    )
    _add_legend(fig, cols_all)
    fig.tight_layout()

    if save:
        _save_fig(fig, "t2_fhv_tripdata_chart")
    plt.show()
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# TABLE — PANDAS STYLER  (Jupyter display)
# ══════════════════════════════════════════════════════════════════════════════

def _fmt_count(v) -> str:
    """Comma-separated int, or em dash for zero / NaN."""
    return "—" if (pd.isna(v) or v == 0) else f"{int(v):,}"


def _fmt_rate(v) -> str:
    return "—" if (pd.isna(v) or v == 0) else f"{v:.2f}%"


def _color_rate_col(series: pd.Series) -> list[str]:
    """
    Colour-map the Rate (%) column by severity.
    Applied post-format so values arrive as strings like '1.80%' or '—'.
    """
    out = []
    for v in series:
        try:
            rate = float(str(v).replace("%", "").replace("—", "0"))
        except ValueError:
            rate = 0.0
        if rate > 5:
            out.append("background-color: #FEE2E2; color: #B91C1C")
        elif rate > 1:
            out.append("background-color: #FED7AA; color: #C2410C")
        elif rate > 0:
            out.append("background-color: #FEF3C7; color: #92400E")
        else:
            out.append("")
    return out


_TABLE_STYLES = [
    {"selector": "caption",
     "props": [("caption-side", "top"), ("font-style", "italic"),
               ("font-size", "0.82em"), ("color", "#6B7280"),
               ("text-align", "left"), ("padding-bottom", "6px")]},
    {"selector": "th",
     "props": [("background-color", "#1F2937"), ("color", "#F9FAFB"),
               ("font-weight", "500"), ("text-align", "right"),
               ("padding", "4px 8px"), ("font-size", "0.82em"),
               ("white-space", "nowrap")]},
    {"selector": "th.row_heading",
     "props": [("background-color", "#1F2937"), ("color", "#F9FAFB"),
               ("text-align", "right"), ("padding", "4px 8px"),
               ("font-size", "0.82em")]},
    {"selector": "th.blank",
     "props": [("background-color", "#1F2937")]},
    {"selector": "td",
     "props": [("text-align", "right"), ("padding", "3px 8px"),
               ("font-size", "0.82em"), ("font-family", "monospace"),
               ("white-space", "nowrap")]},
    {"selector": "tr:nth-child(even) td",
     "props": [("background-color", "#F9FAFB")]},
]


def make_quality_table(
    df: pd.DataFrame,
    ds_name: str,
    include_anomalous: bool = False,
) -> "pd.io.formats.style.Styler":
    """
    Build and return a styled quality table for *ds_name*.

    Call ``display(make_quality_table(df, ds_name))`` in Jupyter to render it.

    Parameters
    ----------
    df                : output of load_data()
    ds_name           : one of DS_CONFIG.keys()
    include_anomalous : if True, years outside the valid range are appended
                        with an amber highlight

    Returns
    -------
    pandas Styler ready for Jupyter display
    """
    cfg = DS_CONFIG[ds_name]
    lo, hi = cfg["valid_range"]

    sub_all  = df[df["dataset"] == ds_name].sort_values("year").set_index("year")
    sub_main = sub_all[(sub_all.index >= lo) & (sub_all.index <= hi)]
    anom_idx = sub_all.index[(sub_all.index < lo) | (sub_all.index > hi)]

    base = sub_all if include_anomalous else sub_main
    # Use sub_all so columns active only in anomalous years still appear
    cols = _active_cols(sub_all)

    # ── Build display frame ──────────────────────────────────────────────────
    disp = base[["total_rows"] + cols + ["total_issues", "issue_rate_%"]].copy()
    disp.index.name = "Year"

    col_rename = {
        "total_rows":   "Total rows",
        "total_issues": "Total issues",
        "issue_rate_%": "Rate (%)",
        **{c: COL_ABBREV[c] for c in cols},
    }
    disp = disp.rename(columns=col_rename)

    fmt_dict: dict = {
        "Total rows":   "{:,.0f}",
        "Total issues": _fmt_count,
        "Rate (%)":     _fmt_rate,
        **{COL_ABBREV[c]: _fmt_count for c in cols},
    }

    n_valid   = len(sub_main)
    n_anom    = len(anom_idx)
    anom_note = (
        f" {n_anom} anomalous year stub(s) appended (amber rows)."
        if include_anomalous and n_anom else ""
    )
    caption = (
        f"{cfg['label']} — data quality violations aggregated by year ({lo}–{hi}). "
        "Issue rate = total violations ÷ total records "
        "(a single record may trigger multiple checks)."
        + anom_note
    )

    # ── Style ────────────────────────────────────────────────────────────────
    styler = (
        disp.style
        .format(fmt_dict, na_rep="—")
        .set_caption(caption)
        .set_table_styles(_TABLE_STYLES)
        .apply(_color_rate_col, subset=["Rate (%)"], axis=0)
    )

    # Highlight anomalous rows in amber
    if include_anomalous and len(anom_idx):
        def _hl_anom(row):
            return (
                ["background-color: #FEF9C3"] * len(row)
                if row.name in anom_idx else [""] * len(row)
            )
        styler = styler.apply(_hl_anom, axis=1)

    return styler


# ══════════════════════════════════════════════════════════════════════════════
# TABLE — MATPLOTLIB FIGURE  (save as PNG / PDF)
# ══════════════════════════════════════════════════════════════════════════════

def render_table_figure(
    df: pd.DataFrame,
    ds_name: str,
    include_anomalous: bool = False,
    save: bool = True,
) -> None:
    """
    Render the quality table as a self-contained matplotlib figure.

    Produces a PNG and PDF suitable for direct inclusion in a report.
    Issue Rate cells are colour-coded; anomalous year rows are amber.

    Parameters
    ----------
    df                : output of load_data()
    ds_name           : one of DS_CONFIG.keys()
    include_anomalous : include year stubs outside the valid range
    save              : write PNG + PDF to FIGURES_DIR
    """
    cfg = DS_CONFIG[ds_name]
    lo, hi = cfg["valid_range"]

    sub_all = df[df["dataset"] == ds_name].sort_values("year").set_index("year")
    base    = sub_all if include_anomalous else sub_all[
        (sub_all.index >= lo) & (sub_all.index <= hi)
    ]
    cols = _active_cols(sub_all)

    # ── Build cell data ──────────────────────────────────────────────────────
    def _cell(col_key: str, val) -> str:
        if col_key == "issue_rate_%":
            return f"{val:.2f}%" if val > 0 else "—"
        return "—" if val == 0 else f"{int(val):,}"

    display_cols = ["total_rows"] + cols + ["total_issues", "issue_rate_%"]
    col_headers  = (
        ["Total rows"]
        + [COL_ABBREV[c] for c in cols]
        + ["Total issues", "Rate (%)"]
    )

    cell_text  = [
        [_cell(c, row[c]) for c in display_cols]
        for _, row in base.iterrows()
    ]
    row_labels = [str(y) for y in base.index]

    n_rows = len(row_labels)
    n_cols = len(col_headers)

    # Figure dimensions scale with table size
    fig_w = max(10.0, 0.88 * (n_cols + 1))   # +1 for the row-label column
    fig_h = max(3.0,  0.29 * (n_rows + 2) + 0.5)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_headers,
        rowLabels=row_labels,
        cellLoc="right",
        rowLoc="right",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.auto_set_column_width(range(n_cols))

    # ── Style header row ─────────────────────────────────────────────────────
    for j in range(n_cols):
        cell = tbl[0, j]
        cell.set_facecolor("#1F2937")
        cell.get_text().set_color("#F9FAFB")
        cell.get_text().set_fontweight("bold")
        cell.set_edgecolor("#374151")

    # ── Style row-label column (year numbers) ─────────────────────────────
    # Row labels in matplotlib tables live at column index -1
    for i in range(1, n_rows + 1):
        try:
            rl = tbl[i, -1]
            rl.set_facecolor("#374151")
            rl.get_text().set_color("#F9FAFB")
            rl.get_text().set_fontweight("bold")
            rl.set_edgecolor("#374151")
        except KeyError:
            pass

    # ── Style data cells ─────────────────────────────────────────────────────
    for i, (year, row) in enumerate(base.iterrows(), start=1):
        is_anom = year < lo or year > hi
        rate    = row["issue_rate_%"]

        for j, col_key in enumerate(display_cols):
            cell = tbl[i, j]
            cell.set_edgecolor("#E5E7EB")

            # Default: zebra striping
            bg = "#F9FAFB" if i % 2 == 0 else "#FFFFFF"

            # Anomalous year rows override everything
            if is_anom:
                bg = "#FEF9C3"
            elif col_key == "issue_rate_%":
                if rate > 5:
                    bg = "#FEE2E2"
                    cell.get_text().set_color("#B91C1C")
                    cell.get_text().set_fontweight("bold")
                elif rate > 1:
                    bg = "#FED7AA"
                    cell.get_text().set_color("#C2410C")
                elif rate > 0:
                    bg = "#FEF3C7"
                    cell.get_text().set_color("#92400E")

            cell.set_facecolor(bg)

    ax.set_title(
        f"{cfg['label']} — data quality issues by year",
        fontsize=9, fontweight="bold", loc="left", pad=10,
    )
    fig.tight_layout()

    if save:
        _save_fig(fig, f"t2_{ds_name}_table")
    plt.show()
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# LATEX TABLE EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def export_latex_table(
    df: pd.DataFrame,
    ds_name: str,
    output_path: Path | None = None,
) -> str:
    """
    Write the quality table as a LaTeX ``longtable`` and return the source.

    The file is saved to *output_path* (defaults to FIGURES_DIR/<stem>.tex).
    Paste the output into your report's ``\\input{...}`` directive.

    Parameters
    ----------
    df          : output of load_data()
    ds_name     : one of DS_CONFIG.keys()
    output_path : override the default save location

    Returns
    -------
    str : LaTeX table source
    """
    cfg = DS_CONFIG[ds_name]
    lo, hi = cfg["valid_range"]

    sub  = _get_ds(df, ds_name, valid_only=True)
    cols = _active_cols(sub)

    tbl = sub[["total_rows"] + cols + ["total_issues", "issue_rate_%"]].copy()

    # Format for LaTeX
    for c in ["total_rows", "total_issues"] + cols:
        tbl[c] = tbl[c].apply(
            lambda v: r"---" if v == 0 else f"{int(v):,}"
        )
    tbl["issue_rate_%"] = tbl["issue_rate_%"].apply(
        lambda v: r"---" if v == 0 else f"{v:.2f}\\%"
    )

    col_rename = {
        "total_rows":   "Total rows",
        "total_issues": "Total issues",
        "issue_rate_%": "Rate",
        **{c: COL_ABBREV[c] for c in cols},
    }
    tbl = tbl.rename(columns=col_rename)
    tbl.index.name = "Year"

    n_cols   = len(tbl.columns) + 1   # +1 for the index column
    col_fmt  = "r" * n_cols

    caption = (
        f"{cfg['label']}---data quality violations by year ({lo}--{hi}). "
        r"Rate = total violations $\div$ total records "
        r"(a single record may trigger multiple checks)."
    )

    latex = tbl.to_latex(
        caption=caption,
        label=f"tab:t2_{ds_name}",
        na_rep="---",
        escape=False,          # we handle special chars manually above
        column_format=col_fmt,
    )

    out = Path(output_path) if output_path else FIGURES_DIR / f"t2_{ds_name}_table.tex"
    out.write_text(latex, encoding="utf-8")
    print(f"  Saved LaTeX table: {out}")
    return latex


# ══════════════════════════════════════════════════════════════════════════════
# OPTIONAL: 2×2 OVERVIEW FIGURE  (all four datasets in one figure)
# ══════════════════════════════════════════════════════════════════════════════

def plot_overview_grid(df: pd.DataFrame, save: bool = True) -> None:
    """
    2×2 grid of mini stacked-bar charts — one panel per dataset.

    FHV excludes placeholder_dropoff so all four panels share a comparable
    scale. A note is added to the FHV panel reminding the reader of this.

    Parameters
    ----------
    df   : output of load_data()
    save : write PNG + PDF to FIGURES_DIR
    """
    ds_keys = list(DS_CONFIG.keys())
    fig, axes = plt.subplots(2, 2, figsize=(14, 7), constrained_layout=True)

    all_active: list[str] = []

    for ax, ds_name in zip(axes.flatten(), ds_keys):
        cfg  = DS_CONFIG[ds_name]
        sub  = _get_ds(df, ds_name, valid_only=True)
        excl = ["placeholder_dropoff"] if ds_name == "fhv_tripdata" else None
        cols = _active_cols(sub, exclude=excl)
        all_active.extend(c for c in cols if c not in all_active)

        _draw_bars(ax, sub, cols)
        suffix = "\n(placeholder dropoff excluded)" if ds_name == "fhv_tripdata" else ""
        ax.set_title(f"{cfg['label']}{suffix}", loc="left", fontsize=9, fontweight="bold")
        ax.axhline(0, color=cfg["color"], lw=1.5, alpha=0.45, zorder=1)

    # Shared legend below all panels
    handles = [
        Patch(facecolor=ISSUE_COLORS[c], label=ISSUE_LABELS[c])
        for c in all_active
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        fontsize=7.5,
        frameon=False,
        title="Issue type",
        title_fontsize=8,
        bbox_to_anchor=(0.5, -0.07),
    )
    fig.suptitle(
        "T2 — Data quality issues by year (all datasets)",
        fontsize=11, fontweight="bold",
    )

    if save:
        _save_fig(fig, "t2_overview_grid")
    plt.show()
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT — generates everything for all four datasets
# ══════════════════════════════════════════════════════════════════════════════

def run_all(results_file: Path | str = RESULTS_FILE) -> None:
    """
    Generate all per-dataset tables and charts plus the 2×2 overview.
    Saves PNG, PDF, and .tex files to FIGURES_DIR.

    Intended to be called from a notebook cell or the command line.
    """
    try:
        from IPython.display import display as ipy_display   # Jupyter
    except ImportError:
        ipy_display = print                                   # plain Python

    df = load_data(results_file)
    print(f"Loaded {len(df)} records across {df['dataset'].nunique()} datasets.\n")

    for ds_name in DS_CONFIG:
        cfg = DS_CONFIG[ds_name]
        sep = "─" * 64
        print(f"\n{sep}\n  {cfg['label']}  ({ds_name})\n{sep}")

        print("  [1/4] Styled table (Jupyter)")
        ipy_display(make_quality_table(df, ds_name))

        # print("  [2/4] Table figure (PNG + PDF)")
        # render_table_figure(df, ds_name, save=True)

        print("  [3/4] Quality chart (PNG + PDF)")
        plot_quality_chart(df, ds_name, save=True)

        print("  [4/4] LaTeX export")
        export_latex_table(df, ds_name)

    print("\n─ 2×2 overview grid ─")
    plot_overview_grid(df, save=True)
    print(f"\nAll outputs saved to: {FIGURES_DIR.resolve()}/")


if __name__ == "__main__":
    run_all()
