"""
analyze_results.py  ·  T6 Streaming Analytics Report

Produces all tables and figures needed for the T6 section of the report:
  - Rolling descriptive statistics for boroughs (mean, std, min, max for
    trip_distance, fare_amount, tip_amount) -- Section 4 of the assignment
  - Rolling descriptive statistics for the 10 most active locations
  - Summary tables comparing the two implementations (Quix vs plain Python)
  - Aggregate cross-check between the two implementations

Source: borough_stats_python.jsonl / location_stats_python.jsonl (from
regular_python_stats.py). These files carry 'group_value' explicitly and
are the primary source. The Quix files are used only for aggregate-level
cross-checking (total records + overall weighted mean) because the Quix
output was missing the per-row group key due to the metadata=True bug
documented in the report.

All heavy reads use DuckDB directly on the JSONL files -- the borough file
alone is ~1.3 GB and pandas.read_json would balloon that several-fold.
Only small aggregated results become pandas DataFrames before plotting.
"""

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from pathlib import Path

from config import OUTPUT_DIR, STATS_FIELDS, TOP10_LOCATIONS

# ── paths ────────────────────────────────────────────────────────────────────
BOROUGH_PY   = f"{OUTPUT_DIR}/borough_stats_python.jsonl"
LOCATION_PY  = f"{OUTPUT_DIR}/location_stats_python.jsonl"
BOROUGH_QUIX = f"{OUTPUT_DIR}/borough_stats.jsonl"
LOCATION_QUIX = f"{OUTPUT_DIR}/location_stats.jsonl"

ASSETS = Path(OUTPUT_DIR) / "report_assets"
ASSETS.mkdir(parents=True, exist_ok=True)

con = duckdb.connect()
con.execute("PRAGMA threads=4")
con.execute("PRAGMA memory_limit='3GB'")

FIELD_LABELS = {
    "trip_distance": "Trip distance (miles)",
    "fare_amount":   "Fare amount ($)",
    "tip_amount":    "Tip amount ($)",
}

# Mapping LocationID -> Zone name for readable axis labels
LOCATION_NAMES = {
    161: "Midtown Ctr",    237: "Upper E Side S",  236: "Upper E Side N",
    132: "JFK Airport",   230: "Times Sq",         138: "LaGuardia",
    162: "Midtown East",  170: "Murray Hill",      142: "Lincoln Sq E",
    79:  "East Village",
}


# ── helpers ──────────────────────────────────────────────────────────────────

def file_ok(path):
    p = Path(path)
    return p.exists() and p.stat().st_size > 100


def savefig(fig, name):
    p = ASSETS / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {p.name}")


# ── 1. summary table ─────────────────────────────────────────────────────────

def summary_table(path, group_type, label):
    """Overall weighted-mean statistics per group across the full year."""
    mean_cols = ", ".join(
        f"ROUND(SUM(count*{f}_mean)/SUM(count),3) AS {f}_mean" for f in STATS_FIELDS
    )
    std_cols = ", ".join(
        f"ROUND(AVG({f}_std),3) AS {f}_std" for f in STATS_FIELDS
    )
    df = con.sql(f"""
        SELECT group_value,
               SUM(count) AS total_events,
               {mean_cols},
               {std_cols}
        FROM read_ndjson_auto('{path}')
        WHERE group_type = '{group_type}'
        GROUP BY group_value
        ORDER BY total_events DESC
    """).df()
    out = ASSETS / f"{label}_summary.csv"
    df.to_csv(out, index=False)
    print(f"  saved {out.name}")
    return df


# ── 2. daily trend (mean of each field per group, across the year) ────────────

def daily_means(path, group_type):
    mean_cols = ", ".join(
        f"SUM(count*{f}_mean)/SUM(count) AS {f}_mean" for f in STATS_FIELDS
    )
    return con.sql(f"""
        SELECT group_value,
               STRFTIME(CAST(window_start AS TIMESTAMP), '%Y-%m-%d') AS day,
               SUM(count) AS n,
               {mean_cols}
        FROM read_ndjson_auto('{path}')
        WHERE group_type = '{group_type}'
        GROUP BY group_value, STRFTIME(CAST(window_start AS TIMESTAMP), '%Y-%m-%d')
        ORDER BY day
    """).df()


def plot_daily_trends(df, label, name_map=None, top_n=None):
    """One figure with 3 subplots (one per STATS_FIELD), all groups overlaid."""
    groups = df["group_value"].unique().tolist()
    if top_n:
        totals = df.groupby("group_value")["n"].sum().sort_values(ascending=False)
        groups = totals.head(top_n).index.tolist()

    fig, axes = plt.subplots(len(STATS_FIELDS), 1,
                              figsize=(13, 3.8 * len(STATS_FIELDS)), sharex=True)
    cmap = plt.cm.tab10.colors

    for ax, field in zip(axes, STATS_FIELDS):
        for i, g in enumerate(groups):
            sub = df[df["group_value"] == g].sort_values("day")
            xs = pd.to_datetime(sub["day"])
            ys = sub[f"{field}_mean"]
            lbl = name_map.get(g, str(g)) if name_map else str(g)
            ax.plot(xs, ys, label=lbl, color=cmap[i % 10], linewidth=0.9)
        ax.set_ylabel(FIELD_LABELS[field], fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.legend(fontsize=7, ncol=3, loc="upper right")

    axes[0].set_title(f"Daily mean rolling statistics by {label} — 2021", fontsize=11)
    axes[-1].set_xlabel("Date")
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    savefig(fig, f"{label}_daily_trends.png")


# ── 3. day-of-week and hour-of-day heatmaps ──────────────────────────────────

def plot_heatmap_dow_hod(path, group_type, group_value, field, label, fname):
    """Mean of `field` by hour-of-day × day-of-week for one group."""
    df = con.sql(f"""
        SELECT
            DAYOFWEEK(CAST(window_start AS TIMESTAMP)) AS dow,
            HOUR(CAST(window_start AS TIMESTAMP))      AS hod,
            SUM(count*{field}_mean)/SUM(count)         AS mean_val
        FROM read_ndjson_auto('{path}')
        WHERE group_type = '{group_type}'
          AND CAST(group_value AS VARCHAR) = '{group_value}'
        GROUP BY dow, hod
        ORDER BY dow, hod
    """).df()

    pivot = df.pivot_table(index="hod", columns="dow", values="mean_val")
    pivot.columns = ["Sun","Mon","Tue","Wed","Thu","Fri","Sat"]
    pivot = pivot.reindex(range(24))

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(pivot, aspect="auto", cmap="YlOrRd", origin="lower")
    ax.set_xticks(range(7)); ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(0, 24, 3)); ax.set_yticklabels([f"{h:02d}:00" for h in range(0,24,3)])
    ax.set_xlabel("Day of week"); ax.set_ylabel("Hour of day")
    ax.set_title(f"Mean {FIELD_LABELS[field]} — {label}\n(by hour of day × day of week, 2021)")
    plt.colorbar(im, ax=ax, label=FIELD_LABELS[field])
    fig.tight_layout()
    savefig(fig, fname)


# ── 4. zoomed single-day rolling windows (proves it's actually rolling) ───────

def plot_zoomed_windows(path, group_type, groups, field, label):
    """Show actual 30-second window granularity for one busy day per group.
    This is the key chart proving the stats are genuinely windowed, not just
    daily aggregates."""
    busiest_day = con.sql(f"""
        SELECT STRFTIME(CAST(window_start AS TIMESTAMP), '%Y-%m-%d') AS day,
               SUM(count) AS n
        FROM read_ndjson_auto('{path}')
        WHERE group_type = '{group_type}'
        GROUP BY day ORDER BY n DESC LIMIT 1
    """).fetchone()[0]

    placeholders = ",".join(f"'{g}'" for g in groups)
    df = con.sql(f"""
        SELECT group_value,
               CAST(window_start AS TIMESTAMP) AS ts,
               {field}_mean, {field}_std,
               count
        FROM read_ndjson_auto('{path}')
        WHERE group_type   = '{group_type}'
          AND STRFTIME(CAST(window_start AS TIMESTAMP), '%Y-%m-%d') = '{busiest_day}'
          AND CAST(group_value AS VARCHAR) IN ({placeholders})
        ORDER BY ts
    """).df()

    if df.empty:
        print(f"  no zoomed data for {label}")
        return

    name_map = LOCATION_NAMES if group_type == "location" else {}
    fig, ax = plt.subplots(figsize=(13, 4.5))
    cmap = plt.cm.tab10.colors
    for i, g in enumerate(df["group_value"].unique()):
        sub = df[df["group_value"] == g]
        lbl = name_map.get(g, str(g))
        ax.plot(sub["ts"], sub[f"{field}_mean"], label=lbl,
                color=cmap[i % 10], linewidth=0.8)
        ax.fill_between(sub["ts"],
                        sub[f"{field}_mean"] - sub[f"{field}_std"],
                        sub[f"{field}_mean"] + sub[f"{field}_std"],
                        color=cmap[i % 10], alpha=0.08)

    ax.set_title(f"30-second rolling windows — {FIELD_LABELS[field]} by {label}\n{busiest_day}", fontsize=11)
    ax.set_xlabel("Time of day")
    ax.set_ylabel(FIELD_LABELS[field])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax.legend(fontsize=7, ncol=3, loc="upper right")
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    savefig(fig, f"{label}_zoomed_{busiest_day}.png")
    return busiest_day


# ── 5. descriptive stats table (report-ready, per-field) ─────────────────────

def descriptive_stats_table(summary_df, label):
    """Reshape summary into a clean per-field table for the report."""
    rows = []
    for field in STATS_FIELDS:
        sub = summary_df[["group_value", f"{field}_mean", f"{field}_std"]].copy()
        sub = sub.rename(columns={
            f"{field}_mean": "mean",
            f"{field}_std": "mean_of_stds",
        })
        sub["attribute"] = field
        rows.append(sub)
    df = pd.concat(rows).pivot_table(
        index="group_value", columns="attribute",
        values=["mean", "mean_of_stds"]
    )
    df.columns = [f"{attr}_{stat}" for stat, attr in df.columns]
    df = df.reset_index().rename(columns={"group_value": "group"})
    out = ASSETS / f"{label}_stats_report_table.csv"
    df.to_csv(out, index=False)
    print(f"  saved {out.name}")
    return df


# ── 6. implementation comparison ─────────────────────────────────────────────

def compare_implementations(quix_path, python_path, group_type, label):
    if not file_ok(quix_path):
        print(f"  [{label}] Quix output file missing/empty — skipping cross-check")
        return

    field = STATS_FIELDS[0]
    q = con.sql(f"""
        SELECT COUNT(*) AS window_records,
               SUM(count) AS total_events,
               ROUND(SUM(count*{field}_mean)/SUM(count),4) AS overall_{field}_mean
        FROM read_ndjson_auto('{quix_path}')
    """).df().assign(implementation="Quix Streams")

    p = con.sql(f"""
        SELECT COUNT(*) AS window_records,
               SUM(count) AS total_events,
               ROUND(SUM(count*{field}_mean)/SUM(count),4) AS overall_{field}_mean
        FROM read_ndjson_auto('{python_path}')
        WHERE group_type = '{group_type}'
    """).df().assign(implementation="Regular Python")

    result = pd.concat([q, p])[["implementation","window_records","total_events",f"overall_{field}_mean"]]
    out = ASSETS / f"{label}_implementation_comparison.csv"
    result.to_csv(out, index=False)
    print(f"  saved {out.name}")
    print(result.to_string(index=False))


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── BOROUGHS ──────────────────────────────────────────────────────────────
    print("\n=== Borough statistics ===")

    print("Summary table...")
    borough_summary = summary_table(BOROUGH_PY, "borough", "borough")
    print(borough_summary[["group_value","total_events","fare_amount_mean","trip_distance_mean","tip_amount_mean"]].to_string(index=False))

    print("\nDescriptive stats table...")
    descriptive_stats_table(borough_summary, "borough")

    print("\nDaily trend charts...")
    borough_daily = daily_means(BOROUGH_PY, "borough")
    plot_daily_trends(borough_daily, "borough")

    print("\nHour-of-day × day-of-week heatmaps (Manhattan fare, Queens distance)...")
    plot_heatmap_dow_hod(BOROUGH_PY, "borough", "Manhattan", "fare_amount",
                         "Manhattan", "manhattan_fare_heatmap.png")
    plot_heatmap_dow_hod(BOROUGH_PY, "borough", "Queens", "trip_distance",
                         "Queens", "queens_distance_heatmap.png")

    print("\nZoomed 30-second windows (top 4 boroughs by volume)...")
    top_boroughs = borough_summary.head(4)["group_value"].tolist()
    busiest = plot_zoomed_windows(BOROUGH_PY, "borough", top_boroughs,
                                  "fare_amount", "borough")

    # ── TOP 10 LOCATIONS ──────────────────────────────────────────────────────
    print("\n=== Top-10 location statistics ===")

    print("Summary table...")
    location_summary = summary_table(LOCATION_PY, "location", "location")
    location_summary["zone"] = location_summary["group_value"].map(
        lambda x: LOCATION_NAMES.get(int(x), str(x))
    )
    print(location_summary[["zone","total_events","fare_amount_mean","trip_distance_mean"]].to_string(index=False))

    print("\nDescriptive stats table...")
    descriptive_stats_table(location_summary, "location")

    print("\nDaily trend charts...")
    location_daily = daily_means(LOCATION_PY, "location")
    plot_daily_trends(location_daily, "location",
                      name_map={k: v for k, v in LOCATION_NAMES.items()})

    print("\nHour-of-day × day-of-week heatmaps (JFK and Midtown Center)...")
    plot_heatmap_dow_hod(LOCATION_PY, "location", "132", "trip_distance",
                         "JFK Airport (loc 132)", "jfk_distance_heatmap.png")
    plot_heatmap_dow_hod(LOCATION_PY, "location", "161", "fare_amount",
                         "Midtown Center (loc 161)", "midtown_fare_heatmap.png")

    print("\nZoomed 30-second windows (top 4 locations by volume)...")
    top_locs = [str(x) for x in location_summary.head(4)["group_value"].tolist()]
    if busiest:
        plot_zoomed_windows(LOCATION_PY, "location", top_locs,
                            "fare_amount", "location")

    # ── IMPLEMENTATION COMPARISON ─────────────────────────────────────────────
    print("\n=== Implementation comparison (Quix vs Python) ===")
    compare_implementations(BOROUGH_QUIX, BOROUGH_PY, "borough", "borough")
    compare_implementations(LOCATION_QUIX, LOCATION_PY, "location", "location")

    print(f"\nDone. All outputs in {ASSETS}")
