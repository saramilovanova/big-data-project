"""
T8: FHVHV Emergence — Impact on Taxis and Smaller Competitors
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Calculates and visualises how the rise of FHVHV operators (Uber, Lyft)
affected Yellow and Green taxi services and smaller FHVHV rivals (Juno, Via),
in both absolute trip counts and relative market share.

FHVHV license codes (hvfhs_license_num):
  HV0002 → Juno   (exited NYC market mid-2019)
  HV0003 → Uber
  HV0004 → Via
  HV0005 → Lyft

Aggregation: DuckDB over T1 partitioned parquets → monthly totals.
All results cached to CSV so re-running plots is instant.

Outputs (data/t8/):
  monthly_by_service.csv    — yellow, green, fhv, fhvhv_total per month
  monthly_by_operator.csv   — per-operator within FHVHV per month
  monthly_combined.csv      — full wide table used for all plots
  plots/
    01_absolute_trends.png
    02_market_share.png
    03_indexed_trends.png
    04_fhvhv_competition.png
"""

import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PATHS  (mirrors T3 layout exactly)
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR   = Path("/d/hpc/projects/FRI/bigdata/students/sm_bv/final_project/data")
YELLOW_DIR = BASE_DIR / "yellow_normalized"
PART_DIR   = BASE_DIR / "partitioned"
OUT_DIR    = BASE_DIR / "t8"
PLOTS_DIR  = OUT_DIR / "plots"

for d in [OUT_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

VALID_YEARS = {
    "yellow": range(2019, 2026),
    "green":  range(2019, 2026),
    "fhv":    range(2019, 2026),
    "fhvhv":  range(2019, 2026),
}

# FHVHV license → company name
LICENSE_MAP = {"HV0002": "Juno", "HV0003": "Uber", "HV0004": "Via", "HV0005": "Lyft"}

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def collect_files(base_dir, valid_years):
    files = []
    for d in sorted(Path(base_dir).iterdir()):
        yr = d.name.replace("year=", "")
        if yr.isdigit() and int(yr) in valid_years:
            files.extend(sorted(d.glob("*.parquet")))
    return [str(f) for f in files]


def to_date(df):
    """Convert integer year + month columns to a proper Period/datetime index."""
    df["date"] = pd.to_datetime(
        df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2)
    )
    return df.drop(columns=["year", "month"]).sort_values("date").reset_index(drop=True)


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1: AGGREGATION
# ═════════════════════════════════════════════════════════════════════════════

def aggregate_all():
    """
    Run two DuckDB queries and produce the wide monthly combined table.
    Results are cached to CSV; subsequent runs skip straight to plotting.
    """
    combined_cache = OUT_DIR / "monthly_combined.csv"
    if combined_cache.exists():
        print("[Cache] Loading monthly_combined.csv ...", flush=True)
        df = pd.read_csv(combined_cache, parse_dates=["date"])
        return df

    con = duckdb.connect()

    FILES = {
        "yellow": collect_files(YELLOW_DIR,                VALID_YEARS["yellow"]),
        "green":  collect_files(PART_DIR / "green_tripdata", VALID_YEARS["green"]),
        "fhv":    collect_files(PART_DIR / "fhv_tripdata",   VALID_YEARS["fhv"]),
        "fhvhv":  collect_files(PART_DIR / "fhvhv_tripdata", VALID_YEARS["fhvhv"]),
    }
    for name, files in FILES.items():
        print(f"  {name}: {len(files)} parquet files", flush=True)

    PCOL = {
        "yellow": "tpep_pickup_datetime",
        "green":  "lpep_pickup_datetime",
        "fhv":    "pickup_datetime",
        "fhvhv":  "pickup_datetime",
    }

    # ── Query 1: monthly total trips per taxi/FHV service ─────────────────
    print("\n[Aggregation] Monthly trips per service ...", flush=True)
    svc_parts = []
    for svc in ["yellow", "green", "fhv"]:
        files = FILES[svc]
        pcol  = PCOL[svc]
        print(f"  {svc} ...", end=" ", flush=True)
        df_svc = con.execute(f"""
            SELECT
                year({pcol})  AS year,
                month({pcol}) AS month,
                COUNT(*)      AS trips
            FROM read_parquet({files}, union_by_name=True)
            WHERE {pcol} IS NOT NULL
              AND year({pcol}) BETWEEN 2019 AND 2025
            GROUP BY 1, 2
            ORDER BY 1, 2
        """).fetchdf()
        df_svc = to_date(df_svc).rename(columns={"trips": svc})
        svc_parts.append(df_svc)
        print(f"{len(df_svc)} months", flush=True)

    # FHVHV total (for cross-check)
    files = FILES["fhvhv"]
    print("  fhvhv total ...", end=" ", flush=True)
    df_fhvhv_total = con.execute(f"""
        SELECT
            year(pickup_datetime)  AS year,
            month(pickup_datetime) AS month,
            COUNT(*) AS fhvhv
        FROM read_parquet({files}, union_by_name=True)
        WHERE pickup_datetime IS NOT NULL
          AND year(pickup_datetime) BETWEEN 2019 AND 2025
        GROUP BY 1, 2
        ORDER BY 1, 2
    """).fetchdf()
    df_fhvhv_total = to_date(df_fhvhv_total)
    print(f"{len(df_fhvhv_total)} months", flush=True)

    # ── Query 2: monthly FHVHV trips per operator ─────────────────────────
    print("\n[Aggregation] Monthly FHVHV trips by operator ...", flush=True)
    df_ops_long = con.execute(f"""
        SELECT
            year(pickup_datetime)  AS year,
            month(pickup_datetime) AS month,
            hvfhs_license_num      AS license,
            COUNT(*)               AS trips
        FROM read_parquet({files}, union_by_name=True)
        WHERE pickup_datetime IS NOT NULL
          AND year(pickup_datetime) BETWEEN 2019 AND 2025
          AND hvfhs_license_num IN ('HV0002', 'HV0003', 'HV0004', 'HV0005')
        GROUP BY 1, 2, 3
        ORDER BY 1, 2, 3
    """).fetchdf()
    df_ops_long["company"] = df_ops_long["license"].map(LICENSE_MAP)
    df_ops_long = to_date(df_ops_long)

    # Pivot to wide: one column per operator
    df_ops = df_ops_long.pivot_table(
        index="date", columns="company", values="trips", aggfunc="sum"
    ).reset_index()
    df_ops.columns.name = None
    for op in ["Uber", "Lyft", "Via", "Juno"]:
        if op not in df_ops.columns:
            df_ops[op] = 0
    df_ops = df_ops.fillna(0)
    print(f"  {len(df_ops)} months × {df_ops.shape[1]-1} operators", flush=True)

    # ── Combine into one wide table ───────────────────────────────────────
    df = svc_parts[0]  # yellow
    for part in svc_parts[1:]:
        df = df.merge(part, on="date", how="outer")
    df = df.merge(df_fhvhv_total, on="date", how="outer")
    df = df.merge(df_ops, on="date", how="outer")
    df = df.sort_values("date").reset_index(drop=True).fillna(0)

    # Derived columns
    df["large_fhvhv"] = df["Uber"] + df["Lyft"]
    df["small_fhvhv"] = df["Via"]  + df["Juno"]
    df["taxi_total"]  = df["yellow"] + df["green"]
    df["all_total"]   = df["yellow"] + df["green"] + df["fhv"] + df["fhvhv"]

    # Save intermediate CSVs
    svc_cols = ["date", "yellow", "green", "fhv", "fhvhv"]
    df[svc_cols].to_csv(OUT_DIR / "monthly_by_service.csv", index=False)
    op_cols = ["date", "Uber", "Lyft", "Via", "Juno"]
    df[op_cols].to_csv(OUT_DIR / "monthly_by_operator.csv", index=False)
    df.to_csv(combined_cache, index=False)

    print(f"\n[Cache] Saved monthly_combined.csv  ({len(df)} months)", flush=True)
    con.close()
    return df


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2: PLOTS
# ═════════════════════════════════════════════════════════════════════════════

# Color palette — consistent across all figures
COLORS = {
    "yellow": "#E8B500",
    "green":  "#2ca02c",
    "fhv":    "#7f7f7f",
    "Uber":   "#1a1a2e",
    "Lyft":   "#FF00BF",
    "Via":    "#17a2b8",
    "Juno":   "#fd7e14",
}

COVID_START = pd.Timestamp("2020-03-01")
COVID_END   = pd.Timestamp("2021-07-01")


def shade_covid(ax, alpha=0.10):
    ax.axvspan(COVID_START, COVID_END, color="gray", alpha=alpha, label="COVID-19 period")


def millions(x, pos):
    return f"{x / 1e6:.1f}M"


def setup():
    plt.rcParams.update({
        "figure.dpi":        150,
        "font.size":         11,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "legend.frameon":    False,
    })


# ── Plot 1: Absolute monthly trips ────────────────────────────────────────

def plot_absolute(df):
    """
    Two-panel figure: top = Yellow + Green; bottom = Uber, Lyft, Via, Juno.
    Same x-axis allows direct visual comparison of scale and timing.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    fmt = mticker.FuncFormatter(millions)

    # Panel 1 — taxis and FHV
    ax1.plot(df["date"], df["yellow"], color=COLORS["yellow"], lw=2,  label="Yellow Taxi")
    ax1.plot(df["date"], df["green"],  color=COLORS["green"],  lw=2,  label="Green Taxi")
    ax1.plot(df["date"], df["fhv"],    color=COLORS["fhv"],    lw=1.5, ls="--", label="FHV (black cars)")
    shade_covid(ax1)
    ax1.yaxis.set_major_formatter(fmt)
    ax1.set_ylabel("Monthly trips")
    ax1.set_title("Traditional Taxi and Black-Car Services", fontweight="bold")
    ax1.legend(loc="upper right")
    ax1.grid(axis="y", alpha=0.25)

    # Panel 2 — FHVHV operators
    ax2.plot(df["date"], df["Uber"],  color=COLORS["Uber"],  lw=2.5, label="Uber (HV0003)")
    ax2.plot(df["date"], df["Lyft"],  color=COLORS["Lyft"],  lw=2,   label="Lyft (HV0005)")
    ax2.plot(df["date"], df["Via"],   color=COLORS["Via"],   lw=1.5, ls="--", label="Via (HV0004)")
    ax2.plot(df["date"], df["Juno"],  color=COLORS["Juno"],  lw=1.5, ls=":",  label="Juno (HV0002)")
    shade_covid(ax2)
    ax2.yaxis.set_major_formatter(fmt)
    ax2.set_ylabel("Monthly trips")
    ax2.set_xlabel("Date")
    ax2.set_title("FHVHV Operators (High-Volume For-Hire)", fontweight="bold")
    ax2.legend(loc="upper left")
    ax2.grid(axis="y", alpha=0.25)

    fig.suptitle(
        "Absolute Monthly Trip Counts by Operator (NYC TLC, 2019–2025)",
        fontsize=13, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "01_absolute_trends.png", bbox_inches="tight")
    plt.close()
    print("  Saved 01_absolute_trends.png", flush=True)


# ── Plot 2: Market share stacked area ────────────────────────────────────

def plot_market_share(df):
    """
    Stacked area chart of each operator's share of total monthly trips.
    Visually shows the substitution of Yellow/Green by FHVHV.
    """
    # Build share columns
    operators = ["yellow", "green", "fhv", "Juno", "Via", "Lyft", "Uber"]
    colors    = [COLORS[k] for k in operators]
    labels    = ["Yellow Taxi", "Green Taxi", "FHV (black cars)",
                 "Juno", "Via", "Lyft", "Uber"]

    shares = {}
    for op in operators:
        shares[op] = df[op] / df["all_total"].replace(0, np.nan) * 100

    shares_arr = np.array([shares[op].fillna(0).values for op in operators])

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.stackplot(df["date"], shares_arr, labels=labels, colors=colors, alpha=0.88)
    shade_covid(ax, alpha=0.06)

    ax.set_ylabel("Share of all on-demand trips (%)")
    ax.set_xlabel("Date")
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.set_title(
        "Market Share of On-Demand Trips by Operator (2019–2025)",
        fontsize=13, fontweight="bold"
    )
    # Legend reversed so top-stack item appears at top
    handles, lbls = ax.get_legend_handles_labels()
    # Add COVID patch manually
    covid_patch = mpatches.Patch(color="gray", alpha=0.25, label="COVID-19 period")
    ax.legend(
        list(reversed(handles)) + [covid_patch],
        list(reversed(lbls))    + ["COVID-19 period"],
        loc="upper left", fontsize=9
    )
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "02_market_share.png", bbox_inches="tight")
    plt.close()
    print("  Saved 02_market_share.png", flush=True)


# ── Plot 3: Indexed trends ────────────────────────────────────────────────

def plot_indexed(df):
    """
    Index Yellow+Green combined and Uber+Lyft combined to their Jan 2019
    values (= 100). Shows the inverse relationship more clearly than
    absolute counts, and is unaffected by the raw scale difference.
    """
    base = df[df["date"] == "2019-01-01"].iloc[0]

    def index_series(series, base_val):
        return series / base_val * 100

    taxi_series  = df["yellow"] + df["green"]
    large_series = df["Uber"]   + df["Lyft"]
    fhv_series   = df["fhv"]

    base_taxi  = (base["yellow"] + base["green"])
    base_large = max(base["Uber"] + base["Lyft"], 1)
    base_fhv   = max(base["fhv"], 1)

    fig, ax = plt.subplots(figsize=(13, 5))

    ax.plot(df["date"], index_series(taxi_series,  base_taxi),
            color=COLORS["yellow"], lw=2.5, label="Yellow + Green (combined)")
    ax.plot(df["date"], index_series(large_series, base_large),
            color=COLORS["Uber"], lw=2.5, label="Uber + Lyft (combined)")
    ax.plot(df["date"], index_series(fhv_series,   base_fhv),
            color=COLORS["fhv"], lw=1.5, ls="--", label="FHV (black cars)")

    ax.axhline(100, color="black", lw=0.8, ls=":", alpha=0.5)
    shade_covid(ax)

    ax.set_ylabel("Index (Jan 2019 = 100)")
    ax.set_xlabel("Date")
    ax.set_title(
        "Indexed Monthly Trips: Traditional Taxis vs. Uber + Lyft (Jan 2019 = 100)",
        fontsize=13, fontweight="bold"
    )
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "03_indexed_trends.png", bbox_inches="tight")
    plt.close()
    print("  Saved 03_indexed_trends.png", flush=True)


# ── Plot 4: FHVHV internal competition ───────────────────────────────────

def plot_fhvhv_competition(df):
    """
    Two sub-panels:
      Left  — absolute trips: Uber/Lyft vs Via/Juno (log scale to show
               the scale difference without flattening small operators)
      Right — small operators' share of total FHVHV trips over time,
               showing their progressive marginalisation.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fmt = mticker.FuncFormatter(millions)

    # Left: absolute (log scale)
    ax1.semilogy(df["date"], df["Uber"],  color=COLORS["Uber"],  lw=2.5, label="Uber")
    ax1.semilogy(df["date"], df["Lyft"],  color=COLORS["Lyft"],  lw=2,   label="Lyft")
    ax1.semilogy(df["date"], df["Via"].replace(0, np.nan),
                 color=COLORS["Via"],  lw=1.5, ls="--", label="Via")
    ax1.semilogy(df["date"], df["Juno"].replace(0, np.nan),
                 color=COLORS["Juno"], lw=1.5, ls=":",  label="Juno")
    shade_covid(ax1)
    ax1.set_ylabel("Monthly trips (log scale)")
    ax1.set_xlabel("Date")
    ax1.set_title("Absolute FHVHV Trips per Operator", fontweight="bold")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.25, which="both")
    ax1.yaxis.set_major_formatter(fmt)

    # Right: small-operator share of FHVHV total
    small_share = (df["small_fhvhv"] / df["fhvhv"].replace(0, np.nan) * 100).fillna(0)
    ax2.fill_between(df["date"], small_share,
                     color="#17a2b8", alpha=0.55, label="Via + Juno share")
    ax2.plot(df["date"], small_share, color="#17a2b8", lw=1.5)
    shade_covid(ax2, alpha=0.10)
    ax2.set_ylabel("Share of FHVHV trips (%)")
    ax2.set_xlabel("Date")
    ax2.set_ylim(0, None)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax2.set_title("Via + Juno Share of Total FHVHV Market", fontweight="bold")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.25)

    fig.suptitle(
        "FHVHV Internal Competition: Large vs. Small Operators",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "04_fhvhv_competition.png", bbox_inches="tight")
    plt.close()
    print("  Saved 04_fhvhv_competition.png", flush=True)


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60, flush=True)
    print("T8: FHVHV Emergence Analysis", flush=True)
    print("=" * 60, flush=True)

    df = aggregate_all()

    print("\n[Summary]", flush=True)
    print(f"  Date range : {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"  Months     : {len(df)}")

    # Sanity check: FHVHV total ≈ Uber+Lyft+Via+Juno
    diff = ((df["fhvhv"] - (df["Uber"] + df["Lyft"] + df["Via"] + df["Juno"])).abs()
            / df["fhvhv"].replace(0, 1) * 100).median()
    print(f"  FHVHV total vs. operator sum (median % diff): {diff:.2f}%")

    print("\n[Plots] Generating ...", flush=True)
    setup()
    plot_absolute(df)
    plot_market_share(df)
    plot_indexed(df)
    plot_fhvhv_competition(df)

    print("\nDone. Outputs in:", OUT_DIR, flush=True)


if __name__ == "__main__":
    main()
