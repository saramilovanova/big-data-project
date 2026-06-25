"""
T3 aggregations.
One combined temporal scan per dataset (year+month+hour+dow in a single pass).
One zone+fare scan per dataset. ~8 total DuckDB scans. 
"""

import zipfile, urllib.request
from pathlib import Path
import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import geopandas as gpd
import dask.dataframe as dd
from dask_jobqueue import SLURMCluster
from dask.distributed import Client

# ── Paths ──────────────────────────────────────────────────────────────────────

BASE_DIR   = Path("/d/hpc/projects/FRI/bigdata/students/sm_bv/final_project/data")
YELLOW_DIR = BASE_DIR / "yellow_normalized"
PART_DIR   = BASE_DIR / "partitioned"
OUT_DIR    = BASE_DIR / "t3"
PLOTS_DIR  = OUT_DIR / "plots"
ZONES_SHP  = OUT_DIR / "taxi_zones"   # extraction target dir

for d in [OUT_DIR, PLOTS_DIR]: d.mkdir(parents=True, exist_ok=True)

VALID_YEARS = {
    "yellow": range(2012, 2027),
    "green":  range(2014, 2027),
    "fhv":    range(2015, 2027),
    "fhvhv":  range(2019, 2027),
}
PCOL = {
    "yellow": "tpep_pickup_datetime",
    "green":  "lpep_pickup_datetime",
    "fhv":    "pickup_datetime",
    "fhvhv":  "pickup_datetime",
}

def collect_files(base_dir, valid_years):
    files = []
    for d in sorted(Path(base_dir).iterdir()):
        yr = d.name.replace("year=", "")
        if yr.isdigit() and int(yr) in valid_years:
            files.extend(sorted(d.glob("*.parquet")))
    return [str(f) for f in files]

FILES = {
    "yellow": collect_files(YELLOW_DIR,                VALID_YEARS["yellow"]),
    "green":  collect_files(PART_DIR/"green_tripdata", VALID_YEARS["green"]),
    "fhv":    collect_files(PART_DIR/"fhv_tripdata",   VALID_YEARS["fhv"]),
    "fhvhv":  collect_files(PART_DIR/"fhvhv_tripdata", VALID_YEARS["fhvhv"]),
}

for name, files in FILES.items():
    print(f"{name}: {len(files)} files", flush=True)

con = duckdb.connect()
con.execute("SET threads TO 4; SET memory_limit='16GB';")

COLORS = {"yellow":"#FFD700","green":"#2ca02c","fhv":"#1f77b4","fhvhv":"#d62728"}

def save(name, df):
    p = OUT_DIR / f"{name}.csv"
    df.to_csv(p, index=True)
    print(f"  saved {p.name}  ({len(df)} rows)", flush=True)

def already_done(*names):
    """Return True if all named CSVs exist — skip the scan."""
    return all((OUT_DIR / f"{n}.csv").exists() for n in names)

def savefig(name):
    p = PLOTS_DIR / f"{name}.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  plot  → {p.name}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# SCAN 1–4: One comprehensive temporal query per dataset
# Returns year+month+hour+dow+trips in one pass — derive all patterns in pandas
# ══════════════════════════════════════════════════════════════════════════════

PATTERN_Y0, PATTERN_Y1 = 2019, 2026   # extended to 2026

if already_done("yearly_trips","dow_trips","covid_monthly","hourly_trips"):
    print("\n[Temporal scans] Core CSVs exist — loading from disk.", flush=True)
    yearly = pd.read_csv(OUT_DIR/"yearly_trips.csv", index_col=0)
    dow_df = pd.read_csv(OUT_DIR/"dow_trips.csv",    index_col=0)
    covid  = pd.read_csv(OUT_DIR/"covid_monthly.csv",index_col=0, parse_dates=True)
    temporal = None  # not needed; monthly regenerated below
else:
    temporal = {}
    for ds, files in FILES.items():
        pcol = PCOL[ds]
        y0, y1 = min(VALID_YEARS[ds]), max(VALID_YEARS[ds])
        print(f"\n[Temporal scan] {ds}...", flush=True)
        df = con.execute(f"""
            SELECT
                year({pcol})        AS year,
                month({pcol})       AS month,
                hour({pcol})        AS hour,
                dayofweek({pcol})   AS dow,
                COUNT(*)            AS trips
            FROM read_parquet({files}, union_by_name=True)
            WHERE {pcol} IS NOT NULL
              AND year({pcol}) BETWEEN {y0} AND {y1}
            GROUP BY year, month, hour, dow
            ORDER BY year, month, hour, dow
        """).fetchdf()
        temporal[ds] = df
        print(f"  {len(df)} rows", flush=True)

    yearly  = pd.DataFrame({ds: t.groupby("year")["trips"].sum() for ds, t in temporal.items()})
    monthly = pd.DataFrame({
        ds: t[t["year"].between(PATTERN_Y0, PATTERN_Y1)].groupby("month")["trips"].sum()
        for ds, t in temporal.items()
    })
    hourly  = pd.DataFrame({
        ds: t[t["year"].between(PATTERN_Y0, PATTERN_Y1)].groupby("hour")["trips"].sum()
        for ds, t in temporal.items()
    })
    dow_df  = pd.DataFrame({
        ds: t[t["year"].between(PATTERN_Y0, PATTERN_Y1)].groupby("dow")["trips"].sum()
        for ds, t in temporal.items()
    })
    covid = pd.DataFrame({
        ds: t[t["year"].between(2019, 2022)].assign(
            date=lambda d: pd.to_datetime(d[["year","month"]].assign(day=1))
        ).groupby("date")["trips"].sum()
        for ds, t in temporal.items()
    })
    save("yearly_trips",  yearly)
    save("monthly_trips", monthly)
    save("hourly_trips",  hourly)
    save("dow_trips",     dow_df)
    save("covid_monthly", covid)
    temporal_loaded = True

# Always define MONTHLY_FILES so hourly/zone sections can use it
def collect_files_range(base_dir, valid_years, y0, y1):
    files = []
    for d in sorted(Path(base_dir).iterdir()):
        yr = d.name.replace("year=", "")
        if yr.isdigit() and y0 <= int(yr) <= y1 and int(yr) in valid_years:
            files.extend(sorted(d.glob("*.parquet")))
    return [str(f) for f in files]

MONTHLY_Y0, MONTHLY_Y1 = 2019, 2026
MONTHLY_FILES = {
    "yellow": collect_files_range(YELLOW_DIR,                VALID_YEARS["yellow"], MONTHLY_Y0, MONTHLY_Y1),
    "green":  collect_files_range(PART_DIR/"green_tripdata", VALID_YEARS["green"],  MONTHLY_Y0, MONTHLY_Y1),
    "fhv":    collect_files_range(PART_DIR/"fhv_tripdata",   VALID_YEARS["fhv"],    MONTHLY_Y0, MONTHLY_Y1),
    "fhvhv":  collect_files_range(PART_DIR/"fhvhv_tripdata", VALID_YEARS["fhvhv"],  MONTHLY_Y0, MONTHLY_Y1),
}

# Monthly — regenerate with 2019-2026 (skipped on future runs once saved)
if already_done("monthly_trips"):
    print("\n[Monthly] CSV exists — loading from disk.", flush=True)
    monthly = pd.read_csv(OUT_DIR/"monthly_trips.csv", index_col=0)
else:
    print("\n[Monthly 2019-2026] recomputing...", flush=True)

    def collect_files_range(base_dir, valid_years, y0, y1):
        files = []
        for d in sorted(Path(base_dir).iterdir()):
            yr = d.name.replace("year=", "")
            if yr.isdigit() and y0 <= int(yr) <= y1 and int(yr) in valid_years:
                files.extend(sorted(d.glob("*.parquet")))
        return [str(f) for f in files]

    MONTHLY_Y0, MONTHLY_Y1 = 2019, 2026
    MONTHLY_FILES = {
        "yellow": collect_files_range(YELLOW_DIR,                VALID_YEARS["yellow"], MONTHLY_Y0, MONTHLY_Y1),
        "green":  collect_files_range(PART_DIR/"green_tripdata", VALID_YEARS["green"],  MONTHLY_Y0, MONTHLY_Y1),
        "fhv":    collect_files_range(PART_DIR/"fhv_tripdata",   VALID_YEARS["fhv"],    MONTHLY_Y0, MONTHLY_Y1),
        "fhvhv":  collect_files_range(PART_DIR/"fhvhv_tripdata", VALID_YEARS["fhvhv"],  MONTHLY_Y0, MONTHLY_Y1),
    }
    for ds, files in MONTHLY_FILES.items():
        print(f"  {ds}: {len(files)} files (2019-2026 only)", flush=True)

    monthly_frames = {}
    for ds, files in MONTHLY_FILES.items():
        pcol = PCOL[ds]
        print(f"  querying {ds}...", flush=True)
        df = con.execute(f"""
            SELECT month({pcol}) AS month, COUNT(*) AS trips
            FROM read_parquet({files}, union_by_name=True)
            WHERE {pcol} IS NOT NULL
            GROUP BY 1 ORDER BY 1
        """).fetchdf().set_index("month")["trips"]
        monthly_frames[ds] = df
    monthly = pd.DataFrame(monthly_frames)
    save("monthly_trips", monthly)
    # make MONTHLY_FILES available for hourly/zone sections below
    globals().update({"MONTHLY_FILES": MONTHLY_FILES})


if already_done("hourly_trips"):
    print("\n[Hourly] CSV exists — loading from disk.", flush=True)
    hourly = pd.read_csv(OUT_DIR/"hourly_trips.csv", index_col=0)
else:
    print("\n[Hourly 2019-2026] recomputing...", flush=True)
    hourly_frames = {}
    for ds, files in MONTHLY_FILES.items():
        pcol = PCOL[ds]
        y0 = max(min(VALID_YEARS[ds]), 2019)
        y1 = min(max(VALID_YEARS[ds]), 2026)
        print(f"  querying {ds}...", flush=True)
        df = con.execute(f"""
            SELECT hour({pcol}) AS hour, COUNT(*) AS trips
            FROM read_parquet({files}, union_by_name=True)
            WHERE {pcol} IS NOT NULL AND year({pcol}) BETWEEN {y0} AND {y1}
            GROUP BY 1 ORDER BY 1
        """).fetchdf().set_index("hour")["trips"]
        hourly_frames[ds] = df
    hourly = pd.DataFrame(hourly_frames)
    save("hourly_trips", hourly)


# ══════════════════════════════════════════════════════════════════════════════
# SCAN 5–7: Fare + distance + zones — one query per relevant dataset
# ══════════════════════════════════════════════════════════════════════════════

FARE_CONF = [
    ("yellow", "tpep_pickup_datetime", "trip_distance", "fare_amount",        "PULocationID", "DOLocationID"),
    ("green",  "lpep_pickup_datetime", "trip_distance", "fare_amount",        "PULocationID", "DOLocationID"),
    ("fhvhv",  "pickup_datetime",      "trip_miles",    "base_passenger_fare","PULocationID", "DOLocationID"),
]

fare_rows = []

if already_done("fare_distance","fare_distribution"):
    print("\n[Fare scans] CSVs exist — loading from disk.", flush=True)
    fare_df = pd.read_csv(OUT_DIR/"fare_distance.csv", index_col=[0,1])
else:
    for ds, pcol, dcol, fcol, pu_col, do_col in FARE_CONF:
        files = FILES[ds]
        y0, y1 = min(VALID_YEARS[ds]), max(VALID_YEARS[ds])
        print(f"\n[Fare+Zone scan] {ds}...", flush=True)
        df = con.execute(f"""
            SELECT
                year({pcol})    AS year,
                {pu_col}        AS pu_zone,
                AVG({fcol})     AS avg_fare,
                AVG({dcol})     AS avg_dist,
                COUNT(*)        AS trips,
                FLOOR({fcol}/5)*5 AS fare_bucket
            FROM read_parquet({files}, union_by_name=True)
            WHERE {pcol} IS NOT NULL
              AND year({pcol}) BETWEEN {y0} AND {y1}
              AND {fcol} > 0 AND {dcol} > 0
            GROUP BY year, pu_zone, fare_bucket
        """).fetchdf()

        by_year = df.groupby("year").apply(
            lambda g: pd.Series({
                "avg_fare": (g["avg_fare"] * g["trips"]).sum() / g["trips"].sum(),
                "avg_dist": (g["avg_dist"] * g["trips"]).sum() / g["trips"].sum(),
            })
        )
        by_year["dataset"] = ds
        fare_rows.append(by_year)

        if ds == "yellow":
            fd = df[df["year"].between(2022, 2023)].groupby("fare_bucket")["trips"].sum()
            fd = fd[(fd.index >= 0) & (fd.index <= 200)]
            save("fare_distribution", fd.rename("cnt").to_frame())

# Zone pickups — skip if all CSVs exist, else recompute
ZONE_CONF = [
    ("yellow", "tpep_pickup_datetime", "PULocationID"),
    ("green",  "lpep_pickup_datetime", "PULocationID"),
    ("fhv",    "pickup_datetime",      "PUlocationID"),
    ("fhvhv",  "pickup_datetime",      "PULocationID"),
]
if already_done("zone_pickups_yellow","zone_pickups_green","zone_pickups_fhv","zone_pickups_fhvhv"):
    print("\n[Zone pickups] CSVs exist — skipping.", flush=True)
else:
    print("\n[Zone pickups 2019-2026] recomputing...", flush=True)
    for ds, pcol, pu_col in ZONE_CONF:
        if already_done(f"zone_pickups_{ds}"):
            print(f"  {ds}: CSV exists — skipping.", flush=True)
            continue
        files = MONTHLY_FILES[ds]
        y0 = max(min(VALID_YEARS[ds]), 2019)
        y1 = min(max(VALID_YEARS[ds]), 2026)
        print(f"  querying {ds}...", flush=True)
        df = con.execute(f"""
            SELECT {pu_col} AS pu_zone, COUNT(*) AS pickups
            FROM read_parquet({files}, union_by_name=True)
            WHERE {pcol} IS NOT NULL
              AND year({pcol}) BETWEEN {y0} AND {y1}
              AND {pu_col} IS NOT NULL
            GROUP BY 1
        """).fetchdf().set_index("pu_zone")
        save(f"zone_pickups_{ds}", df)

# fare_df concat only if fare_rows was populated (i.e. fare scan ran)
if fare_rows:
    fare_df = pd.concat(fare_rows).reset_index().set_index(["dataset","year"])
    save("fare_distance", fare_df)

# Top 10 zones from yellow (always refreshed with new zone_pickups)
pu_z = pd.read_csv(OUT_DIR/"zone_pickups_yellow.csv", index_col=0)
save("top_pu_zones", pu_z.nlargest(10, "pickups"))
save("top_do_zones", pu_z.nlargest(10, "pickups"))


if already_done("fhvhv_company"):
    print("\n[FHVHV company] CSV exists — skipping.", flush=True)
else:
    print("\n[FHVHV company scan]...", flush=True)
    files = FILES["fhvhv"]
    company = con.execute(f"""
        SELECT year(pickup_datetime) AS year, hvfhs_license_num AS company,
               COUNT(*) AS trips,
               SUM(CASE WHEN shared_request_flag IN ('Y','1') THEN 1 ELSE 0 END) AS shared
        FROM read_parquet({files}, union_by_name=True)
        WHERE pickup_datetime IS NOT NULL AND year(pickup_datetime) BETWEEN 2019 AND 2025
        GROUP BY 1, 2 ORDER BY 1, 3 DESC
    """).fetchdf()
    MAP = {"HV0002":"Juno","HV0003":"Uber","HV0004":"Via","HV0005":"Lyft"}
    company["company"] = company["company"].map(MAP).fillna("Other")
    save("fhvhv_company", company.set_index(["year","company"]))


# ══════════════════════════════════════════════════════════════════════════════
# Geographic choropleth maps
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Geographic maps]...", flush=True)

# Try known cluster paths for the taxi zones shapefile
CANDIDATE_PATHS = [
    Path("/d/hpc/projects/FRI/bigdata/students/sm_bv/data/taxi_zones/taxi_zones.shp"),
    Path("/d/hpc/projects/FRI/bigdata/students/sm_bv/final_project/data/t3/taxi_zones/taxi_zones/taxi_zones.shp"),
    Path("/d/hpc/home/sm79111/bigdata/taxi_zones/taxi_zones/taxi_zones.shp"),
]
shp_path = next((p for p in CANDIDATE_PATHS if p.exists()), None)

if shp_path is None:
    print("  Shapefile not found — skipping geographic maps.", flush=True)
    zones_gdf = None
else:
    print(f"  Using shapefile: {shp_path}", flush=True)
    zones_gdf = gpd.read_file(str(shp_path)).set_index("LocationID").to_crs("EPSG:4326")

if zones_gdf is not None:
    for ds, label, cmap in [
        ("yellow","Yellow Taxi","YlOrRd"),
        ("green", "Green Taxi", "YlGn"),
        ("fhv",   "FHV",        "Blues"),
        ("fhvhv", "FHVHV",      "Purples"),
    ]:
        counts = pd.read_csv(OUT_DIR/f"zone_pickups_{ds}.csv", index_col=0)
        gdf = zones_gdf.join(counts, how="left")
        gdf["pickups"] = gdf["pickups"].fillna(0)

        fig, ax = plt.subplots(figsize=(12, 10))
        gdf.plot(column="pickups", cmap=cmap, linewidth=0.3, edgecolor="grey",
                 legend=True, legend_kwds={"label":"Pickups (2019–2026)","shrink":0.6},
                 ax=ax, missing_kwds={"color":"lightgrey"})
        ax.set_title(f"{label} — Pickup Density by Taxi Zone (2019–2026)", fontsize=14)
        ax.axis("off"); savefig(f"map_pickups_{ds}")

    borough = {}
    for ds, label in [("yellow","Yellow"), ("green","Green"), ("fhv","FHV"), ("fhvhv","FHVHV")]:
        counts = pd.read_csv(OUT_DIR/f"zone_pickups_{ds}.csv", index_col=0)
        gdf = zones_gdf.join(counts, how="left")
        borough[label] = gdf.groupby("borough")["pickups"].sum().fillna(0)
    borough_df = pd.DataFrame(borough).fillna(0)
    borough_df.div(1e6).plot(kind="bar", figsize=(11, 5), colormap="tab10")
    plt.title("Pickups by NYC Borough (2019–2026)", fontsize=13)
    plt.xlabel("Borough"); plt.ylabel("Trips (M)")
    plt.xticks(rotation=30, ha="right"); plt.legend(); plt.grid(axis="y", alpha=0.3)
    savefig("map_borough_pickups")
else:
    print("  Skipped choropleth maps (no shapefile).", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Standard plots (all saved to plots/)
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Standard plots]...", flush=True)

fig, ax = plt.subplots(figsize=(13,5))
for col in yearly.columns:
    s = yearly[col].dropna()
    ax.plot(s.index.astype(int), s/1e6, marker="o", label=col, color=COLORS.get(col,"grey"))
ax.set_title("Annual Trip Volume by Dataset", fontsize=13)
ax.set_ylabel("Trips (M)"); ax.set_xlabel("Year")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:.0f}M"))
ax.legend(); ax.grid(True, alpha=0.3); plt.tight_layout(); savefig("annual_volumes")

month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
fig, axes = plt.subplots(1,4,figsize=(18,4))
for ax, col in zip(axes, monthly.columns):
    ax.bar(monthly.index.astype(int), monthly[col]/1e6, color=COLORS.get(col,"grey"), alpha=0.85)
    ax.set_title(col); ax.set_xticks(range(1,13))
    ax.set_xticklabels(month_labels, rotation=45, fontsize=7)
    ax.set_ylabel("Trips (M)"); ax.grid(axis="y", alpha=0.3)
plt.suptitle("Monthly Trip Distribution (2019–2026)", fontsize=13); plt.tight_layout()
savefig("monthly_seasonality")

fig, ax = plt.subplots(figsize=(13,5))
for col in hourly.columns:
    pct = hourly[col] / hourly[col].sum() * 100
    ax.plot(hourly.index.astype(int), pct, marker="o", label=col, color=COLORS.get(col,"grey"))
ax.set_title("Hourly Trip Distribution — % of Daily Volume (2019–2026)", fontsize=13)
ax.set_xlabel("Hour"); ax.set_ylabel("Share (%)")
ax.set_xticks(range(24)); ax.legend(); ax.grid(True, alpha=0.3); plt.tight_layout()
savefig("hourly_distribution")

day_labels = ["Sun","Mon","Tue","Wed","Thu","Fri","Sat"]
fig, axes = plt.subplots(1,4,figsize=(16,4))
for ax, col in zip(axes, dow_df.columns):
    ax.bar(dow_df.index.astype(int), dow_df[col]/1e6, color=COLORS.get(col,"grey"), alpha=0.85)
    ax.set_title(col); ax.set_xticks(range(7))
    ax.set_xticklabels(day_labels, rotation=45); ax.set_ylabel("Trips (M)"); ax.grid(axis="y",alpha=0.3)
plt.suptitle("Day-of-Week Distribution (2019–2026)", fontsize=13); plt.tight_layout()
savefig("dow_distribution")

fd = pd.read_csv(OUT_DIR/"fare_distribution.csv", index_col=0)
fig, ax = plt.subplots(figsize=(13,4))
ax.bar(fd.index.astype(float), fd["cnt"]/1e6, width=4.5,
       color="#FFD700", edgecolor="black", linewidth=0.3)
ax.set_title("Yellow Taxi — Fare Distribution (2022–2026)", fontsize=13)
ax.set_xlabel("Fare (USD)"); ax.set_ylabel("Trips (M)")
ax.grid(axis="y", alpha=0.3); plt.tight_layout(); savefig("fare_distribution")

fig, ax = plt.subplots(figsize=(14,5))
for col in covid.columns:
    s = covid[col].dropna()
    ax.plot(pd.to_datetime(s.index), s/1e6, label=col, color=COLORS.get(col,"grey"))
ax.axvline(pd.Timestamp("2020-03-01"), color="red", linestyle="--", alpha=0.7, label="COVID lockdown")
ax.set_title("Monthly Trip Volume 2019–2022", fontsize=13)
ax.set_ylabel("Trips (M)"); ax.legend(); ax.grid(True, alpha=0.3); plt.tight_layout()
savefig("covid_monthly")

fig, axes = plt.subplots(1,2,figsize=(14,5))
for ds in ["yellow","green","fhvhv"]:
    if ds not in fare_df.index.get_level_values(0): continue
    sub = fare_df.loc[ds]; sub.index = sub.index.astype(int)
    axes[0].plot(sub.index, sub["avg_fare"], marker="o", label=ds, color=COLORS.get(ds,"grey"))
    axes[1].plot(sub.index, sub["avg_dist"], marker="o", label=ds, color=COLORS.get(ds,"grey"))
axes[0].set_title("Average Fare by Year"); axes[0].set_ylabel("USD")
axes[1].set_title("Average Distance by Year"); axes[1].set_ylabel("Miles")
for ax in axes: ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); savefig("fare_distance")


print(f"\nDone. All results in {OUT_DIR}", flush=True)
print(f"Plots saved to {PLOTS_DIR}", flush=True)
