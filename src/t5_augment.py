"""
T5: Augment TLC taxi locations with:
  1. Weather (Open-Meteo hourly archive, NYC 2019-2026)
  2. Schools per taxi zone (spatial join)
  3. Businesses per taxi zone (spatial join)
  4. Attractions/Landmarks per taxi zone (download + spatial join)
  5. Events per zone per day (NYC Parks Events + spatial join)

Outputs (data/t5/):
  zone_spatial.parquet        - per zone: school/business/attraction counts
  weather_hourly.parquet      - per hour: temperature, precipitation, wind, etc.
  events_zone_day.parquet     - per zone+date: event count
  yellow_2023_augmented.parquet - demo: yellow taxi 2023 fully augmented
"""

import urllib.request, json, io
from pathlib import Path
import pandas as pd
import geopandas as gpd
import pyarrow.parquet as pq

# ── Paths ──────────────────────────────────────────────────────────────────────

DATA    = Path("/d/hpc/projects/FRI/bigdata/students/sm_bv/data")
OUT     = Path("/d/hpc/projects/FRI/bigdata/students/sm_bv/final_project/data/t5")
YELLOW_2023 = Path("/d/hpc/projects/FRI/bigdata/students/sm_bv/final_project"
                   "/data/yellow_normalized/2023/part-0.parquet")

ZONES_SHP    = DATA / "taxi_zones/taxi_zones.shp"
SCHOOLS_SHP  = DATA / "schools_and_events/SchoolPoints_APS_2024_08_28/SchoolPoints_APS_2024_08_28.shp"
BUSINESS_CSV = DATA / "business_and_attractions/businesses_20260406.csv"
EVENTS_CSV   = DATA / "schools_and_events/NYC_Parks_Events_Listing_Event_Listing_20260410.csv"
EVENTS_LOC   = DATA / "schools_and_events/NYC_Parks_Events_Listing_Event_Locations_20260410.csv"

OUT.mkdir(parents=True, exist_ok=True)


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode()


# ══════════════════════════════════════════════════════════════════════════════
# 1. WEATHER — Open-Meteo hourly archive for NYC
# ══════════════════════════════════════════════════════════════════════════════

WEATHER_OUT = OUT / "weather_hourly.parquet"

if not WEATHER_OUT.exists():
    print("\n[1] Downloading weather from Open-Meteo (2019-2026)...", flush=True)
    frames = []
    # Split into 2-year chunks to avoid timeouts
    for y_start, y_end in [("2019-01-01","2020-12-31"), ("2021-01-01","2022-12-31"),
                             ("2023-01-01","2024-12-31"), ("2025-01-01","2026-02-28")]:
        url = (
            "https://archive-api.open-meteo.com/v1/archive"
            f"?latitude=40.7128&longitude=-74.0060"
            f"&start_date={y_start}&end_date={y_end}"
            "&hourly=temperature_2m,precipitation,rain,cloudcover,windspeed_10m"
            "&timezone=America%2FNew_York&format=json"
        )
        print(f"  {y_start[:4]}–{y_end[:4]}...", end=" ", flush=True)
        data = json.loads(get(url))
        df = pd.DataFrame({
            "time":             pd.to_datetime(data["hourly"]["time"]),
            "temperature_c":    data["hourly"]["temperature_2m"],
            "precipitation_mm": data["hourly"]["precipitation"],
            "rain_mm":          data["hourly"]["rain"],
            "cloudcover_pct":   data["hourly"]["cloudcover"],
            "windspeed_kmh":    data["hourly"]["windspeed_10m"],
        })
        frames.append(df)
        print(f"{len(df):,} rows", flush=True)
    weather = pd.concat(frames).reset_index(drop=True)
    weather["time"] = weather["time"].dt.floor("h")
    weather.to_parquet(WEATHER_OUT, index=False)
    print(f"  Saved {len(weather):,} hourly records.", flush=True)
else:
    print("\n[1] Weather: loading from disk.", flush=True)
    weather = pd.read_parquet(WEATHER_OUT)


# ══════════════════════════════════════════════════════════════════════════════
# 2. ATTRACTIONS — Individual Landmark Sites (NYC Open Data)
# ══════════════════════════════════════════════════════════════════════════════

ATTRACT_CSV = OUT / "Individual_Landmark_Sites.csv"
attractions_raw = None

if ATTRACT_CSV.exists():
    print("\n[2] Attractions: loading from disk.", flush=True)
    attractions_raw = pd.read_csv(ATTRACT_CSV)
    print(f"  {len(attractions_raw):,} rows", flush=True)
else:
    print("\n[2] Downloading Individual Landmark Sites (LPC)...", flush=True)
    # Try several known NYC Open Data endpoint IDs
    for uid in ["hkd1-vwjz", "7j3b-f9gs", "c8p6-tgfm", "hkd1-vwjz"]:
        url = f"https://data.cityofnewyork.us/api/views/{uid}/rows.csv?accessType=DOWNLOAD"
        try:
            csv_text = get(url)
            # Verify it has the expected geometry column
            if "the_geom" in csv_text[:500]:
                ATTRACT_CSV.write_text(csv_text)
                attractions_raw = pd.read_csv(ATTRACT_CSV)
                print(f"  Saved {ATTRACT_CSV.name} ({len(attractions_raw):,} rows)", flush=True)
                break
            else:
                print(f"  {uid}: wrong dataset (no the_geom), skipping", flush=True)
        except Exception as e:
            print(f"  {uid}: {e}", flush=True)

    if attractions_raw is None:
        print("  Attractions not available — attraction_count will be 0.", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# 3. ZONE SPATIAL FEATURES — schools, businesses, attractions per zone
# ══════════════════════════════════════════════════════════════════════════════

SPATIAL_OUT = OUT / "zone_spatial.parquet"

if not SPATIAL_OUT.exists():
    print("\n[3] Building zone spatial features...", flush=True)

    zones = gpd.read_file(ZONES_SHP)[["LocationID","zone","borough","geometry"]].to_crs("EPSG:4326")

    # ── Schools ──────────────────────────────────────────────────────────────
    schools = gpd.read_file(SCHOOLS_SHP).to_crs("EPSG:4326")
    # keep only primary/high schools
    schools_filt = schools[schools["Latitude"].notna() & schools["Longitude"].notna()].copy()
    schools_geo  = gpd.GeoDataFrame(
        schools_filt,
        geometry=gpd.points_from_xy(schools_filt["Longitude"], schools_filt["Latitude"]),
        crs="EPSG:4326"
    )
    s_join = gpd.sjoin(schools_geo, zones, how="left", predicate="within")
    school_counts = s_join.groupby("LocationID").size().rename("school_count")
    print(f"  Schools: {len(schools_geo):,} points, {len(school_counts)} zones covered", flush=True)

    # ── Businesses ───────────────────────────────────────────────────────────
    biz = pd.read_csv(BUSINESS_CSV)
    biz = biz.dropna(subset=["Latitude","Longitude"])
    biz_geo = gpd.GeoDataFrame(
        biz,
        geometry=gpd.points_from_xy(biz["Longitude"], biz["Latitude"]),
        crs="EPSG:4326"
    )
    b_join = gpd.sjoin(biz_geo, zones, how="left", predicate="within")
    biz_counts = b_join.groupby("LocationID").size().rename("business_count")
    print(f"  Businesses: {len(biz_geo):,} points, {len(biz_counts)} zones covered", flush=True)

    # ── Attractions ───────────────────────────────────────────────────────────
    attr_counts = pd.Series(dtype=int, name="attraction_count")
    if attractions_raw is not None and "the_geom" in attractions_raw.columns:
        try:
            from shapely import wkt as shapely_wkt
            attr = attractions_raw.copy()
            attr["geometry"] = attr["the_geom"].apply(shapely_wkt.loads)
            attr_geo = gpd.GeoDataFrame(attr, geometry="geometry", crs="EPSG:4326")
            attr_geo_proj = attr_geo.to_crs("EPSG:2263")  # NY State Plane
            attr_geo["geometry"] = attr_geo_proj.centroid.to_crs("EPSG:4326")
            # attr_geo = attr_geo.set_geometry(attr_geo.geometry.centroid)
            a_join = gpd.sjoin(attr_geo, zones, how="left", predicate="within")
            attr_counts = a_join.groupby("LocationID").size().rename("attraction_count")
            print(f"  Attractions: {len(attr_geo):,} points, {len(attr_counts)} zones covered", flush=True)
        except Exception as e:
            print(f"  Attractions: skipped ({e})", flush=True)
    else:
        print("  Attractions: not available, attraction_count = 0", flush=True)

    # ── Combine into zone feature table ──────────────────────────────────────
    zone_ids = zones[["LocationID","zone","borough"]].set_index("LocationID")
    zone_features = zone_ids.join(school_counts).join(biz_counts).join(attr_counts)
    zone_features = zone_features.fillna(0).astype(
        {"school_count": int, "business_count": int,
         "attraction_count": int if "attraction_count" in zone_features.columns else float}
    )
    zone_features.to_parquet(SPATIAL_OUT)
    print(f"  Saved zone_spatial.parquet ({len(zone_features)} zones)", flush=True)

else:
    print("\n[3] Zone spatial features: loading from disk.", flush=True)
    zone_features = pd.read_parquet(SPATIAL_OUT)

print(zone_features.describe().to_string(), flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# 4. EVENTS — NYC Parks events per zone per day
# ══════════════════════════════════════════════════════════════════════════════

EVENTS_OUT = OUT / "events_zone_day.parquet"

if not EVENTS_OUT.exists():
    print("\n[4] Building events per zone per day...", flush=True)

    events = pd.read_csv(EVENTS_CSV)
    locs   = pd.read_csv(EVENTS_LOC)

    events_full = events.merge(locs, on="event_id", how="inner")
    events_full["date"] = pd.to_datetime(events_full["date"], errors="coerce")
    events_full = events_full.dropna(subset=["date","lat","long"])

    events_geo = gpd.GeoDataFrame(
        events_full,
        geometry=gpd.points_from_xy(events_full["long"], events_full["lat"]),
        crs="EPSG:4326"
    )
    zones = gpd.read_file(ZONES_SHP)[["LocationID","geometry"]].to_crs("EPSG:4326")
    e_join = gpd.sjoin(events_geo, zones, how="left", predicate="within")

    events_agg = (
        e_join.dropna(subset=["LocationID"])
        .groupby(["LocationID", "date"])
        .size()
        .reset_index(name="event_count")
    )
    events_agg["LocationID"] = events_agg["LocationID"].astype(int)
    events_agg.to_parquet(EVENTS_OUT, index=False)
    print(f"  Saved events_zone_day.parquet ({len(events_agg):,} zone-day rows)", flush=True)

else:
    print("\n[4] Events: loading from disk.", flush=True)
    events_agg = pd.read_parquet(EVENTS_OUT)


# ══════════════════════════════════════════════════════════════════════════════
# 5. AUGMENT all four datasets — year by year
# ══════════════════════════════════════════════════════════════════════════════

PART_DIR  = Path("/d/hpc/projects/FRI/bigdata/students/sm_bv/final_project/data/partitioned")
YELLOW_DIR = Path("/d/hpc/projects/FRI/bigdata/students/sm_bv/final_project/data/yellow_normalized")
AUG_DIR   = OUT / "augmented"
AUG_DIR.mkdir(exist_ok=True)

# pickup column and zone column names per dataset
DATASET_CONF = {
    "yellow": {
        "src_dir":   YELLOW_DIR,
        "pcol":      "tpep_pickup_datetime",
        "pu_col":    "PULocationID",
        "do_col":    "DOLocationID",
        "min_year":  2012, "max_year": 2026,
    },
    "green": {
        "src_dir":   PART_DIR / "green_tripdata",
        "pcol":      "lpep_pickup_datetime",
        "pu_col":    "PULocationID",
        "do_col":    "DOLocationID",
        "min_year":  2014, "max_year": 2026,
    },
    "fhv": {
        "src_dir":   PART_DIR / "fhv_tripdata",
        "pcol":      "pickup_datetime",
        "pu_col":    "PUlocationID",
        "do_col":    "DOlocationID",
        "min_year":  2015, "max_year": 2026,
    },
    "fhvhv": {
        "src_dir":   PART_DIR / "fhvhv_tripdata",
        "pcol":      "pickup_datetime",
        "pu_col":    "PULocationID",
        "do_col":    "DOLocationID",
        "min_year":  2019, "max_year": 2026,
    },
}

CHUNK_SIZE = 2_000_000   # rows per chunk — keep peak RAM ~4 GB per chunk

# Prepare lookup tables once
weather["time"]    = pd.to_datetime(weather["time"])
events_agg["date"] = pd.to_datetime(events_agg["date"]).dt.normalize()
zf = zone_features.reset_index()[["LocationID","school_count","business_count","attraction_count"]]


def augment_df(df, pcol, pu_col, do_col):
    """Apply all augmentations to a single-year DataFrame."""

    # Weather (on pickup hour)
    df["_ph"] = pd.to_datetime(df[pcol]).dt.floor("h")
    df = df.merge(weather.rename(columns={"time":"_ph"}), on="_ph", how="left")
    df = df.drop(columns=["_ph"])

    # Spatial features — pickup zone
    pu_merge = (zf.rename(columns={c: f"pickup_{c}"
                                    for c in ["school_count","business_count","attraction_count"]})
                  .rename(columns={"LocationID": pu_col}))
    df = df.merge(pu_merge, on=pu_col, how="left")

    # Spatial features — dropoff zone
    do_merge = (zf.rename(columns={c: f"dropoff_{c}"
                                    for c in ["school_count","business_count","attraction_count"]})
                  .rename(columns={"LocationID": do_col}))
    df = df.merge(do_merge, on=do_col, how="left")

    # Events (pickup zone + pickup date)
    df["_pd"] = pd.to_datetime(df[pcol]).dt.normalize()
    ev = (events_agg.rename(columns={"LocationID": pu_col,
                                      "date": "_pd",
                                      "event_count": "pickup_event_count"}))
    df = df.merge(ev, on=[pu_col, "_pd"], how="left").drop(columns=["_pd"])

    # Fill nulls for count columns
    for col in ["pickup_school_count","dropoff_school_count",
                "pickup_business_count","dropoff_business_count",
                "pickup_attraction_count","dropoff_attraction_count",
                "pickup_event_count"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df


print("\n[5] Augmenting all four datasets...", flush=True)

for ds_name, conf in DATASET_CONF.items():
    src_dir = conf["src_dir"]
    pcol    = conf["pcol"]
    pu_col  = conf["pu_col"]
    do_col  = conf["do_col"]

    ds_out = AUG_DIR / ds_name
    ds_out.mkdir(exist_ok=True)

    min_yr = conf["min_year"]
    max_yr = conf["max_year"]

    # find valid year directories only
    year_dirs = sorted([
        d for d in src_dir.iterdir()
        if d.is_dir()
        and d.name.replace("year=","").isdigit()
        and min_yr <= int(d.name.replace("year=","")) <= max_yr
    ])

    print(f"\n  {ds_name}: {len(year_dirs)} valid years ({min_yr}–{max_yr})", flush=True)

    for year_dir in year_dirs:
        yr_str = year_dir.name.replace("year=","")
        out_file = ds_out / f"{yr_str}.parquet"
        if out_file.exists():
            print(f"    {yr_str}: already done", flush=True)
            continue
        parquets = list(year_dir.glob("*.parquet"))
        if not parquets:
            continue

        file_mb = parquets[0].stat().st_size / 1e6
        print(f"    {yr_str}: {file_mb:.0f} MB...", end=" ", flush=True)

        if file_mb > 200:
            # stream chunks directly to parquet — never accumulate all in RAM
            import pyarrow as pa
            writer = None
            total_rows = 0
            pf = pq.ParquetFile(parquets[0])
            batch_no = 0
            for batch in pf.iter_batches(batch_size=CHUNK_SIZE):
                batch_no += 1
                print(f"      batch {batch_no}", flush=True)
                chunk = batch.to_pandas()
                chunk = augment_df(chunk, pcol, pu_col, do_col)
                table = pa.Table.from_pandas(chunk, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(str(out_file), table.schema)
                writer.write_table(table)
                total_rows += len(chunk)
                del chunk, table
            if writer:
                writer.close()
            print(f"{total_rows:,} rows → {out_file.name}", flush=True)
        else:
            result = augment_df(pd.read_parquet(parquets[0]), pcol, pu_col, do_col)
            result.to_parquet(out_file, index=False)
            print(f"{len(result):,} rows → {out_file.name}", flush=True)
            del result

print("\nAll augmentations complete.", flush=True)
