"""
T7: City-Wide On-Demand Transportation Demand Forecasting
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Problem Statement (suggestion e):
  Predict hourly trip counts per taxi zone across all 4 NYC TLC datasets
  (Yellow, Green, FHV, FHVHV).  The model uses temporal, weather, spatial,
  and event features — directly leveraging T5 augmentation.

  A unified model (all services + service_type feature) is compared against
  4 separate per-service models, evaluating whether cross-service patterns
  (e.g. the COVID-19 shift from Yellow → FHVHV observed in T3 EDA) add
  predictive value.

  Augmentation impact is also measured: baseline (time + zone features only)
  vs. augmented (+ weather, + spatial, + events from T5).

Algorithms (all three kinds required by the task):
  1. Dask-ML LinearRegression   — native distributed algorithm (Dask-ML)
  2. XGBoost with Dask backend  — third-party native distributed support
  3. SGDRegressor + partial_fit — scikit-learn incremental / out-of-core

Scalability: benchmarked with 2, 4, 8 Dask SLURM workers on Arnes HPC.

Train/test split: temporal — 2021–2023 train, 2024 test.
Target: log(1 + trip_count) per (pickup_zone, year, month, day, hour).

Outputs (data/t7/):
  cache/demand_all_services.parquet   — aggregated demand dataset (cached)
  results.json                        — all metrics, scalability timings
  results_table.csv                   — printable summary table
  xgb_feature_importance_*.csv        — XGBoost gain importances
  dask_report_{N}w.html               — Dask performance reports
  plots/model_comparison.png
  plots/unified_vs_separate.png
  plots/scalability.png
  plots/memory_vs_workers.png
  plots/augmentation_impact.png
  plots/sgd_convergence.png
  plots/feature_importance.png
"""

import os
import sys
import time
import json
import traceback
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

import dask
import dask.dataframe as dd
import dask.array as da
from dask.distributed import Client, wait, performance_report
from dask_jobqueue import SLURMCluster

# Dask-ML (Algorithm 1)
from dask_ml.preprocessing import StandardScaler as DaskStandardScaler
from dask_ml.linear_model import LinearRegression as DaskLinearRegression

# Scikit-learn (Algorithm 3)
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# XGBoost (Algorithm 2)
try:
    # import xgboost as xgb
    from xgboost import dask as dxgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    warnings.warn("XGBoost not available — install with: pip install xgboost")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

PROJECT   = Path("/d/hpc/projects/FRI/bigdata/students/sm_bv/final_project")
AUG_DIR   = PROJECT / "data/t5/augmented"
OUT_DIR   = PROJECT / "data/t7"
PLOTS_DIR = OUT_DIR / "plots"
CACHE_DIR = OUT_DIR / "cache"

for _d in [OUT_DIR, PLOTS_DIR, CACHE_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATASET CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DATASETS = {
    "yellow": {"pcol": "tpep_pickup_datetime", "pu_col": "PULocationID",  "id": 0},
    "green":  {"pcol": "lpep_pickup_datetime",  "pu_col": "PULocationID",  "id": 1},
    "fhv":    {"pcol": "pickup_datetime",        "pu_col": "PUlocationID",  "id": 2},  # lowercase l
    "fhvhv":  {"pcol": "pickup_datetime",        "pu_col": "PULocationID",  "id": 3},
}

TRAIN_YEARS = list(range(2021, 2024))   # 2021–2023
TEST_YEAR   = 2024

COLORS = {
    "yellow": "#FFD700",
    "green":  "#2ca02c",
    "fhv":    "#1f77b4",
    "fhvhv":  "#d62728",
}

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE GROUPS
# ─────────────────────────────────────────────────────────────────────────────

# Base features — time + zone only (no T5 augmentation)
BASE_FEATURES = [
    "hour_sin", "hour_cos",
    "dow_sin",  "dow_cos",
    "month_sin", "month_cos",
    "year_norm",
    "is_weekend", "is_rush_hour", "is_covid",
    "zone_id",
]

# Augmentation features — from T5 (weather + spatial + events)
AUG_FEATURES = [
    "temperature_c", "precipitation_mm", "cloudcover_pct", "windspeed_kmh",
    "is_raining",
    "pickup_school_count", "pickup_business_count",
    "pickup_attraction_count", "pickup_event_count",
]

ALL_FEATURES     = BASE_FEATURES + AUG_FEATURES           # per-service models
UNIFIED_FEATURES = ALL_FEATURES + ["service_type"]         # single unified model

TARGET     = "trip_count"
LOG_TARGET = "log_trip_count"

# ═════════════════════════════════════════════════════════════════════════════
# STEP 1: PREPROCESSING — aggregate raw trip data to zone-hour demand
# ═════════════════════════════════════════════════════════════════════════════

def aggregate_service(service: str, conf: dict, years: list) -> pd.DataFrame | None:
    """
    Load T5-augmented trip-level parquets for one service using Dask,
    aggregate to (PULocationID, year, month, day, hour) → trip_count + features.

    This is the distributed step: raw augmented data easily exceeds a single
    worker's RAM (Yellow alone is >100 GB across all years), so Dask reads
    and aggregates lazily across partitions.

    Fix vs. original draft: a dedicated "_count" column (value=1, sum-aggregated)
    is used for trip counting so that no weather/spatial column is silently
    lost by being used as the count proxy.
    """
    pcol   = conf["pcol"]
    pu_col = conf["pu_col"]

    weather_cols = [
        "temperature_c", "precipitation_mm", "rain_mm",
        "cloudcover_pct", "windspeed_kmh",
    ]
    spatial_cols = [
        "pickup_school_count", "pickup_business_count",
        "pickup_attraction_count", "pickup_event_count",
    ]

    parts = []
    for year in years:
        fpath = AUG_DIR / service / f"{year}.parquet"
        if not fpath.exists():
            print(f"  [SKIP] {fpath} not found", flush=True)
            continue

        print(f"  {service}/{year} ({fpath.stat().st_size/1e6:.0f} MB)...",
              end=" ", flush=True)

        # Dask lazy read — partitioned across multiple workers.
        # Each worker holds only a fraction of the year's data.
        ddf = dd.read_parquet(str(fpath), engine="pyarrow")

        # Standardise pickup zone column name (fhv uses lowercase 'l')
        if pu_col in ddf.columns and pu_col != "PULocationID":
            ddf = ddf.rename(columns={pu_col: "PULocationID"})
        if "PULocationID" not in ddf.columns:
            print(f"[WARN] PU column not found in {service}/{year}", flush=True)
            continue

        # Parse pickup datetime
        ddf[pcol] = dd.to_datetime(ddf[pcol], errors="coerce")
        ddf = ddf.dropna(subset=[pcol, "PULocationID"])

        # Extract temporal components
        ddf["year"]  = ddf[pcol].dt.year
        ddf["month"] = ddf[pcol].dt.month
        ddf["day"]   = ddf[pcol].dt.day
        ddf["hour"]  = ddf[pcol].dt.hour
        ddf["dow"]   = ddf[pcol].dt.dayofweek   # 0=Monday … 6=Sunday

        # Filter to target year (guard against bad timestamps)
        ddf = ddf[ddf["year"] == year]

        # ── Add a dedicated trip counter column ───────────────────────────
        # Using a dedicated _count=1 column (summed per group) avoids the
        # original bug of repurposing a weather column as the trip counter.
        ddf["_count"] = 1

        # Select only the columns we need (minimise data shuffle)
        group_cols = ["PULocationID", "year", "month", "day", "hour", "dow"]
        available_weather  = [c for c in weather_cols if c in ddf.columns]
        available_spatial  = [c for c in spatial_cols  if c in ddf.columns]
        keep = group_cols + ["_count"] + available_weather + available_spatial
        ddf = ddf[[c for c in keep if c in ddf.columns]]

        # Aggregate:
        #   - _count   → sum  (total trips in zone-hour)
        #   - weather  → first  (city-wide: same for every zone in that hour)
        #   - spatial  → first  (constant per zone; stable across time)
        agg_spec = {"_count": "sum"}
        for c in available_weather + available_spatial:
            agg_spec[c] = "first"

        agg = ddf.groupby(group_cols).agg(agg_spec).reset_index()
        agg = agg.rename(columns={"_count": "trip_count"})

        # Compute this year's aggregation (result fits in RAM)
        year_cache = CACHE_DIR / f"{service}_{year}.parquet"
        df_year = agg.compute()
        df_year["service"]      = service
        df_year["service_type"] = conf["id"]

        parts.append(df_year)
        # df_year.to_parquet(year_cache, write_index=False, overwrite=True)
        # del df_year

        print(f"{len(df_year):,} zone-hour rows", flush=True)

    if not parts:
        return None
    return pd.concat(parts, ignore_index=True)


def build_demand_dataset() -> pd.DataFrame:
    """Build the aggregated demand dataset; uses a cached parquet if available."""
    cache_file = CACHE_DIR / "demand_all_services.parquet"

    if cache_file.exists():
        print("\n[Preprocessing] Loading cached aggregated demand data...", flush=True)
        df = pd.read_parquet(str(cache_file))
        print(f"  {len(df):,} zone-hour rows loaded.", flush=True)
        return df

    print("\n[Preprocessing] Aggregating trips → zone-hour demand (all services)...",
          flush=True)
    all_parts = []

    for service, conf in DATASETS.items():
        print(f"\n  {service}:", flush=True)
        df_svc = aggregate_service(service, conf, TRAIN_YEARS + [TEST_YEAR])
        if df_svc is not None:
            all_parts.append(df_svc)
            print(f"  {service} total: {len(df_svc):,} rows", flush=True)

    if not all_parts:
        raise RuntimeError("No data found — check AUG_DIR path and T5 outputs.")

    df = pd.concat(all_parts, ignore_index=True)
    df.to_parquet(str(cache_file), index=False)
    print(f"\n  Cached → {cache_file.name} ({len(df):,} rows)", flush=True)
    return df

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cyclical time encodings, binary flags, and normalised year.
    Directly uses the COVID period identified in T3 EDA (March 2020 – June 2021).

    Works on a copy to avoid mutating the caller's dataframe.
    """
    df = df.copy()

    # Cyclical encodings — capture periodicity without ordinal bias
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]  / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]  / 24)
    df["dow_sin"]   = np.sin(2 * np.pi * df["dow"]   / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["dow"]   / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Year trend (0.0 = 2021, 1.0 = 2023)
    df["year_norm"] = (df["year"] - 2021) / 2.0

    # Zone ID (label-encoded; tree models handle ordinal, LR uses it linearly)
    df["zone_id"] = df["PULocationID"].astype(int)

    # Binary flags
    df["is_weekend"]   = (df["dow"] >= 5).astype(np.int8)
    df["is_rush_hour"] = (
        ((df["hour"] >= 7)  & (df["hour"] <= 9)) |
        ((df["hour"] >= 17) & (df["hour"] <= 19))
    ).astype(np.int8)

    # COVID-19 impact period (T3 EDA: massive demand crash observed here)
    df["is_covid"] = (
        ((df["year"] == 2020) & (df["month"] >= 3)) |
        ((df["year"] == 2021) & (df["month"] <= 6))
    ).astype(np.int8)

    # Rain binary — derived from T5 weather augmentation
    if "rain_mm" in df.columns:
        df["is_raining"] = (df["rain_mm"] > 0.5).astype(np.int8)
    elif "precipitation_mm" in df.columns:
        df["is_raining"] = (df["precipitation_mm"] > 0.5).astype(np.int8)
    else:
        df["is_raining"] = np.int8(0)

    # Log-transform target — trip counts are heavily right-skewed
    df[LOG_TARGET] = np.log1p(df[TARGET].clip(lower=0))

    return df


def get_Xy(df: pd.DataFrame, feature_cols: list,
           log_target: bool = True) -> tuple:
    """
    Extract feature matrix X and target vector y.
    Missing augmentation columns (some years/services may lack them) are
    zero-filled — not in-place on the caller's dataframe.
    """
    target_col = LOG_TARGET if log_target else TARGET
    X_vals = {}
    for col in feature_cols:
        if col in df.columns:
            X_vals[col] = df[col].fillna(0).values
        else:
            X_vals[col] = np.zeros(len(df), dtype=np.float32)

    X = np.column_stack([X_vals[c] for c in feature_cols]).astype(np.float32)
    y = df[target_col].fillna(0).values.astype(np.float32)
    return X, y

# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def eval_metrics(y_true_log: np.ndarray, y_pred_log: np.ndarray,
                 label: str = "") -> dict:
    """
    Evaluate on the original scale (expm1 of log predictions).
    Returns RMSE, MAE, R² in units of trips/hour.
    """
    y_pred = np.maximum(np.expm1(np.asarray(y_pred_log, dtype=np.float64)), 0)
    y_true = np.expm1(np.asarray(y_true_log, dtype=np.float64))

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))

    print(f"  [{label}]  RMSE={rmse:.2f}  MAE={mae:.2f}  R²={r2:.4f}", flush=True)
    return {"label": label, "rmse": rmse, "mae": mae, "r2": r2}


def mem_gb() -> float:
    """Current process + children RSS in GB (approximate)."""
    proc = psutil.Process()
    rss  = proc.memory_info().rss
    for child in proc.children(recursive=True):
        try:
            rss += child.memory_info().rss
        except Exception:
            pass
    return rss / 1e9

# ═════════════════════════════════════════════════════════════════════════════
# ALGORITHM 1: Dask-ML LinearRegression (native distributed algorithm)
# ═════════════════════════════════════════════════════════════════════════════

def run_dask_lr(client, X_tr, y_tr, X_te, y_te,
                n_workers: int, label: str) -> dict:
    """
    Dask-ML LinearRegression: data is chunked across workers, gradient
    computations are distributed, and results are aggregated by the scheduler.
    This is a native distributed algorithm from the dask_ml library.

    The chunk size is tuned so each worker holds ~4 partitions, giving
    sufficient parallelism while avoiding excessive task-graph overhead.
    """
    print(f"\n  [Dask-ML LR | {n_workers}w] Training ({len(X_tr):,} rows)...",
          flush=True)
    t0 = time.time()
    m0 = mem_gb()

    chunk = max(5_000, len(X_tr) // (n_workers * 4))

    X_da = da.from_array(X_tr, chunks=(chunk, -1))
    y_da = da.from_array(y_tr, chunks=chunk)

    # Scale and distribute data to workers
    scaler  = DaskStandardScaler()
    X_da_s  = scaler.fit_transform(X_da)
    X_da_s  = client.persist(X_da_s)
    y_da_p  = client.persist(y_da)
    wait([X_da_s, y_da_p])

    model = DaskLinearRegression(max_iter=300, tol=1e-4, solver="lbfgs")
    model.fit(X_da_s, y_da_p)

    X_te_s = scaler.transform(da.from_array(X_te, chunks=(len(X_te), -1)))
    y_pred = model.predict(X_te_s).compute()

    elapsed  = time.time() - t0
    mem_used = max(0.0, mem_gb() - m0)

    metrics = eval_metrics(y_te, y_pred, label=label)
    metrics.update({
        "algorithm":       "Dask-ML LinearRegression",
        "n_workers":       n_workers,
        "time_s":          elapsed,
        "memory_delta_gb": mem_used,
    })
    print(f"    Time: {elapsed:.1f}s  |  Mem Δ: {mem_used:.2f} GB", flush=True)
    return metrics

# ═════════════════════════════════════════════════════════════════════════════
# ALGORITHM 2: XGBoost with native Dask support (third-party distributed)
# ═════════════════════════════════════════════════════════════════════════════

def run_xgboost_dask(client, X_tr, y_tr, X_te, y_te,
                     n_workers: int, label: str,
                     n_rounds: int = 200) -> tuple | None:
    """
    XGBoost's xgb.dask module provides native distributed training:
    each Dask worker holds a partition of DMatrix and builds local trees,
    which are then reduced (AllReduce) across the cluster.
    tree_method='hist' is required for the distributed backend.

    Returns (metrics_dict, booster, y_pred_log).
    """
    if not HAS_XGB:
        print("  [XGBoost] SKIP — not installed", flush=True)
        return None

    print(f"\n  [XGBoost/Dask | {n_workers}w] Training ({len(X_tr):,} rows, "
          f"{n_rounds} rounds)...", flush=True)
    t0 = time.time()
    m0 = mem_gb()

    chunk  = max(5_000, len(X_tr) // (n_workers * 4))
    X_da   = da.from_array(X_tr, chunks=(chunk, -1))
    y_da   = da.from_array(y_tr, chunks=chunk)
    dtrain = dxgb.DaskDMatrix(client, X_da, y_da)

    params = {
        "objective":        "reg:squarederror",
        "max_depth":        6,
        "learning_rate":    0.08,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_lambda":       1.0,
        "tree_method":      "hist",   # mandatory for distributed XGBoost
        "device":           "cpu",
        "verbosity":        1,
    }

    output  = dxgb.train(
        client, params, dtrain,
        num_boost_round=n_rounds,
        evals=[(dtrain, "train")],
        verbose_eval=50,
    )
    booster = output["booster"]

    X_te_da = da.from_array(X_te, chunks=(len(X_te), -1))
    dtest   = dxgb.DaskDMatrix(client, X_te_da)
    y_pred  = dxgb.predict(client, booster, dtest).compute()

    elapsed  = time.time() - t0
    mem_used = max(0.0, mem_gb() - m0)

    metrics = eval_metrics(y_te, y_pred, label=label)
    metrics.update({
        "algorithm":       "XGBoost/Dask",
        "n_workers":       n_workers,
        "time_s":          elapsed,
        "memory_delta_gb": mem_used,
        "n_rounds":        n_rounds,
    })
    print(f"    Time: {elapsed:.1f}s  |  Mem Δ: {mem_used:.2f} GB", flush=True)

    # Save feature importances (gain = total reduction in loss per feature)
    safe_label = label.replace(" ", "_").replace("(", "").replace(")", "")[:60]
    fi_path    = OUT_DIR / f"xgb_feature_importance_{safe_label}.csv"
    fi_dict    = booster.get_score(importance_type="gain")
    pd.DataFrame(fi_dict.items(), columns=["feature", "importance"]).sort_values(
        "importance", ascending=False
    ).to_csv(fi_path, index=False)

    return metrics, booster, y_pred

# ═════════════════════════════════════════════════════════════════════════════
# ALGORITHM 3: SGDRegressor with partial_fit (out-of-core / incremental)
# ═════════════════════════════════════════════════════════════════════════════

def run_sgd_partial_fit(X_tr, y_tr, X_te, y_te,
                        label: str,
                        n_epochs: int   = 5,
                        chunk_size: int = 100_000) -> tuple:
    """
    SGDRegressor processes training data in equal-sized chunks using
    partial_fit(), simulating out-of-core learning: a single worker never
    loads the full dataset into RAM at once.

    Each epoch randomly shuffles the data before chunking to ensure
    stochastic behaviour.  The scaler is fit only on the first chunk and
    then applied incrementally — matching the out-of-core paradigm.

    Runs on the scheduler node (single process); n_workers is reported as 1.
    """
    print(f"\n  [SGD partial_fit] Training ({len(X_tr):,} rows, "
          f"{n_epochs} epochs, chunk={chunk_size:,})...", flush=True)
    t0 = time.time()
    m0 = mem_gb()

    scaler = SklearnStandardScaler()
    model  = SGDRegressor(
        loss          = "squared_error",
        max_iter      = 1,
        tol           = None,
        warm_start    = True,
        learning_rate = "adaptive",
        eta0          = 0.01,
        random_state  = 42,
    )

    n_chunks    = int(np.ceil(len(X_tr) / chunk_size))
    epoch_rmses = []
    scaler_fit  = False

    for epoch in range(n_epochs):
        rng      = np.random.RandomState(epoch)
        idx      = rng.permutation(len(X_tr))
        X_s, y_s = X_tr[idx], y_tr[idx]

        for i in range(n_chunks):
            Xc = X_s[i * chunk_size : (i + 1) * chunk_size]
            yc = y_s[i * chunk_size : (i + 1) * chunk_size]

            if not scaler_fit:
                Xc_s       = scaler.fit_transform(Xc)
                scaler_fit = True
            else:
                Xc_s = scaler.transform(Xc)

            model.partial_fit(Xc_s, yc)

        # Per-epoch validation (log-scale RMSE for convergence monitoring)
        X_te_s    = scaler.transform(X_te)
        y_pred_ep = model.predict(X_te_s)
        ep_rmse   = float(np.sqrt(mean_squared_error(y_te, y_pred_ep)))
        epoch_rmses.append(ep_rmse)
        print(f"    Epoch {epoch+1}/{n_epochs}  RMSE(log)={ep_rmse:.5f}", flush=True)

    # Final evaluation on original scale
    X_te_s = scaler.transform(X_te)
    y_pred = model.predict(X_te_s)

    elapsed  = time.time() - t0
    mem_used = max(0.0, mem_gb() - m0)

    metrics = eval_metrics(y_te, y_pred, label=label)
    metrics.update({
        "algorithm":       "SGDRegressor+partial_fit",
        "n_workers":       1,            # runs on scheduler node
        "time_s":          elapsed,
        "memory_delta_gb": mem_used,
        "n_epochs":        n_epochs,
        "n_chunks":        n_chunks,
        "chunk_size":      chunk_size,
        "epoch_log_rmses": epoch_rmses,
    })
    print(f"    Time: {elapsed:.1f}s  |  Mem Δ: {mem_used:.2f} GB", flush=True)
    return metrics, scaler, model

# ═════════════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT RUNNER
# ═════════════════════════════════════════════════════════════════════════════

def run_all(client, n_workers: int,
            train_df: pd.DataFrame,
            test_df:  pd.DataFrame) -> dict:
    """
    Run the full ML pipeline for a given number of Dask workers:
      • All 3 algorithms on the unified model (Section A)
      • Per-service separate XGBoost models (Section B)
      • Augmentation impact: with vs. without T5 features (Section C)

    Returns a results dict with all metrics and scalability info.
    """
    results = {
        "n_workers":   n_workers,
        "metrics":     [],
        "scalability": {},
    }
    t_start   = time.time()
    mem_start = mem_gb()

    # ── Prepare unified train/test arrays ─────────────────────────────────
    constant_cols = [
        c for c in UNIFIED_FEATURES
        if train_df[c].nunique() <= 1
    ]

    print("Removing constant columns:", constant_cols)

    features = [
        c for c in UNIFIED_FEATURES
        if c not in constant_cols
    ]
    # X_tr_u, y_tr_u = get_Xy(train_df, UNIFIED_FEATURES)
    # X_te_u, y_te_u = get_Xy(test_df,  UNIFIED_FEATURES)
    X_tr_u, y_tr_u = get_Xy(train_df, features)
    X_te_u, y_te_u = get_Xy(test_df, features)
    print(f"\n  Train: {len(X_tr_u):,}  |  Test: {len(X_te_u):,}  |  "
          f"Features: {len(features)} (of {len(UNIFIED_FEATURES)}, {len(constant_cols)} constant removed)"
          f"  |  Mem: {mem_start:.2f} GB", flush=True)

    # ═════════════════════════════════════════════════════════════════════
    # SECTION A: UNIFIED MODEL — all 4 services stacked
    # ═════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 60, flush=True)
    print("[SECTION A] UNIFIED MODEL — all services", flush=True)
    print("─" * 60, flush=True)

    # --- Algorithm 1: Dask-ML LinearRegression ---------------------------
    try:
        m = run_dask_lr(client, X_tr_u, y_tr_u, X_te_u, y_te_u,
                        n_workers, label="Dask-ML LR (unified)")
        results["metrics"].append(m)
    except Exception as e:
        print(f"  [ERROR] Dask-ML LR: {e}", flush=True)
        traceback.print_exc()

    # --- Algorithm 2: XGBoost/Dask ---------------------------------------
    xgb_unified_pred = None
    try:
        out = run_xgboost_dask(client, X_tr_u, y_tr_u, X_te_u, y_te_u,
                               n_workers, label="XGBoost (unified)",
                               n_rounds=200)
        if out:
            m, _, xgb_unified_pred = out
            results["metrics"].append(m)
    except Exception as e:
        print(f"  [ERROR] XGBoost unified: {e}", flush=True)
        traceback.print_exc()

    # --- Algorithm 3: SGDRegressor + partial_fit -------------------------
    try:
        m, _, _ = run_sgd_partial_fit(
            X_tr_u, y_tr_u, X_te_u, y_te_u,
            label="SGD partial_fit (unified)",
            n_epochs=5, chunk_size=200_000,
        )
        results["metrics"].append(m)
    except Exception as e:
        print(f"  [ERROR] SGD partial_fit: {e}", flush=True)
        traceback.print_exc()

    # ═════════════════════════════════════════════════════════════════════
    # SECTION B: SEPARATE PER-SERVICE MODELS (XGBoost)
    # Directly tests whether cross-service patterns (e.g. COVID-era Yellow
    # → FHVHV shift from T3 EDA) improve prediction.
    # ═════════════════════════════════════════════════════════════════════
    if HAS_XGB:
        print("\n" + "─" * 60, flush=True)
        print("[SECTION B] SEPARATE MODELS — per-service XGBoost", flush=True)
        print("─" * 60, flush=True)

        sep_preds  = []
        sep_truths = []

        for service in DATASETS:
            tr_s = train_df[train_df["service"] == service]
            te_s = test_df[test_df["service"]  == service]

            if len(te_s) == 0:
                print(f"  {service}: no 2024 test data, skipping", flush=True)
                continue

            X_tr_s, y_tr_s = get_Xy(tr_s, ALL_FEATURES)
            X_te_s, y_te_s = get_Xy(te_s, ALL_FEATURES)

            try:
                out = run_xgboost_dask(
                    client, X_tr_s, y_tr_s, X_te_s, y_te_s,
                    n_workers, label=f"XGBoost ({service})", n_rounds=150,
                )
                if out:
                    m, _, y_pred_svc = out
                    results["metrics"].append(m)
                    sep_preds.append(np.expm1(y_pred_svc.astype(np.float64)))
                    sep_truths.append(np.expm1(y_te_s.astype(np.float64)))
            except Exception as e:
                print(f"  [ERROR] XGBoost {service}: {e}", flush=True)
                traceback.print_exc()

        # Aggregate separate-model performance across all services
        if sep_preds and sep_truths:
            all_p = np.maximum(np.concatenate(sep_preds),  0)
            all_t = np.concatenate(sep_truths)

            rmse_s = float(np.sqrt(mean_squared_error(all_t, all_p)))
            mae_s  = float(mean_absolute_error(all_t, all_p))
            r2_s   = float(r2_score(all_t, all_p))

            sep_summary = {
                "label":     "XGBoost (4 separate — aggregated)",
                "algorithm": "XGBoost/Dask separate",
                "n_workers": n_workers,
                "rmse": rmse_s, "mae": mae_s, "r2": r2_s,
            }
            print(f"\n  [4-separate combined]  "
                  f"RMSE={rmse_s:.2f}  MAE={mae_s:.2f}  R²={r2_s:.4f}", flush=True)
            results["metrics"].append(sep_summary)

    # ═════════════════════════════════════════════════════════════════════
    # SECTION C: AUGMENTATION IMPACT — with vs. without T5 features
    # Directly quantifies the value of adding weather / spatial / events.
    # ═════════════════════════════════════════════════════════════════════
    if HAS_XGB:
        print("\n" + "─" * 60, flush=True)
        print("[SECTION C] AUGMENTATION IMPACT (XGBoost unified)", flush=True)
        print("─" * 60, flush=True)

        base_feats = BASE_FEATURES + ["service_type"]
        X_tr_b, y_tr_b = get_Xy(train_df, base_feats)
        X_te_b, y_te_b = get_Xy(test_df,  base_feats)

        try:
            out = run_xgboost_dask(
                client, X_tr_b, y_tr_b, X_te_b, y_te_b,
                n_workers, label="XGBoost (no augmentation)", n_rounds=150,
            )
            if out:
                m, _, _ = out
                m["augmentation_used"] = False
                results["metrics"].append(m)
        except Exception as e:
            print(f"  [ERROR] Baseline XGBoost: {e}", flush=True)
            traceback.print_exc()

        # Tag the full unified result for augmentation comparison
        for m in results["metrics"]:
            if m.get("label") == "XGBoost (unified)":
                m["augmentation_used"] = True

    # ── Scalability record ────────────────────────────────────────────────
    total_t = time.time() - t_start
    results["scalability"] = {
        "total_time_s":    total_t,
        "start_memory_gb": mem_start,
        "final_memory_gb": mem_gb(),
        "n_workers":       n_workers,
    }
    print(f"\n  Total experiment time ({n_workers} workers): {total_t:.1f}s",
          flush=True)
    return results

# ═════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═════════════════════════════════════════════════════════════════════════════

def make_plots(all_results: list):
    plt.rcParams.update({
        "figure.dpi": 150, "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    last = all_results[-1]

    # ── 1. Algorithm comparison (unified model) ─────────────────────────
    algo_metrics = [
        m for m in last["metrics"]
        if "rmse" in m
        and m.get("label", "").endswith("(unified)")
    ]
    if algo_metrics:
        labels = [m["label"].replace(" (unified)", "") for m in algo_metrics]
        rmses  = [m["rmse"] for m in algo_metrics]
        maes   = [m["mae"]  for m in algo_metrics]
        r2s    = [max(0.0, m["r2"]) for m in algo_metrics]
        colors = ["#4C72B0", "#DD8452", "#55A868"][:len(labels)]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, vals, title, xlabel in zip(
            axes,
            [rmses, maes, r2s],
            ["RMSE (trips/hr) ↓", "MAE (trips/hr) ↓", "R² ↑"],
            ["RMSE", "MAE", "R²"],
        ):
            ax.barh(labels, vals, color=colors)
            ax.set_xlabel(xlabel)
            ax.set_title(title)
            ax.grid(axis="x", alpha=0.3)
            if xlabel == "R²":
                ax.set_xlim(0, 1)

        plt.suptitle("Algorithm Comparison — Unified Demand Forecasting Model",
                     fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "model_comparison.png", bbox_inches="tight")
        plt.close()
        print("  Saved model_comparison.png", flush=True)

    # ── 2. Unified vs Separate models (XGBoost) ──────────────────────────
    unified_m  = next((m for m in last["metrics"]
                       if m.get("label") == "XGBoost (unified)"), None)
    separate_m = next((m for m in last["metrics"]
                       if "4 separate" in m.get("label", "")), None)

    if unified_m and separate_m:
        labels2  = ["Unified\n(1 model, 4 services)", "Separate\n(4 models aggregated)"]
        rmses2   = [unified_m["rmse"], separate_m["rmse"]]
        r2s2     = [max(0, unified_m["r2"]), max(0, separate_m["r2"])]
        bar_clr  = ["#4C72B0", "#DD8452"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.bar(labels2, rmses2, color=bar_clr, width=0.5)
        ax1.set_ylabel("RMSE (trips/hr)"); ax1.set_title("RMSE ↓")
        ax1.grid(axis="y", alpha=0.3)
        ax2.bar(labels2, r2s2, color=bar_clr, width=0.5)
        ax2.set_ylabel("R²"); ax2.set_title("R² ↑"); ax2.set_ylim(0, 1)
        ax2.grid(axis="y", alpha=0.3)

        plt.suptitle("Unified vs. 4 Separate Models (XGBoost)", fontweight="bold")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "unified_vs_separate.png", bbox_inches="tight")
        plt.close()
        print("  Saved unified_vs_separate.png", flush=True)

    # ── 3. Scalability: wall time and speedup vs workers ─────────────────
    if len(all_results) > 1:
        wkrs  = [r["n_workers"]                   for r in all_results]
        times = [r["scalability"]["total_time_s"] for r in all_results]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.plot(wkrs, times, "o-", lw=2, ms=8, color="#4C72B0", label="Actual")
        ideal = [times[0] * wkrs[0] / w for w in wkrs]
        ax1.plot(wkrs, ideal, "--", color="gray", label="Ideal linear speedup")
        ax1.set_xlabel("Workers"); ax1.set_ylabel("Total wall time (s)")
        ax1.set_title("Scalability: Wall Time vs Workers")
        ax1.legend(); ax1.grid(alpha=0.3); ax1.set_xticks(wkrs)

        speedup = [times[0] / t for t in times]
        ideal_s = [w / wkrs[0] for w in wkrs]
        ax2.plot(wkrs, speedup, "o-", lw=2, ms=8, color="#DD8452", label="Actual speedup")
        ax2.plot(wkrs, ideal_s, "--", color="gray", label="Ideal speedup")
        ax2.set_xlabel("Workers"); ax2.set_ylabel("Speedup factor (×)")
        ax2.set_title("Speedup Relative to Minimum Workers")
        ax2.legend(); ax2.grid(alpha=0.3); ax2.set_xticks(wkrs)

        plt.suptitle("Cluster Scalability — Arnes HPC (SLURM)", fontweight="bold")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "scalability.png", bbox_inches="tight")
        plt.close()
        print("  Saved scalability.png", flush=True)

    # ── 4. Memory usage vs workers ────────────────────────────────────────
    if len(all_results) > 1:
        wkrs = [r["n_workers"] for r in all_results]
        mems = [r["scalability"].get("final_memory_gb", 0) for r in all_results]

        plt.figure(figsize=(7, 4))
        plt.bar([str(w) for w in wkrs], mems, color="#55A868", alpha=0.85)
        plt.xlabel("Workers"); plt.ylabel("Total Memory (GB)")
        plt.title("Total Memory Consumption vs Number of Workers")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "memory_vs_workers.png", bbox_inches="tight")
        plt.close()
        print("  Saved memory_vs_workers.png", flush=True)

    # ── 5. Augmentation impact ────────────────────────────────────────────
    base_m = next((m for m in last["metrics"]
                   if "no augmentation" in m.get("label", "")), None)
    aug_m  = next((m for m in last["metrics"]
                   if m.get("label") == "XGBoost (unified)"), None)

    if base_m and aug_m:
        cats      = ["Baseline\n(time + zone)", "Augmented\n(+ weather + spatial + events)"]
        rmses_aug = [base_m["rmse"], aug_m["rmse"]]
        r2s_aug   = [max(0, base_m["r2"]), max(0, aug_m["r2"])]
        clrs      = ["#999999", "#2ca02c"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.bar(cats, rmses_aug, color=clrs, width=0.5)
        ax1.set_ylabel("RMSE (trips/hr)")
        ax1.grid(axis="y", alpha=0.3)
        ax2.bar(cats, r2s_aug, color=clrs, width=0.5)
        ax2.set_ylabel("R²"); ax2.set_ylim(0, 1); ax2.grid(axis="y", alpha=0.3)

        pct = (base_m["rmse"] - aug_m["rmse"]) / max(base_m["rmse"], 1e-6) * 100
        ax1.set_title(f"RMSE ↓  ({pct:+.1f}% from augmentation)")
        ax2.set_title("R² ↑")
        plt.suptitle("Impact of T5 Data Augmentation on Forecast Quality",
                     fontweight="bold")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "augmentation_impact.png", bbox_inches="tight")
        plt.close()
        print("  Saved augmentation_impact.png", flush=True)

    # ── 6. SGD convergence curve ─────────────────────────────────────────
    sgd_m = next((m for m in last["metrics"]
                  if "SGD" in m.get("algorithm", "")
                  and "epoch_log_rmses" in m), None)
    if sgd_m:
        epochs = list(range(1, len(sgd_m["epoch_log_rmses"]) + 1))
        plt.figure(figsize=(7, 4))
        plt.plot(epochs, sgd_m["epoch_log_rmses"], "o-",
                 color="darkorange", lw=2, ms=7)
        plt.xlabel("Epoch"); plt.ylabel("Test RMSE (log scale)")
        plt.title("SGD partial_fit Convergence per Epoch (unified model)")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "sgd_convergence.png", bbox_inches="tight")
        plt.close()
        print("  Saved sgd_convergence.png", flush=True)

    # ── 7. XGBoost feature importance (unified model) ─────────────────────
    fi_candidates = list(OUT_DIR.glob("xgb_feature_importance_*unified*.csv"))
    if fi_candidates:
        fi_path = fi_candidates[0]
        fi_df   = pd.read_csv(fi_path).head(20)
        plt.figure(figsize=(8, 6))
        plt.barh(fi_df["feature"][::-1], fi_df["importance"][::-1], color="#4C72B0")
        plt.xlabel("Gain (feature importance)")
        plt.title("Top-20 Feature Importances — XGBoost Unified Model")
        plt.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "feature_importance.png", bbox_inches="tight")
        plt.close()
        print("  Saved feature_importance.png", flush=True)

    # ── 8. Printable results table ────────────────────────────────────────
    algo_all = [m for m in last["metrics"] if "rmse" in m]
    if algo_all:
        tbl_cols = ["label", "algorithm", "n_workers",
                    "rmse", "mae", "r2", "time_s", "memory_delta_gb"]
        df_tbl = pd.DataFrame([
            {c: m.get(c) for c in tbl_cols} for m in algo_all
        ]).rename(columns={
            "label": "Model", "algorithm": "Algorithm",
            "n_workers": "Workers", "rmse": "RMSE",
            "mae": "MAE", "r2": "R²",
            "time_s": "Time (s)", "memory_delta_gb": "Mem Δ (GB)",
        })
        df_tbl.to_csv(OUT_DIR / "results_table.csv", index=False)
        print("\n  === RESULTS TABLE ===", flush=True)
        print(df_tbl.to_string(index=False), flush=True)

# ═════════════════════════════════════════════════════════════════════════════
# CLUSTER SETUP
# ═════════════════════════════════════════════════════════════════════════════

def make_cluster(n_workers: int, cores: int, mem: str, walltime: str):
    """
    Create a SLURMCluster and wait for workers to connect.

    The main SLURM job (run_t7.sh) is the scheduler/orchestrator.
    Each worker is spawned as its own sub-SLURM job by SLURMCluster.
    This is the standard Dask-on-HPC pattern.

    IMPORTANT — processes=1 is mandatory:
      Without it, dask_jobqueue defaults to processes=cores (here: 4), which
      means each SLURM job spawns 4 Dask worker processes that share the node's
      memory and cores. scale(n) then means "n worker processes", not "n SLURM
      jobs", so scale(2) and scale(4) both resolve to 1 SLURM job and produce
      identical clusters — making the scalability sweep meaningless.
      With processes=1, every scale(n) spawns exactly n SLURM jobs, each with
      1 worker, `cores` threads, and `mem` RAM, giving a true 2→4→8-node sweep.
    """
    cluster = SLURMCluster(
        queue     = "all",
        cores     = cores,
        memory    = mem,
        walltime  = walltime,
        processes = 1,        # 1 Dask worker per SLURM job — required for a
                              # meaningful worker-count scalability benchmark
        job_extra_directives=[
            "--nodes=1",
            f"--output={PROJECT}/logs/t7_worker_%j.out",
        ],
        env_extra=[
            "source /d/hpc/projects/FRI/bigdata/students/sm_bv/.venv/bin/activate",
        ],
    )
    cluster.scale(n_workers)
    client = Client(cluster)
    print(f"  Dask dashboard: {client.dashboard_link}", flush=True)
    client.wait_for_workers(n_workers, timeout=600)
    actual = len(client.scheduler_info()["workers"])
    print(f"  {actual}/{n_workers} workers ready.", flush=True)
    return cluster, client

# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="T7 — City-Wide Demand Forecasting")
    parser.add_argument("--workers",          type=int, nargs="+", default=[2, 4, 8],
                        help="Worker counts to benchmark (scalability sweep)")
    parser.add_argument("--cores-per-worker", type=int, default=4)
    parser.add_argument("--mem-per-worker",   type=str, default="16GB")
    parser.add_argument("--walltime",         type=str, default="01:00:00")
    parser.add_argument("--skip-scalability", action="store_true",
                        help="Run only the largest worker count (fastest option)")
    parser.add_argument("--rebuild-cache",    action="store_true",
                        help="Delete cached demand dataset and rebuild from scratch")
    args = parser.parse_args()

    # Optionally bust the demand cache
    cache_file = CACHE_DIR / "demand_all_services.parquet"
    if args.rebuild_cache and cache_file.exists():
        cache_file.unlink()
        print("[Cache] Deleted cached demand dataset — will rebuild.", flush=True)

    # Create logs directory
    (PROJECT / "logs").mkdir(exist_ok=True)

    worker_counts = (
        [max(args.workers)] if args.skip_scalability
        else sorted(args.workers)
    )

    print("=" * 70, flush=True)
    print("T7: City-Wide On-Demand Transportation Demand Forecasting", flush=True)
    print(f"  Problem  : Predict hourly trips per zone (all 4 services)", flush=True)
    print(f"  Workers  : {worker_counts}", flush=True)
    print(f"  Train    : {TRAIN_YEARS[0]}–{TRAIN_YEARS[-1]}", flush=True)
    print(f"  Test     : {TEST_YEAR}", flush=True)
    print(f"  Features : {len(UNIFIED_FEATURES)} (unified) / {len(ALL_FEATURES)} (per-service)", flush=True)
    print("=" * 70, flush=True)

    # ── Preprocessing (cached after first run) ────────────────────────────
    demand_df = build_demand_dataset()
    demand_df = engineer_features(demand_df)

    # Ensure all feature columns exist (fill missing with 0)
    for col in UNIFIED_FEATURES:
        if col not in demand_df.columns:
            demand_df[col] = 0.0
    demand_df = demand_df.fillna(0.0)

    train_df = demand_df[demand_df["year"].isin(TRAIN_YEARS)].reset_index(drop=True)
    test_df  = demand_df[demand_df["year"] == TEST_YEAR].reset_index(drop=True)

    print(f"\nDataset split:", flush=True)
    print(f"  Train : {len(train_df):,} zone-hour samples "
          f"({TRAIN_YEARS[0]}–{TRAIN_YEARS[-1]})", flush=True)
    print(f"  Test  : {len(test_df):,}  zone-hour samples ({TEST_YEAR})", flush=True)
    ram_mb = demand_df.memory_usage(deep=True).sum() / 1e6
    print(f"  RAM   : ~{ram_mb:.0f} MB — note: raw trip data (T5 inputs) is "
          f"orders of magnitude larger; the Dask aggregation in preprocessing "
          f"is what requires distributed memory.", flush=True)

    all_results = []

    # ── Scalability sweep: vary Dask worker count ─────────────────────────
    for n_workers in worker_counts:
        print("\n" + "=" * 70, flush=True)
        print(f"EXPERIMENT: {n_workers} WORKERS", flush=True)
        print("=" * 70, flush=True)

        cluster, client = make_cluster(
            n_workers, args.cores_per_worker,
            args.mem_per_worker, args.walltime,
        )

        try:
            report_path = str(OUT_DIR / f"dask_report_{n_workers}w.html")
            with performance_report(filename=report_path):
                result = run_all(client, n_workers, train_df, test_df)
            all_results.append(result)
        except Exception as e:
            print(f"[FATAL ERROR] {e}", flush=True)
            traceback.print_exc()
        finally:
            client.close()
            cluster.close()

        # Persist intermediate results after every worker-count experiment
        with open(OUT_DIR / "results.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"  results.json updated ({n_workers}w done)", flush=True)

    # ── Generate all plots ────────────────────────────────────────────────
    if all_results:
        print("\n[Plots] Generating...", flush=True)
        make_plots(all_results)

    # ── Final summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 70, flush=True)
    print("FINAL SUMMARY", flush=True)
    print("=" * 70, flush=True)
    if all_results:
        last = all_results[-1]
        for m in sorted(last["metrics"], key=lambda x: x.get("rmse", 9999)):
            if "rmse" in m:
                print(
                    f"  {m.get('label','?'):55s}  "
                    f"RMSE={m['rmse']:7.2f}  "
                    f"MAE={m['mae']:7.2f}  "
                    f"R²={m['r2']:+.4f}  "
                    f"Time={m.get('time_s', 0):.0f}s",
                    flush=True,
                )
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()