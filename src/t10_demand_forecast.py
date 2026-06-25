"""
T10: City-Wide Demand Forecasting — Distributed CPU+GPU Processing

Extends T7 by replacing every algorithm with its GPU-accelerated equivalent,
using the same demand dataset (reads from the T7 cache) and identical feature
engineering, evaluation, and plotting code.

GPU Algorithms (all three required categories):
  1. cuML dask LinearRegression — native multi-GPU distributed (cuML dask)
  2. XGBoost + device='cuda'   — third-party native distributed GPU support
  3. cuML MBSGDRegressor        — GPU-accelerated mini-batch SGD (single GPU)

Cluster:  LocalCUDACluster — all GPU workers on one SLURM node.
          Scalability sweep: 1 -> 2 -> 4 GPUs (auto-detected from --gpus arg).

Key differences vs T7:
  - SLURMCluster -> LocalCUDACluster (intra-node multi-GPU, no sub-jobs)
  - numpy arrays -> cupy-backed dask arrays for GPU workers
  - pynvml used for VRAM tracking alongside psutil for CPU RAM
  - MBSGDRegressor replaces SGDRegressor (cuML has no partial_fit;
    all data fits in 32 GB V100S VRAM, so out-of-core is not required on GPU)
  - Added CPU vs GPU wall-time comparison plot

Outputs (data/t10/):
  results.json                  — all metrics and GPU scalability timings
  results_table.csv             — printable summary table
  xgb_feature_importance_*.csv  — XGBoost gain importances
  dask_report_{N}gpu.html       — Dask performance reports
  plots/model_comparison.png
  plots/unified_vs_separate.png
  plots/scalability.png
  plots/gpu_memory.png
  plots/augmentation_impact.png
  plots/sgd_convergence.png     — MBSGDRegressor loss curve (if available)
  plots/feature_importance.png
  plots/cpu_vs_gpu.png          — T7 vs T10 wall-time comparison
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

# ── GPU imports ───────────────────────────────────────────────────────────────
import cupy as cp
from dask_cuda import LocalCUDACluster

import dask
import dask.dataframe as dd
import dask.array as da
from dask.distributed import Client, wait, performance_report

# cuML — Algorithm 1 (multi-GPU distributed LinearRegression)
from cuml.dask.linear_model import LinearRegression as cuMLLinearRegression

# cuML — Algorithm 3 (single-GPU mini-batch SGD)
from cuml.linear_model import MBSGDRegressor
import cuml.preprocessing

# Scikit-learn — CPU scaler used before GPU transfer (reliable baseline)
from sklearn.preprocessing import StandardScaler as SklearnScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# XGBoost — Algorithm 2 (native distributed GPU via device='cuda')
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    warnings.warn("XGBoost not available — pip install xgboost")

# pynvml — VRAM tracking (installed as nvidia-ml-py with RAPIDS)
try:
    import pynvml
    pynvml.nvmlInit()
    NVML_OK = True
except Exception:
    NVML_OK = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

PROJECT   = Path("/d/hpc/projects/FRI/bigdata/students/sm_bv/final_project")
AUG_DIR   = PROJECT / "data/t5/augmented"
OUT_DIR   = PROJECT / "data/t10"
PLOTS_DIR = OUT_DIR / "plots"
# Read the already-built demand cache from T7 — no need to re-aggregate
CACHE_DIR = PROJECT / "data/t7/cache"
T7_RESULTS = PROJECT / "data/t7/results.json"   # for CPU vs GPU comparison plot

for _d in [OUT_DIR, PLOTS_DIR, CACHE_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATASET CONFIGURATION  (identical to T7)
# ─────────────────────────────────────────────────────────────────────────────

DATASETS = {
    "yellow": {"pcol": "tpep_pickup_datetime", "pu_col": "PULocationID",  "id": 0},
    "green":  {"pcol": "lpep_pickup_datetime",  "pu_col": "PULocationID",  "id": 1},
    "fhv":    {"pcol": "pickup_datetime",        "pu_col": "PUlocationID",  "id": 2},
    "fhvhv":  {"pcol": "pickup_datetime",        "pu_col": "PULocationID",  "id": 3},
}

TRAIN_YEARS = list(range(2021, 2024))   # 2021–2023  (matches T7 cache)
TEST_YEAR   = 2024

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE GROUPS  (identical to T7)
# ─────────────────────────────────────────────────────────────────────────────

BASE_FEATURES = [
    "hour_sin", "hour_cos",
    "dow_sin",  "dow_cos",
    "month_sin", "month_cos",
    "year_norm",
    "is_weekend", "is_rush_hour", "is_covid",
    "zone_id",
]

AUG_FEATURES = [
    "temperature_c", "precipitation_mm", "cloudcover_pct", "windspeed_kmh",
    "is_raining",
    "pickup_school_count", "pickup_business_count",
    "pickup_attraction_count", "pickup_event_count",
]

ALL_FEATURES     = BASE_FEATURES + AUG_FEATURES
UNIFIED_FEATURES = ALL_FEATURES + ["service_type"]

TARGET     = "trip_count"
LOG_TARGET = "log_trip_count"

# ═════════════════════════════════════════════════════════════════════════════
# DATA PIPELINE — identical to T7; reads from T7 cache
# ═════════════════════════════════════════════════════════════════════════════

def build_demand_dataset() -> pd.DataFrame:
    """Load the aggregated demand dataset from T7 cache (already built)."""
    cache_file = CACHE_DIR / "demand_all_services.parquet"
    if not cache_file.exists():
        raise FileNotFoundError(
            f"T7 cache not found at {cache_file}. Run T7 first to build it."
        )
    print("\n[Data] Loading T7 demand cache...", flush=True)
    df = pd.read_parquet(str(cache_file))
    print(f"  {len(df):,} zone-hour rows loaded.", flush=True)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering — identical to T7."""
    df = df.copy()
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]  / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]  / 24)
    df["dow_sin"]   = np.sin(2 * np.pi * df["dow"]   / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["dow"]   / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["year_norm"] = (df["year"] - 2021) / 2.0
    df["zone_id"]   = df["PULocationID"].astype(int)
    df["is_weekend"]   = (df["dow"] >= 5).astype(np.int8)
    df["is_rush_hour"] = (
        ((df["hour"] >= 7)  & (df["hour"] <= 9)) |
        ((df["hour"] >= 17) & (df["hour"] <= 19))
    ).astype(np.int8)
    df["is_covid"] = (
        ((df["year"] == 2020) & (df["month"] >= 3)) |
        ((df["year"] == 2021) & (df["month"] <= 6))
    ).astype(np.int8)
    if "rain_mm" in df.columns:
        df["is_raining"] = (df["rain_mm"] > 0.5).astype(np.int8)
    elif "precipitation_mm" in df.columns:
        df["is_raining"] = (df["precipitation_mm"] > 0.5).astype(np.int8)
    else:
        df["is_raining"] = np.int8(0)
    df[LOG_TARGET] = np.log1p(df[TARGET].clip(lower=0))
    return df


def get_Xy(df: pd.DataFrame, feature_cols: list,
           log_target: bool = True) -> tuple:
    """Extract numpy feature matrix and target vector — identical to T7."""
    target_col = LOG_TARGET if log_target else TARGET
    X_vals = {
        col: (df[col].fillna(0).values if col in df.columns
              else np.zeros(len(df), dtype=np.float32))
        for col in feature_cols
    }
    X = np.column_stack([X_vals[c] for c in feature_cols]).astype(np.float32)
    y = df[target_col].fillna(0).values.astype(np.float32)
    return X, y

# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def eval_metrics(y_true_log: np.ndarray, y_pred_log: np.ndarray,
                 label: str = "") -> dict:
    """Evaluate on original scale — identical to T7."""
    y_pred = np.maximum(np.expm1(np.asarray(y_pred_log, dtype=np.float64)), 0)
    y_true = np.expm1(np.asarray(y_true_log, dtype=np.float64))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    print(f"  [{label}]  RMSE={rmse:.2f}  MAE={mae:.2f}  R²={r2:.4f}", flush=True)
    return {"label": label, "rmse": rmse, "mae": mae, "r2": r2}


def cpu_mem_gb() -> float:
    """Current process + children RSS in GB."""
    proc = psutil.Process()
    rss  = proc.memory_info().rss
    for child in proc.children(recursive=True):
        try:
            rss += child.memory_info().rss
        except Exception:
            pass
    return rss / 1e9


def gpu_mem_gb(device_indices=None) -> float:
    """Total GPU VRAM used across specified device indices, in GB."""
    if not NVML_OK:
        return 0.0
    try:
        count = pynvml.nvmlDeviceGetCount()
        if device_indices is None:
            device_indices = range(count)
        total = sum(
            pynvml.nvmlDeviceGetMemoryInfo(
                pynvml.nvmlDeviceGetHandleByIndex(i)
            ).used
            for i in device_indices if i < count
        )
        return total / 1e9
    except Exception:
        return 0.0

# ═════════════════════════════════════════════════════════════════════════════
# ALGORITHM 1: cuML dask LinearRegression — native multi-GPU distributed
# ═════════════════════════════════════════════════════════════════════════════

def run_cuml_lr(client, X_tr, y_tr, X_te, y_te,
                n_gpus: int, label: str, feature_cols: list) -> dict:
    """
    cuML's dask LinearRegression distributes gradient computation across all
    GPU workers using the same lbfgs-style solver as Dask-ML LR, but on GPU.

    Data pipeline:
      numpy (CPU) -> sklearn StandardScaler (CPU, reliable) ->
      scaled numpy -> dask array -> map_blocks(cp.asarray) -> GPU workers

    The map_blocks(cp.asarray) step converts each CPU chunk to a CuPy array
    lazily on the assigned GPU worker, avoiding a single-GPU memory bottleneck.
    The cuMLLinearRegression(client=client) call distributes the solve across
    all n_gpus workers.
    """
    print(f"\n  [cuML LR | {n_gpus} GPU] Training ({len(X_tr):,} rows)...",
          flush=True)
    t0  = time.time()
    gm0 = gpu_mem_gb(range(n_gpus))
    cm0 = cpu_mem_gb()

    chunk = max(5_000, len(X_tr) // (n_gpus * 4))

    # Scale on CPU first — avoids cuML scaler version compatibility issues
    scaler   = SklearnScaler()
    X_tr_s   = scaler.fit_transform(X_tr).astype(np.float32)
    X_te_s   = scaler.transform(X_te).astype(np.float32)

    # Create dask arrays; map_blocks converts each partition to CuPy on the worker
    X_da = da.from_array(X_tr_s, chunks=(chunk, -1)).map_blocks(
        cp.asarray, dtype=cp.float32)
    y_da = da.from_array(y_tr, chunks=chunk).map_blocks(
        cp.asarray, dtype=cp.float32)

    X_da = client.persist(X_da)
    y_da = client.persist(y_da)
    wait([X_da, y_da])

    model  = cuMLLinearRegression(client=client, fit_intercept=True)
    model.fit(X_da, y_da)

    X_te_da  = da.from_array(X_te_s, chunks=(len(X_te_s), -1)).map_blocks(
        cp.asarray, dtype=cp.float32)
    y_pred_da = model.predict(X_te_da)
    y_pred    = cp.asnumpy(y_pred_da.compute())

    elapsed    = time.time() - t0
    gpu_delta  = max(0.0, gpu_mem_gb(range(n_gpus)) - gm0)
    cpu_delta  = max(0.0, cpu_mem_gb() - cm0)

    metrics = eval_metrics(y_te, y_pred, label=label)
    metrics.update({
        "algorithm":        "cuML LinearRegression (GPU)",
        "n_gpus":           n_gpus,
        "time_s":           elapsed,
        "gpu_memory_gb":    gpu_delta,
        "cpu_memory_gb":    cpu_delta,
    })
    print(f"    Time: {elapsed:.1f}s  |  VRAM Δ: {gpu_delta:.2f} GB", flush=True)
    return metrics

# ═════════════════════════════════════════════════════════════════════════════
# ALGORITHM 2: XGBoost + device='cuda' — third-party native distributed GPU
# ═════════════════════════════════════════════════════════════════════════════

def run_xgboost_gpu(client, X_tr, y_tr, X_te, y_te,
                    n_gpus: int, label: str,
                    n_rounds: int = 200) -> tuple | None:
    """
    XGBoost's Dask backend with device='cuda' enables native multi-GPU
    distributed training via NCCL AllReduce. Each Dask worker owns one GPU
    and builds local trees; gradients are reduced across GPUs after each round.

    The interface is identical to the T7 CPU version — only device='cuda'
    is changed. This demonstrates XGBoost's transparent CPU<->GPU portability.

    Returns (metrics_dict, booster, y_pred_log).
    """
    if not HAS_XGB:
        print("  [XGBoost GPU] SKIP — xgboost not installed", flush=True)
        return None

    print(f"\n  [XGBoost GPU | {n_gpus} GPU] Training ({len(X_tr):,} rows, "
          f"{n_rounds} rounds)...", flush=True)
    t0  = time.time()
    gm0 = gpu_mem_gb(range(n_gpus))
    cm0 = cpu_mem_gb()

    chunk  = max(5_000, len(X_tr) // (n_gpus * 4))
    X_da   = da.from_array(X_tr, chunks=(chunk, -1))
    y_da   = da.from_array(y_tr, chunks=chunk)
    dtrain = xgb.dask.DaskDMatrix(client, X_da, y_da)

    params = {
        "objective":        "reg:squarederror",
        "max_depth":        6,
        "learning_rate":    0.08,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_lambda":       1.0,
        "tree_method":      "hist",
        "device":           "cuda",   # <- only change vs T7 XGBoost
        "verbosity":        1,
    }

    output  = xgb.dask.train(
        client, params, dtrain,
        num_boost_round=n_rounds,
        evals=[(dtrain, "train")],
        verbose_eval=50,
    )
    booster = output["booster"]

    X_te_da = da.from_array(X_te, chunks=(len(X_te), -1))
    dtest   = xgb.dask.DaskDMatrix(client, X_te_da)
    y_pred  = xgb.dask.predict(client, booster, dtest).compute()

    elapsed   = time.time() - t0
    gpu_delta = max(0.0, gpu_mem_gb(range(n_gpus)) - gm0)
    cpu_delta = max(0.0, cpu_mem_gb() - cm0)

    metrics = eval_metrics(y_te, y_pred, label=label)
    metrics.update({
        "algorithm":     "XGBoost/GPU",
        "n_gpus":        n_gpus,
        "time_s":        elapsed,
        "gpu_memory_gb": gpu_delta,
        "cpu_memory_gb": cpu_delta,
        "n_rounds":      n_rounds,
    })
    print(f"    Time: {elapsed:.1f}s  |  VRAM Δ: {gpu_delta:.2f} GB", flush=True)

    # Save feature importances
    safe_label = label.replace(" ", "_").replace("(", "").replace(")", "")[:60]
    fi_path    = OUT_DIR / f"xgb_feature_importance_{safe_label}.csv"
    fi_dict    = booster.get_score(importance_type="gain")
    pd.DataFrame(fi_dict.items(), columns=["feature", "importance"]).sort_values(
        "importance", ascending=False
    ).to_csv(fi_path, index=False)

    return metrics, booster, y_pred

# ═════════════════════════════════════════════════════════════════════════════
# ALGORITHM 3: cuML MBSGDRegressor — GPU-accelerated mini-batch SGD
# ═════════════════════════════════════════════════════════════════════════════

def run_cuml_sgd(X_tr, y_tr, X_te, y_te,
                 label: str,
                 n_epochs: int   = 5,
                 chunk_size: int = 200_000) -> tuple:
    """
    cuML MBSGDRegressor: GPU-accelerated mini-batch stochastic gradient descent.

    Key difference vs T7 SGDRegressor (partial_fit):
      - cuML has no partial_fit; the model is fitted for all epochs at once.
      - All data fits in 32 GB V100S VRAM (~3.7 GB after float32 cast), so
        out-of-core loading is not required on GPU — this is the architectural
        trade-off: GPU trades out-of-core flexibility for massive compute speedup.
      - Data is scaled on CPU (sklearn), then transferred to GPU as a single
        CuPy array, which MBSGDRegressor processes in batches of chunk_size.

    For convergence monitoring, we fit for 1 epoch at a time in a Python loop
    (each call re-initialises weights — this mimics the T7 SGD epoch structure
    and allows per-epoch RMSE tracking for the convergence plot).
    """
    print(f"\n  [cuML MBSGD | 1 GPU] Training ({len(X_tr):,} rows, "
          f"{n_epochs} epochs, batch={chunk_size:,})...", flush=True)
    t0  = time.time()
    gm0 = gpu_mem_gb([0])
    cm0 = cpu_mem_gb()

    # Scale on CPU, transfer full arrays to GPU as CuPy
    scaler   = SklearnScaler()
    X_tr_s   = scaler.fit_transform(X_tr).astype(np.float32)
    X_te_s   = scaler.transform(X_te).astype(np.float32)

    X_tr_cp = cp.asarray(X_tr_s)
    y_tr_cp = cp.asarray(y_tr)
    X_te_cp = cp.asarray(X_te_s)

    epoch_rmses = []
    model = None

    for epoch in range(n_epochs):
        # Shuffle on GPU for each epoch
        idx      = cp.random.permutation(len(X_tr_cp))
        X_shuf   = X_tr_cp[idx]
        y_shuf   = y_tr_cp[idx]

        # Re-initialise and fit for 1 epoch — cuML has no warm_start/partial_fit
        # Each epoch is an independent fit; not truly incremental but demonstrates
        # GPU-accelerated SGD with epoch-level convergence monitoring.
        model = MBSGDRegressor(
            loss          = "squared_loss",
            penalty       = "l2",
            alpha         = 1e-4,
            fit_intercept = True,
            epochs        = 1,
            eta0          = 0.01,
            learning_rate = "adaptive",
            batch_size    = chunk_size,
            random_state  = epoch,
        )
        model.fit(X_shuf, y_shuf)

        y_pred_cp = model.predict(X_te_cp)
        y_pred_np = cp.asnumpy(y_pred_cp)
        ep_rmse   = float(np.sqrt(mean_squared_error(y_te, y_pred_np)))
        epoch_rmses.append(ep_rmse)
        print(f"    Epoch {epoch+1}/{n_epochs}  RMSE(log)={ep_rmse:.5f}", flush=True)

    # Final prediction from last epoch's model
    y_pred_final = cp.asnumpy(model.predict(X_te_cp))

    elapsed   = time.time() - t0
    gpu_delta = max(0.0, gpu_mem_gb([0]) - gm0)
    cpu_delta = max(0.0, cpu_mem_gb() - cm0)

    metrics = eval_metrics(y_te, y_pred_final, label=label)
    metrics.update({
        "algorithm":        "cuML MBSGDRegressor (GPU)",
        "n_gpus":           1,
        "time_s":           elapsed,
        "gpu_memory_gb":    gpu_delta,
        "cpu_memory_gb":    cpu_delta,
        "n_epochs":         n_epochs,
        "chunk_size":       chunk_size,
        "epoch_log_rmses":  epoch_rmses,
    })
    print(f"    Time: {elapsed:.1f}s  |  VRAM Δ: {gpu_delta:.2f} GB", flush=True)
    return metrics, scaler, model

# ═════════════════════════════════════════════════════════════════════════════
# CLUSTER SETUP — LocalCUDACluster (single node, multiple GPUs)
# ═════════════════════════════════════════════════════════════════════════════

def make_gpu_cluster(n_gpus: int):
    """
    LocalCUDACluster starts one Dask worker per GPU on the current node.
    Unlike T7's SLURMCluster, no sub-jobs are submitted — all workers run
    within the same SLURM allocation, which is the standard pattern for
    intra-node multi-GPU distributed computing.

    Each worker gets:
      - 1 dedicated GPU (pinned via CUDA_VISIBLE_DEVICES per worker)
      - threads_per_worker CPU threads for data loading
      - rmm_pool_size of pre-allocated GPU memory (speeds up allocations)

    CUDA_VISIBLE_DEVICES="0,1,...,n_gpus-1" is set automatically by
    LocalCUDACluster based on n_workers.
    """
    cluster = LocalCUDACluster(
        n_workers          = n_gpus,
        threads_per_worker = 4,
        memory_limit       = "24GB",      # CPU RAM per worker
        rmm_pool_size      = "8GB",       # pre-allocated GPU memory pool
        # device_memory_limit not set — use full 32GB V100S VRAM per worker
    )
    client = Client(cluster)
    client.wait_for_workers(n_gpus, timeout=120)
    actual = len(client.scheduler_info()["workers"])
    print(f"  {actual}/{n_gpus} GPU workers ready", flush=True)
    print(f"  Dashboard: {client.dashboard_link}", flush=True)
    return cluster, client

# ═════════════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT RUNNER
# ═════════════════════════════════════════════════════════════════════════════

def run_all_gpu(client, n_gpus: int,
                train_df: pd.DataFrame,
                test_df:  pd.DataFrame) -> dict:
    """
    GPU equivalent of T7's run_all:
      Section A: unified model — all 3 GPU algorithms
      Section B: per-service XGBoost GPU
      Section C: augmentation impact (XGBoost GPU, with vs. without T5 features)
    """
    results = {"n_gpus": n_gpus, "metrics": [], "scalability": {}}
    t_start = time.time()
    gm_start = gpu_mem_gb(range(n_gpus))
    cm_start = cpu_mem_gb()

    # Remove constant columns (same logic as T7)
    constant_cols = [c for c in UNIFIED_FEATURES if train_df[c].nunique() <= 1]
    if constant_cols:
        print(f"  Removing constant columns: {constant_cols}", flush=True)
    features = [c for c in UNIFIED_FEATURES if c not in constant_cols]

    X_tr_u, y_tr_u = get_Xy(train_df, features)
    X_te_u, y_te_u = get_Xy(test_df,  features)
    print(f"\n  Train: {len(X_tr_u):,}  |  Test: {len(X_te_u):,}  |  "
          f"Features: {len(features)} (of {len(UNIFIED_FEATURES)}, "
          f"{len(constant_cols)} constant removed)  |  "
          f"GPU RAM: {gm_start:.2f} GB used", flush=True)

    # ═══ SECTION A: UNIFIED MODEL ════════════════════════════════════════
    print("\n" + "─" * 60, flush=True)
    print("[SECTION A] UNIFIED MODEL — GPU algorithms", flush=True)
    print("─" * 60, flush=True)

    # Algorithm 1: cuML dask LinearRegression (multi-GPU)
    try:
        m = run_cuml_lr(client, X_tr_u, y_tr_u, X_te_u, y_te_u,
                        n_gpus, label="cuML LR (unified)", feature_cols=features)
        results["metrics"].append(m)
    except Exception as e:
        print(f"  [ERROR] cuML LR: {e}", flush=True)
        traceback.print_exc()

    # Algorithm 2: XGBoost GPU (multi-GPU via NCCL AllReduce)
    try:
        out = run_xgboost_gpu(client, X_tr_u, y_tr_u, X_te_u, y_te_u,
                              n_gpus, label="XGBoost GPU (unified)",
                              n_rounds=200)
        if out:
            m, _, _ = out
            results["metrics"].append(m)
    except Exception as e:
        print(f"  [ERROR] XGBoost GPU unified: {e}", flush=True)
        traceback.print_exc()

    # Algorithm 3: cuML MBSGDRegressor (single GPU, batch-based)
    try:
        m, _, _ = run_cuml_sgd(
            X_tr_u, y_tr_u, X_te_u, y_te_u,
            label="cuML MBSGD (unified)",
            n_epochs=5, chunk_size=200_000,
        )
        results["metrics"].append(m)
    except Exception as e:
        print(f"  [ERROR] cuML MBSGD: {e}", flush=True)
        traceback.print_exc()

    # ═══ SECTION B: PER-SERVICE XGBoost GPU ══════════════════════════════
    if HAS_XGB:
        print("\n" + "─" * 60, flush=True)
        print("[SECTION B] SEPARATE MODELS — per-service XGBoost GPU", flush=True)
        print("─" * 60, flush=True)

        sep_preds  = []
        sep_truths = []
        svc_features = [c for c in ALL_FEATURES if c not in constant_cols]

        for service in DATASETS:
            tr_s = train_df[train_df["service"] == service]
            te_s = test_df[test_df["service"]  == service]
            if len(te_s) == 0:
                print(f"  {service}: no 2024 test data, skipping", flush=True)
                continue

            X_tr_s, y_tr_s = get_Xy(tr_s, svc_features)
            X_te_s, y_te_s = get_Xy(te_s, svc_features)

            try:
                out = run_xgboost_gpu(
                    client, X_tr_s, y_tr_s, X_te_s, y_te_s,
                    n_gpus, label=f"XGBoost GPU ({service})", n_rounds=150,
                )
                if out:
                    m, _, y_pred_svc = out
                    results["metrics"].append(m)
                    sep_preds.append(np.expm1(y_pred_svc.astype(np.float64)))
                    sep_truths.append(np.expm1(y_te_s.astype(np.float64)))
            except Exception as e:
                print(f"  [ERROR] XGBoost GPU {service}: {e}", flush=True)
                traceback.print_exc()

        if sep_preds and sep_truths:
            all_p = np.maximum(np.concatenate(sep_preds), 0)
            all_t = np.concatenate(sep_truths)
            sep_summary = {
                "label":        "XGBoost GPU (4 separate — aggregated)",
                "algorithm":    "XGBoost/GPU separate",
                "n_gpus":       n_gpus,
                "rmse":  float(np.sqrt(mean_squared_error(all_t, all_p))),
                "mae":   float(mean_absolute_error(all_t, all_p)),
                "r2":    float(r2_score(all_t, all_p)),
            }
            print(f"\n  [4-separate GPU combined]  "
                  f"RMSE={sep_summary['rmse']:.2f}  "
                  f"MAE={sep_summary['mae']:.2f}  "
                  f"R²={sep_summary['r2']:.4f}", flush=True)
            results["metrics"].append(sep_summary)

    # ═══ SECTION C: AUGMENTATION IMPACT ══════════════════════════════════
    if HAS_XGB:
        print("\n" + "─" * 60, flush=True)
        print("[SECTION C] AUGMENTATION IMPACT (XGBoost GPU)", flush=True)
        print("─" * 60, flush=True)

        base_feats = [c for c in (BASE_FEATURES + ["service_type"])
                      if c not in constant_cols]
        X_tr_b, y_tr_b = get_Xy(train_df, base_feats)
        X_te_b, y_te_b = get_Xy(test_df,  base_feats)

        try:
            out = run_xgboost_gpu(
                client, X_tr_b, y_tr_b, X_te_b, y_te_b,
                n_gpus, label="XGBoost GPU (no augmentation)", n_rounds=150,
            )
            if out:
                m, _, _ = out
                m["augmentation_used"] = False
                results["metrics"].append(m)
        except Exception as e:
            print(f"  [ERROR] Baseline XGBoost GPU: {e}", flush=True)
            traceback.print_exc()

        for m in results["metrics"]:
            if m.get("label") == "XGBoost GPU (unified)":
                m["augmentation_used"] = True

    # ── Scalability record ────────────────────────────────────────────────
    total_t = time.time() - t_start
    results["scalability"] = {
        "total_time_s":     total_t,
        "start_gpu_mem_gb": gm_start,
        "final_gpu_mem_gb": gpu_mem_gb(range(n_gpus)),
        "start_cpu_mem_gb": cm_start,
        "final_cpu_mem_gb": cpu_mem_gb(),
        "n_gpus":           n_gpus,
    }
    print(f"\n  Total experiment time ({n_gpus} GPU): {total_t:.1f}s", flush=True)
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

    # ── 1. Algorithm comparison (unified model) ───────────────────────────
    algo_metrics = [
        m for m in last["metrics"]
        if "rmse" in m and m.get("label", "").endswith("(unified)")
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
            ax.set_xlabel(xlabel); ax.set_title(title)
            ax.grid(axis="x", alpha=0.3)
            if xlabel == "R²":
                ax.set_xlim(0, 1)
        plt.suptitle("GPU Algorithm Comparison — Unified Demand Forecasting Model",
                     fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "model_comparison.png", bbox_inches="tight")
        plt.close()
        print("  Saved model_comparison.png", flush=True)

    # ── 2. Unified vs Separate (XGBoost GPU) ─────────────────────────────
    unified_m  = next((m for m in last["metrics"]
                       if m.get("label") == "XGBoost GPU (unified)"), None)
    separate_m = next((m for m in last["metrics"]
                       if "4 separate" in m.get("label", "")), None)
    if unified_m and separate_m:
        labels2 = ["Unified\n(1 model, 4 services)", "Separate\n(4 models aggregated)"]
        bar_clr = ["#4C72B0", "#DD8452"]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.bar(labels2, [unified_m["rmse"], separate_m["rmse"]],
                color=bar_clr, width=0.5)
        ax1.set_ylabel("RMSE (trips/hr)"); ax1.set_title("RMSE ↓")
        ax1.grid(axis="y", alpha=0.3)
        ax2.bar(labels2, [max(0, unified_m["r2"]), max(0, separate_m["r2"])],
                color=bar_clr, width=0.5)
        ax2.set_ylabel("R²"); ax2.set_title("R² ↑"); ax2.set_ylim(0, 1)
        ax2.grid(axis="y", alpha=0.3)
        plt.suptitle("Unified vs. 4 Separate Models (XGBoost GPU)", fontweight="bold")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "unified_vs_separate.png", bbox_inches="tight")
        plt.close()
        print("  Saved unified_vs_separate.png", flush=True)

    # ── 3. GPU scalability: wall time and speedup ─────────────────────────
    if len(all_results) > 1:
        gpus  = [r["n_gpus"]                      for r in all_results]
        times = [r["scalability"]["total_time_s"]  for r in all_results]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(gpus, times, "o-", lw=2, ms=8, color="#4C72B0", label="Actual")
        ideal = [times[0] * gpus[0] / g for g in gpus]
        ax1.plot(gpus, ideal, "--", color="gray", label="Ideal linear speedup")
        ax1.set_xlabel("GPUs"); ax1.set_ylabel("Total wall time (s)")
        ax1.set_title("Scalability: Wall Time vs GPUs")
        ax1.legend(); ax1.grid(alpha=0.3); ax1.set_xticks(gpus)

        speedup = [times[0] / t for t in times]
        ideal_s = [g / gpus[0] for g in gpus]
        ax2.plot(gpus, speedup, "o-", lw=2, ms=8, color="#DD8452",
                 label="Actual speedup")
        ax2.plot(gpus, ideal_s, "--", color="gray", label="Ideal speedup")
        ax2.set_xlabel("GPUs"); ax2.set_ylabel("Speedup factor (×)")
        ax2.set_title("Speedup Relative to 1 GPU")
        ax2.legend(); ax2.grid(alpha=0.3); ax2.set_xticks(gpus)

        plt.suptitle("GPU Scalability — Arnes HPC (V100S)", fontweight="bold")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "scalability.png", bbox_inches="tight")
        plt.close()
        print("  Saved scalability.png", flush=True)

    # ── 4. GPU VRAM usage vs GPU count ────────────────────────────────────
    if len(all_results) > 1:
        gpus = [r["n_gpus"] for r in all_results]
        vram = [r["scalability"].get("final_gpu_mem_gb", 0) for r in all_results]
        plt.figure(figsize=(7, 4))
        plt.bar([str(g) for g in gpus], vram, color="#9467bd", alpha=0.85)
        plt.xlabel("GPUs"); plt.ylabel("Total VRAM Used (GB)")
        plt.title("GPU Memory Consumption vs Number of GPUs")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "gpu_memory.png", bbox_inches="tight")
        plt.close()
        print("  Saved gpu_memory.png", flush=True)

    # ── 5. Augmentation impact (XGBoost GPU) ─────────────────────────────
    base_m = next((m for m in last["metrics"]
                   if "no augmentation" in m.get("label", "")), None)
    aug_m  = next((m for m in last["metrics"]
                   if m.get("label") == "XGBoost GPU (unified)"), None)
    if base_m and aug_m:
        cats = ["Baseline\n(time + zone)", "Augmented\n(+ weather + spatial + events)"]
        clrs = ["#999999", "#2ca02c"]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.bar(cats, [base_m["rmse"], aug_m["rmse"]], color=clrs, width=0.5)
        ax1.set_ylabel("RMSE (trips/hr)"); ax1.grid(axis="y", alpha=0.3)
        ax2.bar(cats, [max(0, base_m["r2"]), max(0, aug_m["r2"])],
                color=clrs, width=0.5)
        ax2.set_ylabel("R²"); ax2.set_ylim(0, 1); ax2.grid(axis="y", alpha=0.3)
        pct = (base_m["rmse"] - aug_m["rmse"]) / max(base_m["rmse"], 1e-6) * 100
        ax1.set_title(f"RMSE ↓  ({pct:+.1f}% from augmentation)")
        ax2.set_title("R² ↑")
        plt.suptitle("Impact of T5 Data Augmentation — XGBoost GPU",
                     fontweight="bold")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "augmentation_impact.png", bbox_inches="tight")
        plt.close()
        print("  Saved augmentation_impact.png", flush=True)

    # ── 6. cuML MBSGD convergence curve ──────────────────────────────────
    sgd_m = next((m for m in last["metrics"]
                  if "MBSGD" in m.get("algorithm", "")
                  and "epoch_log_rmses" in m), None)
    if sgd_m:
        epochs = list(range(1, len(sgd_m["epoch_log_rmses"]) + 1))
        plt.figure(figsize=(7, 4))
        plt.plot(epochs, sgd_m["epoch_log_rmses"], "o-",
                 color="darkorange", lw=2, ms=7)
        plt.xlabel("Epoch"); plt.ylabel("Test RMSE (log scale)")
        plt.title("cuML MBSGDRegressor — Epoch RMSE (unified model)")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "sgd_convergence.png", bbox_inches="tight")
        plt.close()
        print("  Saved sgd_convergence.png", flush=True)

    # ── 7. XGBoost feature importance ────────────────────────────────────
    fi_candidates = list(OUT_DIR.glob("xgb_feature_importance_*unified*.csv"))
    if fi_candidates:
        fi_df = pd.read_csv(fi_candidates[0]).head(20)
        plt.figure(figsize=(8, 6))
        plt.barh(fi_df["feature"][::-1], fi_df["importance"][::-1], color="#4C72B0")
        plt.xlabel("Gain (feature importance)")
        plt.title("Top-20 Feature Importances — XGBoost GPU Unified Model")
        plt.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "feature_importance.png", bbox_inches="tight")
        plt.close()
        print("  Saved feature_importance.png", flush=True)

    # ── 8. CPU vs GPU wall-time comparison (T7 vs T10) ───────────────────
    if T7_RESULTS.exists():
        try:
            with open(T7_RESULTS) as f:
                t7_data = json.load(f)

            # Extract XGBoost unified times from T7 (last run = most workers)
            t7_last   = t7_data[-1]["metrics"] if t7_data else []
            gpu_last  = last["metrics"]

            algo_pairs = [
                ("Dask-ML LR\n/ cuML LR",
                 "Dask-ML LR (unified)", "cuML LR (unified)"),
                ("XGBoost CPU\n/ XGBoost GPU",
                 "XGBoost (unified)", "XGBoost GPU (unified)"),
                ("SGD CPU\n/ cuML MBSGD",
                 "SGD partial_fit (unified)", "cuML MBSGD (unified)"),
            ]

            labels_cmp, t7_times, t10_times = [], [], []
            for lbl, t7_lbl, t10_lbl in algo_pairs:
                t7_m  = next((m for m in t7_last  if m.get("label") == t7_lbl),  None)
                t10_m = next((m for m in gpu_last  if m.get("label") == t10_lbl), None)
                if t7_m and t10_m:
                    labels_cmp.append(lbl)
                    t7_times.append(t7_m.get("time_s", 0))
                    t10_times.append(t10_m.get("time_s", 0))

            if labels_cmp:
                x    = np.arange(len(labels_cmp))
                w    = 0.35
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.bar(x - w/2, t7_times,  w, label="T7 CPU",       color="#4C72B0")
                ax.bar(x + w/2, t10_times, w, label="T10 GPU",       color="#DD8452")
                ax.set_xticks(x); ax.set_xticklabels(labels_cmp)
                ax.set_ylabel("Wall time (s)")
                ax.set_title("CPU vs GPU Wall Time — Unified Model (best worker/GPU count)")
                ax.legend(); ax.grid(axis="y", alpha=0.3)
                plt.tight_layout()
                plt.savefig(PLOTS_DIR / "cpu_vs_gpu.png", bbox_inches="tight")
                plt.close()
                print("  Saved cpu_vs_gpu.png", flush=True)
        except Exception as e:
            print(f"  [WARN] CPU vs GPU plot skipped: {e}", flush=True)

    # ── 9. Results table ─────────────────────────────────────────────────
    algo_all = [m for m in last["metrics"] if "rmse" in m]
    if algo_all:
        tbl_cols = ["label", "algorithm", "n_gpus",
                    "rmse", "mae", "r2", "time_s", "gpu_memory_gb"]
        df_tbl = pd.DataFrame([
            {c: m.get(c) for c in tbl_cols} for m in algo_all
        ]).rename(columns={
            "label": "Model", "algorithm": "Algorithm", "n_gpus": "GPUs",
            "rmse": "RMSE", "mae": "MAE", "r2": "R²",
            "time_s": "Time (s)", "gpu_memory_gb": "VRAM Δ (GB)",
        })
        df_tbl.to_csv(OUT_DIR / "results_table.csv", index=False)
        print("\n  === RESULTS TABLE ===", flush=True)
        print(df_tbl.to_string(index=False), flush=True)

# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="T10 — GPU Demand Forecasting")
    parser.add_argument("--gpus", type=int, nargs="+", default=None,
                        help="GPU counts to sweep. Default: auto-detect (1 -> max)")
    parser.add_argument("--skip-scalability", action="store_true",
                        help="Run only the maximum GPU count")
    args = parser.parse_args()

    # Auto-detect available GPU count
    try:
        max_gpus = cp.cuda.runtime.getDeviceCount()
    except Exception:
        max_gpus = 1

    if args.gpus:
        gpu_counts = sorted(g for g in args.gpus if g <= max_gpus)
    else:
        # Build sweep: 1, 2, 4 stopping at max_gpus
        gpu_counts = sorted({g for g in [1, 2, 4] if g <= max_gpus})

    if args.skip_scalability:
        gpu_counts = [max(gpu_counts)]

    print("=" * 70, flush=True)
    print("T10: City-Wide Demand Forecasting — GPU Edition", flush=True)
    print(f"  GPUs available : {max_gpus} × Tesla V100S-32GB", flush=True)
    print(f"  Sweep          : {gpu_counts}", flush=True)
    print(f"  Train          : {TRAIN_YEARS[0]}–{TRAIN_YEARS[-1]}", flush=True)
    print(f"  Test           : {TEST_YEAR}", flush=True)
    print("=" * 70, flush=True)

    (PROJECT / "logs").mkdir(exist_ok=True)

    # Load and prepare data (reads T7 cache — no re-aggregation needed)
    demand_df = build_demand_dataset()
    demand_df = engineer_features(demand_df)
    for col in UNIFIED_FEATURES:
        if col not in demand_df.columns:
            demand_df[col] = 0.0
    demand_df = demand_df.fillna(0.0)

    train_df = demand_df[demand_df["year"].isin(TRAIN_YEARS)].reset_index(drop=True)
    test_df  = demand_df[demand_df["year"] == TEST_YEAR].reset_index(drop=True)

    print(f"\nDataset split:", flush=True)
    print(f"  Train : {len(train_df):,} zone-hour samples", flush=True)
    print(f"  Test  : {len(test_df):,}  zone-hour samples", flush=True)
    ram_mb = demand_df.memory_usage(deep=True).sum() / 1e6
    print(f"  RAM   : ~{ram_mb:.0f} MB (fits in single V100S VRAM for GPU SGD)", flush=True)

    all_results = []

    for n_gpus in gpu_counts:
        print("\n" + "=" * 70, flush=True)
        print(f"EXPERIMENT: {n_gpus} GPU(s)", flush=True)
        print("=" * 70, flush=True)

        cluster, client = make_gpu_cluster(n_gpus)

        try:
            report_path = str(OUT_DIR / f"dask_report_{n_gpus}gpu.html")
            with performance_report(filename=report_path):
                result = run_all_gpu(client, n_gpus, train_df, test_df)
            all_results.append(result)
        except Exception as e:
            print(f"[FATAL ERROR] {e}", flush=True)
            traceback.print_exc()
        finally:
            client.close()
            cluster.close()
            # Free all CuPy memory pools between runs
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()

        with open(OUT_DIR / "results.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"  results.json updated ({n_gpus} GPU done)", flush=True)

    if all_results:
        print("\n[Plots] Generating...", flush=True)
        make_plots(all_results)

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
                    f"Time={m.get('time_s', 0):.0f}s  "
                    f"VRAM={m.get('gpu_memory_gb', 0):.2f}GB",
                    flush=True,
                )
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
