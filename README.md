# Large-Scale Processing, Analysis and Forecasting of NYC TLC Trip Data

**Sara Milovanova & Biljana Vitanova**

Big Data Project

---

## Overview

This project implements an end-to-end distributed data engineering and machine learning pipeline over the NYC Taxi and Limousine Commission (TLC) trip record dataset - 4.07 billion trips across four service types (Yellow Taxi, Green Taxi, FHV, FHVHV) spanning 2009-2026 (84.5 GB on disk after repartitioning).

All compute-intensive tasks run on the **Arnes HPC cluster** via SLURM using Dask Distributed, DuckDB, PyArrow, XGBoost, and Kafka. The full pipeline covers data acquisition, repartitioning, quality assessment, exploratory analysis, storage benchmarking, contextual augmentation, stream processing, distributed machine learning, and market emergence analysis.

---

## Repository Structure

```
.
├── src/                        # Python source scripts (one per task)
│   ├── t0_download.py          # T0: parallel download + schema comparison (Dask Bag)
│   ├── t1_repartition.py       # T1: year-based Hive partitioning (Green, FHV, FHVHV)
│   ├── t1_yellow_normalize.py  # T1: Yellow Taxi schema normalization (2009–2026)
│   ├── t2_quality.py           # T2: data quality analysis (Dask Bag map-reduce)
│   ├── t2_quality_plots.py     # T2: quality visualization
│   ├── t3_aggregations.py      # T3: EDA aggregations via DuckDB
│   ├── t4_formats.py           # T4: storage format benchmark
│   ├── t5_augment.py           # T5: weather, spatial, and event augmentation
│   ├── t6/                     # T6: Kafka stream processing pipeline
│   │   ├── docker-compose.yaml #     Kafka + ksqlDB cluster definition
│   │   ├── prepare_stream_source.py
│   │   ├── producer.py
│   │   ├── quix_streams.py     #     Quix Streams tumbling-window consumer
│   │   ├── regular_python_stats.py
│   │   ├── stream_clustering.py #    Online K-Means (k=5)
│   │   ├── config.py
│   │   ├── sinks.py
│   │   ├── analyze_clusters.py
│   │   ├── analyze_results.py
│   │   ├── basic_consumer.py
│   │   ├── find_top_locations.py
│   │   ├── read_topic.py
│   │   └── reset_topics.py
│   ├── t7_demand_forecast.py   # T7: distributed ML demand forecasting (Dask + XGBoost)
│   ├── t8_analysis.py          # T8: FHVHV market emergence analysis
│   └── t10_demand_forecast.py  # T10: GPU demand forecasting (CuML/RAPIDS - incomplete)
│
├── scripts/                    # SLURM submission scripts
│   ├── run_t1.sh               # T1: repartition (Green, FHV, FHVHV)
│   ├── run_t1_yellow.sh        # T1: Yellow normalization
│   ├── run_t2.sh               # T2: quality analysis
│   ├── run_t3.sh               # T3: EDA aggregations
│   ├── run_t4.sh               # T4: format benchmark
│   ├── run_t5.sh               # T5: augmentation
│   ├── run_t7.sh               # T7: demand forecasting scalability sweep
│   ├── run_t8.sh               # T8: FHVHV emergence analysis
│   └── run_t10.sh              # T10: GPU run (incomplete)
│
├── notebooks/                  # Analysis notebooks (lightweight - load precomputed results)
│   ├── T0.ipynb                # T0: schema comparison display
│   ├── T1.ipynb                # T1: repartitioning analysis and statistics
│   ├── T2.ipynb                # T2: quality issue visualization
│   ├── T4.ipynb                # T4: benchmark results
│   └── T7.ipynb                # T7: ML results, scalability plots, feature importance
│
├── results/                    # Precomputed outputs (CSVs, JSONs, plots, LaTeX tables)
│   ├── T0_schema_comparison.json
│   ├── t2_quality_results.json
│   ├── t2_figures/             # Per-dataset quality charts and LaTeX tables
│   ├── t3/                     # EDA CSVs (monthly, hourly, spatial, fare) + plots
│   ├── t4/                     # Benchmark CSV + plots
│   ├── t6_outputs/             # Rolling stats, cluster centroids, heatmaps
│   ├── t7_data/                # ML metrics JSON, scalability plots, feature importances
│   └── t8/                     # Monthly trip counts by service/operator + plots
│
└── requirements.txt            # Python dependencies
```

---

## Data

**Raw TLC data is not stored in this repository.** On the Arnes HPC cluster, the directory layout is:

```
/d/hpc/projects/FRI/bigdata/
├── data/Taxi/                  # Shared: original monthly .parquet files (all years)
└── students/sm_bv/
    ├── taxi_new/               # T0 output: newly downloaded files (2025-02 – 2026-02)
    └── final_project/data/
        ├── partitioned/        # T1 output: year-partitioned Hive layout (Green, FHV, FHVHV)
        ├── yellow_normalized/  # T1 output: canonical 25-column Yellow Taxi schema
        └── t5/augmented/       # T5 output: trip records enriched with weather + spatial
```


---

## Requirements

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Key packages: `dask`, `dask-ml`, `dask-jobqueue`, `xgboost`, `scikit-learn`, `pyarrow`, `duckdb`, `geopandas`, `confluent-kafka`, `quixstreams`, `matplotlib`.

For T6, Docker is required to run the Kafka cluster defined in `src/t6/docker-compose.yaml`.

---

## Authors

Sara Milovanova, Biljana Vitanova  
Faculty of Computer and Information Science, University of Ljubljana