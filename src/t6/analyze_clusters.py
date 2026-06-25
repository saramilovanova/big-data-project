"""
Builds the clustering section of the T6 report from:
  - cluster_centres.jsonl: periodic snapshots of the 5 cluster centres
    (one record per 1000 events), showing how they evolved as the stream
    was processed -- the key evidence that this was genuinely online/
    streaming K-Means, not batch K-Means applied after the fact.
  - clusters.jsonl: per-trip cluster assignments (subset), used for
    borough/source breakdown per cluster.

Cluster feature columns come from CLUSTER_FEATURES in config.py:
  trip_distance, fare_amount, tip_amount, total_amount,
  trip_duration_min, is_fhvhv

Outputs land in OUTPUT_DIR/report_assets/ alongside the rolling-stats charts.
"""

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np
from pathlib import Path

from config import OUTPUT_DIR, CLUSTER_FEATURES, N_CLUSTERS

con = duckdb.connect()
ASSETS_DIR = Path(OUTPUT_DIR) / "report_assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

CENTRES_FILE = f"{OUTPUT_DIR}/cluster_centres.jsonl"
CLUSTERS_FILE = f"{OUTPUT_DIR}/clusters.jsonl"

CLUSTER_LABELS = {
    0: "Cluster 0",
    1: "Cluster 1",
    2: "Cluster 2",
    3: "Cluster 3",
    4: "Cluster 4",
}


# -----------------------------------------------------------------------
# 1. Final cluster characterisation table
# -----------------------------------------------------------------------

def build_characterisation_table():
    """Use the last snapshot in cluster_centres.jsonl as the final cluster
    state, and annotate each cluster with a plain-English description based
    on its feature values."""

    raw = con.sql(f"""
        SELECT event_count, timestamp, centres
        FROM read_ndjson_auto('{CENTRES_FILE}')
        ORDER BY event_count DESC
        LIMIT 1
    """).fetchone()

    centres = raw[2]  # list of dicts, one per cluster

    rows = []
    for c in sorted(centres, key=lambda x: x["cluster_id"]):
        row = {
            "cluster_id": c["cluster_id"],
            "count_in_last_snapshot": c.get("count", "?"),
        }
        for f in CLUSTER_FEATURES:
            row[f] = round(c.get(f, float("nan")), 3)

        # Plain-English interpretation keys for the report
        dist = c.get("trip_distance", 0)
        fare = c.get("fare_amount", 0)
        fhv  = c.get("is_fhvhv", 0)
        dur  = c.get("trip_duration_min", 0)
        if fhv < 0.5:
            vehicle = "Yellow Taxi"
        elif fhv > 0.85:
            vehicle = "FHVHV (Uber/Lyft)"
        else:
            vehicle = "Mixed"

        if dist < 2:
            trip_type = "short local"
        elif dist < 6:
            trip_type = "medium"
        elif dist < 12:
            trip_type = "long"
        else:
            trip_type = "airport/outer-borough"

        row["interpretation"] = f"{vehicle}, {trip_type} trip (~{dist:.1f} mi, ~${fare:.0f} fare)"
        rows.append(row)

    df = pd.DataFrame(rows)
    out = ASSETS_DIR / "cluster_characterisation.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {out}")
    return df


# -----------------------------------------------------------------------
# 2. Cluster centre evolution over the stream
# -----------------------------------------------------------------------

def plot_centre_evolution():
    """For each feature, plot how each cluster's centre value changed as
    more events were processed -- the most direct visual proof that this
    was streaming (online) K-Means."""

    df = con.sql(f"""
        SELECT event_count, UNNEST(centres) AS c
        FROM read_ndjson_auto('{CENTRES_FILE}')
        ORDER BY event_count
    """).df()

    # Unpack the struct column into individual feature columns
    centres_expanded = pd.json_normalize(df["c"])
    centres_expanded["event_count"] = df["event_count"].values
    centres_expanded = centres_expanded.rename(columns={"cluster_id": "cluster_id"})

    plot_features = ["trip_distance", "fare_amount", "trip_duration_min"]
    colors = cm.tab10.colors

    fig, axes = plt.subplots(len(plot_features), 1, figsize=(11, 4 * len(plot_features)), sharex=True)

    for ax, feature in zip(axes, plot_features):
        for cid in range(N_CLUSTERS):
            sub = centres_expanded[centres_expanded["cluster_id"] == cid].sort_values("event_count")
            if sub.empty or feature not in sub.columns:
                continue
            ax.plot(sub["event_count"], sub[feature],
                    label=f"Cluster {cid}", color=colors[cid], linewidth=1.2)
        ax.set_ylabel(feature)
        ax.set_title(f"Online K-Means: {feature} centre evolution over stream")
        ax.legend(fontsize=8, ncol=N_CLUSTERS)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Events processed")
    fig.tight_layout()
    out = ASSETS_DIR / "cluster_centre_evolution.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


# -----------------------------------------------------------------------
# 3. Final cluster sizes (bar chart)
# -----------------------------------------------------------------------

def plot_cluster_sizes():
    df = con.sql(f"""
        SELECT cluster_id, COUNT(*) AS n
        FROM read_ndjson_auto('{CLUSTERS_FILE}')
        GROUP BY cluster_id
        ORDER BY cluster_id
    """).df()

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [cm.tab10.colors[i] for i in df["cluster_id"]]
    ax.bar(df["cluster_id"].astype(str), df["n"], color=colors)
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Number of assigned trips")
    ax.set_title("Trip count per cluster (sampled stream)")
    for i, row in df.iterrows():
        ax.text(i, row["n"] + 5, str(int(row["n"])), ha="center", fontsize=9)
    fig.tight_layout()
    out = ASSETS_DIR / "cluster_sizes.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


# -----------------------------------------------------------------------
# 4. Borough breakdown per cluster
# -----------------------------------------------------------------------

def plot_cluster_borough_breakdown():
    df = con.sql(f"""
        SELECT cluster_id, PU_Borough, COUNT(*) AS n
        FROM read_ndjson_auto('{CLUSTERS_FILE}')
        WHERE PU_Borough IS NOT NULL
        GROUP BY cluster_id, PU_Borough
        ORDER BY cluster_id, n DESC
    """).df()

    pivot = df.pivot_table(index="PU_Borough", columns="cluster_id", values="n", fill_value=0)
    # Normalise to % within each cluster
    pivot_pct = pivot.div(pivot.sum(axis=0), axis=1) * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    pivot_pct.T.plot(kind="bar", ax=ax, colormap="tab10")
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("% of trips in cluster from each borough")
    ax.set_title("Borough composition per cluster")
    ax.legend(title="Borough", fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.set_xticklabels([str(c) for c in pivot_pct.columns], rotation=0)
    fig.tight_layout()
    out = ASSETS_DIR / "cluster_borough_breakdown.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


# -----------------------------------------------------------------------
# 5. Source (Yellow vs FHVHV) breakdown per cluster
# -----------------------------------------------------------------------

def plot_cluster_source_breakdown():
    df = con.sql(f"""
        SELECT cluster_id, source, COUNT(*) AS n
        FROM read_ndjson_auto('{CLUSTERS_FILE}')
        WHERE source IS NOT NULL
        GROUP BY cluster_id, source
        ORDER BY cluster_id
    """).df()

    pivot = df.pivot_table(index="source", columns="cluster_id", values="n", fill_value=0)
    pivot_pct = pivot.div(pivot.sum(axis=0), axis=1) * 100

    fig, ax = plt.subplots(figsize=(8, 4))
    pivot_pct.T.plot(kind="bar", ax=ax, color=["#f7ca18", "#3498db"])
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("% of trips from each source")
    ax.set_title("Yellow Taxi vs FHVHV composition per cluster")
    ax.legend(title="Source", fontsize=9)
    ax.set_xticklabels([str(c) for c in pivot_pct.columns], rotation=0)
    fig.tight_layout()
    out = ASSETS_DIR / "cluster_source_breakdown.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


# -----------------------------------------------------------------------
# 6. Scatter: trip_distance vs fare_amount coloured by cluster
# -----------------------------------------------------------------------

def plot_cluster_scatter():
    # Cap at 10k rows for the scatter -- we only need a sample for visual clarity
    df = con.sql(f"""
        SELECT cluster_id, trip_distance, fare_amount
        FROM read_ndjson_auto('{CLUSTERS_FILE}')
        WHERE trip_distance IS NOT NULL AND fare_amount IS NOT NULL
          AND trip_distance < 40 AND fare_amount < 120
        LIMIT 10000
    """).df()

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = cm.tab10.colors
    for cid in sorted(df["cluster_id"].unique()):
        sub = df[df["cluster_id"] == cid]
        ax.scatter(sub["trip_distance"], sub["fare_amount"],
                   c=[colors[cid]], label=f"Cluster {cid}",
                   alpha=0.25, s=6, rasterized=True)
    ax.set_xlabel("Trip distance (miles)")
    ax.set_ylabel("Fare amount ($)")
    ax.set_title("Trip distance vs fare amount, coloured by cluster (sampled)")
    ax.legend(fontsize=9, markerscale=3)
    fig.tight_layout()
    out = ASSETS_DIR / "cluster_scatter_dist_vs_fare.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    print("=== Cluster characterisation ===")
    char_df = build_characterisation_table()
    print(char_df[["cluster_id", "trip_distance", "fare_amount", "trip_duration_min", "is_fhvhv", "interpretation"]].to_string(index=False))

    print("\n=== Cluster centre evolution ===")
    plot_centre_evolution()

    print("\n=== Cluster sizes ===")
    plot_cluster_sizes()

    print("\n=== Borough breakdown ===")
    plot_cluster_borough_breakdown()

    print("\n=== Source breakdown ===")
    plot_cluster_source_breakdown()

    print("\n=== Scatter plot ===")
    plot_cluster_scatter()

    print(f"\nAll done. See {ASSETS_DIR}")
