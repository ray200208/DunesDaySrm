import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans, MiniBatchKMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


def normalize_series(s: pd.Series, higher_is_better: bool = True) -> pd.Series:
    s_min, s_max = s.min(), s.max()
    if np.isclose(s_max, s_min):
        return pd.Series(np.ones(len(s)), index=s.index)
    norm = (s - s_min) / (s_max - s_min)
    return norm if higher_is_better else (1.0 - norm)


def run_clustering_tournament():
    start_total = time.perf_counter()
    print("=" * 85)
    print("STEP 2C: 4-ALGORITHM ROUTE CLUSTERING TOURNAMENT (K=4)")
    print("=" * 85)

    input_path = Path("data/aggregated_hourly_demand.parquet")
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[1/4] Reading {input_path.as_posix()}...")
    df = pd.read_parquet(input_path)
    print(
        f"Loaded {len(df):,} hourly records across {df['route_num'].nunique():,} unique routes."
    )

    # 1. Build Route Feature Matrix
    print("[2/4] Building 27-feature Route Profile Matrix...")
    hourly_pivot = df.pivot_table(
        index="route_num",
        columns="hour",
        values="passenger_demand",
        aggfunc="mean",
        fill_value=0.0,
    )
    hourly_pivot = hourly_pivot.reindex(columns=range(24), fill_value=0.0)
    hourly_pivot.columns = [f"hour_{h}" for h in range(24)]

    route_stats = df.groupby("route_num").agg(
        avg_total_demand=("passenger_demand", "mean"),
        mean_active_trips=("active_trips", "mean"),
        mean_ticket_price=("avg_ticket_price", "mean"),
    )

    peak_df = df[df["is_peak_hour"] == 1]
    peak_stats = peak_df.groupby("route_num")["passenger_demand"].mean().rename(
        "avg_peak_demand"
    )

    route_stats = route_stats.join(peak_stats)
    route_stats["avg_peak_demand"] = route_stats["avg_peak_demand"].fillna(
        route_stats["avg_total_demand"]
    )
    route_stats["peak_demand_ratio"] = (
        route_stats["avg_peak_demand"]
        / route_stats["avg_total_demand"].replace(0.0, np.nan)
    ).fillna(1.0)

    feature_matrix = hourly_pivot.join(
        route_stats[["mean_active_trips", "mean_ticket_price", "peak_demand_ratio"]]
    ).fillna(0.0)

    print(
        f"  -> Route Feature Matrix shape: {feature_matrix.shape[0]} routes x {feature_matrix.shape[1]} features"
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_matrix)

    # 2. Test K=4 across all 4 clustering algorithms
    print("[3/4] Benchmarking 4 Clustering Algorithms (K=4)...")
    algorithms = {
        "KMeans": KMeans(n_clusters=4, random_state=42, n_init=10),
        "MiniBatchKMeans": MiniBatchKMeans(
            n_clusters=4, random_state=42, batch_size=256, n_init=10
        ),
        "GaussianMixture": GaussianMixture(n_components=4, random_state=42),
        "Agglomerative": AgglomerativeClustering(n_clusters=4, linkage="ward"),
    }

    leaderboard_rows = []
    cluster_assignments = {}

    for name, algo in algorithms.items():
        t0 = time.perf_counter()
        if hasattr(algo, "fit_predict"):
            labels = algo.fit_predict(X_scaled)
        else:
            algo.fit(X_scaled)
            labels = algo.predict(X_scaled)
        fit_time_s = time.perf_counter() - t0

        sil = float(silhouette_score(X_scaled, labels))
        ch = float(calinski_harabasz_score(X_scaled, labels))
        db = float(davies_bouldin_score(X_scaled, labels))

        leaderboard_rows.append(
            {
                "Algorithm": name,
                "Silhouette_Score": round(sil, 4),
                "Calinski_Harabasz": round(ch, 2),
                "Davies_Bouldin": round(db, 4),
                "Fit_Time_seconds": round(fit_time_s, 4),
            }
        )
        cluster_assignments[name] = labels

    lb_df = pd.DataFrame(leaderboard_rows)
    # Compute overall Composite_Score (average of min-max normalized Silhouette (+), CH (+), DB (-))
    lb_df["Composite_Score"] = (
        normalize_series(lb_df["Silhouette_Score"], higher_is_better=True)
        + normalize_series(lb_df["Calinski_Harabasz"], higher_is_better=True)
        + normalize_series(lb_df["Davies_Bouldin"], higher_is_better=False)
    ) / 3.0
    lb_df["Composite_Score"] = lb_df["Composite_Score"].round(4)

    lb_df = lb_df.sort_values(
        by=["Composite_Score", "Silhouette_Score"], ascending=False
    ).reset_index(drop=True)
    lb_df.insert(0, "Rank", range(1, len(lb_df) + 1))

    print("\n" + "=" * 85)
    print("4-ALGORITHM CLUSTERING LEADERBOARD TABLE (K=4)")
    print("-" * 85)
    print(lb_df.to_string(index=False))
    print("=" * 85)

    winner_name = lb_df.iloc[0]["Algorithm"]
    winner_comp = lb_df.iloc[0]["Composite_Score"]
    winner_sil = lb_df.iloc[0]["Silhouette_Score"]
    winner_ch = lb_df.iloc[0]["Calinski_Harabasz"]
    winner_db = lb_df.iloc[0]["Davies_Bouldin"]

    print(
        f"\nWINNING ALGORITHM: {winner_name} "
        f"(Composite Score: {winner_comp:.4f} | Silhouette: {winner_sil:.4f} | "
        f"Calinski-Harabasz: {winner_ch:.2f} | Davies-Bouldin: {winner_db:.4f})"
    )

    # 3. Profile the 4 winning clusters
    print("\n[4/4] Profiling the 4 Operational Route Clusters...")
    raw_labels = cluster_assignments[winner_name]

    profile_df = route_stats.copy()
    profile_df["raw_cluster"] = raw_labels

    # Order clusters by average total demand so Cluster 0 -> 3 follow a logical hierarchy
    cluster_order = (
        profile_df.groupby("raw_cluster")["avg_total_demand"]
        .mean()
        .sort_values(ascending=True)
        .index.tolist()
    )
    cluster_remap = {old_id: new_id for new_id, old_id in enumerate(cluster_order)}
    profile_df["route_cluster"] = profile_df["raw_cluster"].map(cluster_remap)

    cluster_labels_map = {
        0: "Cluster 0: Low-Volume Feeder",
        1: "Cluster 1: Peak-Heavy Arterial",
        2: "Cluster 2: High-Density Trunk",
        3: "Cluster 3: All-Day Mega Corridor",
    }
    profile_df["cluster_label"] = profile_df["route_cluster"].map(cluster_labels_map)

    summary_rows = []
    for cid in range(4):
        sub = profile_df[profile_df["route_cluster"] == cid]
        sample_routes = ", ".join(sub.index.astype(str)[:5].tolist())
        summary_rows.append(
            {
                "Cluster_ID": cid,
                "Operational_Label": cluster_labels_map[cid],
                "Routes_Count": len(sub),
                "Avg_Hourly_Pax": round(float(sub["avg_total_demand"].mean()), 2),
                "Peak_Pax": round(float(sub["avg_peak_demand"].mean()), 2),
                "Peak_Demand_Ratio": round(
                    float(sub["peak_demand_ratio"].mean()), 3
                ),
                "Mean_Active_Trips": round(
                    float(sub["mean_active_trips"].mean()), 2
                ),
                "Mean_Ticket_Price_INR": round(
                    float(sub["mean_ticket_price"].mean()), 2
                ),
                "Sample_Routes": sample_routes,
            }
        )

    cluster_summary_df = pd.DataFrame(summary_rows)
    print("\n" + "=" * 85)
    print(f"OPERATIONAL PROFILE OF DISCOVERED CLUSTERS ({winner_name})")
    print("-" * 85)
    print(cluster_summary_df.to_string(index=False))
    print("=" * 85)

    # 4. Save route cluster assignments
    out_df = (
        profile_df[
            [
                "route_cluster",
                "cluster_label",
                "avg_total_demand",
                "avg_peak_demand",
                "peak_demand_ratio",
                "mean_active_trips",
                "mean_ticket_price",
            ]
        ]
        .join(hourly_pivot)
        .reset_index()
    )

    parquet_out = data_dir / "route_clusters.parquet"
    csv_out = data_dir / "route_clusters.csv"
    out_df.to_parquet(parquet_out, index=False)
    out_df.to_csv(csv_out, index=False)

    elapsed = time.perf_counter() - start_total
    print(
        f"\nSaved {len(out_df):,} route cluster assignments to {parquet_out.as_posix()} and {csv_out.as_posix()}"
    )
    print(f"Total runtime: {elapsed:.2f} seconds\n")


if __name__ == "__main__":
    run_clustering_tournament()

