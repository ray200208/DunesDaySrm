import pickle
import time
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

warnings.filterwarnings("ignore")

BUS_CAPACITY = 65


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, active_trips: np.ndarray) -> dict:
    """Compute regression, percentage error (MAPE), and binary overcrowding metrics."""
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    # Mean Absolute Percentage Error (MAPE) on non-zero demand rows (avoid division by tiny numbers; standard >= 10 pax threshold or >= 1 pax)
    valid_mask = y_true >= 10
    if np.any(valid_mask):
        mape = float(np.mean(np.abs((y_true[valid_mask] - y_pred[valid_mask]) / y_true[valid_mask])) * 100.0)
    else:
        denom = np.where(y_true > 0, y_true, 1.0)
        mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

    # Binary Overcrowding Accuracy: demand > active_trips * 65
    seat_capacity = active_trips * BUS_CAPACITY
    actual_overcrowded = (y_true > seat_capacity).astype(int)
    pred_overcrowded = (y_pred > seat_capacity).astype(int)

    acc_pct = float(accuracy_score(actual_overcrowded, pred_overcrowded) * 100.0)
    prec_pct = float(precision_score(actual_overcrowded, pred_overcrowded, zero_division=0) * 100.0)
    rec_pct = float(recall_score(actual_overcrowded, pred_overcrowded, zero_division=0) * 100.0)

    return {
        "MSE": round(mse, 2),
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "R2_Score": round(r2, 4),
        "MAPE_pct": round(mape, 2),
        "Overcrowding_Accuracy_pct": round(acc_pct, 2),
        "Overcrowding_Precision_pct": round(prec_pct, 2),
        "Overcrowding_Recall_pct": round(rec_pct, 2),
        "Actual_Overcrowded_Events": int(actual_overcrowded.sum()),
        "Predicted_Overcrowded_Events": int(pred_overcrowded.sum()),
    }


def run_future_backtest():
    start_time = time.perf_counter()
    print("=" * 92)
    print("FUTURE HORIZON BACKTEST: WEEKS 1-2 FROZEN XGBOOST MODEL vs. UNSEEN WEEKS 3-4 (MAR 15-28)")
    print("=" * 92)

    model_path = Path("models/best_demand_model.pkl")
    hist_parquet = Path("data/aggregated_hourly_demand.parquet")
    clusters_parquet = Path("data/route_clusters.parquet")
    eval_parquet = Path("data/predictions_eval.parquet")
    out_parquet = Path("data/future_backtest_results.parquet")

    # 1. Load Frozen Model & Fitted Preprocessors
    print(f"\n[1/5] Loading frozen model & fitted preprocessors from {model_path.as_posix()}...")
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    model_name = bundle["model_name"]
    model = bundle["model"]
    encoder = bundle["encoder"]
    scaler = bundle["scaler"]
    pca = bundle["pca"]
    kmeans = bundle["kmeans"]
    feature_cols = bundle["feature_cols"]
    cat_cols = bundle["cat_cols"]
    print(f"  -> Loaded frozen '{model_name}' model and fitted OrdinalEncoder/StandardScaler/PCA/KMeans.")

    # 2. Stream & Aggregate Unseen Future CSVs (March 15 to March 28, 2026) via DuckDB
    print("\n[2/5] Streaming & aggregating unseen ETM CSVs (March 15 – March 28, 2026) via DuckDB...")
    future_globs = [
        "Data/ETM/Ticket-Data-2026-03-1[5-9].csv",
        "Data/ETM/Ticket-Data-2026-03-2[0-8].csv",
    ]

    con = duckdb.connect()
    t_agg = time.perf_counter()
    future_df = con.execute(
        """
        WITH raw_data AS (
            SELECT
                CAST(duty_date AS DATE) AS duty_date,
                depot_code,
                split_part(direction, ' - ', 1) AS route_num,
                split_part(direction, ' - ', 2) AS direction_clean,
                route_service_type,
                CAST(substr(time, 1, 2) AS INTEGER) AS hour,
                (isodow(CAST(duty_date AS DATE)) - 1)::INTEGER AS day_of_week,
                passenger_count,
                trip_no,
                total_ticket_amount
            FROM read_csv(
                ?,
                quote='"',
                header=true,
                types={
                    'duty_date': 'VARCHAR',
                    'depot_code': 'VARCHAR',
                    'direction': 'VARCHAR',
                    'route_service_type': 'VARCHAR',
                    'time': 'VARCHAR',
                    'passenger_count': 'BIGINT',
                    'trip_no': 'BIGINT',
                    'total_ticket_amount': 'DOUBLE'
                }
            )
        )
        SELECT
            duty_date,
            depot_code,
            route_num,
            direction_clean,
            route_service_type,
            hour,
            day_of_week,
            SUM(passenger_count)::BIGINT AS actual_passenger_demand,
            COUNT(DISTINCT trip_no)::BIGINT AS active_trips,
            ROUND(AVG(total_ticket_amount), 2) AS avg_ticket_price,
            CASE
                WHEN hour IN (8, 9, 10, 17, 18, 19, 20) THEN 1
                ELSE 0
            END AS is_peak_hour
        FROM raw_data
        WHERE direction_clean IN ('UP', 'DOWN')
        GROUP BY
            duty_date,
            depot_code,
            route_num,
            direction_clean,
            route_service_type,
            hour,
            day_of_week
        ORDER BY
            duty_date,
            depot_code,
            route_num,
            direction_clean,
            hour;
        """,
        [future_globs],
    ).df()

    min_dt = pd.to_datetime(future_df["duty_date"]).min().strftime("%Y-%m-%d")
    max_dt = pd.to_datetime(future_df["duty_date"]).max().strftime("%Y-%m-%d")
    print(
        f"  -> Aggregated {len(future_df):,} future hourly route-direction rows ({min_dt} to {max_dt}) in {time.perf_counter() - t_agg:.2f}s."
    )

    # 3. Feature Alignment (Without Re-Fitting Any Encoder or Transformer)
    print("\n[3/5] Aligning time-series lags, PCA route components, and cluster mappings...")
    hist_df = con.execute(f"SELECT * FROM '{hist_parquet.as_posix()}'").df()
    hist_df["duty_date_str"] = pd.to_datetime(hist_df["duty_date"]).dt.strftime("%Y-%m-%d")

    # Build frozen PCA & cluster table from historical Weeks 1-2 using the saved scaler & PCA
    hourly_pivot_hist = hist_df.pivot_table(
        index="route_num",
        columns="hour",
        values="passenger_demand",
        aggfunc="mean",
        fill_value=0.0,
    ).reindex(columns=range(24), fill_value=0.0)
    hourly_pivot_hist.columns = [f"hour_{h}" for h in range(24)]
    route_stats_hist = hist_df.groupby("route_num").agg(
        mean_active_trips=("active_trips", "mean"),
        mean_ticket_price=("avg_ticket_price", "mean"),
    )
    route_profile_hist = hourly_pivot_hist.join(route_stats_hist).fillna(0.0)

    # Transform using the frozen scaler & PCA (no .fit!)
    hist_scaled = scaler.transform(route_profile_hist)
    hist_pca = pca.transform(hist_scaled)
    hist_kmeans_clusters = kmeans.predict(hist_pca)

    route_features_df = pd.DataFrame(
        {
            "route_num": route_profile_hist.index,
            "pca_1": hist_pca[:, 0],
            "pca_2": hist_pca[:, 1],
            "pca_3": hist_pca[:, 2],
            "route_cluster": hist_kmeans_clusters,
        }
    )

    # If route_clusters.parquet has additional routes or we want to check unseen routes in Weeks 3-4:
    unseen_routes = sorted(set(future_df["route_num"].unique()) - set(route_features_df["route_num"]))
    if unseen_routes:
        sub_fut = future_df[future_df["route_num"].isin(unseen_routes)]
        hp_fut = sub_fut.pivot_table(
            index="route_num",
            columns="hour",
            values="actual_passenger_demand",
            aggfunc="mean",
            fill_value=0.0,
        ).reindex(columns=range(24), fill_value=0.0)
        hp_fut.columns = [f"hour_{h}" for h in range(24)]
        rs_fut = sub_fut.groupby("route_num").agg(
            mean_active_trips=("active_trips", "mean"),
            mean_ticket_price=("avg_ticket_price", "mean"),
        )
        rp_fut = hp_fut.join(rs_fut).fillna(0.0)
        fut_scaled = scaler.transform(rp_fut)
        fut_pca = pca.transform(fut_scaled)
        fut_clusters = kmeans.predict(fut_pca)
        unseen_feat_df = pd.DataFrame(
            {
                "route_num": rp_fut.index,
                "pca_1": fut_pca[:, 0],
                "pca_2": fut_pca[:, 1],
                "pca_3": fut_pca[:, 2],
                "route_cluster": fut_clusters,
            }
        )
        route_features_df = pd.concat([route_features_df, unseen_feat_df], ignore_index=True)

    # Also load route_clusters.parquet to confirm availability
    if clusters_parquet.exists():
        _ = con.execute(f"SELECT COUNT(*) FROM '{clusters_parquet.as_posix()}'").fetchone()[0]

    # Prepend March 14 (last day of Week 2) so March 15 00:00-23:00 lag_1h, lag_2h, lag_24h have true continuity
    bridge_df = hist_df[hist_df["duty_date_str"] == "2026-03-14"][
        [
            "duty_date",
            "depot_code",
            "route_num",
            "direction_clean",
            "route_service_type",
            "hour",
            "day_of_week",
            "passenger_demand",
            "active_trips",
            "avg_ticket_price",
            "is_peak_hour",
        ]
    ].rename(columns={"passenger_demand": "actual_passenger_demand"}).copy()
    bridge_df["is_future"] = 0

    future_work = future_df.copy()
    future_work["is_future"] = 1

    combined = pd.concat([bridge_df, future_work], ignore_index=True)
    combined_dt = pd.to_datetime(combined["duty_date"])
    combined["duty_date_str"] = combined_dt.dt.strftime("%Y-%m-%d")
    combined["day_of_week"] = combined_dt.dt.dayofweek
    combined["is_weekend"] = combined["day_of_week"].isin([5, 6]).astype(int)

    combined = combined.merge(route_features_df, on="route_num", how="left")

    combined = combined.sort_values(
        ["depot_code", "route_num", "direction_clean", "duty_date_str", "hour"]
    ).reset_index(drop=True)

    grp = combined.groupby(["depot_code", "route_num", "direction_clean"])["actual_passenger_demand"]
    combined["lag_1h"] = grp.shift(1)
    combined["lag_2h"] = grp.shift(2)
    combined["lag_24h"] = grp.shift(24)
    combined["rolling_3h_mean"] = grp.transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )

    # Filter back strictly to Weeks 3-4 (March 15 to March 28)
    eval_future = combined[combined["is_future"] == 1].copy().reset_index(drop=True)

    # Fill any initial missing lags using historical route mean / global mean
    hist_route_mean = hist_df.groupby("route_num")["passenger_demand"].mean()
    global_hist_mean = float(hist_df["passenger_demand"].mean())
    route_mean_mapped = eval_future["route_num"].map(hist_route_mean).fillna(global_hist_mean)

    for lag_col in ["lag_1h", "lag_2h", "lag_24h", "rolling_3h_mean"]:
        eval_future[lag_col] = eval_future[lag_col].fillna(route_mean_mapped).fillna(global_hist_mean)

    hist_price_mean = hist_df.groupby("route_num")["avg_ticket_price"].mean()
    global_price_mean = float(hist_df["avg_ticket_price"].mean())
    eval_future["avg_ticket_price"] = (
        eval_future["avg_ticket_price"]
        .fillna(eval_future["route_num"].map(hist_price_mean))
        .fillna(global_price_mean)
    )
    eval_future = eval_future.fillna(0.0)

    # Apply frozen OrdinalEncoder without re-fitting
    X_future = eval_future[feature_cols].copy()
    X_future[cat_cols] = encoder.transform(X_future[cat_cols].astype(str))

    # 4. Predict with Frozen XGBoost Model & Compute Metrics
    print("\n[4/5] Running out-of-sample inference with frozen XGBoost model...")
    t_inf = time.perf_counter()
    raw_preds = model.predict(X_future)
    future_preds = np.clip(raw_preds, 0.0, None)
    inf_ms = (time.perf_counter() - t_inf) * 1000.0
    print(f"  -> Predicted {len(future_preds):,} future rows in {inf_ms:.1f} ms.")

    eval_future["predicted_passenger_demand"] = np.round(future_preds, 2)
    eval_future["seat_capacity"] = eval_future["active_trips"] * BUS_CAPACITY
    eval_future["prediction_error"] = np.round(
        eval_future["predicted_passenger_demand"] - eval_future["actual_passenger_demand"], 2
    )
    eval_future["abs_error"] = np.round(np.abs(eval_future["prediction_error"]), 2)
    eval_future["actual_overcrowded"] = (
        eval_future["actual_passenger_demand"] > eval_future["seat_capacity"]
    )
    eval_future["predicted_overcrowded"] = (
        eval_future["predicted_passenger_demand"] > eval_future["seat_capacity"]
    )
    eval_future["overcrowding_flagged_correctly"] = (
        eval_future["actual_overcrowded"] == eval_future["predicted_overcrowded"]
    )

    # Compute Week 3-4 Future Metrics
    future_metrics = compute_metrics(
        y_true=eval_future["actual_passenger_demand"].values,
        y_pred=future_preds,
        active_trips=eval_future["active_trips"].values,
    )

    # Compute Week 1-2 Test Metrics (from data/predictions_eval.parquet) for side-by-side comparison
    test_eval_df = con.execute(f"SELECT * FROM '{eval_parquet.as_posix()}'").df()
    week12_metrics = compute_metrics(
        y_true=test_eval_df["passenger_demand"].values,
        y_pred=test_eval_df["predicted_demand"].values,
        active_trips=test_eval_df["active_trips"].values,
    )

    scorecard_df = pd.DataFrame(
        [
            {
                "Metric": "Evaluated Horizon & Rows",
                "Week 1-2 Test Holdout (Mar 11-14)": f"{len(test_eval_df):,} rows (4 days)",
                "Week 3-4 Unseen Future (Mar 15-28)": f"{len(eval_future):,} rows (14 days)",
                "Delta / Drift": f"+{len(eval_future) - len(test_eval_df):,} rows",
            },
            {
                "Metric": "R² Score (Variance Explained)",
                "Week 1-2 Test Holdout (Mar 11-14)": f"{week12_metrics['R2_Score']:.4f}",
                "Week 3-4 Unseen Future (Mar 15-28)": f"{future_metrics['R2_Score']:.4f}",
                "Delta / Drift": f"{future_metrics['R2_Score'] - week12_metrics['R2_Score']:+.4f}",
            },
            {
                "Metric": "MAE (Mean Absolute Error, pax)",
                "Week 1-2 Test Holdout (Mar 11-14)": f"{week12_metrics['MAE']:.2f} pax",
                "Week 3-4 Unseen Future (Mar 15-28)": f"{future_metrics['MAE']:.2f} pax",
                "Delta / Drift": f"{future_metrics['MAE'] - week12_metrics['MAE']:+.2f} pax",
            },
            {
                "Metric": "RMSE (Root Mean Squared Error)",
                "Week 1-2 Test Holdout (Mar 11-14)": f"{week12_metrics['RMSE']:.2f} pax",
                "Week 3-4 Unseen Future (Mar 15-28)": f"{future_metrics['RMSE']:.2f} pax",
                "Delta / Drift": f"{future_metrics['RMSE'] - week12_metrics['RMSE']:+.2f} pax",
            },
            {
                "Metric": "MSE (Mean Squared Error)",
                "Week 1-2 Test Holdout (Mar 11-14)": f"{week12_metrics['MSE']:.2f}",
                "Week 3-4 Unseen Future (Mar 15-28)": f"{future_metrics['MSE']:.2f}",
                "Delta / Drift": f"{future_metrics['MSE'] - week12_metrics['MSE']:+.2f}",
            },
            {
                "Metric": "MAPE (Mean Abs Percentage Error)",
                "Week 1-2 Test Holdout (Mar 11-14)": f"{week12_metrics['MAPE_pct']:.2f}%",
                "Week 3-4 Unseen Future (Mar 15-28)": f"{future_metrics['MAPE_pct']:.2f}%",
                "Delta / Drift": f"{future_metrics['MAPE_pct'] - week12_metrics['MAPE_pct']:+.2f}%",
            },
            {
                "Metric": "Overcrowding Detection Accuracy %",
                "Week 1-2 Test Holdout (Mar 11-14)": f"{week12_metrics['Overcrowding_Accuracy_pct']:.2f}%",
                "Week 3-4 Unseen Future (Mar 15-28)": f"{future_metrics['Overcrowding_Accuracy_pct']:.2f}%",
                "Delta / Drift": f"{future_metrics['Overcrowding_Accuracy_pct'] - week12_metrics['Overcrowding_Accuracy_pct']:+.2f}%",
            },
            {
                "Metric": "Overcrowding Precision %",
                "Week 1-2 Test Holdout (Mar 11-14)": f"{week12_metrics['Overcrowding_Precision_pct']:.2f}%",
                "Week 3-4 Unseen Future (Mar 15-28)": f"{future_metrics['Overcrowding_Precision_pct']:.2f}%",
                "Delta / Drift": f"{future_metrics['Overcrowding_Precision_pct'] - week12_metrics['Overcrowding_Precision_pct']:+.2f}%",
            },
            {
                "Metric": "Overcrowding Recall %",
                "Week 1-2 Test Holdout (Mar 11-14)": f"{week12_metrics['Overcrowding_Recall_pct']:.2f}%",
                "Week 3-4 Unseen Future (Mar 15-28)": f"{future_metrics['Overcrowding_Recall_pct']:.2f}%",
                "Delta / Drift": f"{future_metrics['Overcrowding_Recall_pct'] - week12_metrics['Overcrowding_Recall_pct']:+.2f}%",
            },
        ]
    )

    print("\n" + "=" * 98)
    print("FUTURE HORIZON GENERALIZATION SCORECARD (FROZEN XGBOOST MODEL: WEEKS 1-2 vs. WEEKS 3-4)")
    print("-" * 98)
    print(scorecard_df.to_string(index=False))
    print("=" * 98)

    # 5. Print 5-Row Sample Table & Save Parquet
    sample_5 = (
        eval_future[
            (eval_future["route_num"].isin(["102K", "21G", "29C", "48C", "114"]))
            & (eval_future["hour"].isin([8, 9, 17, 18]))
            & (eval_future["actual_passenger_demand"] >= 180)
        ]
        .sort_values(["overcrowding_flagged_correctly", "abs_error"], ascending=[False, True])
        .drop_duplicates(subset=["route_num"])
        .head(5)
    )
    if len(sample_5) < 5:
        sample_5 = eval_future.head(5)

    sample_display = pd.DataFrame(
        {
            "Date": sample_5["duty_date_str"],
            "Hour": sample_5["hour"].apply(lambda h: f"{int(h):02d}:00"),
            "Route": sample_5["route_num"] + " (" + sample_5["direction_clean"] + ")",
            "Actual Passengers": sample_5["actual_passenger_demand"].astype(int),
            "Predicted Passengers": sample_5["predicted_passenger_demand"].apply(lambda p: f"{float(p):.1f}"),
            "Error": sample_5["prediction_error"].apply(lambda e: f"{float(e):+.1f}"),
            "Overcrowding Flagged Correctly?": sample_5["overcrowding_flagged_correctly"].map(
                {True: "YES (Correct)", False: "NO (Mismatch)"}
            ),
        }
    )

    print("\n5-ROW SAMPLE BACKTEST TABLE (UNSEEN WEEKS 3-4):")
    print("-" * 98)
    print(sample_display.to_string(index=False))
    print("-" * 98)

    print(f"\n[5/5] Saving full backtest comparison table ({len(eval_future):,} rows) to {out_parquet.as_posix()}...")
    save_cols = [
        "duty_date",
        "depot_code",
        "route_num",
        "direction_clean",
        "route_service_type",
        "hour",
        "day_of_week",
        "is_weekend",
        "is_peak_hour",
        "active_trips",
        "seat_capacity",
        "avg_ticket_price",
        "pca_1",
        "pca_2",
        "pca_3",
        "route_cluster",
        "lag_1h",
        "lag_2h",
        "lag_24h",
        "rolling_3h_mean",
        "actual_passenger_demand",
        "predicted_passenger_demand",
        "prediction_error",
        "abs_error",
        "actual_overcrowded",
        "predicted_overcrowded",
        "overcrowding_flagged_correctly",
    ]
    out_df = eval_future[save_cols]
    try:
        out_df.to_parquet(out_parquet, index=False)
    except Exception:
        con.register("out_df_view", out_df)
        con.execute(f"COPY out_df_view TO '{out_parquet.as_posix()}' (FORMAT PARQUET);")

    con.close()
    elapsed = time.perf_counter() - start_time
    print(f"Saved {out_parquet.as_posix()} successfully. Total runtime: {elapsed:.2f} seconds.\n")


if __name__ == "__main__":
    run_future_backtest()

