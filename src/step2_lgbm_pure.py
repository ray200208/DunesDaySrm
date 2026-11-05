import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OrdinalEncoder

warnings.filterwarnings("ignore")

XGB_BENCHMARK = {
    "Model": "XGBoost (PCA+KMeans Benchmark)",
    "MSE": 2357.89,
    "RMSE": 48.5581,
    "MAE": 31.5086,
    "R2_Score": 0.8089,
    "Training_Time_seconds": 8.10,
    "Inference_Time_ms": 266.87,
}


def run_pure_lgbm_benchmark():
    start_total = time.perf_counter()
    print("=" * 80)
    print("STEP 2B: PURE LIGHTGBM BENCHMARK (NO PCA) VS. XGBOOST")
    print("=" * 80)

    input_path = Path("data/aggregated_hourly_demand.parquet")
    models_dir = Path("models")
    data_dir = Path("data")
    models_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[1/4] Reading {input_path.as_posix()}...")
    df = pd.read_parquet(input_path)

    # Temporal features
    print("[2/4] Engineering pure temporal & lag features (NO PCA)...")
    duty_dt = pd.to_datetime(df["duty_date"])
    df["duty_date_str"] = duty_dt.dt.strftime("%Y-%m-%d")
    df["day_of_week"] = duty_dt.dt.dayofweek  # Monday=0..Sunday=6
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Sort identically for deterministic lag computation and split
    df = df.sort_values(
        ["depot_code", "route_num", "direction_clean", "duty_date_str", "hour"]
    ).reset_index(drop=True)

    grp = df.groupby(["route_num", "direction_clean"], sort=False)[
        "passenger_demand"
    ]
    df["lag_1h"] = grp.shift(1)
    df["lag_2h"] = grp.shift(2)
    lag_3h = grp.shift(3)
    df["lag_24h"] = grp.shift(24)
    # Vectorized 3-hour rolling mean of prior 3 hours (equivalent to shift(1).rolling(3, min_periods=1).mean())
    df["rolling_3h_mean"] = pd.concat(
        [df["lag_1h"], df["lag_2h"], lag_3h], axis=1
    ).mean(axis=1)

    route_mean_demand = df.groupby("route_num", sort=False)[
        "passenger_demand"
    ].transform("mean")
    global_mean_demand = float(df["passenger_demand"].mean())
    for col in ["lag_1h", "lag_2h", "lag_24h", "rolling_3h_mean"]:
        df[col] = df[col].fillna(route_mean_demand).fillna(global_mean_demand)

    route_mean_price = df.groupby("route_num", sort=False)[
        "avg_ticket_price"
    ].transform("mean")
    df["avg_ticket_price"] = (
        df["avg_ticket_price"]
        .fillna(route_mean_price)
        .fillna(float(df["avg_ticket_price"].mean()))
    )

    # Categorical encoding & Temporal Split
    print("[3/4] Applying OrdinalEncoder and Temporal Train/Test Split...")
    cat_cols = ["depot_code", "route_num", "direction_clean", "route_service_type"]
    feature_cols = [
        # Temporal
        "hour",
        "day_of_week",
        "is_weekend",
        "is_peak_hour",
        # Time-series lags
        "lag_1h",
        "lag_2h",
        "lag_24h",
        "rolling_3h_mean",
        # Route context
        "avg_ticket_price",
        "active_trips",
        # Categoricals
        "depot_code",
        "route_num",
        "direction_clean",
        "route_service_type",
    ]
    target_col = "passenger_demand"

    train_mask = df["duty_date_str"] <= "2026-03-10"
    test_mask = df["duty_date_str"] > "2026-03-10"

    train_df = df.loc[train_mask].copy()
    test_df = df.loc[test_mask].copy()
    print(
        f"  -> Train Set: {len(train_df):,} rows | Test Set: {len(test_df):,} rows"
    )

    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1
    )
    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()

    X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols].astype(str))
    X_test[cat_cols] = encoder.transform(X_test[cat_cols].astype(str))

    y_train = train_df[target_col].values
    y_test = test_df[target_col].values

    print("[4/4] Training Pure LightGBM Models...")
    lgbm_models = {
        "LGBM_Pure_Base": LGBMRegressor(
            n_estimators=200,
            learning_rate=0.08,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ),
        "LGBM_Pure_Tuned": LGBMRegressor(
            n_estimators=450,
            learning_rate=0.04,
            num_leaves=70,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ),
    }

    results = []
    trained_models = {}
    predictions = {}

    for name, model in lgbm_models.items():
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        preds = model.predict(X_test)
        inf_time_ms = (time.perf_counter() - t1) * 1000.0

        preds_clipped = np.clip(preds, 0.0, None)
        mse = float(mean_squared_error(y_test, preds_clipped))
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_test, preds_clipped))
        r2 = float(r2_score(y_test, preds_clipped))

        row = {
            "Model": name,
            "MSE": round(mse, 2),
            "RMSE": round(rmse, 4),
            "MAE": round(mae, 4),
            "R2_Score": round(r2, 4),
            "Training_Time_seconds": round(train_time_s, 2),
            "Inference_Time_ms": round(inf_time_ms, 2),
        }
        results.append(row)
        trained_models[name] = model
        predictions[name] = preds_clipped

    # Combine with previous XGBoost benchmark for comparison table
    all_rows = results + [XGB_BENCHMARK]
    comparison_df = (
        pd.DataFrame(all_rows)
        .sort_values(by="RMSE", ascending=True)
        .reset_index(drop=True)
    )

    print("\n" + "=" * 80)
    print("PURE LIGHTGBM (NO PCA) VS. XGBOOST BENCHMARK TABLE")
    print("-" * 80)
    print(comparison_df.to_string(index=False))
    print("=" * 80)

    # Determine best pure LightGBM model
    pure_df = pd.DataFrame(results).sort_values(by="RMSE", ascending=True)
    best_pure_row = pure_df.iloc[0]
    best_pure_name = best_pure_row["Model"]
    best_pure_rmse = best_pure_row["RMSE"]
    best_pure_r2 = best_pure_row["R2_Score"]

    xgb_rmse = XGB_BENCHMARK["RMSE"]
    xgb_r2 = XGB_BENCHMARK["R2_Score"]
    rmse_diff = best_pure_rmse - xgb_rmse

    print("\nCOMPARISON VERDICT:")
    if best_pure_rmse < xgb_rmse:
        print(
            f"  [WINNER: {best_pure_name}] Pure LightGBM WITHOUT PCA beat the previous XGBoost benchmark!"
        )
        print(
            f"  -> RMSE improved from {xgb_rmse:.4f} to {best_pure_rmse:.4f} ({abs(rmse_diff):.4f} lower RMSE)"
        )
        print(f"  -> R2 improved from {xgb_r2:.4f} to {best_pure_r2:.4f}")

        model_out_path = models_dir / "best_demand_model.pkl"
        with open(model_out_path, "wb") as f:
            pickle.dump(
                {
                    "model_name": best_pure_name,
                    "model": trained_models[best_pure_name],
                    "encoder": encoder,
                    "feature_cols": feature_cols,
                    "cat_cols": cat_cols,
                    "comparison_table": comparison_df,
                },
                f,
            )

        eval_df = test_df.drop(columns=["duty_date_str"]).copy()
        eval_df["predicted_demand"] = np.round(predictions[best_pure_name], 2)
        denom = eval_df["active_trips"].replace(0, np.nan) * 65.0
        eval_df["demand_pressure_pct"] = np.round(
            (eval_df["predicted_demand"] / denom) * 100.0, 2
        ).fillna(0.0)

        eval_out_path = data_dir / "predictions_eval.parquet"
        eval_df.to_parquet(eval_out_path, index=False)
        print(
            f"  -> Overwrote {model_out_path.as_posix()} and {eval_out_path.as_posix()} with {best_pure_name}."
        )
    else:
        print(
            f"  [WINNER: XGBoost (PCA+KMeans Benchmark)] Previous XGBoost remains the champion."
        )
        print(
            f"  -> Best Pure LightGBM ({best_pure_name}) achieved RMSE={best_pure_rmse:.4f}, R2={best_pure_r2:.4f} "
            f"vs. XGBoost RMSE={xgb_rmse:.4f}, R2={xgb_r2:.4f} (+{rmse_diff:.4f} RMSE gap)."
        )
        print(
            "  -> Existing models/best_demand_model.pkl and data/predictions_eval.parquet retained."
        )

    total_elapsed = time.perf_counter() - start_total
    print(f"\nTotal runtime: {total_elapsed:.2f} seconds\n")


if __name__ == "__main__":
    run_pure_lgbm_benchmark()

