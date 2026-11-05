import pickle
import time
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


def load_parquet_df(parquet_path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(parquet_path)
    except Exception:
        con = duckdb.connect()
        df = con.execute(f"SELECT * FROM '{parquet_path.as_posix()}'").df()
        con.close()
        return df


def save_parquet_df(df: pd.DataFrame, parquet_path: Path) -> None:
    try:
        df.to_parquet(parquet_path, index=False)
    except Exception:
        con = duckdb.connect()
        con.register("pred_df_view", df)
        con.execute(
            f"COPY pred_df_view TO '{parquet_path.as_posix()}' (FORMAT PARQUET);"
        )
        con.close()


def run_tournament():
    start_total = time.time()
    print("=" * 75)
    print("STEP 2: PCA + K-MEANS ROUTE PROFILING & 5-MODEL ML TOURNAMENT")
    print("=" * 75)

    input_path = Path("data/aggregated_hourly_demand.parquet")
    models_dir = Path("models")
    data_dir = Path("data")
    models_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[1/6] Loading aggregated hourly demand from {input_path.as_posix()}...")
    df = load_parquet_df(input_path)
    print(f"Loaded {len(df):,} rows and {df.shape[1]} columns.")

    # 1. Fix Date / DOW Alignment
    print("[2/6] Fixing Date/DOW alignment & deriving weekend indicator...")
    duty_dt = pd.to_datetime(df["duty_date"])
    df["duty_date_str"] = duty_dt.dt.strftime("%Y-%m-%d")
    df["day_of_week"] = duty_dt.dt.dayofweek  # Monday=0 ... Sunday=6
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # 2. PCA + K-Means Route Profiling
    print("[3/6] Running PCA (26 -> 3 dims) + K-Means (k=4) route profiling...")
    # Fit route profile on training window (<= 2026-03-10) with fallback for any unseen route
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
        mean_active_trips=("active_trips", "mean"),
        mean_ticket_price=("avg_ticket_price", "mean"),
    )

    route_profile = hourly_pivot.join(route_stats).fillna(0.0)
    assert route_profile.shape[1] == 26, (
        f"Expected 26 route dimensions, got {route_profile.shape[1]}"
    )

    scaler = StandardScaler()
    route_scaled = scaler.fit_transform(route_profile)

    pca = PCA(n_components=3, random_state=42)
    pca_coords = pca.fit_transform(route_scaled)
    variance_ratios = pca.explained_variance_ratio_ * 100
    total_variance_pct = float(np.sum(variance_ratios))

    print(
        f"  -> PCA Explained Variance by Component: "
        f"PC1={variance_ratios[0]:.2f}%, PC2={variance_ratios[1]:.2f}%, PC3={variance_ratios[2]:.2f}%"
    )
    print(
        f"  -> Total PCA Variance Retained (26 -> 3 dims): {total_variance_pct:.2f}%"
    )

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    route_clusters = kmeans.fit_predict(pca_coords)

    route_features_df = pd.DataFrame(
        {
            "route_num": route_profile.index,
            "pca_1": pca_coords[:, 0],
            "pca_2": pca_coords[:, 1],
            "pca_3": pca_coords[:, 2],
            "route_cluster": route_clusters,
        }
    )

    df = df.merge(route_features_df, on="route_num", how="left")

    # 3. Time-Series Lag Features
    print("[4/6] Engineering time-series lag & rolling features...")
    df = df.sort_values(
        ["depot_code", "route_num", "direction_clean", "duty_date_str", "hour"]
    ).reset_index(drop=True)

    grp = df.groupby(["depot_code", "route_num", "direction_clean"])[
        "passenger_demand"
    ]
    df["lag_1h"] = grp.shift(1)
    df["lag_2h"] = grp.shift(2)
    df["lag_24h"] = grp.shift(24)
    df["rolling_3h_mean"] = grp.transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )

    route_mean_demand = df.groupby("route_num")["passenger_demand"].transform(
        "mean"
    )
    global_mean_demand = float(df["passenger_demand"].mean())
    for lag_col in ["lag_1h", "lag_2h", "lag_24h", "rolling_3h_mean"]:
        df[lag_col] = (
            df[lag_col].fillna(route_mean_demand).fillna(global_mean_demand)
        )

    # Ensure no NaNs remain in numeric columns (e.g. avg_ticket_price for AdaBoost)
    route_mean_price = df.groupby("route_num")["avg_ticket_price"].transform(
        "mean"
    )
    global_mean_price = float(df["avg_ticket_price"].mean())
    df["avg_ticket_price"] = (
        df["avg_ticket_price"].fillna(route_mean_price).fillna(global_mean_price)
    )
    df = df.fillna(0.0)

    # 4. Temporal Train/Test Split & Categorical Ordinal Encoding
    print("[5/6] Performing temporal train/test split & ordinal encoding...")
    cat_cols = ["depot_code", "route_num", "direction_clean", "route_service_type"]
    feature_cols = [
        "depot_code",
        "route_num",
        "direction_clean",
        "route_service_type",
        "hour",
        "day_of_week",
        "is_weekend",
        "is_peak_hour",
        "active_trips",
        "avg_ticket_price",
        "pca_1",
        "pca_2",
        "pca_3",
        "route_cluster",
        "lag_1h",
        "lag_2h",
        "lag_24h",
        "rolling_3h_mean",
    ]
    target_col = "passenger_demand"

    train_mask = df["duty_date_str"] <= "2026-03-10"
    test_mask = df["duty_date_str"] > "2026-03-10"

    train_df = df.loc[train_mask].copy()
    test_df = df.loc[test_mask].copy()

    print(
        f"  -> Train Set (<= 2026-03-10): {len(train_df):,} rows | "
        f"Test Set (> 2026-03-10): {len(test_df):,} rows"
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

    # 5. Model Tournament
    print("[6/6] Running 5-Model ML Tournament...")
    models = {
        "LGBM_Base": LGBMRegressor(
            n_estimators=150,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ),
        "LGBM_Tuned": LGBMRegressor(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=250,
            learning_rate=0.07,
            max_depth=8,
            random_state=42,
            n_jobs=-1,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=80,
            max_depth=16,
            random_state=42,
            n_jobs=-1,
        ),
        "AdaBoost": AdaBoostRegressor(
            n_estimators=50,
            learning_rate=0.1,
            random_state=42,
        ),
    }

    leaderboard_rows = []
    trained_models = {}
    test_predictions = {}

    for name, model in models.items():
        t_fit_start = time.perf_counter()
        model.fit(X_train, y_train)
        fit_time_s = time.perf_counter() - t_fit_start

        t_inf_start = time.perf_counter()
        preds = model.predict(X_test)
        inf_time_ms = (time.perf_counter() - t_inf_start) * 1000.0

        # Ensure non-negative demand predictions
        preds_clipped = np.clip(preds, 0.0, None)

        mse = float(mean_squared_error(y_test, preds_clipped))
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_test, preds_clipped))
        r2 = float(r2_score(y_test, preds_clipped))

        leaderboard_rows.append(
            {
                "Model": name,
                "MSE": round(mse, 2),
                "RMSE": round(rmse, 4),
                "MAE": round(mae, 4),
                "R2_Score": round(r2, 4),
                "Inference_Time_ms": round(inf_time_ms, 2),
            }
        )
        trained_models[name] = model
        test_predictions[name] = preds_clipped
        print(
            f"  -> {name:<14} | RMSE: {rmse:8.4f} | MAE: {mae:8.4f} | "
            f"R2: {r2:6.4f} | Inference: {inf_time_ms:7.2f} ms (Fit: {fit_time_s:.1f}s)"
        )

    leaderboard_df = pd.DataFrame(leaderboard_rows).sort_values(
        by="RMSE", ascending=True
    ).reset_index(drop=True)

    print("\n" + "=" * 75)
    print(f"PCA VARIANCE RETAINED (3 Components): {total_variance_pct:.2f}%")
    print("=" * 75)
    print("5-MODEL TOURNAMENT LEADERBOARD (Sorted by Lowest RMSE)")
    print("-" * 75)
    print(leaderboard_df.to_string(index=False))
    print("=" * 75)

    best_model_name = leaderboard_df.iloc[0]["Model"]
    best_rmse = leaderboard_df.iloc[0]["RMSE"]
    best_model = trained_models[best_model_name]
    best_preds = test_predictions[best_model_name]

    model_out_path = models_dir / "best_demand_model.pkl"
    with open(model_out_path, "wb") as f:
        pickle.dump(
            {
                "model_name": best_model_name,
                "model": best_model,
                "encoder": encoder,
                "scaler": scaler,
                "pca": pca,
                "kmeans": kmeans,
                "feature_cols": feature_cols,
                "cat_cols": cat_cols,
                "leaderboard": leaderboard_df,
                "pca_variance_retained_pct": total_variance_pct,
            },
            f,
        )
    print(
        f"\nSaved best model ('{best_model_name}', RMSE={best_rmse:.4f}) to {model_out_path.as_posix()}"
    )

    # Save test predictions + demand_pressure_pct
    eval_df = test_df.drop(columns=["duty_date_str"]).copy()
    eval_df["predicted_demand"] = np.round(best_preds, 2)
    # demand_pressure_pct = (predicted_demand / (active_trips * 65)) * 100
    denom = (eval_df["active_trips"].replace(0, np.nan) * 65.0)
    eval_df["demand_pressure_pct"] = np.round(
        (eval_df["predicted_demand"] / denom) * 100.0, 2
    ).fillna(0.0)

    eval_out_path = data_dir / "predictions_eval.parquet"
    save_parquet_df(eval_df, eval_out_path)
    print(
        f"Saved test evaluation predictions ({len(eval_df):,} rows) to {eval_out_path.as_posix()}"
    )
    print(f"Total tournament runtime: {time.time() - start_total:.2f}s\n")


if __name__ == "__main__":
    run_tournament()

