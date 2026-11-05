# ACTIVE TASK
Create and run `src/backtest_future_accuracy.py` to evaluate our frozen XGBoost model (trained on Weeks 1–2: March 01–14) against the completely unseen future data in Weeks 3–4 (March 15–28) and print the real-world out-of-sample accuracy.

# CONSTRAINTS
- Data Ingestion via DuckDB (Future Window):
  - Stream and aggregate raw CSVs from `Data/ETM/Ticket-Data-2026-03-1[5-9].csv` and `Data/ETM/Ticket-Data-2026-03-2*.csv` (March 15 to March 28).
  - Use the exact same grouping logic as Step 1:
    - Group by `duty_date`, `depot_code`, `route_num`, `direction_clean`, `route_service_type`, `hour`, and `day_of_week`.
    - Compute `actual_passenger_demand`: `SUM(passenger_count)` and `active_trips`: `COUNT(DISTINCT trip_no)`.
- Feature Alignment:
  - Generate the same features used during training:
    - `hour`, `day_of_week` (Monday=0..Sunday=6), `is_weekend`, `is_peak_hour`.
    - `lag_1h`, `lag_2h`, `lag_24h`, and `rolling_3h_mean`.
    - PCA route components (`pca_1`, `pca_2`, `pca_3`) and `route_cluster` mapped from `data/route_clusters.parquet` (or the fitted encoders).
  - Apply the fitted encoders from `models/` without re-fitting.
- Load Frozen Model:
  - Load `models/best_demand_model.pkl`.
  - Predict `predicted_passenger_demand` across all future rows.
- Compute Real-World Accuracy & Operational Metrics:
  1. Regression Metrics: `RMSE`, `MAE`, `MSE`, and `R2_Score`.
  2. Percentage Error: Mean Absolute Percentage Error (`MAPE`).
  3. Binary Overcrowding Accuracy:
     - `Actual_Overcrowded`: actual_demand > active_trips * 65.
     - `Predicted_Overcrowded`: predicted_demand > active_trips * 65.
     - Calculate **Precision, Recall, and Accuracy %** for correctly anticipating overcrowding events.
- Output Presentation:
  - Print a formatted "Future Horizon Generalization Scorecard" comparing Week 1-2 Test Metrics vs. Week 3-4 Future Metrics.
  - Print a 5-row sample table showing: `Date`, `Hour`, `Route`, `Actual Passengers`, `Predicted Passengers`, `Error`, and `Overcrowding Flagged Correctly?`.
  - Save the full backtest comparison table to `data/future_backtest_results.parquet`.

# DEFINITION OF DONE
- Running `python src/backtest_future_accuracy.py` completes cleanly in under 60 seconds without memory errors.
- Terminal outputs the Future Horizon Generalization Scorecard with R², MAE, and Overcrowding Detection Accuracy.
- `data/future_backtest_results.parquet` is created on disk.