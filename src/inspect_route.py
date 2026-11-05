from pathlib import Path
import numpy as np
import pandas as pd

BUS_CAPACITY = 65


def inspect_route_102k():
    print("=" * 95)
    print("ROUTE INSPECTION: 24-HOUR DEMAND, CAPACITY & CROWDING BREAKDOWN")
    print("Target: Route 102K | Direction: DOWN | Depot: KA (Kannagi Nagar Depot)")
    print("=" * 95)

    eval_path = Path("data/predictions_eval.parquet")
    df = pd.read_parquet(eval_path)
    df["duty_date_str"] = pd.to_datetime(df["duty_date"]).dt.strftime("%Y-%m-%d")

    # Filter strictly for route_num == '102K' (or containing '102K'), direction_clean == 'DOWN', depot_code == 'KA'
    sub = df[
        (df["route_num"].astype(str) == "102K")
        & (df["direction_clean"].astype(str) == "DOWN")
        & (df["depot_code"].astype(str) == "KA")
    ].copy()

    if sub.empty:
        sub = df[
            (df["route_num"].astype(str).str.contains("102K", na=False))
            & (df["direction_clean"].astype(str) == "DOWN")
            & (df["depot_code"].astype(str) == "KA")
        ].copy()

    available_dates = sorted(sub["duty_date_str"].unique().tolist())
    target_date = "2026-03-12" if "2026-03-12" in available_dates else available_dates[0]

    day_df = sub[sub["duty_date_str"] == target_date].sort_values("hour").reset_index(drop=True)

    table_df = pd.DataFrame()
    table_df["hour"] = day_df["hour"].astype(int)
    table_df["actual_passengers"] = day_df["passenger_demand"].round(0).astype(int)
    table_df["predicted_passengers"] = day_df["predicted_demand"].astype(float).round(1)
    table_df["active_trips"] = day_df["active_trips"].astype(int)
    table_df["planned_capacity"] = table_df["active_trips"] * BUS_CAPACITY
    table_df["demand_pressure_pct"] = (
        (table_df["predicted_passengers"] / table_df["planned_capacity"].replace(0, np.nan)) * 100.0
    ).astype(float).round(1).fillna(0.0)
    table_df["status"] = np.where(
        table_df["demand_pressure_pct"] > 100.0, "OVERCROWDED", "NORMAL"
    )

    print(f"\nRepresentative Weekday Selected: {target_date} (Available Test Dates: {', '.join(available_dates)})")
    print("-" * 95)
    print(table_df.to_string(index=False))
    print("-" * 95)

    # Identify peak crowding hour (by demand_pressure_pct, tie-broken by predicted_passengers)
    peak_idx = table_df["demand_pressure_pct"].idxmax()
    peak_row = table_df.loc[peak_idx]

    max_pax_idx = table_df["actual_passengers"].idxmax()
    max_pax_row = table_df.loc[max_pax_idx]

    total_actual = int(table_df["actual_passengers"].sum())
    total_pred = float(table_df["predicted_passengers"].sum())
    total_trips = int(table_df["active_trips"].sum())
    total_cap = int(table_df["planned_capacity"].sum())
    overcrowded_hours = table_df[table_df["status"] == "OVERCROWDED"]["hour"].tolist()

    print("\nSUMMARY FOR ROUTE 102K (DOWN) - KANNAGI NAGAR DEPOT (KA):")
    print(f"  * Representative Date         : {target_date} (Thursday)")
    print(f"  * Operating Hours Recorded    : {len(table_df)} hours (Hours {table_df['hour'].min():02d}:00 to {table_df['hour'].max():02d}:00)")
    print(f"  * Total Daily Actual Demand   : {total_actual:,} passengers")
    print(f"  * Total Daily Predicted Demand: {total_pred:,.1f} passengers")
    print(f"  * Total Daily Active Trips    : {total_trips} bus runs ({total_cap:,} total seats @ 65 seats/bus)")
    print(f"  * Overcrowded Hours (>100%)   : {len(overcrowded_hours)} hours -> {overcrowded_hours}")
    print(
        f"  * Peak Crowding Hour (Pressure): Hour {int(peak_row['hour']):02d}:00 "
        f"| Predicted: {peak_row['predicted_passengers']:.1f} pax | Actual: {int(peak_row['actual_passengers'])} pax "
        f"| Active Trips: {int(peak_row['active_trips'])} ({int(peak_row['planned_capacity'])} cap) "
        f"| Pressure: {peak_row['demand_pressure_pct']:.1f}% ({peak_row['status']})"
    )
    print(
        f"  * Peak Actual Ridership Hour  : Hour {int(max_pax_row['hour']):02d}:00 "
        f"| Actual: {int(max_pax_row['actual_passengers'])} pax | Predicted: {max_pax_row['predicted_passengers']:.1f} pax "
        f"| Active Trips: {int(max_pax_row['active_trips'])} ({int(max_pax_row['planned_capacity'])} cap) "
        f"| Pressure: {max_pax_row['demand_pressure_pct']:.1f}% ({max_pax_row['status']})"
    )
    print("=" * 95 + "\n")


if __name__ == "__main__":
    inspect_route_102k()
