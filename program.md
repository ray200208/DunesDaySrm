# ACTIVE TASK
Create and run `src/inspect_route.py` to filter and display the full 24-hour demand, capacity, and crowding predictions specifically for Route `102K`, direction `DOWN`, from depot `KA` (Kannagi Nagar).

# CONSTRAINTS
- Read `data/predictions_eval.parquet`.
- Filter strictly for:
  - `route_num == '102K'` (or containing '102K')
  - `direction_clean == 'DOWN'`
  - `depot_code == 'KA'`
- If test set has multiple dates (March 11–14), select one representative weekday (e.g., `2026-03-12` or `2026-03-11`).
- Sort by `hour` (0 to 23).
- Print a clear console table with these columns:
  - `hour`: Hour of the day
  - `actual_passengers`: Actual recorded ticket demand
  - `predicted_passengers`: XGBoost predicted demand
  - `active_trips`: Scheduled trips on the road
  - `planned_capacity`: Total passenger capacity (`active_trips * 65`)
  - `demand_pressure_pct`: (`predicted / capacity * 100`)
  - `status`: 'OVERCROWDED' if pressure > 100% else 'NORMAL'
- Print the peak crowding hour and summarize the exact passenger numbers for Route 102K.

# DEFINITION OF DONE
- Running `python src/inspect_route.py` completes cleanly.
- Console displays the hourly breakdown table for Route 102K DOWN from KA depot.