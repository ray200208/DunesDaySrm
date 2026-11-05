import duckdb
import pandas as pd


def run_sanity_audit():
    print("=" * 95)
    print("DATA SANITY AUDIT: INVESTIGATING 'active_trips' COLLISION ON ROUTES 102K & 114")
    print("=" * 95)

    raw_csv = "Data/ETM/Ticket-Data-2026-03-12.csv"
    agg_parquet = "data/aggregated_hourly_demand.parquet"

    con = duckdb.connect()

    # 1. Inspect what aggregated_hourly_demand.parquet recorded on 2026-03-12 for 102K and 114
    print("\n[1/3] Checking existing `data/aggregated_hourly_demand.parquet` (2026-03-12)...")
    agg_df = con.execute(f"""
        SELECT
            duty_date,
            depot_code,
            route_num,
            direction_clean,
            hour,
            passenger_demand,
            active_trips AS recorded_active_trips,
            ROUND(passenger_demand / (active_trips * 65.0) * 100.0, 1) AS recorded_pressure_pct
        FROM '{agg_parquet}'
        WHERE duty_date = DATE '2026-03-12'
          AND route_num IN ('102K', '114')
          AND hour IN (7, 13)
        ORDER BY route_num, hour, direction_clean;
    """).df()
    print(agg_df.to_string(index=False))

    # 2. Inspect Raw CSV between 13:00:00 and 13:59:59 (plus hour 7 for 114 alert comparison)
    print(f"\n[2/3] Auditing Raw ETM CSV (`{raw_csv}`) for Routes 102K & 114 (13:00:00 - 13:59:59)...")
    raw_13h_df = con.execute(f"""
        WITH raw_filtered AS (
            SELECT
                depot_code,
                depot,
                trim(replace(replace(direction, ' - UP', ''), ' - DOWN', '')) AS route_num,
                CASE WHEN direction LIKE '% - UP' THEN 'UP' ELSE 'DOWN' END AS direction_clean,
                route_service_type,
                fleet_no,
                schedule_no,
                waybill_no,
                trip_no,
                passenger_count,
                time
            FROM read_csv('{raw_csv}', quote='"', header=true, dateformat='%Y-%m-%d')
            WHERE trim(replace(replace(direction, ' - UP', ''), ' - DOWN', '')) IN ('102K', '114')
              AND CAST(time AS TIME) BETWEEN TIME '13:00:00' AND TIME '13:59:59'
        )
        SELECT
            route_num,
            direction_clean,
            depot_code,
            COUNT(*) AS raw_ticket_rows,
            SUM(passenger_count)::BIGINT AS total_passengers,
            COUNT(DISTINCT trip_no)::BIGINT AS distinct_trip_no_naive,
            COUNT(DISTINCT fleet_no)::BIGINT AS distinct_fleet_no_buses,
            COUNT(DISTINCT schedule_no)::BIGINT AS distinct_schedule_no,
            COUNT(DISTINCT (fleet_no || '-' || CAST(trip_no AS VARCHAR)))::BIGINT AS true_unique_bus_trips,
            ROUND(SUM(passenger_count) * 1.0 / NULLIF(COUNT(DISTINCT trip_no), 0), 1) AS naive_pax_per_trip,
            ROUND(SUM(passenger_count) * 1.0 / NULLIF(COUNT(DISTINCT fleet_no), 0), 1) AS true_avg_pax_per_bus,
            ROUND((SUM(passenger_count) * 1.0 / (NULLIF(COUNT(DISTINCT trip_no), 0) * 65.0)) * 100.0, 1) AS naive_pressure_pct,
            ROUND((SUM(passenger_count) * 1.0 / (NULLIF(COUNT(DISTINCT (fleet_no || '-' || CAST(trip_no AS VARCHAR))), 0) * 65.0)) * 100.0, 1) AS true_pressure_pct
        FROM raw_filtered
        GROUP BY route_num, direction_clean, depot_code
        ORDER BY route_num, direction_clean, depot_code;
    """).df()
    print(raw_13h_df.to_string(index=False))

    # 3. Also audit across both Alert Windows (13:00 for 102K and 07:00 for 114)
    print("\n[3/3] Auditing Exact High-Priority Alert Windows (102K @ 13:00 & 114 @ 07:00)...")
    alert_audit_df = con.execute(f"""
        WITH raw_filtered AS (
            SELECT
                depot_code,
                trim(replace(replace(direction, ' - UP', ''), ' - DOWN', '')) AS route_num,
                CASE WHEN direction LIKE '% - UP' THEN 'UP' ELSE 'DOWN' END AS direction_clean,
                EXTRACT(HOUR FROM CAST(time AS TIME))::INTEGER AS hour,
                fleet_no,
                schedule_no,
                trip_no,
                passenger_count
            FROM read_csv('{raw_csv}', quote='"', header=true, dateformat='%Y-%m-%d')
            WHERE (
                (trim(replace(replace(direction, ' - UP', ''), ' - DOWN', '')) = '102K' AND EXTRACT(HOUR FROM CAST(time AS TIME)) = 13)
                OR
                (trim(replace(replace(direction, ' - UP', ''), ' - DOWN', '')) = '114' AND EXTRACT(HOUR FROM CAST(time AS TIME)) = 7)
            )
        )
        SELECT
            route_num,
            direction_clean,
            depot_code,
            hour,
            SUM(passenger_count)::BIGINT AS total_passengers,
            COUNT(DISTINCT trip_no)::BIGINT AS naive_trip_no_count,
            COUNT(DISTINCT fleet_no)::BIGINT AS actual_unique_buses,
            COUNT(DISTINCT schedule_no)::BIGINT AS distinct_schedules,
            COUNT(DISTINCT (fleet_no || '-' || CAST(trip_no AS VARCHAR)))::BIGINT AS true_unique_bus_trips,
            ROUND(SUM(passenger_count) * 1.0 / NULLIF(COUNT(DISTINCT fleet_no), 0), 1) AS true_pax_per_bus,
            ROUND((SUM(passenger_count) * 1.0 / (NULLIF(COUNT(DISTINCT (fleet_no || '-' || CAST(trip_no AS VARCHAR))), 0) * 65.0)) * 100.0, 1) AS corrected_pressure_pct
        FROM raw_filtered
        GROUP BY route_num, direction_clean, depot_code, hour
        ORDER BY total_passengers DESC;
    """).df()
    print(alert_audit_df.to_string(index=False))

    print("\n" + "=" * 95)
    print("ROOT CAUSE DIAGNOSIS: WHY THE 800-900 PASSENGER SPIKE & >200% PRESSURE APPEARED")
    print("=" * 95)
    print(
        "1. `trip_no` IS A SHIFT SEQUENCE NUMBER (1, 2, 3, 4...), NOT A UNIQUE TRIP ID:\n"
        "   In the raw ETM ticket table, every bus (`fleet_no`) numbers its daily runs sequentially\n"
        "   as trip_no = 1, 2, 3, 4, etc. When 10+ buses operate on Route 102K during 13:00-13:59,\n"
        "   almost all of them are simultaneously on their 3rd, 4th, 5th, or 6th trip of the day.\n"
        "   Using `COUNT(DISTINCT trip_no)` collapses all 10+ buses into just 4 distinct integers!\n"
    )
    print(
        "2. WHY 800-922 PASSENGERS APPEARED:\n"
        "   The 922 passengers on Route 102K (DOWN, 13:00-13:59) are cumulative boardings across ALL\n"
        "   active buses on that route during the hour (and across all stage-to-stage turnovers),\n"
        "   NOT 922 people crammed into 4 buses. Dividing by `COUNT(DISTINCT fleet_no)` (actual buses)\n"
        "   or `COUNT(DISTINCT (fleet_no || '-' || trip_no))` (true unique bus trips) reveals a much\n"
        "   more realistic per-bus passenger load."
    )
    print("=" * 95 + "\n")

    con.close()


if __name__ == "__main__":
    run_sanity_audit()

