import time
from pathlib import Path
import duckdb


def run_aggregation():
    start_time = time.time()
    print("Starting ETM data preprocessing & hourly aggregation...")

    # Ensure output directories exist
    Path("src").mkdir(parents=True, exist_ok=True)
    Path("data").mkdir(parents=True, exist_ok=True)

    parquet_out = Path("data/aggregated_hourly_demand.parquet")
    csv_out = Path("data/aggregated_hourly_demand.csv")

    # Files to process: March 01 to March 14, 2026
    files = [
        "Data/ETM/Ticket-Data-2026-03-0*.csv",
        "Data/ETM/Ticket-Data-2026-03-1[0-4].csv",
    ]

    con = duckdb.connect()

    aggregation_sql = f"""
    CREATE OR REPLACE TABLE aggregated_hourly_demand AS
    WITH raw_data AS (
        SELECT 
            CAST(duty_date AS DATE) AS duty_date,
            depot_code,
            depot,
            regexp_extract(direction, '(.*) - (UP|DOWN)', 1) AS route_num,
            regexp_extract(direction, '(.*) - (UP|DOWN)', 2) AS direction_clean,
            route_service_type,
            EXTRACT(HOUR FROM CAST(time AS TIME))::INTEGER AS hour,
            (isodow(CAST(duty_date AS DATE)) - 1)::INTEGER AS day_of_week,
            passenger_count,
            trip_no,
            total_ticket_amount
        FROM read_csv(?, quote='"', header=true, types={{'duty_date': 'VARCHAR'}})
    )
    SELECT 
        duty_date,
        depot_code,
        depot,
        route_num,
        direction_clean,
        route_service_type,
        hour,
        day_of_week,
        SUM(passenger_count)::BIGINT AS passenger_demand,
        COUNT(DISTINCT trip_no)::BIGINT AS active_trips,
        ROUND(AVG(total_ticket_amount), 2) AS avg_ticket_price,
        CASE 
            WHEN hour IN (8, 9, 10, 17, 18, 19, 20) THEN 1 
            ELSE 0 
        END AS is_peak_hour
    FROM raw_data
    GROUP BY 
        duty_date,
        depot_code,
        depot,
        route_num,
        direction_clean,
        route_service_type,
        hour,
        day_of_week
    ORDER BY 
        duty_date,
        route_num,
        direction_clean,
        hour;
    """

    print("Aggregating 14 days of ETM data using DuckDB streaming...")
    con.execute(aggregation_sql, [files])

    print(f"Exporting to Parquet: {parquet_out}...")
    con.execute(f"COPY aggregated_hourly_demand TO '{parquet_out.as_posix()}' (FORMAT PARQUET);")

    print(f"Exporting to CSV: {csv_out}...")
    con.execute(f"COPY aggregated_hourly_demand TO '{csv_out.as_posix()}' (HEADER, DELIMITER ',');")

    # Fetch summary statistics
    summary = con.execute("""
        SELECT 
            COUNT(*) AS total_rows,
            MIN(duty_date) AS min_date,
            MAX(duty_date) AS max_date
        FROM aggregated_hourly_demand;
    """).fetchone()

    total_rows, min_date, max_date = summary

    # Fetch first 5 rows
    first_5 = con.execute("SELECT * FROM aggregated_hourly_demand LIMIT 5;").df()

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("ETM HOURLY DEMAND AGGREGATION COMPLETE")
    print("=" * 60)
    print(f"Total Aggregated Row Count: {total_rows:,}")
    print(f"Date Range: {min_date} to {max_date}")
    print(f"Elapsed Time: {elapsed:.2f} seconds")
    print("\nFirst 5 Rows:")
    print(first_5.to_string(index=False))
    print("=" * 60 + "\n")

    con.close()


if __name__ == "__main__":
    run_aggregation()