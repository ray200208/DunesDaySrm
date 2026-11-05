import duckdb

con = duckdb.connect()
files = [
    "Data/ETM/Ticket-Data-2026-03-0*.csv",
    "Data/ETM/Ticket-Data-2026-03-1[0-4].csv",
]
sql = """
WITH raw_data AS (
    SELECT 
        strptime(duty_date, '%Y-%m-%d')::DATE AS duty_date,
        depot_code,
        depot,
        regexp_extract(direction, '(.*) - (UP|DOWN)', 1) AS route_num,
        regexp_extract(direction, '(.*) - (UP|DOWN)', 2) AS direction_clean,
        route_service_type,
        EXTRACT(HOUR FROM CAST(time AS TIME))::INTEGER AS hour,
        (isodow(strptime(duty_date, '%Y-%m-%d')::DATE) - 1)::INTEGER AS day_of_week,
        passenger_count,
        trip_no,
        total_ticket_amount
    FROM read_csv(?, quote='"', header=true, types={'duty_date': 'VARCHAR'})
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
LIMIT 5
"""
print(con.execute(sql, [files]).df())