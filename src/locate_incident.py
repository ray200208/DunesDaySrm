import duckdb
import pandas as pd


def locate_incident():
    print("=" * 95)
    print("LOCATING INCIDENT BNC-2026-001 (LEAD BUS J1576 & TRAILING BUS K5068)")
    print("=" * 95)

    con = duckdb.connect()

    # 1. Search across Data/ETM/Ticket-Data-2026-03-1*.csv to find which CSV file(s) contain both buses around 16:00
    print("\n[1/3] Scanning `Data/ETM/Ticket-Data-2026-03-1*.csv` for fleet_no IN ('J1576', 'K5068') around 16:00:00...")
    file_matches = con.execute("""
        SELECT
            replace(filename, '\\', '/') AS csv_filename,
            duty_date,
            COUNT(DISTINCT fleet_no) AS matched_buses,
            COUNT(*) AS matching_ticket_rows
        FROM read_csv('Data/ETM/Ticket-Data-2026-03-1*.csv', quote='"', header=true, filename=true, dateformat='%Y-%m-%d')
        WHERE fleet_no IN ('J1576', 'K5068')
          AND CAST(time AS TIME) BETWEEN TIME '15:55:00' AND TIME '17:15:00'
          AND direction LIKE '114%'
        GROUP BY filename, duty_date
        HAVING COUNT(DISTINCT fleet_no) = 2
        ORDER BY duty_date;
    """).df()

    print("\nEXACT CSV FILE WHERE BOTH BUSES APPEAR:")
    print(file_matches.to_string(index=False))

    target_csv = str(file_matches.iloc[0]["csv_filename"])

    # 2. Compute exact CSV row index (and Excel 1-based line number, where line 1 is header) within that file
    print(f"\n[2/3] Extracting exact row indices and first 3 tickets per bus from `{target_csv}`...")
    tickets_df = con.execute(f"""
        WITH numbered_csv AS (
            SELECT
                ROW_NUMBER() OVER () AS data_row_index,
                ROW_NUMBER() OVER () + 1 AS excel_line_number,
                id,
                depot_code,
                depot,
                fleet_no,
                schedule_no,
                trip_no,
                ticket_no,
                duty_date,
                CAST(time AS VARCHAR) AS time,
                route_service_type,
                direction,
                source,
                destination,
                passenger_count
            FROM read_csv('{target_csv}', quote='"', header=true, dateformat='%Y-%m-%d')
        ),
        incident_tickets AS (
            SELECT
                *,
                ROW_NUMBER() OVER (PARTITION BY fleet_no ORDER BY CAST(time AS TIME), ticket_no) AS bus_ticket_rank,
                COUNT(*) OVER (PARTITION BY fleet_no) AS total_trip_tickets,
                SUM(passenger_count) OVER (PARTITION BY fleet_no) AS total_trip_passengers
            FROM numbered_csv
            WHERE (
                (fleet_no = 'J1576' AND trip_no = 12)
                OR
                (fleet_no = 'K5068' AND trip_no = 10)
            )
            AND direction = '114 - DOWN'
        )
        SELECT
            '{target_csv}' AS csv_file,
            excel_line_number,
            data_row_index,
            id AS ticket_id,
            fleet_no,
            trip_no,
            duty_date,
            time,
            ticket_no,
            route_service_type,
            direction,
            source,
            destination,
            passenger_count,
            total_trip_tickets,
            total_trip_passengers
        FROM incident_tickets
        WHERE bus_ticket_rank <= 3
        ORDER BY fleet_no, time, ticket_no;
    """).df()

    print("\n" + "=" * 95)
    print(f"FIRST 3 TICKETS FOR EACH BUS IN `{target_csv}` (INCIDENT BNC-2026-001)")
    print("-" * 95)
    display_cols = [
        "csv_file",
        "excel_line_number",
        "data_row_index",
        "fleet_no",
        "trip_no",
        "duty_date",
        "time",
        "ticket_no",
        "route_service_type",
        "passenger_count",
        "source",
        "destination",
    ]
    print(tickets_df[display_cols].to_string(index=False))
    print("=" * 95)

    print("\n[3/3] Trip Summary for Excel Lookup:")
    for fleet in ["J1576", "K5068"]:
        sub = tickets_df[tickets_df["fleet_no"] == fleet]
        if not sub.empty:
            r0 = sub.iloc[0]
            role = "Lead Bus" if fleet == "J1576" else "Trailing Bus"
            print(
                f"  * {role} `{fleet}` (Route {r0['direction']}, Trip #{r0['trip_no']}):\n"
                f"    - CSV File          : {r0['csv_file']}\n"
                f"    - First Excel Row   : Line {r0['excel_line_number']} (0-based Data Row Index: {r0['data_row_index'] - 1}, 1-based Data Row: {r0['data_row_index']})\n"
                f"    - First Ticket Time : {r0['duty_date']} @ {r0['time']} (ticket_no={r0['ticket_no']}, id={r0['ticket_id']})\n"
                f"    - Total Run Volume  : {r0['total_trip_passengers']} passengers across {r0['total_trip_tickets']} tickets"
            )
    print("=" * 95 + "\n")
    con.close()


if __name__ == "__main__":
    locate_incident()
