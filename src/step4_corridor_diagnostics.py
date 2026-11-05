import json
import time
from pathlib import Path
from typing import Any, Dict, List

import duckdb
import numpy as np
import pandas as pd

BUS_CAPACITY = 65
TARGET_ROUTES = ["102K", "102P", "114", "48C", "29C", "19B", "21G", "48CCT"]


def run_corridor_diagnostics():
    start_time = time.perf_counter()
    print("=" * 95)
    print("STEP 4: STOP-BY-STOP CUMULATIVE LOAD ACCUMULATOR & BUS BUNCHING DIAGNOSTICS")
    print("=" * 95)

    raw_csv = Path("Data/ETM/Ticket-Data-2026-03-12.csv")
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    bottlenecks_out = data_dir / "corridor_bottlenecks.json"
    bunching_out = data_dir / "bunching_incidents.json"

    con = duckdb.connect()

    print(f"\n[1/4] Ingesting ticket events from {raw_csv.as_posix()} via DuckDB...")
    target_list_sql = ", ".join(f"'{r}'" for r in TARGET_ROUTES)

    # Create an in-memory filtered table of ticket events with clean route/direction and stage NaN imputation
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE raw_tickets AS
        SELECT
            id,
            depot_code,
            depot,
            fleet_no,
            schedule_no,
            trip_no,
            trim(replace(replace(direction, ' - UP', ''), ' - DOWN', '')) AS route_num,
            CASE WHEN direction LIKE '% - UP' THEN 'UP' ELSE 'DOWN' END AS direction_clean,
            route_service_type,
            CAST(time AS TIME) AS ticket_time,
            EXTRACT(HOUR FROM CAST(time AS TIME))::INTEGER AS hour,
            trim(source) AS source_stop,
            trim(destination) AS dest_stop,
            trim(Trip_Origin) AS trip_origin,
            trim(Trip_Destination) AS trip_destination,
            CAST(from_stage AS INTEGER) AS from_stage,
            -- Impute NULL/NaN in to_stage with from_stage + 4 per constraint
            COALESCE(CAST(ROUND(to_stage) AS INTEGER), CAST(from_stage AS INTEGER) + 4) AS to_stage_imputed,
            CAST(passenger_count AS INTEGER) AS passenger_count
        FROM read_csv('{raw_csv.as_posix()}', quote='"', header=true, dateformat='%Y-%m-%d')
        WHERE from_stage IS NOT NULL
          AND passenger_count > 0;
    """)

    total_tickets = con.execute("SELECT COUNT(*) FROM raw_tickets").fetchone()[0]
    target_tickets = con.execute(
        f"SELECT COUNT(*) FROM raw_tickets WHERE route_num IN ({target_list_sql})"
    ).fetchone()[0]
    print(
        f"  -> Processed {total_tickets:,} valid ticket rows ({target_tickets:,} rows on target corridors: {', '.join(TARGET_ROUTES)})"
    )

    # -------------------------------------------------------------------------
    # FEATURE 1: Stop-by-Stop Cumulative Load Accumulator
    # -------------------------------------------------------------------------
    print("[2/4] Computing Stop-by-Stop Cumulative Passenger Occupancy & Crush-Load Bottlenecks...")

    # Map stage numbers to canonical stop names per (route_num, direction_clean)
    stage_names_df = con.execute(f"""
        WITH stage_stops AS (
            SELECT route_num, direction_clean, from_stage AS stage_no, source_stop AS stop_name, SUM(passenger_count) AS w
            FROM raw_tickets
            WHERE route_num IN ({target_list_sql})
              AND source_stop IS NOT NULL
              AND source_stop != ''
              AND LOWER(source_stop) NOT LIKE '%ticket%'
            GROUP BY route_num, direction_clean, from_stage, source_stop
            UNION ALL
            SELECT route_num, direction_clean, to_stage_imputed AS stage_no, dest_stop AS stop_name, SUM(passenger_count) AS w
            FROM raw_tickets
            WHERE route_num IN ({target_list_sql})
              AND dest_stop IS NOT NULL
              AND dest_stop != ''
              AND LOWER(dest_stop) NOT LIKE '%ticket%'
            GROUP BY route_num, direction_clean, to_stage_imputed, dest_stop
        ),
        ranked AS (
            SELECT
                route_num,
                direction_clean,
                stage_no,
                stop_name,
                SUM(w) AS total_w,
                ROW_NUMBER() OVER (PARTITION BY route_num, direction_clean, stage_no ORDER BY SUM(w) DESC) AS rn
            FROM stage_stops
            GROUP BY route_num, direction_clean, stage_no, stop_name
        )
        SELECT route_num, direction_clean, stage_no, stop_name
        FROM ranked
        WHERE rn = 1;
    """).df()

    stage_name_map = {
        (row["route_num"], row["direction_clean"], int(row["stage_no"])): str(
            row["stop_name"]
        )
        for _, row in stage_names_df.iterrows()
    }

    # Aggregate boardings and alightings per individual bus trip (route, direction, depot, fleet_no, trip_no)
    # Normalizing stage order along direction of travel:
    # For each ticket, passengers are on board from stage_start = LEAST(from_stage, to_stage)
    # and alight at stage_end = GREATEST(from_stage, to_stage) (if equal, stage_start + 4).
    trip_stages_df = con.execute(f"""
        WITH normalized_tickets AS (
            SELECT
                route_num,
                direction_clean,
                depot_code,
                depot,
                fleet_no,
                trip_no,
                MIN(ticket_time) OVER (PARTITION BY route_num, direction_clean, fleet_no, trip_no) AS trip_start_time,
                LEAST(from_stage, to_stage_imputed) AS board_stage,
                CASE
                    WHEN GREATEST(from_stage, to_stage_imputed) = LEAST(from_stage, to_stage_imputed)
                    THEN LEAST(from_stage, to_stage_imputed) + 4
                    ELSE GREATEST(from_stage, to_stage_imputed)
                END AS alight_stage,
                passenger_count
            FROM raw_tickets
            WHERE route_num IN ({target_list_sql})
        ),
        events AS (
            SELECT
                route_num, direction_clean, depot_code, depot, fleet_no, trip_no, trip_start_time,
                board_stage AS stage_no,
                SUM(passenger_count) AS boardings,
                0 AS alightings
            FROM normalized_tickets
            GROUP BY route_num, direction_clean, depot_code, depot, fleet_no, trip_no, trip_start_time, board_stage
            UNION ALL
            SELECT
                route_num, direction_clean, depot_code, depot, fleet_no, trip_no, trip_start_time,
                alight_stage AS stage_no,
                0 AS boardings,
                SUM(passenger_count) AS alightings
            FROM normalized_tickets
            GROUP BY route_num, direction_clean, depot_code, depot, fleet_no, trip_no, trip_start_time, alight_stage
        )
        SELECT
            route_num,
            direction_clean,
            depot_code,
            depot,
            fleet_no,
            trip_no,
            CAST(trip_start_time AS VARCHAR) AS trip_start_time,
            stage_no,
            SUM(boardings)::INTEGER AS boardings,
            SUM(alightings)::INTEGER AS alightings
        FROM events
        GROUP BY route_num, direction_clean, depot_code, depot, fleet_no, trip_no, trip_start_time, stage_no
        ORDER BY route_num, direction_clean, fleet_no, trip_no, stage_no;
    """).df()

    # Compute running occupancy[stage] = previous_occupancy + boardings[stage] - alightings[stage]
    trip_bottlenecks: List[Dict[str, Any]] = []
    for (route_num, direction, depot_code, depot, fleet_no, trip_no, start_t), grp in trip_stages_df.groupby(
        ["route_num", "direction_clean", "depot_code", "depot", "fleet_no", "trip_no", "trip_start_time"],
        sort=False,
    ):
        grp_sorted = grp.sort_values("stage_no")
        stages = grp_sorted["stage_no"].to_numpy()
        boards = grp_sorted["boardings"].to_numpy()
        alights = grp_sorted["alightings"].to_numpy()

        total_trip_pax = int(boards.sum())
        # Filter out corrupted or multi-shift merged anomalies so single-bus physics remain realistic
        if total_trip_pax < 40 or len(stages) < 2:
            continue

        occ = 0
        peak_occ = -1
        peak_idx = 0
        stage_profile = []

        for i in range(len(stages)):
            s_no = int(stages[i])
            b_cnt = int(boards[i])
            a_cnt = int(alights[i])
            occ = max(0, occ + b_cnt - a_cnt)
            next_s_no = int(stages[i + 1]) if i + 1 < len(stages) else s_no + 1
            from_name = stage_name_map.get((route_num, direction, s_no), f"Stage {s_no}")
            to_name = stage_name_map.get((route_num, direction, next_s_no), f"Stage {next_s_no}")

            status = "STAGE_CRUSH_LOAD" if occ > BUS_CAPACITY else "NORMAL"
            stage_profile.append(
                {
                    "stage_no": s_no,
                    "next_stage_no": next_s_no,
                    "from_stop": from_name,
                    "to_stop": to_name,
                    "boardings": b_cnt,
                    "alightings": a_cnt,
                    "in_vehicle_occupancy": occ,
                    "load_factor_pct": round((occ / float(BUS_CAPACITY)) * 100.0, 1),
                    "status": status,
                }
            )
            if occ > peak_occ and i < len(stages) - 1 and from_name != to_name:
                peak_occ = occ
                peak_idx = i

        # Keep realistic single-bus crush loads (66 to 135 peak on-board passengers)
        if 66 <= peak_occ <= 135:
            pk = stage_profile[peak_idx]
            trip_bottlenecks.append(
                {
                    "route_num": route_num,
                    "direction": direction,
                    "depot": f"{depot} ({depot_code})",
                    "fleet_no": fleet_no,
                    "trip_no": int(trip_no),
                    "trip_start_time": start_t,
                    "bottleneck_from_stage": pk["stage_no"],
                    "bottleneck_to_stage": pk["next_stage_no"],
                    "bottleneck_segment": f"Stage {pk['stage_no']} ({pk['from_stop']}) -> Stage {pk['next_stage_no']} ({pk['to_stop']})",
                    "peak_onboard_passengers": int(peak_occ),
                    "crush_load_factor_pct": round((peak_occ / float(BUS_CAPACITY)) * 100.0, 1),
                    "total_trip_boardings": total_trip_pax,
                    "status": "STAGE_CRUSH_LOAD" if peak_occ > BUS_CAPACITY else "NORMAL",
                    "crush_stage_count": sum(
                        1 for s in stage_profile if s["status"] == "STAGE_CRUSH_LOAD"
                    ),
                    "stage_by_stage_profile": stage_profile,
                }
            )

    # Sort by peak_onboard_passengers descending and pick distinct routes for Top 5 showcase
    bottlenecks_df = pd.DataFrame(trip_bottlenecks).sort_values(
        "peak_onboard_passengers", ascending=False
    )
    top_distinct_bottlenecks = (
        bottlenecks_df.drop_duplicates(subset=["route_num", "direction"])
        .drop_duplicates(subset=["route_num"])
        .head(5)
    )
    if len(top_distinct_bottlenecks) < 5:
        top_distinct_bottlenecks = bottlenecks_df.drop_duplicates(
            subset=["route_num", "direction"]
        ).head(5)

    top_5_records = top_distinct_bottlenecks.to_dict(orient="records")

    # -------------------------------------------------------------------------
    # FEATURE 2: Traffic-Induced Bus Bunching & Load Imbalance
    # -------------------------------------------------------------------------
    print("[3/4] Detecting Traffic-Induced Bus Bunching (Delta_t < 6 min & Lead Load > 2.0x Trailing)...")

    bunching_df = con.execute(f"""
        WITH bus_runs AS (
            SELECT
                route_num,
                direction_clean,
                depot_code,
                depot,
                fleet_no,
                trip_no,
                MIN(ticket_time) AS first_ticket_time,
                MAX(ticket_time) AS last_ticket_time,
                SUM(passenger_count)::INTEGER AS total_pax,
                ANY_VALUE(trip_origin) AS origin_terminal,
                ANY_VALUE(trip_destination) AS dest_terminal
            FROM raw_tickets
            WHERE route_num IN ({target_list_sql})
            GROUP BY route_num, direction_clean, depot_code, depot, fleet_no, trip_no
            HAVING SUM(passenger_count) BETWEEN 15 AND 185
               AND COUNT(*) >= 10
        ),
        ordered_runs AS (
            SELECT
                route_num,
                direction_clean,
                depot_code,
                depot,
                origin_terminal,
                dest_terminal,
                LAG(fleet_no) OVER w AS lead_fleet_no,
                LAG(trip_no) OVER w AS lead_trip_no,
                LAG(first_ticket_time) OVER w AS lead_start_time,
                LAG(total_pax) OVER w AS lead_pax,
                fleet_no AS trailing_fleet_no,
                trip_no AS trailing_trip_no,
                first_ticket_time AS trailing_start_time,
                total_pax AS trailing_pax,
                ROUND(date_diff('second', LAG(first_ticket_time) OVER w, first_ticket_time) / 60.0, 2) AS gap_minutes
            FROM bus_runs
            WINDOW w AS (PARTITION BY route_num, direction_clean ORDER BY first_ticket_time, fleet_no)
        )
        SELECT
            route_num,
            direction_clean AS direction,
            depot || ' (' || depot_code || ')' AS depot,
            COALESCE(origin_terminal, 'Origin Terminal') AS origin_terminal,
            COALESCE(dest_terminal, 'Destination Terminal') AS dest_terminal,
            lead_fleet_no,
            lead_trip_no,
            CAST(lead_start_time AS VARCHAR) AS lead_start_time,
            lead_pax,
            trailing_fleet_no,
            trailing_trip_no,
            CAST(trailing_start_time AS VARCHAR) AS trailing_start_time,
            trailing_pax,
            gap_minutes,
            ROUND(lead_pax * 1.0 / NULLIF(trailing_pax, 0), 2) AS load_imbalance_ratio
        FROM ordered_runs
        WHERE lead_fleet_no IS NOT NULL
          AND lead_fleet_no != trailing_fleet_no
          AND gap_minutes > 0.2
          AND gap_minutes < 6.0
          AND lead_pax > 2.0 * trailing_pax
          AND lead_pax >= 65
        ORDER BY (lead_pax - trailing_pax) DESC, gap_minutes ASC;
    """).df()

    # Select representative bunching incidents across distinct routes first
    distinct_bunching = (
        bunching_df.drop_duplicates(subset=["route_num", "direction"]).head(8)
    )
    bunching_records: List[Dict[str, Any]] = []
    for idx, row in distinct_bunching.reset_index(drop=True).iterrows():
        route_num = str(row["route_num"])
        direction = str(row["direction"])
        origin_term = str(row["origin_terminal"])
        dest_term = str(row["dest_terminal"])
        lead_bus = str(row["lead_fleet_no"])
        trail_bus = str(row["trailing_fleet_no"])
        gap_min = float(row["gap_minutes"])
        lead_pax = int(row["lead_pax"])
        trail_pax = int(row["trailing_pax"])
        ratio = float(row["load_imbalance_ratio"])

        bunching_records.append(
            {
                "incident_id": f"BNC-2026-{idx + 1:03d}",
                "flag": "BUS_BUNCHING",
                "route_num": route_num,
                "direction": direction,
                "depot": str(row["depot"]),
                "corridor": f"{origin_term} -> {dest_term}",
                "lead_bus": {
                    "fleet_no": lead_bus,
                    "trip_no": int(row["lead_trip_no"]),
                    "first_ticket_time": str(row["lead_start_time"]),
                    "passenger_count": lead_pax,
                },
                "trailing_bus": {
                    "fleet_no": trail_bus,
                    "trip_no": int(row["trailing_trip_no"]),
                    "first_ticket_time": str(row["trailing_start_time"]),
                    "passenger_count": trail_pax,
                },
                "observed_gap_minutes": gap_min,
                "load_imbalance_ratio": ratio,
                "passenger_load_gap": lead_pax - trail_pax,
                "operational_response": {
                    "lead_bus_action": (
                        f"Lead Bus ({lead_bus}): Run limited-stop / skip minor stops along Route {route_num} ({direction}) to recover schedule."
                    ),
                    "trailing_bus_action": (
                        f"Trailing Bus ({trail_bus}): Hold 3 minutes at {origin_term} major terminal to restore uniform 15-minute headway."
                    ),
                },
            }
        )

    # -------------------------------------------------------------------------
    # EXPORT & PRINT CONSOLE DIAGNOSTICS
    # -------------------------------------------------------------------------
    print("[4/4] Exporting JSON Diagnostics to `data/`...")

    bottlenecks_payload = {
        "generated_at": "2026-11-05T23:45:00+05:30",
        "source_file": raw_csv.as_posix(),
        "crush_load_threshold_pax": BUS_CAPACITY,
        "to_stage_nan_imputation": "COALESCE(to_stage, from_stage + 4)",
        "total_crush_load_trips_detected": len(bottlenecks_df),
        "top_5_corridor_bottlenecks": top_5_records,
    }
    bottlenecks_out.write_text(
        json.dumps(bottlenecks_payload, indent=2), encoding="utf-8"
    )

    bunching_payload = {
        "generated_at": "2026-11-05T23:45:00+05:30",
        "source_file": raw_csv.as_posix(),
        "criteria": {
            "max_gap_minutes": 6.0,
            "min_lead_to_trailing_load_ratio": 2.0,
            "flag": "BUS_BUNCHING",
        },
        "total_bunching_pairs_detected": len(bunching_df),
        "incidents": bunching_records,
    }
    bunching_out.write_text(
        json.dumps(bunching_payload, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 95)
    print("TOP 5 WORST STOP-TO-STOP BOTTLENECK SEGMENTS IN CHENNAI (STAGE_CRUSH_LOAD > 65 PAX)")
    print("-" * 95)
    top5_display = pd.DataFrame(
        [
            {
                "Rank": i + 1,
                "Route": f"{r['route_num']} ({r['direction']})",
                "Bus_Run": f"{r['fleet_no']} (Trip {r['trip_no']} @ {r['trip_start_time']})",
                "Peak_Bottleneck_Segment": r["bottleneck_segment"],
                "Peak_OnBoard_Pax": r["peak_onboard_passengers"],
                "Load_Factor_%": f"{r['crush_load_factor_pct']:.1f}%",
                "Status": r["status"],
            }
            for i, r in enumerate(top_5_records)
        ]
    )
    print(top5_display.to_string(index=False))
    print("=" * 95)

    print("\n" + "=" * 95)
    print("DETECTED TRAFFIC-INDUCED BUS BUNCHING INCIDENTS (GAP < 6 MIN & LEAD > 2.0x TRAILING)")
    print("-" * 95)
    bunch_display = pd.DataFrame(
        [
            {
                "Incident": r["incident_id"],
                "Route": f"{r['route_num']} ({r['direction']})",
                "Lead_Bus(Time/Pax)": f"{r['lead_bus']['fleet_no']} ({r['lead_bus']['first_ticket_time']} | {r['lead_bus']['passenger_count']} pax)",
                "Trailing_Bus(Time/Pax)": f"{r['trailing_bus']['fleet_no']} ({r['trailing_bus']['first_ticket_time']} | {r['trailing_bus']['passenger_count']} pax)",
                "Gap_Min": f"{r['observed_gap_minutes']:.2f}m",
                "Load_Ratio": f"{r['load_imbalance_ratio']:.2f}x",
                "Flag": r["flag"],
            }
            for r in bunching_records[:6]
        ]
    )
    print(bunch_display.to_string(index=False))
    print("-" * 95)
    for r in bunching_records[:3]:
        print(f"  [{r['incident_id']}] Route {r['route_num']} ({r['direction']}) | Gap: {r['observed_gap_minutes']:.2f} min | Ratio: {r['load_imbalance_ratio']:.2f}x")
        print(f"    -> {r['operational_response']['lead_bus_action']}")
        print(f"    -> {r['operational_response']['trailing_bus_action']}")
    print("=" * 95)

    elapsed = time.perf_counter() - start_time
    print(
        f"\nSaved {bottlenecks_out.as_posix()} and {bunching_out.as_posix()} in {elapsed:.2f} seconds.\n"
    )
    con.close()


if __name__ == "__main__":
    run_corridor_diagnostics()
