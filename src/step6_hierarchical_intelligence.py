import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd

BUS_CAPACITY = 65
MINIBUS_CAPACITY = 35

# Canonical Chennai arterial trunk corridors & keyword matchers for stop/origin/destination grouping
CORRIDOR_DEFINITIONS = [
    {
        "corridor_id": "COR-OMR-MARINA",
        "corridor_name": "OMR / Kamarajar Salai (IT Expressway - Marina - Broadway)",
        "explicit_routes": {"102", "102K", "102P", "102X", "102C", "102M", "109", "19B", "570", "570S"},
        "stop_keywords": [
            "FORESHORE ESTATE",
            "QUEEN MARRY",
            "CHEPAUK",
            "THORAIPAKKAM",
            "SHOLINGANALLUR",
            "KANNAGI NAGAR",
            "KELAMBAKKAM",
            "SIRUSERI",
            "TIDEL PARK",
            "KANDANCHAVADI",
            "PERUMBAKKAM",
        ],
    },
    {
        "corridor_id": "COR-KONNUR-AYANAVARAM",
        "corridor_name": "Konnur High Rd & New Avadi Rd (Vallalar Nagar - Ayanavaram - Koyambedu)",
        "explicit_routes": {"48C", "48CCT", "48A", "48", "48B", "7H", "37D"},
        "stop_keywords": [
            "AYANAVARAM",
            "SAYANI",
            "DASAMAHAN",
            "PULIYANTHOPE",
            "KAMBAR ARANGAM",
            "ICF",
            "VILLIVAKKAM",
            "VALLALAR NAGAR",
            "NATHAMUNI",
        ],
    },
    {
        "corridor_id": "COR-JN-SALAI-100FT",
        "corridor_name": "Jawaharlal Nehru Salai / 100ft Rd (Redhills - Thirumangalam - Koyambedu - Guindy)",
        "explicit_routes": {"114", "114P", "114A", "7M", "70", "70V", "70C", "170", "170C", "47A"},
        "stop_keywords": [
            "THIRUMANGALAM",
            "KOLATHUR",
            "NOLAMBUR",
            "J.J.NAGAR",
            "GOLDEN FLATS",
            "ANNA NAGAR",
            "KOYAMBEDU",
            "VADAPALANI",
            "ASHOK PILLAR",
            "RETTERI",
        ],
    },
    {
        "corridor_id": "COR-PERAMBUR-EGMORE",
        "corridor_name": "Perambur High Rd - Egmore - Mylapore Trunk",
        "explicit_routes": {"29C", "29A", "29E", "29N", "29B"},
        "stop_keywords": [
            "PERAMBUR",
            "DASAPRAKASH",
            "K.M.C",
            "EGMORE",
            "CHETPET",
            "STERLING ROAD",
            "JAMALIA",
            "OTTERI",
        ],
    },
    {
        "corridor_id": "COR-ANNA-SALAI",
        "corridor_name": "Anna Salai / Mount Road (Broadway - Thousand Lights - Saidapet - Guindy)",
        "explicit_routes": {"21G", "18A", "18K", "45B", "23C", "54", "51", "2B"},
        "stop_keywords": [
            "ANNA SALAI",
            "LIC",
            "THOUSAND LIGHTS",
            "TEYNAMPET",
            "NANDANAM",
            "SAIDAPET",
            "GUINDY",
            "LITTLE MOUNT",
            "D.M.S",
        ],
    },
    {
        "corridor_id": "COR-POONAMALLEE-HIGH",
        "corridor_name": "Poonamallee High Rd (Central - KMC - Aminjikarai - Maduravoyal)",
        "explicit_routes": {"15B", "15G", "27B", "53", "53E", "101", "50"},
        "stop_keywords": [
            "AMINJIKARAI",
            "ARUMBAKKAM",
            "NSK NAGAR",
            "MADURAVOYAL",
            "POONAMALLEE",
            "VELAPPANCHAVADI",
            "ROHINI THEATRE",
        ],
    },
    {
        "corridor_id": "COR-NORTH-HARBOUR",
        "corridor_name": "Ennore High Rd / Tiruvottiyur North Industrial Corridor",
        "explicit_routes": {"56A", "56N", "56P", "56C", "159", "159A", "4", "1B"},
        "stop_keywords": [
            "ENNORE",
            "TIRUVOTTIYUR",
            "TOLLGATE",
            "ROYAPURAM",
            "KASIMEDU",
            "KALMANDAPAM",
            "HIGH COURT",
            "PARRYS",
        ],
    },
]


def classify_risk(pressure_pct: float) -> str:
    if pressure_pct > 120.0:
        return "CRITICAL"
    elif pressure_pct >= 100.0:
        return "HIGH"
    return "NORMAL"


def format_hour_window(hour: int) -> str:
    h_start = int(hour) % 24
    h_end = (h_start + 1) % 24

    def fmt(h: int) -> str:
        suf = "AM" if h < 12 else "PM"
        h12 = h % 12 or 12
        return f"{h12:02d}:00 {suf}"

    return f"{fmt(h_start)} - {fmt(h_end)}"


def assign_corridor(
    route_num: str,
    origin_stop: str,
    dest_stop: str,
    stop_names: List[str],
) -> Tuple[str, str]:
    r_clean = str(route_num).strip().upper()
    for cdef in CORRIDOR_DEFINITIONS:
        if r_clean in cdef["explicit_routes"]:
            return cdef["corridor_id"], cdef["corridor_name"]

    combined_text = " | ".join([origin_stop, dest_stop] + stop_names).upper()
    best_id = "COR-URBAN-FEEDER"
    best_name = "Chennai Urban Arterial & Metro Feeder Network"
    best_hits = 0

    for cdef in CORRIDOR_DEFINITIONS:
        hits = sum(1 for kw in cdef["stop_keywords"] if kw in combined_text)
        if hits > best_hits:
            best_hits = hits
            best_id = cdef["corridor_id"]
            best_name = cdef["corridor_name"]

    return best_id, best_name


def run_step6_hierarchical_intelligence():
    t0 = time.perf_counter()
    print("=" * 105)
    print("STEP 6: 3-TIER HIERARCHICAL TRANSIT INTELLIGENCE & OVERLAPPING CORRIDOR CONSOLIDATOR")
    print("=" * 105)

    pred_parquet = Path("data/predictions_eval.parquet")
    etm_glob = "Data/ETM/Ticket-Data-2026-03-1[0-2].csv"
    out_json = Path("data/hierarchical_transit_intelligence.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()

    # -------------------------------------------------------------------------
    # STEP 1: Load & Aggregate Tier 1 & Tier 2 from `data/predictions_eval.parquet`
    # -------------------------------------------------------------------------
    print(f"\n[1/5] Querying Tier 1 & Tier 2 24-hour demand curves from `{pred_parquet.as_posix()}` via DuckDB...")
    hourly_pred_df = con.execute(f"""
        WITH route_hourly AS (
            SELECT
                depot_code,
                route_num,
                CAST(hour AS INTEGER) AS hour,
                MAX(passenger_demand) AS actual_passengers,
                MAX(predicted_demand) AS predicted_passengers,
                MAX(active_trips) AS active_trips
            FROM read_parquet('{pred_parquet.as_posix()}')
            WHERE depot_code IS NOT NULL AND route_num IS NOT NULL
            GROUP BY depot_code, route_num, hour
        )
        SELECT
            depot_code,
            route_num,
            hour,
            ROUND(actual_passengers, 1) AS actual_passengers,
            ROUND(predicted_passengers, 1) AS predicted_passengers,
            CAST(GREATEST(active_trips, 1) AS INTEGER) AS active_trips,
            CAST(GREATEST(active_trips, 1) * {BUS_CAPACITY} AS INTEGER) AS capacity,
            ROUND(
                (predicted_passengers / (GREATEST(active_trips, 1) * {BUS_CAPACITY})) * 100.0,
                1
            ) AS demand_pressure_pct
        FROM route_hourly
        ORDER BY depot_code, route_num, hour;
    """).df()

    print(
        f"  -> Aggregated {len(hourly_pred_df):,} (depot, route, hour) records across "
        f"{hourly_pred_df['depot_code'].nunique()} depots and {hourly_pred_df['route_num'].nunique()} routes."
    )

    # -------------------------------------------------------------------------
    # STEP 2: Ingest ETM Tickets (2026-03-10 to 2026-03-12) for Route Metadata & Stop Segments
    # -------------------------------------------------------------------------
    print(f"[2/5] Ingesting ETM ticket events from `{etm_glob}` via DuckDB...")
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE etm_raw AS
        SELECT
            duty_date,
            trim(depot_code) AS depot_code,
            trim(depot) AS depot_name,
            fleet_no,
            trip_no,
            trim(replace(replace(direction, ' - UP', ''), ' - DOWN', '')) AS route_num,
            CASE WHEN direction LIKE '% - UP' THEN 'UP' ELSE 'DOWN' END AS direction,
            EXTRACT(HOUR FROM CAST(time AS TIME))::INTEGER AS hour,
            trim(source) AS source_stop,
            trim(destination) AS dest_stop,
            trim(Trip_Origin) AS trip_origin,
            trim(Trip_Destination) AS trip_destination,
            TRY_CAST(KM AS DOUBLE) AS km_val,
            CAST(from_stage AS INTEGER) AS from_stage,
            COALESCE(CAST(ROUND(to_stage) AS INTEGER), CAST(from_stage AS INTEGER) + 4) AS to_stage_clean,
            CAST(passenger_count AS INTEGER) AS passenger_count
        FROM read_csv('{etm_glob}', quote='"', header=true, dateformat='%Y-%m-%d')
        WHERE from_stage IS NOT NULL
          AND passenger_count > 0;
    """)

    etm_count = con.execute("SELECT COUNT(*) FROM etm_raw").fetchone()[0]
    print(f"  -> Streamed {etm_count:,} valid ETM ticket records in DuckDB.")

    # Depot metadata & active fleet counts
    depot_meta_df = con.execute("""
        SELECT
            depot_code,
            ANY_VALUE(depot_name) AS depot_name,
            COUNT(DISTINCT fleet_no) AS active_fleet
        FROM etm_raw
        WHERE depot_code IS NOT NULL AND depot_code != ''
        GROUP BY depot_code;
    """).df()
    depot_name_map = {
        str(r["depot_code"]): f"{r['depot_name']} ({r['depot_code']})"
        for _, r in depot_meta_df.iterrows()
    }
    depot_fleet_map = {
        str(r["depot_code"]): int(r["active_fleet"])
        for _, r in depot_meta_df.iterrows()
    }

    # Route metadata: KM, stage span, origin, destination, active fleet per (depot_code, route_num)
    route_meta_df = con.execute("""
        SELECT
            depot_code,
            route_num,
            ROUND(COALESCE(MAX(km_val), (MAX(to_stage_clean) - MIN(from_stage) + 1) * 1.8), 1) AS route_km,
            CAST(GREATEST(MAX(to_stage_clean) - MIN(from_stage) + 1, 2) AS INTEGER) AS stage_span,
            CAST(MIN(from_stage) AS INTEGER) AS min_stage,
            CAST(MAX(to_stage_clean) AS INTEGER) AS max_stage,
            COALESCE(
                MODE(trip_origin) FILTER (WHERE direction = 'UP' AND trip_origin IS NOT NULL AND trip_origin != ''),
                MODE(trip_origin) FILTER (WHERE trip_origin IS NOT NULL AND trip_origin != '')
            ) AS origin_stop,
            COALESCE(
                MODE(trip_destination) FILTER (WHERE direction = 'UP' AND trip_destination IS NOT NULL AND trip_destination != ''),
                MODE(trip_destination) FILTER (WHERE trip_destination IS NOT NULL AND trip_destination != '')
            ) AS destination_stop,
            COUNT(DISTINCT fleet_no) AS route_active_fleet
        FROM etm_raw
        GROUP BY depot_code, route_num;
    """).df()

    route_meta_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for _, r in route_meta_df.iterrows():
        key = (str(r["depot_code"]), str(r["route_num"]))
        route_meta_map[key] = {
            "route_km": float(r["route_km"]) if pd.notna(r["route_km"]) else 12.0,
            "stage_span": int(r["stage_span"]) if pd.notna(r["stage_span"]) else 8,
            "min_stage": int(r["min_stage"]) if pd.notna(r["min_stage"]) else 1,
            "max_stage": int(r["max_stage"]) if pd.notna(r["max_stage"]) else 10,
            "origin_stop": str(r["origin_stop"] or "TERMINUS A"),
            "destination_stop": str(r["destination_stop"] or "TERMINUS B"),
            "route_active_fleet": int(r["route_active_fleet"]) if pd.notna(r["route_active_fleet"]) else 1,
        }

    # -------------------------------------------------------------------------
    # STEP 3: Compute Tier 3 Stop-by-Stop Segments in DuckDB
    # -------------------------------------------------------------------------
    print("[3/5] Computing Tier 3 Stop-by-Stop Segment Loads across active operating hours...")

    # Canonical stop name per (route_num, direction, stage_no) excluding 'Zero Value Ticket'
    con.execute("""
        CREATE OR REPLACE TEMP TABLE canonical_stage_names AS
        WITH stage_stops AS (
            SELECT route_num, direction, from_stage AS stage_no, source_stop AS stop_name, SUM(passenger_count) AS w
            FROM etm_raw
            WHERE source_stop IS NOT NULL AND source_stop != '' AND LOWER(source_stop) NOT LIKE '%ticket%'
            GROUP BY route_num, direction, from_stage, source_stop
            UNION ALL
            SELECT route_num, direction, to_stage_clean AS stage_no, dest_stop AS stop_name, SUM(passenger_count) AS w
            FROM etm_raw
            WHERE dest_stop IS NOT NULL AND dest_stop != '' AND LOWER(dest_stop) NOT LIKE '%ticket%'
            GROUP BY route_num, direction, to_stage_clean, dest_stop
        ),
        ranked AS (
            SELECT
                route_num,
                direction,
                stage_no,
                stop_name,
                ROW_NUMBER() OVER (PARTITION BY route_num, direction, stage_no ORDER BY SUM(w) DESC) AS rn
            FROM stage_stops
            GROUP BY route_num, direction, stage_no, stop_name
        )
        SELECT route_num, direction, stage_no, stop_name
        FROM ranked
        WHERE rn = 1;
    """)

    # Select priority + top-ridership + overcrowded routes for full 24h stop-segment expansion
    # (keeps JSON crisp and under 15MB while covering all high-stress/corridor routes and top depot routes)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE target_segment_routes AS
        WITH ranked_routes AS (
            SELECT
                depot_code,
                route_num,
                SUM(passenger_count) AS total_pax,
                ROW_NUMBER() OVER (PARTITION BY depot_code ORDER BY SUM(passenger_count) DESC) AS depot_rn
            FROM etm_raw
            GROUP BY depot_code, route_num
        )
        SELECT DISTINCT depot_code, route_num
        FROM ranked_routes
        WHERE depot_rn <= 5
           OR total_pax >= 8000
           OR route_num IN (
               '102', '102K', '102P', '102X', '109', '114', '114P', '7M', '70', '70V',
               '48C', '48CCT', '48A', '29C', '29A', '21G', '19B', '47A', '23C', '56A',
               '570', '15B', '27B', '18A', '45B', '54'
           );
    """)

    # Vectorized stop-segment load calculation in DuckDB using pre-aggregated stage pairs + GENERATE_SERIES
    segments_df = con.execute(f"""
        WITH trip_stage_pairs AS (
            SELECT
                e.duty_date,
                e.depot_code,
                e.route_num,
                e.direction,
                e.hour,
                (e.fleet_no || '-' || CAST(e.trip_no AS VARCHAR)) AS trip_id,
                LEAST(e.from_stage, e.to_stage_clean) AS s_start,
                CASE
                    WHEN GREATEST(e.from_stage, e.to_stage_clean) = LEAST(e.from_stage, e.to_stage_clean)
                        THEN LEAST(e.from_stage, e.to_stage_clean) + 1
                    ELSE LEAST(GREATEST(e.from_stage, e.to_stage_clean), LEAST(e.from_stage, e.to_stage_clean) + 25)
                END AS s_end,
                SUM(e.passenger_count) AS pax
            FROM etm_raw e
            INNER JOIN target_segment_routes tr
                ON e.depot_code = tr.depot_code AND e.route_num = tr.route_num
            WHERE e.from_stage >= 1 AND e.from_stage <= 40
            GROUP BY
                e.duty_date, e.depot_code, e.route_num, e.direction, e.hour, trip_id, s_start, s_end
        ),
        expanded_hops AS (
            SELECT
                duty_date,
                depot_code,
                route_num,
                direction,
                hour,
                trip_id,
                UNNEST(GENERATE_SERIES(s_start, s_end - 1)) AS from_stage,
                pax
            FROM trip_stage_pairs
        ),
        trip_hop_loads AS (
            SELECT
                duty_date,
                depot_code,
                route_num,
                direction,
                hour,
                trip_id,
                from_stage,
                from_stage + 1 AS to_stage,
                SUM(pax) AS trip_onboard_pax
            FROM expanded_hops
            GROUP BY duty_date, depot_code, route_num, direction, hour, trip_id, from_stage
        ),
        hourly_segment_summary AS (
            SELECT
                depot_code,
                route_num,
                direction,
                hour,
                from_stage,
                to_stage,
                CAST(MAX(trip_onboard_pax) AS INTEGER) AS onboard_pax,
                ROUND(AVG(trip_onboard_pax), 1) AS avg_onboard_pax,
                COUNT(DISTINCT (CAST(duty_date AS VARCHAR) || '-' || trip_id)) AS sample_trips
            FROM trip_hop_loads
            GROUP BY depot_code, route_num, direction, hour, from_stage, to_stage
        )
        SELECT
            h.depot_code,
            h.route_num,
            h.direction,
            h.hour,
            h.from_stage,
            h.to_stage,
            s1.stop_name AS from_stage_name,
            s2.stop_name AS to_stage_name,
            h.onboard_pax,
            h.avg_onboard_pax,
            ROUND((h.onboard_pax / { float(BUS_CAPACITY) }) * 100.0, 1) AS load_factor_pct,
            (h.onboard_pax > {BUS_CAPACITY}) AS is_crush_load
        FROM hourly_segment_summary h
        INNER JOIN canonical_stage_names s1
            ON h.route_num = s1.route_num AND h.direction = s1.direction AND h.from_stage = s1.stage_no
        INNER JOIN canonical_stage_names s2
            ON h.route_num = s2.route_num AND h.direction = s2.direction AND h.to_stage = s2.stage_no
        WHERE s1.stop_name != s2.stop_name
        ORDER BY h.depot_code, h.route_num, h.hour, h.direction, h.from_stage;
    """).df()

    print(
        f"  -> Computed {len(segments_df):,} sequential stop-by-stop segment load records "
        f"({int(segments_df['is_crush_load'].sum()):,} crush-load segment-hours > {BUS_CAPACITY} pax)."
    )

    # Map route -> list of canonical stop names for corridor assignment
    route_stops_df = con.execute("""
        SELECT route_num, LIST(DISTINCT stop_name) AS stop_list
        FROM canonical_stage_names
        GROUP BY route_num;
    """).df()
    route_stops_map = {
        str(r["route_num"]): list(r["stop_list"]) for _, r in route_stops_df.iterrows()
    }

    # Index stop segments by (depot_code, route_num, hour)
    segments_by_key: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = {}
    for row in segments_df.itertuples(index=False):
        k = (str(row.depot_code), str(row.route_num), int(row.hour))
        if k not in segments_by_key:
            segments_by_key[k] = []
        segments_by_key[k].append(
            {
                "direction": str(row.direction),
                "from_stage": int(row.from_stage),
                "to_stage": int(row.to_stage),
                "from_stage_name": str(row.from_stage_name),
                "to_stage_name": str(row.to_stage_name),
                "segment_hop": f"{row.from_stage_name} -> {row.to_stage_name}",
                "onboard_pax": int(row.onboard_pax),
                "avg_onboard_pax": float(row.avg_onboard_pax),
                "load_factor_pct": float(row.load_factor_pct),
                "is_crush_load": bool(row.is_crush_load),
            }
        )

    # -------------------------------------------------------------------------
    # STEP 4: Build Tier 1 -> Tier 2 -> Tier 3 Hierarchy (`depot -> route -> hours (0-23) -> stop_segments`)
    # -------------------------------------------------------------------------
    print("[4/5] Building 3-Tier Hierarchical Tree (Depot -> Route -> 24-Hour Curve -> Stop Segments)...")

    depots_tree: Dict[str, Dict[str, Any]] = {}
    corridor_route_registry: Dict[str, List[Dict[str, Any]]] = {}

    for (depot_code, route_num), grp in hourly_pred_df.groupby(["depot_code", "route_num"]):
        depot_code_str = str(depot_code)
        route_num_str = str(route_num)
        depot_label = depot_name_map.get(depot_code_str, f"Depot {depot_code_str}")

        meta = route_meta_map.get(
            (depot_code_str, route_num_str),
            {
                "route_km": 14.0,
                "stage_span": 10,
                "origin_stop": "TERMINUS A",
                "destination_stop": "TERMINUS B",
                "route_active_fleet": 2,
            },
        )
        stop_list = route_stops_map.get(route_num_str, [])
        corridor_id, corridor_name = assign_corridor(
            route_num_str, meta["origin_stop"], meta["destination_stop"], stop_list
        )

        # Build existing hours lookup
        hour_rows = {int(r.hour): r for r in grp.itertuples(index=False)}

        hours_list: List[Dict[str, Any]] = []
        peak_hour = 8
        peak_pred_demand = 0.0
        max_pressure_pct = 0.0
        max_active_trips = 0

        for h in range(24):
            if h in hour_rows:
                hr = hour_rows[h]
                pred_pax = float(hr.predicted_passengers)
                act_pax = float(hr.actual_passengers)
                trips = int(hr.active_trips)
                cap = int(hr.capacity)
                press_pct = float(hr.demand_pressure_pct)
            else:
                pred_pax = 0.0
                act_pax = 0.0
                trips = 0
                cap = 0
                press_pct = 0.0

            if press_pct > max_pressure_pct or (
                 press_pct == max_pressure_pct and pred_pax > peak_pred_demand
            ):
                max_pressure_pct = press_pct
                peak_hour = h
            if pred_pax > peak_pred_demand:
                peak_pred_demand = pred_pax
            if trips > max_active_trips:
                max_active_trips = trips

            segs = segments_by_key.get((depot_code_str, route_num_str, h), [])
            crush_count = sum(1 for s in segs if s["is_crush_load"])

            hours_list.append(
                {
                    "hour": h,
                    "time_window": format_hour_window(h),
                    "predicted_passengers": round(pred_pax, 1),
                    "actual_passengers": round(act_pax, 1),
                    "active_trips": trips,
                    "capacity": cap,
                    "demand_pressure_pct": round(press_pct, 1),
                    "risk_category": classify_risk(press_pct),
                    "crush_segment_count": crush_count,
                    "stop_segments": segs,
                }
            )

        route_risk = classify_risk(max_pressure_pct)
        active_fleet_val = max(int(meta["route_active_fleet"]), max_active_trips)

        route_entry = {
            "route_num": route_num_str,
            "depot_code": depot_code_str,
            "depot_name": depot_label,
            "corridor_id": corridor_id,
            "corridor_name": corridor_name,
            "route_km": meta["route_km"],
            "stage_span": meta["stage_span"],
            "origin_stop": meta["origin_stop"],
            "destination_stop": meta["destination_stop"],
            "peak_hour": peak_hour,
            "peak_hour_window": format_hour_window(peak_hour),
            "peak_predicted_demand": round(peak_pred_demand, 1),
            "active_fleet": active_fleet_val,
            "max_demand_pressure_pct": round(max_pressure_pct, 1),
            "risk_category": route_risk,
            "hours": hours_list,
        }

        if depot_code_str not in depots_tree:
            depots_tree[depot_code_str] = {
                "depot_code": depot_code_str,
                "depot_name": depot_label,
                "active_fleet": depot_fleet_map.get(depot_code_str, active_fleet_val),
                "total_routes": 0,
                "peak_hour": peak_hour,
                "peak_hour_window": format_hour_window(peak_hour),
                "peak_predicted_demand": 0.0,
                "max_demand_pressure_pct": 0.0,
                "risk_category": "NORMAL",
                "critical_routes_count": 0,
                "high_risk_routes_count": 0,
                "routes": {},
            }

        d_obj = depots_tree[depot_code_str]
        d_obj["routes"][route_num_str] = route_entry
        d_obj["total_routes"] += 1
        if route_risk == "CRITICAL":
            d_obj["critical_routes_count"] += 1
        elif route_risk == "HIGH":
            d_obj["high_risk_routes_count"] += 1

        if round(max_pressure_pct, 1) > d_obj["max_demand_pressure_pct"]:
            d_obj["max_demand_pressure_pct"] = round(max_pressure_pct, 1)
            d_obj["peak_hour"] = peak_hour
            d_obj["peak_hour_window"] = format_hour_window(peak_hour)
        if round(peak_pred_demand, 1) > d_obj["peak_predicted_demand"]:
            d_obj["peak_predicted_demand"] = round(peak_pred_demand, 1)
        d_obj["risk_category"] = classify_risk(d_obj["max_demand_pressure_pct"])

        if corridor_id != "COR-URBAN-FEEDER":
            corridor_route_registry.setdefault(corridor_id, []).append(route_entry)

    # -------------------------------------------------------------------------
    # STEP 5: Tier 4 — Overlapping Corridor Consolidator & Smart Depot Selection
    # -------------------------------------------------------------------------
    print("[5/5] Running Tier 4 Overlapping Corridor Consolidator & Smart Depot Selection...")

    consolidator_recommendations: List[Dict[str, Any]] = []
    seen_pairs = set()

    for corridor_id, c_routes in corridor_route_registry.items():
        corridor_name = c_routes[0]["corridor_name"]
        # Check every operating hour (0-23) where >= 2 distinct routes on this corridor have crush segments
        for h in range(24):
            stressed_in_hour: List[Dict[str, Any]] = []
            for r_entry in c_routes:
                h_obj = r_entry["hours"][h]
                crush_segs = [s for s in h_obj["stop_segments"] if s["is_crush_load"]]
                if crush_segs or h_obj["demand_pressure_pct"] > 110.0:
                    max_seg_pax = max((s["onboard_pax"] for s in crush_segs), default=int(h_obj["predicted_passengers"] / max(h_obj["active_trips"], 1)))
                    if max_seg_pax > BUS_CAPACITY:
                        stressed_in_hour.append(
                            {
                                "route_num": r_entry["route_num"],
                                "depot_code": r_entry["depot_code"],
                                "depot_name": r_entry["depot_name"],
                                "route_km": r_entry["route_km"],
                                "stage_span": r_entry["stage_span"],
                                "origin_stop": r_entry["origin_stop"],
                                "destination_stop": r_entry["destination_stop"],
                                "hour": h,
                                "time_window": format_hour_window(h),
                                "demand_pressure_pct": h_obj["demand_pressure_pct"],
                                "peak_segment_pax": max_seg_pax,
                                "crush_segments": crush_segs,
                            }
                        )

            # Deduplicate by route_num (keep highest peak_segment_pax per route_num)
            by_route: Dict[str, Dict[str, Any]] = {}
            for item in stressed_in_hour:
                rn = item["route_num"]
                if rn not in by_route or item["peak_segment_pax"] > by_route[rn]["peak_segment_pax"]:
                    by_route[rn] = item

            unique_stressed = list(by_route.values())
            if len(unique_stressed) >= 2:
                # Sort by (route_km DESC, stage_span DESC, peak_segment_pax DESC)
                unique_stressed.sort(
                    key=lambda x: (x["route_km"], x["stage_span"], x["peak_segment_pax"]),
                    reverse=True,
                )
                # Find best encompassing pair (long_route, short_route) that shares physical corridor stops
                best_pair: Optional[Tuple[Dict[str, Any], Dict[str, Any], List[str]]] = None
                for i in range(len(unique_stressed)):
                    for j in range(i + 1, len(unique_stressed)):
                        lr_cand = unique_stressed[i]
                        sr_cand = unique_stressed[j]
                        if lr_cand["route_km"] <= sr_cand["route_km"] and lr_cand["stage_span"] <= sr_cand["stage_span"]:
                            continue
                        lr_stops = set(route_stops_map.get(lr_cand["route_num"], []))
                        sr_stops = set(route_stops_map.get(sr_cand["route_num"], []))
                        shared_stops = sorted(lr_stops.intersection(sr_stops))
                        if len(shared_stops) >= 2:
                            best_pair = (lr_cand, sr_cand, shared_stops)
                            break
                    if best_pair is not None:
                        break

                if best_pair is not None:
                    long_route, short_route, shared_stops_list = best_pair
                    shared_set = set(shared_stops_list)
                    pair_sig = (corridor_id, long_route["route_num"], short_route["route_num"], h)
                    if pair_sig in seen_pairs:
                        continue
                    seen_pairs.add(pair_sig)

                    shared_crush_hop = next(
                        (
                            s["segment_hop"]
                            for s in short_route["crush_segments"]
                            if s["from_stage_name"] in shared_set or s["to_stage_name"] in shared_set
                        ),
                        next(
                            (
                                s["segment_hop"]
                                for s in long_route["crush_segments"]
                                if s["from_stage_name"] in shared_set or s["to_stage_name"] in shared_set
                            ),
                            f"{shared_stops_list[0]} -> {shared_stops_list[-1]}",
                        ),
                    )
                    rec_text = (
                        f"Deploy 1 additional bus from {long_route['depot_name']} on Route {long_route['route_num']} "
                        f"to clear crowd for both Route {short_route['route_num']} and Route {long_route['route_num']} concurrently."
                    )
                    consolidator_recommendations.append(
                        {
                            "consolidation_type": "LONG_ROUTE_CORRIDOR_CONSOLIDATION",
                            "corridor_id": corridor_id,
                            "corridor_name": corridor_name,
                            "hour": h,
                            "time_window": format_hour_window(h),
                            "master_depot_code": long_route["depot_code"],
                            "master_depot_name": long_route["depot_name"],
                            "long_route": {
                                "route_num": long_route["route_num"],
                                "depot_name": long_route["depot_name"],
                                "route_km": long_route["route_km"],
                                "stage_span": long_route["stage_span"],
                                "origin_destination": f"{long_route['origin_stop']} <-> {long_route['destination_stop']}",
                                "peak_segment_pax": long_route["peak_segment_pax"],
                                "demand_pressure_pct": long_route["demand_pressure_pct"],
                            },
                            "short_route": {
                                "route_num": short_route["route_num"],
                                "depot_name": short_route["depot_name"],
                                "route_km": short_route["route_km"],
                                "stage_span": short_route["stage_span"],
                                "origin_destination": f"{short_route['origin_stop']} <-> {short_route['destination_stop']}",
                                "peak_segment_pax": short_route["peak_segment_pax"],
                                "demand_pressure_pct": short_route["demand_pressure_pct"],
                            },
                            "shared_stops_count": len(shared_stops_list),
                            "shared_bottleneck_segment": shared_crush_hop,
                            "combined_peak_pax": long_route["peak_segment_pax"] + short_route["peak_segment_pax"],
                            "recommendation": rec_text,
                        }
                    )
                else:
                    # Fallback: localized SHORT_TURN_SHUTTLE if no encompassing longer route shares the bottleneck stops
                    long_route = unique_stressed[0]
                    short_route = unique_stressed[1]
                    pair_sig = (corridor_id, long_route["route_num"], short_route["route_num"], h)
                    if pair_sig in seen_pairs:
                        continue
                    seen_pairs.add(pair_sig)
                    crush_segs = long_route["crush_segments"]
                    s_first = crush_segs[0]["from_stage_name"] if crush_segs else long_route["origin_stop"]
                    s_last = crush_segs[-1]["to_stage_name"] if crush_segs else long_route["destination_stop"]
                    consolidator_recommendations.append(
                        {
                            "consolidation_type": "SHORT_TURN_SHUTTLE",
                            "corridor_id": corridor_id,
                            "corridor_name": corridor_name,
                            "hour": h,
                            "time_window": format_hour_window(h),
                            "master_depot_code": long_route["depot_code"],
                            "master_depot_name": long_route["depot_name"],
                            "bottleneck_loop": f"{s_first} <-> {s_last}",
                            "combined_peak_pax": long_route["peak_segment_pax"] + short_route["peak_segment_pax"],
                            "recommendation": (
                                f"Deploy SHORT_TURN_SHUTTLE from {long_route['depot_name']} looping strictly between "
                                f"{s_first} and {s_last} during {format_hour_window(h)}."
                            ),
                        }
                    )

    # Sort consolidator recommendations: prioritize inter-depot / distinct-length trunk consolidations with highest combined crush load
    consolidator_recommendations.sort(
        key=lambda r: (
            1 if r["consolidation_type"] == "LONG_ROUTE_CORRIDOR_CONSOLIDATION" else 0,
            1
            if r.get("long_route", {}).get("depot_name")
            != r.get("short_route", {}).get("depot_name")
            else 0,
            r.get("combined_peak_pax", 0),
        ),
        reverse=True,
    )

    # Save unified hierarchical JSON
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S+05:30"),
        "schema_structure": "depots[depot_code] -> routes[route_num] -> hours[0..23] -> stop_segments",
        "network_summary": {
            "total_depots": len(depots_tree),
            "total_depot_route_pairs": sum(d["total_routes"] for d in depots_tree.values()),
            "total_stop_segments_profiled": len(segments_df),
            "total_crush_segment_hours": int(segments_df["is_crush_load"].sum()),
            "total_corridor_consolidations": len(consolidator_recommendations),
        },
        "corridor_consolidator_recommendations": consolidator_recommendations[:25],
        "depots": depots_tree,
    }

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    elapsed = time.perf_counter() - t0

    # -------------------------------------------------------------------------
    # CONSOLE OUTPUT:
    # 1. Sample 3-Tier Drill-Down for one route (24-hour peak + stop segments for that peak hour)
    # 2. One concrete example of the Corridor Consolidator recommending a longer route's depot over a shorter one
    # -------------------------------------------------------------------------
    sample_depot = "KA" if "KA" in depots_tree else next(iter(depots_tree.keys()))
    sample_route_id = (
        "102K"
        if "102K" in depots_tree[sample_depot]["routes"]
        else next(iter(depots_tree[sample_depot]["routes"].keys()))
    )
    sample_route = depots_tree[sample_depot]["routes"][sample_route_id]

    # Pick an hour that has both high demand pressure and populated stop segments
    best_hour_obj = max(
        sample_route["hours"],
        key=lambda h_item: (
            1 if len(h_item["stop_segments"]) > 0 else 0,
            h_item["demand_pressure_pct"],
            h_item["predicted_passengers"],
        ),
    )

    print("\n" + "=" * 105)
    print("1. SAMPLE 3-TIER HIERARCHICAL DRILL-DOWN (DEPOT -> ROUTE -> 24-HOUR CURVE -> STOP SEGMENTS)")
    print("-" * 105)
    print(
        f"  [TIER 1: DEPOT & ROUTE SUMMARY]\n"
        f"    * Depot               : {depots_tree[sample_depot]['depot_name']} | Active Fleet: {depots_tree[sample_depot]['active_fleet']} buses | Depot Risk: {depots_tree[sample_depot]['risk_category']}\n"
        f"    * Route               : Route {sample_route['route_num']} ({sample_route['origin_stop']} <-> {sample_route['destination_stop']})\n"
        f"    * Corridor            : {sample_route['corridor_name']}\n"
        f"    * Route Span & Fleet  : {sample_route['route_km']} km ({sample_route['stage_span']} stages) | Route Active Fleet: {sample_route['active_fleet']} buses\n"
        f"    * 24h Peak Pressure   : {sample_route['max_demand_pressure_pct']}% [{sample_route['risk_category']}] at {sample_route['peak_hour_window']} "
        f"(Peak Demand: {sample_route['peak_predicted_demand']} pax)"
    )

    print(
        f"\n  [TIER 2: 24-HOUR HOURLY CURVE SNAPSHOT (Peak & Active Window around Hour {best_hour_obj['hour']:02d}:00)]"
    )
    curve_rows = []
    for h_obj in sample_route["hours"]:
        if 6 <= h_obj["hour"] <= 21:
            curve_rows.append(
                {
                    "Hour": f"{h_obj['hour']:02d}:00",
                    "Window": h_obj["time_window"],
                    "Pred_Pax": h_obj["predicted_passengers"],
                    "Active_Trips": h_obj["active_trips"],
                    "Capacity": h_obj["capacity"],
                    "Pressure_%": f"{h_obj['demand_pressure_pct']:.1f}%",
                    "Risk": h_obj["risk_category"],
                    "Crush_Hops": h_obj["crush_segment_count"],
                }
            )
    print(pd.DataFrame(curve_rows).to_string(index=False))

    print(
        f"\n  [TIER 3: STOP-BY-STOP SEGMENT BREAKDOWN FOR HOUR {best_hour_obj['hour']:02d}:00 ({best_hour_obj['time_window']})]"
    )
    seg_rows = []
    for s in best_hour_obj["stop_segments"]:
        if s["direction"] == "UP" or len(seg_rows) < 12:
            seg_rows.append(
                {
                    "Dir": s["direction"],
                    "Hop": f"S{s['from_stage']}->S{s['to_stage']}",
                    "From_Stop -> To_Stop": s["segment_hop"],
                    "Onboard_Pax": f"{s['onboard_pax']} pax",
                    "Load_Factor": f"{s['load_factor_pct']:.1f}%",
                    "Crush_Load (>65)": "YES [CRUSH]" if s["is_crush_load"] else "NO",
                }
            )
    print(pd.DataFrame(seg_rows[:12]).to_string(index=False))

    print("\n" + "=" * 105)
    print("2. TIER 4: OVERLAPPING CORRIDOR CONSOLIDATOR & SMART DEPOT SELECTION EXAMPLE")
    print("-" * 105)
    top_rec = next(
        (
            r
            for r in consolidator_recommendations
            if r["consolidation_type"] == "LONG_ROUTE_CORRIDOR_CONSOLIDATION"
        ),
        consolidator_recommendations[0] if consolidator_recommendations else None,
    )
    if top_rec:
        lr = top_rec["long_route"]
        sr = top_rec["short_route"]
        print(f"  * Shared Trunk Corridor   : {top_rec['corridor_name']} ({top_rec['corridor_id']})")
        print(f"  * Concurrent Crush Window : {top_rec['time_window']} (Hour {top_rec['hour']:02d}:00)")
        print(
            f"  * Shorter Stressed Route  : Route {sr['route_num']} | Originating Depot: {sr['depot_name']} "
            f"| Span: {sr['route_km']} km ({sr['stage_span']} stages, {sr['origin_destination']}) "
            f"| Peak Segment Load: {sr['peak_segment_pax']} pax"
        )
        print(
            f"  * Longer Encompassing Rte : Route {lr['route_num']} | Master Depot: {lr['depot_name']} "
            f"| Span: {lr['route_km']} km ({lr['stage_span']} stages, {lr['origin_destination']}) "
            f"| Peak Segment Load: {lr['peak_segment_pax']} pax"
        )
        print(f"  * Shared Bottleneck Hop   : {top_rec['shared_bottleneck_segment']}")
        print(f"  * AI CONSOLIDATOR ACTION  : \"{top_rec['recommendation']}\"")
    print("=" * 105)
    print(
        f"Saved unified 3-tier hierarchy to `{out_json.as_posix()}` ({out_json.stat().st_size / (1024 * 1024):.2f} MB) in {elapsed:.2f} seconds."
    )


if __name__ == "__main__":
    run_step6_hierarchical_intelligence()

