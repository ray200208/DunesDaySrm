import sys
from pathlib import Path

src_dir = Path(__file__).resolve().parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from app import (
    XAIExplainInput,
    app,
    explain_recommendation,
    get_recommendations,
    get_routes,
    get_tier2_route,
    get_tier3_route_hour,
    init_duckdb_engine,
)

if __name__ == "__main__":
    import uvicorn

    init_duckdb_engine()
    print("FlowLink Explainable AI Operations Server active at http://127.0.0.1:8000", flush=True)
    if "--self-test" in sys.argv:
        res = get_routes(mode="MTC", depot="ALL", day_of_week=0, hour=8)
        t2 = get_tier2_route("KA:102K", day_of_week=3)
        t3 = get_tier3_route_hour("KA:102K", 17, day_of_week=3)
        xai = explain_recommendation(
            XAIExplainInput(route_num="KA:102K", day_of_week=3, hour=17, question_key="why_flagged")
        )
        print(
            f"[SELF-TEST] Routes={res['total_routes']} | Depots={len(res['depots'])} | "
            f"102K Runs={len(t3['bus_runs'])} | 102K Segments={len(t3['stop_segments'])}"
        )
        print(f"  * XAI: {xai['answer']}")
    else:
        uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
