"""Compute and persist a monitoring snapshot via the API (what a daily scheduler would run).

    python -m scripts.monitoring_snapshot
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from src.config import ARTIFACTS_DIR


def main() -> None:
    from api.main import create_app

    key = json.loads((ARTIFACTS_DIR / "demo_api_keys.json").read_text(encoding="utf-8"))["rahul.risk"]["api_key"]
    with TestClient(create_app()) as c:
        r = c.post("/monitoring/run", headers={"X-API-Key": key})
        r.raise_for_status()
        s = r.json()
    dd, pdz, perf, rt = s["data_drift"], s["prediction_drift"], s["performance"], s["retraining"]
    print(f"Model v{s['model_version']} | window {s['window_days']} days")
    if dd.get("status") == "ok":
        print(f"DATA DRIFT: dataset_drift={dd['dataset_drift']} drift_share={dd['drift_share']} "
              f"(n_current={dd['n_current']})")
        for f in dd["features"][:8]:
            print(f"   {f['feature']:30s} psi={f['psi']!s:8s} p={f['p_value']!s:10s} {f['severity']}")
    else:
        print("DATA DRIFT:", dd)
    print("PREDICTION DRIFT:", {k: pdz.get(k) for k in ("status", "psi", "drifted", "current_distribution")})
    print("PERFORMANCE:", perf)
    print("RETRAINING:", rt)


if __name__ == "__main__":
    main()
