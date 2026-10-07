"""Choose Race defaults from fits on the actual deployment hardware."""

from __future__ import annotations

import json
import os
from pathlib import Path

from data import preset

LADDERS = {
    "PC": (50, 80, 100, 125, 150, 200),
    "FCI": (50, 75, 100, 125, 150),
    "DirectLiNGAM": (15, 20, 25, 30, 40, 50, 70),
}


def calibrate(pool, output):
    """Probe sequentially; record unmet targets instead of implying a guaranteed speed."""
    evidence, methods = [], {}
    for method, ladder in LADDERS.items():
        candidates = []
        for size in ladder:
            data, _, _ = preset(method, size, 7, max(ladder))
            ours = pool.run("Andrey", method, data)
            other = pool.run("causal-learn", method, data)
            row = {"method": method, "d": size, "seed": 7}
            for name, result in (("andrey", ours), ("causal_learn", other)):
                row[name] = {key: value for key, value in result.items() if key != "marks"}
            evidence.append(row)
            if ours["status"] == other["status"] == "ok":
                candidates.append(row)
                if other["seconds"] >= 5 or ours["seconds"] >= 1:
                    break
            else:
                break
        if not candidates:
            first = evidence[-1]
            outcomes = "; ".join(
                f"{name}: {result.get('error') or result['status']}"
                for name, result in (
                    ("Andrey", first["andrey"]),
                    ("causal-learn", first["causal_learn"]),
                )
            )
            raise RuntimeError(
                f"No {method} size finished during host calibration. "
                f"At d = {first['d']}, {outcomes}."
            )
        target = [
            r
            for r in candidates
            if r["andrey"]["seconds"] < 1 and 5 <= r["causal_learn"]["seconds"] < 60
        ]
        chosen = (target or candidates)[-1]
        methods[method] = {"default": chosen["d"], "cap": chosen["d"], "target_met": bool(target)}
    payload = {
        "space": os.environ.get("SPACE_ID"),
        "hardware": os.environ.get("ANDREY_DEMO_HARDWARE")
        or ("cpu-basic" if os.environ.get("SPACE_ID") else "preview server"),
        "threads": 1,
        "methods": methods,
        "probes": evidence,
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(payload, indent=2) + "\n")
    return payload
