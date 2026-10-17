"""Offline decomposition of the archived fixed expert objective; no experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import linregress


def reach(design):
    return max([0] + [e["N"] for e in design["experiments"]
                      if e.get("unsolved_fraction", 1) < 0.5])


def decompose(experiments):
    ns = np.asarray([e["N"] for e in experiments])
    medians = np.asarray([e["median_step"] for e in experiments])
    power = linregress(np.log(ns), np.log(medians))
    exponential = linregress(ns, np.log(medians))
    r2 = np.asarray([power.rvalue ** 2, exponential.rvalue ** 2])
    evidence = 1 / (1 - np.minimum(r2, 0.999))
    weights = np.exp((evidence - evidence.max()) / 10.0)
    weights /= weights.sum()
    predictions = np.exp([
        power.intercept + power.slope * np.log(2000),
        exponential.intercept + exponential.slope * 2000,
    ])
    blend = float(weights @ predictions)
    return {
        "power_slope": float(power.slope),
        "exponential_slope": float(exponential.slope),
        "r_squared": r2.tolist(),
        "weights": weights.tolist(),
        "predictions_at_2000": predictions.tolist(),
        "blended_prediction": blend,
        "objective0": float(np.log10(blend)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path,
                        default=Path(__file__).resolve().parent / "artifacts/data")
    parser.add_argument("--core", type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.data_root = args.data_root.resolve()
    root = args.data_root / "campaign"
    if not (root / "database/database.json").is_file():
        parser.error("Extract supporting_data.zip into reproduce/artifacts/, or provide --data-root.")
    if args.data_root in args.output.resolve().parents:
        parser.error("Write output outside the archived data directory.")
    analysis = args.data_root / "analysis"
    code = (root / "objectives/objective_0.py").read_text()
    namespace = {}
    exec(compile(code, "objective_0.py", "exec"), namespace)
    objective = namespace["objective"]
    designs = json.loads((root / "database/database.json").read_text())["designs"]
    index = {d["design_id"]: d for d in designs}
    scores = {d["design_id"]: float(objective(d["experiments"])) for d in designs}
    ordering = sorted(index, key=lambda i: (scores[i], i))
    cached = json.loads((analysis / "score_matrix.json").read_text())["score_matrix"]["0"]
    max_error = max(abs(scores[i] - cached[str(i)]) for i in index)
    assert max_error < 1e-10, max_error
    stability = json.loads((analysis / "winner_stability.json").read_text())
    out = {
        "scope": "Frozen original 414-design archive; offline reanalysis only.",
        "objective0_sha256": hashlib.sha256(code.encode()).hexdigest(),
        "baseline_byte_identical": (args.core / "domain_knowledge/objective_baseline.py").read_bytes()
                                   == (root / "objectives/objective_0.py").read_bytes(),
        "max_score_difference_from_cached_audit": max_error,
        "cases": {},
        "top10_demonstrated_reach": {},
        "observed_failure_candidates": [],
    }
    for did in (247, 340, 353, 393, 380):
        design = index[did]
        item = decompose(design["experiments"])
        item.update({
            "name": design["short_name"],
            "fixed_expert_rank": ordering.index(did) + 1,
            "demonstrated_reach": reach(design),
            "max_attempted_N": max(e["N"] for e in design["experiments"]),
            "n_records": len(design["experiments"]),
            "observed_failed_records": [e for e in design["experiments"]
                                        if e.get("unsolved_fraction", 1) >= 0.5],
            "ranks_from_existing_audit": stability["ranks_of_interest"].get(str(did)),
        })
        out["cases"][str(did)] = item
    for name, ids in stability["top10"].items():
        out["top10_demonstrated_reach"][name] = [
            {"design_id": i, "reach": reach(index[i])} for i in ids]
    for rank, did in enumerate(ordering, 1):
        failed = [e for e in index[did]["experiments"]
                  if e.get("unsolved_fraction", 1) >= 0.5]
        if failed:
            out["observed_failure_candidates"].append({
                "fixed_expert_rank": rank, "design_id": did,
                "objective0": scores[did], "reach": reach(index[did]),
                "failed_sizes": [e["N"] for e in failed],
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(f"Verified {len(scores)} objective-0 scores; wrote {args.output}.")


if __name__ == "__main__":
    main()
