from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from .data import FEATURES, generate_dataset, load_dataset
from .monitoring import drift
from .registry import Registry, write_json
from .training import TrainConfig

def bootstrap_demo(root: Path):
    root = Path(root)
    registry = Registry(root)
    if registry.active_id():
        return registry.status()
    dataset = root/"synthetic.csv"
    if not dataset.exists():
        generate_dataset(dataset)
    first = registry.register(dataset, TrainConfig(seed=17))
    registry.promote(first["id"], None)
    second = registry.register(dataset, TrainConfig(seed=29, profile="compact"))
    registry.promote(second["id"], first["id"])
    # Seal only after the release is chosen, once per dataset, and never use it for a gate.
    registry.final_test(second["id"], dataset)
    sample = load_dataset(dataset)
    reference = sample.loc[sample.split.eq("train"), FEATURES]
    registry.record_drift(second["id"], dict(drift(reference, second["reference"]), scenario="reference"))
    shifted = reference.copy()
    shifted["housing_instability"] = (shifted.housing_instability+.45).clip(0, 1)
    shifted["service_access"] = .02
    registry.record_drift(second["id"], dict(drift(shifted, second["reference"]), scenario="shifted"))
    return registry.status()

def main():
    parser = argparse.ArgumentParser(description="Study Atlas synthetic model lifecycle")
    parser.add_argument("command", choices=["demo", "train", "promote", "rollback", "final-test", "status", "report"])
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("ATLAS_DATA_DIR", "runtime")))
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--profile", choices=["balanced", "compact"], default="balanced")
    parser.add_argument("--model-id")
    parser.add_argument("--expected-active")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    registry = Registry(args.root)
    if args.command == "demo":
        result = bootstrap_demo(args.root)
        if not args.verbose:
            result = dict(active=result["active"], models=len(result["models"]), registry=str(registry.db_path.resolve()),
                          synthetic=True, final_test_sealed=bool(result["final_test_reports"]),
                          drift_scenarios={r.get("scenario"):r["alert"] for r in result["drift_reports"]},
                          next="python -m uvicorn atlas_lifecycle.api:app --host 127.0.0.1 --port 8740")
    elif args.command == "train":
        result = registry.register(args.root/"synthetic.csv", TrainConfig(seed=args.seed, profile=args.profile))
    elif args.command == "promote":
        result = registry.promote(args.model_id, args.expected_active)
    elif args.command == "rollback":
        result = registry.rollback(args.expected_active)
    elif args.command == "final-test":
        result = registry.final_test(args.model_id, args.root/"synthetic.csv")
    elif args.command == "report":
        from .reporting import build_report
        result = build_report(args.root)
    else:
        result = registry.status()
    print(json.dumps(result, indent=2, allow_nan=False))

if __name__ == "__main__":
    main()
