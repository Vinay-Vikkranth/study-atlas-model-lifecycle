from __future__ import annotations
import hashlib
import io
import json
from pathlib import Path
import numpy as np
import pandas as pd

FEATURES = ["population_density", "housing_instability", "income_index", "service_access", "employment_index", "transit_access"]
SCHEMA_VERSION = "synthetic-spatial-v1"
SPLITS = {"train": list(range(18)), "calibration": list(range(18, 22)), "validation": list(range(22, 26)), "test": list(range(26, 30))}

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def stable_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

def generate_dataset(path: Path, seed: int = 20261108, rows_per_city: int = 72) -> dict:
    """Fictional groups and coordinates; no research records or real city names."""
    rng = np.random.default_rng(seed)
    records = []
    for city in range(30):
        shift = rng.normal(0, 0.25)
        for cell in range(rows_per_city):
            x = rng.uniform(0, 1, len(FEATURES))
            logit = -1.3 + 2.7*x[1] + 1.0*x[0] - 1.5*x[2] - 1.2*x[3] - 0.9*x[4] + 0.4*x[5] + shift
            y = int(rng.random() < 1/(1+np.exp(-logit)))
            row = dict(zip(FEATURES, x))
            row.update(grid_id=f"SYN-{city:02d}-{cell:03d}", city=f"Fictional-{city:02d}",
                       split=next(k for k, v in SPLITS.items() if city in v), outcome=y,
                       map_x=float(city % 6 + (cell % 9)/12), map_y=float(city // 6 + (cell // 9)/12))
            records.append(row)
    frame = pd.DataFrame(records)
    # Missing observations exercise fold-contained imputation. Labels never drive missingness.
    for column in FEATURES:
        mask = rng.random(len(frame)) < .025
        frame.loc[mask, column] = np.nan
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.10f")
    manifest = dict(schema=SCHEMA_VERSION, synthetic=True, seed=seed, rows=len(frame),
                    sha256=digest(path), features=FEATURES, split_counts=frame.groupby("split").size().to_dict(),
                    geography={s: sorted(frame.loc[frame.split.eq(s), "city"].unique().tolist()) for s in SPLITS},
                    label="Fictional binary outcome; not an incident forecast", coordinates="Arbitrary diagram coordinates, not real geography")
    path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest

def load_dataset(path: Path) -> pd.DataFrame:
    manifest = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    raw = path.read_bytes()
    snapshot_sha = hashlib.sha256(raw).hexdigest()
    if manifest.get("sha256") != snapshot_sha or manifest.get("synthetic") is not True:
        raise ValueError("Dataset integrity or synthetic provenance check failed")
    frame = pd.read_csv(io.BytesIO(raw))
    frame.attrs["dataset_sha256"] = snapshot_sha
    required = set(FEATURES + ["grid_id", "city", "split", "outcome", "map_x", "map_y"])
    if set(frame) != required or not frame.grid_id.is_unique:
        raise ValueError("Dataset schema or unique grid identifier check failed")
    if set(frame.split) != set(SPLITS) or not frame.outcome.isin([0, 1]).all():
        raise ValueError("Invalid partitions or binary labels")
    if frame.groupby("city").split.nunique().max() != 1:
        raise ValueError("Geographic leakage: a city appears in multiple partitions")
    if frame.groupby("split").city.nunique().min() < 3:
        raise ValueError("Each partition requires at least three distinct cities")
    values = frame[FEATURES].to_numpy(dtype=float)
    if np.isinf(values).any() or ((values < 0) | (values > 1)).any():
        raise ValueError("Feature values outside the synthetic schema")
    return frame
