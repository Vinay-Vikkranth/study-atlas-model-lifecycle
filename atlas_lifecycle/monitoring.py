from __future__ import annotations
import numpy as np
import pandas as pd
from .data import FEATURES

def drift(frame: pd.DataFrame, reference: dict) -> dict:
    if len(frame) < 30:
        raise ValueError("At least 30 rows are required for a drift check")
    result = []
    for feature in FEATURES:
        values = frame[feature]
        baseline = reference[feature]
        expected = np.array(baseline["counts"], dtype=float) + .5
        observed = np.histogram(values.dropna(), bins=baseline["bins"])[0].astype(float) + .5
        expected /= expected.sum()
        observed /= observed.sum()
        psi = float(np.sum((observed-expected)*np.log(observed/expected)))
        missing = float(values.isna().mean())
        alert = psi > .20 or missing > baseline["missing_rate"] + .10
        result.append(dict(feature=feature, psi=psi, missing_rate=missing, reference_missing_rate=baseline["missing_rate"], alert=alert))
    return dict(rows=len(frame), method="PSI with fixed training-only bins, additive 0.5 smoothing", psi_threshold=.20,
                missing_rate_increase_threshold=.10, features=result, alert=any(x["alert"] for x in result),
                meaning="Input distribution change only; not proof of label drift or model-quality deterioration")
