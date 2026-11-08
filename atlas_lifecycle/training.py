from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, f1_score, log_loss, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from .data import FEATURES, stable_hash

@dataclass(frozen=True)
class TrainConfig:
    seed: int = 17
    profile: str = "balanced"
    folds: int = 3
    minimum_auc: float = .60
    minimum_f1: float = .40
    maximum_brier: float = .30

class CalibratedModel:
    """A sigmoid is fitted on separate calibration cities; base model remains frozen."""
    def __init__(self, pipeline, sigmoid):
        self.pipeline = pipeline
        self.sigmoid = sigmoid

    def predict_proba(self, frame):
        with threadpool_limits(limits=1):
            raw = np.clip(self.pipeline.predict_proba(frame)[:, 1], 1e-6, 1-1e-6)
        logits = np.log(raw/(1-raw)).reshape(-1, 1)
        positive = self.sigmoid.predict_proba(logits)[:, 1]
        return np.column_stack([1-positive, positive])

def candidates(config: TrainConfig) -> dict[str, Pipeline]:
    shallow = config.profile == "compact"
    models = {
        "logistic_regression": LogisticRegression(C=.35 if shallow else 1.0, max_iter=400, random_state=config.seed),
        "random_forest": RandomForestClassifier(n_estimators=64, max_depth=4 if shallow else 7, min_samples_leaf=5, n_jobs=1, random_state=config.seed),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=45 if shallow else 70, max_depth=2, random_state=config.seed),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=45 if shallow else 70, max_leaf_nodes=9 if shallow else 15, l2_regularization=2, random_state=config.seed),
        "extra_trees": ExtraTreesClassifier(n_estimators=64, max_depth=5 if shallow else 8, min_samples_leaf=5, n_jobs=1, random_state=config.seed),
    }
    return {name: Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)), ("scale", StandardScaler()), ("classifier", model)]) for name, model in models.items()}

def metrics(y, probability, threshold: float) -> dict[str, Any]:
    prediction = probability >= threshold
    observed, predicted = calibration_curve(y, probability, n_bins=8, strategy="uniform")
    return dict(roc_auc=float(roc_auc_score(y, probability)), average_precision=float(average_precision_score(y, probability)),
                f1=float(f1_score(y, prediction, zero_division=0)), precision=float(precision_score(y, prediction, zero_division=0)),
                recall=float(recall_score(y, prediction, zero_division=0)), brier=float(brier_score_loss(y, probability)),
                log_loss=float(log_loss(y, probability, labels=[0, 1])), threshold=float(threshold),
                confusion_matrix=confusion_matrix(y, prediction, labels=[0, 1]).tolist(),
                calibration=[dict(predicted=float(p), observed=float(o)) for p, o in zip(predicted, observed)])

def select_threshold(y, probability) -> float:
    # Stable tie break chooses the larger threshold. This function only receives validation labels.
    choices = [(float(f1_score(y, probability >= t, zero_division=0)), float(t)) for t in np.linspace(.1, .9, 81)]
    return max(choices)[1]

def check_gates(validation: dict, config: dict) -> dict:
    checks = {"validation_auc": validation["roc_auc"] >= config["minimum_auc"],
              "validation_f1": validation["f1"] >= config["minimum_f1"],
              "validation_brier": validation["brier"] <= config["maximum_brier"]}
    return dict(passed=all(checks.values()), checks=checks, selection_partition="validation", test_metrics_used=False)

def train(frame: pd.DataFrame, config: TrainConfig):
    with threadpool_limits(limits=1):
        return _train(frame, config)

def _train(frame: pd.DataFrame, config: TrainConfig):
    if config.profile not in ("balanced", "compact") or not 2 <= config.folds <= 6:
        raise ValueError("Invalid training profile or fold count")
    partitions = {name: frame.loc[frame.split.eq(name)].copy() for name in ("train", "calibration", "validation")}
    # The test partition is deliberately never inspected by training/model selection.
    train_frame = partitions["train"]
    X, y, groups = train_frame[FEATURES], train_frame.outcome.to_numpy(), train_frame.city.to_numpy()
    cv = GroupKFold(n_splits=config.folds)
    fold_indices = list(cv.split(X, y, groups))
    fold_audit = []
    for fold, (fit, validation) in enumerate(fold_indices):
        fold_audit.append(dict(fold=fold, fit_cities=sorted(set(groups[fit])), validation_cities=sorted(set(groups[validation])),
                               fit_rows=len(fit), validation_rows=len(validation),
                               fit_row_ids_hash=stable_hash(sorted(train_frame.iloc[fit].grid_id.tolist()))))
    fitted = []
    for name, pipeline in candidates(config).items():
        scores, preprocessing_audit = [], []
        for fit, validation in fold_indices:
            model = clone(pipeline).fit(X.iloc[fit], y[fit])
            probability = model.predict_proba(X.iloc[validation])[:, 1]
            scores.append(float(roc_auc_score(y[validation], probability)))
            preprocessing_audit.append(dict(imputer_medians=model.named_steps["impute"].statistics_.tolist(),
                                             scaler_mean=model.named_steps["scale"].mean_.tolist()))
        fitted.append((float(np.mean(scores)), name, pipeline, scores, preprocessing_audit))
    # Pick the family using training-only grouped CV. Calibration and validation never choose the family.
    _, family, pipeline, _, _ = max(fitted, key=lambda x: (x[0], x[1]))
    pipeline.fit(X, y)
    calibration = partitions["calibration"]
    raw = np.clip(pipeline.predict_proba(calibration[FEATURES])[:, 1], 1e-6, 1-1e-6)
    sigmoid = LogisticRegression(C=100, random_state=config.seed).fit(np.log(raw/(1-raw)).reshape(-1, 1), calibration.outcome)
    model = CalibratedModel(pipeline, sigmoid)
    validation = partitions["validation"]
    probability = model.predict_proba(validation[FEATURES])[:, 1]
    threshold = select_threshold(validation.outcome, probability)
    validation_metrics = metrics(validation.outcome, probability, threshold)
    report = dict(config=asdict(config), config_hash=stable_hash(asdict(config)), selected_family=family,
                  feature_schema=FEATURES, grouped_cv=[dict(family=n, mean_auc=auc, fold_auc=scores, preprocessing=audit) for auc,n,_,scores,audit in fitted],
                  folds=fold_audit, fit_geography=sorted(train_frame.city.unique().tolist()),
                  calibration_geography=sorted(calibration.city.unique().tolist()), validation_geography=sorted(validation.city.unique().tolist()),
                  calibration_method="sigmoid fitted only on separate calibration cities", threshold_selection="F1 on validation cities only; deterministic tie break",
                  validation=validation_metrics, gates=check_gates(validation_metrics, asdict(config)),
                  fitted_preprocessing=dict(imputer_medians=pipeline.named_steps["impute"].statistics_.tolist(),
                                            scaler_mean=pipeline.named_steps["scale"].mean_.tolist()),
                  test_accessed=False, synthetic=True)
    return model, report
