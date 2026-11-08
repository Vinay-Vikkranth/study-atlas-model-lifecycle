from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .registry import Registry, write_json

def build_report(root: Path, destination: Path | None = None):
    destination = destination or Path("docs/evidence")
    destination.mkdir(parents=True, exist_ok=True)
    status = Registry(root).status()
    active = next(m for m in status["models"] if m["id"] == status["active"])
    summary = dict(synthetic_only=True, active=status["active"], dataset_sha256=active["dataset_sha256"],
                   training_code_sha256=active.get("training_code_sha256"),
                   models=[{key:m[key] for key in ("id", "selected_family", "config", "config_hash", "artifact_sha256", "validation", "gates", "fit_geography", "calibration_geography", "validation_geography")} for m in status["models"]],
                   grouped_cv=[dict(family=c["family"], mean_auc=c["mean_auc"], fold_auc=c["fold_auc"]) for c in active["grouped_cv"]],
                   final_test_reports=status["final_test_reports"], drift_reports=status["drift_reports"],
                   historical_research_not_reproduced=True)
    write_json(destination/"experiment.json", summary)
    plt.rcParams.update({"font.family":"DejaVu Sans", "axes.spines.top":False, "axes.spines.right":False, "axes.labelcolor":"#314e46", "text.color":"#153630", "axes.titleweight":"bold", "figure.facecolor":"#f7faf7", "axes.facecolor":"#f7faf7"})
    fig, ax = plt.subplots(figsize=(8.5,4.2))
    cv = sorted(active["grouped_cv"], key=lambda c:c["mean_auc"])
    ax.barh([c["family"].replace("_", " ") for c in cv], [c["mean_auc"] for c in cv], color="#087f73")
    ax.set(xlim=(0,1), xlabel="Mean training-only grouped CV ROC AUC", title="Five trained candidate families · synthetic experiment")
    fig.tight_layout();fig.savefig(destination/"grouped-validation.png", dpi=160);plt.close(fig)
    fig, ax = plt.subplots(figsize=(7,4.8))
    ax.plot([0,1],[0,1],"--",color="#a9bcb1", label="Perfect calibration")
    for m in status["models"]:
        points=m["validation"]["calibration"]
        ax.plot([p["predicted"] for p in points],[p["observed"] for p in points], "o-", label=f"{m['id'][:8]} / {m['config']['profile']}")
    ax.set(xlabel="Mean calibrated probability", ylabel="Observed synthetic positive fraction", xlim=(0,1), ylim=(0,1), title="Validation reliability diagram · no test-set tuning")
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(destination/"calibration.png", dpi=160);plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.5,4.5))
    shifted=next(r for r in status["drift_reports"] if r.get("scenario")=="shifted")
    features=shifted["features"]
    ax.barh([f["feature"].replace("_"," ") for f in features], [f["psi"] for f in features], color=["#bc7c35" if f["alert"] else "#087f73" for f in features])
    ax.axvline(.2,color="#bc7c35",linestyle="--",label="PSI threshold 0.20")
    ax.set(xlabel="Population Stability Index",title="Controlled synthetic input shift · fixed training bins")
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(destination/"drift.png",dpi=160);plt.close(fig)
    return dict(report=str(destination/"experiment.json"), charts=[str(p) for p in destination.glob("*.png")])
