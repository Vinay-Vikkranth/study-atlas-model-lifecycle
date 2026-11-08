from __future__ import annotations
import json
import os
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import joblib
import pandas as pd
from .data import FEATURES, digest, load_dataset, stable_hash
from .training import TrainConfig, check_gates, metrics, train

def now():
    return datetime.now(timezone.utc).isoformat()

def write_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)

class Registry:
    """Durable local-volume registry. SQLite transactions serialize release changes."""
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root/"models").mkdir(exist_ok=True)
        self.db_path = self.root/"registry.sqlite3"
        self.lock = threading.RLock()
        with self.db() as db:
            db.executescript('''
            CREATE TABLE IF NOT EXISTS models(id TEXT PRIMARY KEY, created TEXT NOT NULL, manifest TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS active(singleton INTEGER PRIMARY KEY CHECK(singleton=1), model_id TEXT NOT NULL REFERENCES models(id));
            CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY AUTOINCREMENT, created TEXT NOT NULL, kind TEXT NOT NULL, model_id TEXT, details TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS test_reports(test_sha TEXT PRIMARY KEY, model_id TEXT NOT NULL, created TEXT NOT NULL, report TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS drift_reports(id TEXT PRIMARY KEY, created TEXT NOT NULL, model_id TEXT NOT NULL, report TEXT NOT NULL);
            ''')

    @contextmanager
    def db(self):
        db = sqlite3.connect(self.db_path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA journal_mode=WAL")
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def event(db, kind, model_id, details):
        db.execute("INSERT INTO events(created,kind,model_id,details) VALUES(?,?,?,?)", (now(), kind, model_id, json.dumps(details, allow_nan=False)))

    def register(self, dataset: Path, config: TrainConfig) -> dict:
        frame = load_dataset(dataset)
        dataset_sha = frame.attrs["dataset_sha256"]
        training_code_sha = stable_hash({name:digest(Path(__file__).parent/name) for name in ("data.py", "training.py", "__init__.py")})
        model, report = train(frame, config)
        if digest(dataset) != dataset_sha:
            raise ValueError("Dataset changed during training; registration cancelled")
        identity = stable_hash(dict(data=dataset_sha, config=report["config_hash"], code=training_code_sha))[:20]
        artifact = self.root/"models"/f"{identity}.joblib"
        with self.lock, self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            if digest(dataset) != dataset_sha:
                raise ValueError("Dataset changed during training; registration cancelled")
            existing = db.execute("SELECT manifest FROM models WHERE id=?", (identity,)).fetchone()
            if existing:
                return json.loads(existing[0])
            temporary = artifact.with_suffix(f".{uuid.uuid4().hex}.tmp")
            joblib.dump(model, temporary)
            os.replace(temporary, artifact)
            baseline = {}
            reference = frame.loc[frame.split.eq("train"), FEATURES]
            for feature in FEATURES:
                baseline[feature] = dict(mean=float(reference[feature].mean()), std=float(reference[feature].std()),
                                         missing_rate=float(reference[feature].isna().mean()),
                                         bins=[float(x) for x in __import__("numpy").linspace(0, 1, 11)],
                                         counts=__import__("numpy").histogram(reference[feature].dropna(), bins=__import__("numpy").linspace(0, 1, 11))[0].tolist())
            manifest = dict(id=identity, created=now(), artifact=f"models/{identity}.joblib", artifact_sha256=digest(artifact),
                            dataset_sha256=dataset_sha, code_version="1.0.0", training_code_sha256=training_code_sha, schema_version="synthetic-spatial-v1",
                            reference=baseline, **report)
            db.execute("INSERT INTO models VALUES(?,?,?)", (identity, manifest["created"], json.dumps(manifest, allow_nan=False)))
            self.event(db, "registered", identity, dict(dataset_sha256=dataset_sha, gates=report["gates"]))
        return manifest

    def get(self, identity: str) -> dict:
        with self.db() as db:
            row = db.execute("SELECT manifest FROM models WHERE id=?", (identity,)).fetchone()
        if not row:
            raise KeyError(identity)
        return json.loads(row[0])

    def list_models(self) -> list[dict]:
        with self.db() as db:
            return [json.loads(row[0]) for row in db.execute("SELECT manifest FROM models ORDER BY created DESC")]

    def active_id(self) -> str | None:
        with self.db() as db:
            row = db.execute("SELECT model_id FROM active WHERE singleton=1").fetchone()
        return row[0] if row else None

    def load_model(self, identity: str):
        manifest = self.get(identity)
        expected = f"models/{identity}.joblib"
        if manifest["artifact"] != expected or not identity.isalnum() or len(identity) != 20:
            raise ValueError("Invalid model artifact path")
        artifact = self.root / expected
        if artifact.is_symlink() or digest(artifact) != manifest["artifact_sha256"]:
            raise ValueError("Model artifact integrity check failed")
        # Only locally generated artifacts in this trusted, non-uploadable registry are loadable.
        return joblib.load(artifact), manifest

    def promote(self, identity: str, expected_active: str | None, reason="validation gates passed") -> dict:
        _, manifest = self.load_model(identity)
        if not check_gates(manifest["validation"], manifest["config"])["passed"]:
            raise ValueError("Validation release gates failed")
        with self.lock, self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT model_id FROM active WHERE singleton=1").fetchone()
            previous = row[0] if row else None
            if previous != expected_active:
                raise ValueError("Active release changed; refresh before promotion")
            if previous == identity:
                raise ValueError("This model is already active")
            db.execute("INSERT INTO active VALUES(1,?) ON CONFLICT(singleton) DO UPDATE SET model_id=excluded.model_id", (identity,))
            self.event(db, "promoted", identity, dict(previous=previous, reason=reason, test_metrics_used=False))
        return dict(active=identity, previous=previous)

    def rollback(self, expected_active: str) -> dict:
        with self.lock, self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT model_id FROM active WHERE singleton=1").fetchone()
            if not row or row[0] != expected_active:
                raise ValueError("Active release changed; refresh before rollback")
            row = db.execute("SELECT details FROM events WHERE kind IN ('promoted','rolled_back') AND model_id=? ORDER BY seq DESC LIMIT 1", (expected_active,)).fetchone()
            if not row or not json.loads(row[0]).get("previous"):
                raise ValueError("There is no prior release to restore")
            target = json.loads(row[0])["previous"]
            _, target_manifest = self.load_model(target)
            if not check_gates(target_manifest["validation"], target_manifest["config"])["passed"]:
                raise ValueError("Rollback target validation gates failed")
            db.execute("UPDATE active SET model_id=? WHERE singleton=1", (target,))
            self.event(db, "rolled_back", target, dict(previous=expected_active))
        return dict(active=target, previous=expected_active)

    def final_test(self, identity: str, dataset: Path) -> dict:
        """One sealed final-test report per dataset hash, not one score per candidate."""
        frame = load_dataset(dataset)
        dataset_sha = frame.attrs["dataset_sha256"]
        model, manifest = self.load_model(identity)
        if manifest["dataset_sha256"] != dataset_sha:
            raise ValueError("Only the active approved release and its exact dataset can be tested")
        test = frame.loc[frame.split.eq("test")].sort_values("grid_id")
        test_sha = stable_hash(json.loads(test.to_json(orient="records", double_precision=15)))
        with self.lock, self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            active = db.execute("SELECT model_id FROM active WHERE singleton=1").fetchone()
            if not active or active[0] != identity:
                raise ValueError("Only the active approved release can be tested")
            previous = db.execute("SELECT model_id,report FROM test_reports WHERE test_sha=?", (test_sha,)).fetchone()
            if previous:
                if previous[0] != identity:
                    raise ValueError("Final test is already sealed for a different release; do not tune on it")
                return json.loads(previous[1])
            report = dict(model_id=identity, dataset_sha256=dataset_sha, test_partition_sha256=test_sha, created=now(), partition="test",
                          cities=sorted(test.city.unique().tolist()), rows=len(test),
                          metrics=metrics(test.outcome, model.predict_proba(test[FEATURES])[:, 1], manifest["validation"]["threshold"]),
                          release_already_selected=True, immutable=True, used_for_promotion=False)
            db.execute("INSERT INTO test_reports VALUES(?,?,?,?)", (test_sha, identity, report["created"], json.dumps(report)))
            self.event(db, "final_test_sealed", identity, dict(dataset_sha=dataset_sha, test_partition_sha256=test_sha))
        return report

    def status(self) -> dict:
        with self.db() as db:
            events = [dict(row) for row in db.execute("SELECT * FROM events ORDER BY seq DESC LIMIT 30")]
            tests = [json.loads(row[0]) for row in db.execute("SELECT report FROM test_reports")]
            drifts = [json.loads(row[0]) for row in db.execute("SELECT report FROM drift_reports ORDER BY created DESC LIMIT 10")]
        for event in events:
            event["details"] = json.loads(event["details"])
        return dict(active=self.active_id(), models=self.list_models(), events=events, final_test_reports=tests, drift_reports=drifts)

    def record_drift(self, identity: str, report: dict) -> dict:
        report = dict(report, id=uuid.uuid4().hex, model_id=identity, created=now())
        with self.db() as db:
            db.execute("INSERT INTO drift_reports VALUES(?,?,?,?)", (report["id"], report["created"], identity, json.dumps(report)))
            self.event(db, "drift_alert" if report["alert"] else "drift_checked", identity, dict(report_id=report["id"], alert=report["alert"]))
        return report
