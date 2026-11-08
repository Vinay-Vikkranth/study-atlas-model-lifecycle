from __future__ import annotations
import hmac
import json
import logging
import os
import threading
import time
import uuid
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from . import __version__
from .data import FEATURES, load_dataset
from .monitoring import drift
from .registry import Registry
from .training import TrainConfig

FRONTEND = Path(__file__).resolve().parents[1] / "frontend"
LOG = logging.getLogger("atlas")

class Features(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)
    population_density: float | None = Field(default=None, ge=0, le=1)
    housing_instability: float | None = Field(default=None, ge=0, le=1)
    income_index: float | None = Field(default=None, ge=0, le=1)
    service_access: float | None = Field(default=None, ge=0, le=1)
    employment_index: float | None = Field(default=None, ge=0, le=1)
    transit_access: float | None = Field(default=None, ge=0, le=1)

class Prediction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    records: list[Features] = Field(min_length=1, max_length=256)

class Promotion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: str = Field(pattern=r"^[a-f0-9]{20}$")
    expected_active: str | None = Field(default=None, pattern=r"^[a-f0-9]{20}$")

class Rollback(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_active: str = Field(pattern=r"^[a-f0-9]{20}$")

class TrainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seed: int = Field(default=41, ge=0, le=2**31-1)
    profile: Literal["balanced", "compact"] = "balanced"

class DriftRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["reference", "shifted"] = "reference"

def create_app(root: str | Path | None = None, admin_token: str | None = None, bootstrap: bool | None = None):
    root = Path(root or os.environ.get("ATLAS_DATA_DIR", "runtime"))
    registry = Registry(root)
    token = admin_token if admin_token is not None else os.environ.get("ATLAS_ADMIN_TOKEN", "")
    enable_bootstrap = bootstrap if bootstrap is not None else os.environ.get("ATLAS_BOOTSTRAP_DEMO", "0") == "1"
    counters = Counter()
    counter_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app):
        if enable_bootstrap:
            from .cli import bootstrap_demo
            bootstrap_demo(root)
        yield

    app = FastAPI(title="Study Atlas Model Lifecycle", version=__version__, lifespan=lifespan)
    app.state.registry = registry

    def administrator(authorization: str | None = Header(default=None)):
        if not token:
            raise HTTPException(403, "Administration is disabled on this deployment")
        supplied = authorization.removeprefix("Bearer ") if authorization else ""
        if not authorization or not authorization.startswith("Bearer ") or not hmac.compare_digest(supplied.encode("utf-8"), token.encode("utf-8")):
            raise HTTPException(401, "A valid administrator bearer token is required")

    @app.exception_handler(KeyError)
    async def missing(request, error):
        return JSONResponse(status_code=404, content={"detail": "Unknown model or record"})

    @app.exception_handler(ValueError)
    async def invalid(request, error):
        return JSONResponse(status_code=409, content={"detail": str(error)})

    @app.middleware("http")
    async def observation(request: Request, call_next):
        started = time.perf_counter()
        request_id = str(uuid.uuid4())
        response = await call_next(request)
        elapsed = time.perf_counter()-started
        with counter_lock:
            counters["requests"] += 1
            counters["failures"] += int(response.status_code >= 500)
            counters["latency_us"] += round(elapsed*1e6)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        LOG.info(json.dumps(dict(request_id=request_id, path=request.url.path, status=response.status_code, latency_ms=round(elapsed*1000, 3))))
        return response

    @app.get("/api/health")
    def health():
        active = registry.active_id()
        if active:
            try:
                registry.load_model(active)
            except (ValueError, KeyError, OSError):
                raise HTTPException(503, "Active model failed its integrity check")
        return dict(ok=active is not None, version=__version__, active=active, synthetic=True)

    @app.get("/api/ready")
    def ready():
        if not registry.active_id():
            raise HTTPException(503, "No promoted model is available")
        try:
            registry.load_model(registry.active_id())
        except (ValueError, KeyError, OSError):
            raise HTTPException(503, "Active model is unavailable or failed its integrity check")
        return dict(ready=True)

    @app.get("/api/overview")
    def overview():
        status = registry.status()
        dataset = root / "synthetic.csv"
        manifest = json.loads(dataset.with_suffix(".manifest.json").read_text()) if dataset.exists() else None
        return dict(**status, dataset=manifest, features=FEATURES, administration_enabled=bool(token),
                    notice="Synthetic lifecycle demonstration only. Not a real incident forecast, causal analysis, policy recommendation or law enforcement tool.",
                    historical_repository="https://github.com/Vinay-Vikkranth/gun-violence-geospatial-ml")

    @app.get("/api/hexes")
    def hexes(limit: int = Query(default=90, ge=1, le=300)):
        frame = load_dataset(root / "synthetic.csv")
        # Preserve Study Atlas grid/city/scenario schema, with arbitrary synthetic diagram coordinates.
        return dict(items=json.loads(frame.head(limit).to_json(orient="records")), coordinates="Arbitrary synthetic diagram coordinates")

    @app.post("/api/predict")
    def predict(body: Prediction):
        identity = registry.active_id()
        if not identity:
            raise HTTPException(503, "No promoted model is available")
        model, manifest = registry.load_model(identity)
        frame = pd.DataFrame([row.model_dump() for row in body.records], columns=FEATURES).astype(float)
        probability = model.predict_proba(frame)[:, 1]
        threshold = manifest["validation"]["threshold"]
        return dict(model_id=identity, dataset_sha256=manifest["dataset_sha256"], artifact_sha256=manifest["artifact_sha256"],
                    threshold=threshold, synthetic=True, predictions=[dict(probability=float(p), above_threshold=bool(p >= threshold)) for p in probability])

    @app.post("/api/train", dependencies=[Depends(administrator)])
    def train_release(body: TrainRequest):
        return registry.register(root/"synthetic.csv", TrainConfig(seed=body.seed, profile=body.profile))

    @app.post("/api/promote", dependencies=[Depends(administrator)])
    def promote(body: Promotion):
        return registry.promote(body.model_id, body.expected_active)

    @app.post("/api/rollback", dependencies=[Depends(administrator)])
    def rollback(body: Rollback):
        return registry.rollback(body.expected_active)

    @app.post("/api/final-test/{model_id}", dependencies=[Depends(administrator)])
    def seal_test(model_id: str):
        return registry.final_test(model_id, root/"synthetic.csv")

    @app.post("/api/drift/check", dependencies=[Depends(administrator)])
    def check_drift(body: DriftRequest):
        identity = registry.active_id()
        if not identity:
            raise HTTPException(503, "No promoted model")
        manifest = registry.get(identity)
        frame = load_dataset(root/"synthetic.csv")
        sample = frame.loc[frame.split.eq("train"), FEATURES].copy()
        if body.mode == "shifted":
            sample["housing_instability"] = (sample.housing_instability + .45).clip(0, 1)
            sample["service_access"] = .02
        return registry.record_drift(identity, dict(drift(sample, manifest["reference"]), scenario=body.mode))

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics_endpoint():
        with counter_lock:
            snapshot = dict(counters)
        return "\n".join(["# TYPE atlas_requests_total counter", f"atlas_requests_total {snapshot.get('requests', 0)}",
                          "# TYPE atlas_server_errors_total counter", f"atlas_server_errors_total {snapshot.get('failures', 0)}",
                          "# TYPE atlas_request_duration_seconds_sum counter", f"atlas_request_duration_seconds_sum {snapshot.get('latency_us', 0)/1e6}", ""])

    @app.get("/")
    def home():
        return FileResponse(FRONTEND/"index.html")

    app.mount("/static", StaticFiles(directory=FRONTEND), name="static")
    return app

app = create_app()
