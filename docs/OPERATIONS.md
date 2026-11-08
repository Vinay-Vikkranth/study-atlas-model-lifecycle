# Operations and reproducibility

## Local run

Python 3.12 is the tested runtime. Keep a virtual environment separate from the repository's generated data.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python scripts/start.py
```

On macOS or Linux use `.venv/bin/python` in place of `.venv\Scripts\python`. Open `http://127.0.0.1:8740`. The start script generates two synthetic candidate releases, promotes the approved versions, seals one final-test report and records normal/shifted drift batches on the first run. Subsequent starts retain the registry. The startup script does not create a cloud account or make paid API calls.

`ATLAS_DATA_DIR` selects the persistent state folder. `PORT` defaults to 8740 and `ATLAS_HOST` defaults to loopback. Set a strong locally generated `ATLAS_ADMIN_TOKEN` **before** startup to enable administration. Leave it empty on a public read-only demonstration. The dashboard keeps a supplied token only in browser memory; it is not saved to localStorage or included in log lines. Use HTTPS at the platform ingress before remote token use.

```powershell
$env:ATLAS_ADMIN_TOKEN = [guid]::NewGuid().ToString('N')
.venv\Scripts\python scripts/start.py
```

The `.env.example` is documentation only; the application reads process environment variables, not an automatically loaded `.env` file.

## Release lifecycle

```text
python -m atlas_lifecycle.cli train --seed 41 --profile balanced
python -m atlas_lifecycle.cli status
python -m atlas_lifecycle.cli promote --model-id NEW_ID --expected-active CURRENT_ID
python -m atlas_lifecycle.cli rollback --expected-active NEW_ID
```

Replace IDs with the full 20-character IDs returned by the registry. Promotion checks the actual validation metrics against the recorded configuration. A failed candidate cannot be activated; self-promotion is rejected so it cannot erase useful rollback history. Both promotion and rollback verify artifact integrity. Rollback also rechecks the target's release gates. Concurrent active-pointer changes produce a conflict instead of silently overwriting a newer release.

The CLI is a trusted local administration tool and intentionally has no bearer-token prompt. Protect operating-system access to the state volume. HTTP mutations require a bearer token; public deployments should leave administration disabled. Tokens are not bundled or generated as fixed public defaults.

## Durable state and recovery

SQLite WAL contains the active pointer, model manifests, append-only application audit events, drift reports and sealed final-test results. Model files are atomically replaced before their registry transaction commits. Concurrent registrations serialize their artifact/manifest publication and reuse an existing identical model ID. An interrupted registration may leave an unreferenced artifact; it does not change the active release. Do not delete the registry volume to recover a model; inspect the audit history and roll back to a verified prior artifact.

Back up the entire data directory while the service is stopped, or use SQLite's online backup API plus referenced model files. Copying only the main SQLite file while WAL writes are active is not a valid backup. Restoring a copy must preserve artifacts and the sealed test reports together. Tests exercise a fresh Registry instance loading the exact prior model after restart.

## Validation and measured load

```text
python -m unittest discover -s tests -v
python -m atlas_lifecycle.cli demo
python -m atlas_lifecycle.cli report
python scripts/benchmark.py --requests 80 --concurrency 4
```

The benchmark starts its own bounded loopback HTTP service against an isolated state copy, checks the current release under load, rolls back, measures the prior version, then promotes the original release and confirms exact prediction recovery. It also verifies malformed input returns 422 and unauthenticated administration returns 401 when a token is configured. Only the process it started is terminated. It does not load test a public host or reevaluate the sealed test set.

Reported p50/p95 include client connection setup, schema validation, artifact hashing/loading and inference. They are local end-to-end measurements, not cloud latency or pure model compute. Concurrency and request counts are recorded. The current implementation prioritizes auditability and loads/verifies an artifact per request; it does not claim BentoML adaptive batching or a tuned production model cache.

On restricted Windows execution environments, asyncio/TestClient needs local loopback permission for its event-loop socketpair. A blocked socketpair is an execution environment failure, not a successful test run. The published test evidence was obtained with loopback enabled.

## Containers and cloud recipe

```text
docker compose up --build
```

The container runs as UID 10001, drops capabilities, serves one worker and persists state in the named volume. Readiness is `/api/ready`; liveness/integrity is `/api/health`; metrics are `/metrics`. Docker execution is validated by GitHub Actions when that workflow actually succeeds, not by the presence of a Dockerfile.

`render.yaml` is a public read-only deployment recipe. It does not create a service. Its free ephemeral filesystem loses registry changes on replacement; bootstrap can recreate a synthetic demo, but this is **not durable deployment evidence**. Use a persistent disk or an equivalent approved persistent service before claiming durable cloud releases. Any cloud resource or charge requires separately scoped approval. No live cloud service is asserted by this project.

This is a single-service portfolio architecture. Do not place multiple replicas over unrelated local SQLite volumes and claim a shared model registry. A larger deployment would move manifests/events to a transactional shared store and model files to authenticated object storage, with release generation checks and a rollout controller.
