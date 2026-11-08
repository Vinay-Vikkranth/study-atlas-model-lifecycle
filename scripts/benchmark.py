"""Bounded real HTTP load, version change and rollback proof against an isolated copy."""
from __future__ import annotations
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import shutil
import socket
import statistics
import subprocess
import sys
import time
import uuid
import httpx
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from atlas_lifecycle.cli import bootstrap_demo
from atlas_lifecycle.data import FEATURES
from atlas_lifecycle.registry import write_json

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path("runtime"))
    parser.add_argument("--requests",type=int,default=80)
    parser.add_argument("--concurrency",type=int,default=4)
    parser.add_argument("--output",type=Path,default=Path("docs/evidence/benchmark.json"))
    args=parser.parse_args()
    if not 10 <= args.requests <= 1000 or not 1 <= args.concurrency <= 16:
        parser.error("Use 10..1000 requests and 1..16 concurrency")
    bootstrap_demo(args.root)
    isolated=args.root.parent/f"benchmark-{uuid.uuid4().hex[:8]}"
    shutil.copytree(args.root,isolated)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1",0));port=sock.getsockname()[1]
    token=uuid.uuid4().hex
    env=dict(os.environ,ATLAS_DATA_DIR=str(isolated.resolve()),ATLAS_ADMIN_TOKEN=token)
    logfile=(isolated/"http-server.log").open("w",encoding="utf-8")
    process=subprocess.Popen([sys.executable,"-m","uvicorn","atlas_lifecycle.api:app","--host","127.0.0.1","--port",str(port),"--workers","1"],env=env,stdout=logfile,stderr=subprocess.STDOUT)
    base=f"http://127.0.0.1:{port}"
    body={"records":[{f:.5 for f in FEATURES}]}
    headers={"Authorization":f"Bearer {token}"}
    def measure():
        def request(_):
            started=time.perf_counter()
            with httpx.Client(timeout=15,trust_env=False) as client:
                response=client.post(base+"/api/predict",json=body)
                return dict(ms=(time.perf_counter()-started)*1000,status=response.status_code,
                            model_id=response.json().get("model_id"))
        started=time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(args.concurrency) as pool:
            rows=list(pool.map(request,range(args.requests)))
        latency=sorted(r["ms"] for r in rows)
        return dict(requests=len(rows),concurrency=args.concurrency,p50_ms=statistics.median(latency),
                    p95_ms=latency[min(len(latency)-1,int(.95*len(latency)))],
                    errors=sum(r["status"]!=200 for r in rows),
                    elapsed_seconds=time.perf_counter()-started,served_versions=sorted(set(r["model_id"] for r in rows)))
    try:
        with httpx.Client(timeout=30,trust_env=False) as client:
            for _ in range(100):
                try:
                    if client.get(base+"/api/ready").status_code==200:break
                except httpx.HTTPError:pass
                if process.poll() is not None:raise RuntimeError("HTTP benchmark server exited")
                time.sleep(.1)
            else:raise RuntimeError("HTTP benchmark server did not become ready")
            initial=client.post(base+"/api/predict",json=body).json()
            current=measure()
            rollback=client.post(base+"/api/rollback",headers=headers,json={"expected_active":initial["model_id"]})
            rollback.raise_for_status()
            restored_id=rollback.json()["active"]
            restored=measure()
            restore=client.post(base+"/api/promote",headers=headers,json={"model_id":initial["model_id"],"expected_active":restored_id})
            restore.raise_for_status()
            after=client.post(base+"/api/predict",json=body).json()
            same=initial["model_id"]==after["model_id"] and initial["predictions"]==after["predictions"]
            invalid=client.post(base+"/api/predict",json={"records":[{"income_index":5.0}]}).status_code
            unauthorized=client.post(base+"/api/rollback",json={"expected_active":after["model_id"]}).status_code
            report=dict(synthetic_only=True,transport="real loopback HTTP",timing="Client-observed latency includes connection setup, validation, model hash verification, loading and inference; not cloud latency",current=current,prior_release=restored,
                        restored_prediction_identical=same,version_changed=initial["model_id"]!=restored_id,
                        malformed_input_status=invalid,unauthorized_mutation_status=unauthorized,
                        final_test_not_reevaluated=True)
            if current["errors"] or restored["errors"] or not same or invalid!=422 or unauthorized!=401:
                raise RuntimeError(f"Benchmark acceptance failed: {report}")
            args.output.parent.mkdir(parents=True,exist_ok=True)
            write_json(args.output,report)
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig,ax=plt.subplots(figsize=(7,4))
            labels=["Current p50","Current p95","Prior p50","Prior p95"]
            ax.bar(labels,[current["p50_ms"],current["p95_ms"],restored["p50_ms"],restored["p95_ms"]],color=["#087f73","#76aa94","#b58b58","#d3b88b"])
            ax.set(ylabel="Client-observed milliseconds",title="Synthetic model serving · real local HTTP load")
            fig.tight_layout();fig.savefig(args.output.with_suffix(".png"),dpi=160);plt.close(fig)
            print(json.dumps(report,indent=2))
    finally:
        process.terminate()
        try:process.wait(timeout=15)
        except subprocess.TimeoutExpired:process.kill();process.wait()
        logfile.close()

if __name__=="__main__":main()
