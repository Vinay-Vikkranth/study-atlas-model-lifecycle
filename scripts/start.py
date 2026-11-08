"""One-command synthetic bootstrap and same-origin dashboard/API server."""
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas_lifecycle.cli import bootstrap_demo

if __name__ == "__main__":
    root=Path(os.environ.get("ATLAS_DATA_DIR","runtime"))
    status=bootstrap_demo(root)
    print(f"Study Atlas ready: {len(status['models'])} synthetic releases; active {status['active'][:8]}",flush=True)
    os.execv(sys.executable,[sys.executable,"-m","uvicorn","atlas_lifecycle.api:app","--host",os.environ.get("ATLAS_HOST","127.0.0.1"),"--port",os.environ.get("PORT","8740"),"--workers","1"])
