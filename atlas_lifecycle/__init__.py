"""Study Atlas lifecycle lab. All bundled data and predictions are synthetic."""
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
__version__ = "1.0.0"
