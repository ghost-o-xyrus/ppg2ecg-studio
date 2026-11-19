"""RDDM: bench.models.rddm (unchanged from Track 1), PyTorch on the CPU, fixed seed.

The authors' diffusion.py imports neurokit2 at the top but never calls it; when it is not installed an
empty stand-in module is registered (as bench.models.cardiogan does for tensorflow_addons), which saves a
large dependency.
"""
import sys
import types

from bench.models.rddm import RDDM
from studio.engine.isolate import repo_imports

ID = "rddm"
CLASH = ("model", "diffusion", "utils", "data", "metrics")      # module names other repositories use too


class RDDMIsolated(RDDM):
    def load(self):
        with repo_imports(self.repo_dir, CLASH):
            super().load()


def build(ctx):
    try:
        import neurokit2  # noqa: F401
    except ImportError:
        sys.modules.setdefault("neurokit2", types.ModuleType("neurokit2"))
    p = ctx.spec.params
    m = RDDMIsolated(repo_dir=str(ctx.weights_dir / "repo"), weights_dir=str(ctx.weights_dir), n_steps=int(p.get("n_steps", 10)),
             device="cpu", seed=0, batch_size=int(p.get("batch_size", 128)))
    ctx.info["device"] = "CPU"
    return m
