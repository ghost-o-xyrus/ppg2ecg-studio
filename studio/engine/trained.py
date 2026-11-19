"""The four models we trained with their authors' code (Chiu, CLEP-GAN, P2E-WGAN, Yalcin), run through ONNX.

bench.models.trained (unchanged from Track 1) keeps each model's preprocessing, normalisation and stitching;
only the network call moves from PyTorch to ONNX Runtime. Setup exports the network from the authors'
PyTorch code (weights/<id>/repo, pinned commit) and our weights (weights/<id>/weights.pth + meta.json), and
checks the export against PyTorch.
"""
from __future__ import annotations

import numpy as np

from .onnx_run import GpuFallback, export_torch, freeze

EXPORT_TOL = 1e-3          # max |onnx - pytorch| accepted at setup


def onnx_adapter(model_id: str, ctx, batch_size: int = 256):
    from bench.models.trained import TRAINED
    base = TRAINED[model_id]

    class Onnx(GpuFallback, base):
        framework = "onnx"

        def __init__(self):
            super().__init__(str(ctx.weights_dir), device="cpu", batch_size=batch_size)
            self.onnx_path = self.dir / "model.onnx"
            self.model_id = model_id
            self.want_device = ctx.device

        def load(self):
            self._open(self.want_device)
            ctx.info["device"] = self.device_label

        def infer(self, windows):
            return self._run(windows, self.batch_size)

        def weights_bytes(self):
            return self.onnx_path.stat().st_size

    Onnx.__name__ = Onnx.__qualname__ = base.__name__ + "Onnx"
    return Onnx()


def export(model_id: str, ctx, say=print) -> float:
    """PyTorch (authors' code + our weights) -> weights/<id>/model.onnx; -> max abs difference."""
    from bench.models.trained import TRAINED
    ad = TRAINED[model_id](str(ctx.weights_dir), device="cpu")
    ad.load()
    import torch
    freeze([v for v in vars(ad).values() if isinstance(v, torch.nn.Module)])   # networks + P2E's smoother
    diff, x = export_torch(ad._forward, ad.window_len, ctx.weights_dir / "model.onnx")
    say(f"    ONNX export checked: max |ONNX - PyTorch| = {diff:.1e} on {len(x)} windows")
    if not np.isfinite(diff) or diff > EXPORT_TOL:
        (ctx.weights_dir / "model.onnx").unlink(missing_ok=True)
        raise RuntimeError(f"ONNX output differs from PyTorch by {diff:.2e} (limit {EXPORT_TOL:g})")
    return diff
