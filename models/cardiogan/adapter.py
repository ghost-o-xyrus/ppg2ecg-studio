"""CardioGAN: bench.models.cardiogan (unchanged from Track 1, float32) with the generator run by ONNX Runtime.

setup() converts the authors' TF2 checkpoint to ONNX in a temporary TensorFlow 2.15 environment
(tools/convert_cardiogan.py), checks the ONNX output against TensorFlow, and deletes the checkpoint.
"""
import shutil
import subprocess

import numpy as np

from bench.models.cardiogan import CardioGAN
from studio.engine.onnx_run import GpuFallback, run, session

ID = "cardiogan"
TF_PACKAGES = ["tensorflow==2.15.1", "tf2onnx==1.16.1", "numpy<2", "scipy<1.14", "pandas<2.3", "pyyaml"]
CHECK_TOL = 1e-3


class CardioGANOnnx(GpuFallback, CardioGAN):
    framework = "onnx"

    def __init__(self, ctx):
        super().__init__(repo_codes_dir=str(ctx.weights_dir / "repo" / "codes"), weights_dir=str(ctx.weights_dir),
                         filter_output=True, dtype="float32", batch_size=256)
        self.onnx_path, self.model_id, self.want_device, self.ctx = ctx.weights_dir / "model.onnx", ID, ctx.device, ctx

    def load(self):
        self._open(self.want_device)
        self.ctx.info["device"] = self.device_label

    def infer(self, windows):
        return self._run(windows, self.batch_size).astype(np.float64)

    def weights_bytes(self):
        return self.onnx_path.stat().st_size


def build(ctx):
    return CardioGANOnnx(ctx)


def setup(ctx, say):
    d = ctx.weights_dir
    idx = next(d.rglob("ckpt-1.index"), None)
    if idx is None:
        raise FileNotFoundError("ckpt-1.index not found in the downloaded CardioGAN weights")
    ckpt = idx.parent
    ref = d / "convert_ref.npz"
    say("    converting the TF2 checkpoint to ONNX (TensorFlow 2.15 is installed temporarily, a few minutes) ...")
    with ctx.temp_env(TF_PACKAGES) as py:
        subprocess.run([str(py), str(ctx.root / "tools" / "convert_cardiogan.py"), "--codes", str(d / "repo" / "codes"),
                        "--ckpt", str(ckpt), "--out", str(d / "model.onnx"), "--ref", str(ref)], check=True)
    with np.load(ref) as z:
        x, y, e64 = z["x"], z["y"], float(z["fp32_vs_fp64"])
    s, _ = session(d / "model.onnx", "cpu")
    diff = float(np.max(np.abs(run(s, x) - y)))
    say(f"    ONNX export checked: max |ONNX - TensorFlow| = {diff:.1e}  (TensorFlow float32 vs float64: {e64:.1e})")
    ref.unlink()
    if diff > CHECK_TOL:
        (d / "model.onnx").unlink(missing_ok=True)
        raise RuntimeError(f"ONNX output differs from TensorFlow by {diff:.2e}")
    shutil.rmtree(ckpt, ignore_errors=True)           # the checkpoint is not needed to run the model
