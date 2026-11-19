"""ONNX Runtime for the static networks: export from PyTorch at setup, sessions on the best provider at run time.

Provider order: CUDA (when onnxruntime-gpu is installed and an NVIDIA GPU is present) -> DirectML (any GPU
Windows has) -> CPU. A model that cannot be created or run on the GPU falls back to the CPU; setup.py
remembers that choice per model in weights/device.json, so the next start goes straight to the CPU.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .. import config

GPU_PROVIDERS = ("CUDAExecutionProvider", "DmlExecutionProvider")
LABEL = {"CUDAExecutionProvider": "GPU (CUDA)", "DmlExecutionProvider": "GPU (DirectML)",
         "CPUExecutionProvider": "CPU"}
DEVICE_FILE = config.WEIGHTS_DIR / "device.json"
OPSET = 17


def remembered() -> dict:
    try:
        return json.loads(DEVICE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def remember(model_id: str, device: str) -> None:
    d = remembered()
    d[model_id] = device
    try:
        DEVICE_FILE.parent.mkdir(parents=True, exist_ok=True)
        DEVICE_FILE.write_text(json.dumps(d, indent=1), encoding="utf-8")
    except OSError:
        pass


def _quiet(ort):
    """ONNX Runtime logs a failed GPU kernel at error level; the fallback below handles it."""
    try:
        ort.set_default_logger_severity(4)
    except Exception:  # noqa: BLE001
        pass


def providers(device: str = "auto") -> list[str]:
    """Providers to try, best first, for device 'auto' | 'gpu' | 'cpu'."""
    import onnxruntime as ort
    avail = ort.get_available_providers()
    order = [] if device == "cpu" else [p for p in GPU_PROVIDERS if p in avail]
    return order + ["CPUExecutionProvider"]


def _options(provider):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 4
    if provider == "DmlExecutionProvider":      # DirectML needs these two off
        so.enable_mem_pattern = False
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return so


def session(path, device: str = "auto", model_id: str | None = None):
    """-> (InferenceSession, label). Tries each provider in order; a remembered CPU choice skips the GPU."""
    import onnxruntime as ort
    _quiet(ort)
    if model_id and remembered().get(model_id) == "cpu":
        device = "cpu"
    errors = []
    for p in providers(device):
        try:
            prov = [p] if p == "CPUExecutionProvider" else [p, "CPUExecutionProvider"]
            s = ort.InferenceSession(str(path), _options(p), providers=prov)
            if s.get_providers()[0] != p:       # the provider refused the graph and ORT silently used the CPU
                raise RuntimeError(f"{p} not used")
            return s, LABEL[p]
        except Exception as e:  # noqa: BLE001 - any failure means: try the next provider
            errors.append(f"{p}: {e}")
    raise RuntimeError("no ONNX Runtime provider could load " + Path(path).name + ": " + "; ".join(errors))


def run(sess, windows: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """(N, L) float -> (N, L) float32, in batches; the output is cut to the input length like the PyTorch
    adapters do (P2E-WGAN's smoother adds one sample)."""
    name = sess.get_inputs()[0].name
    x = np.ascontiguousarray(windows, dtype=np.float32)
    out = []
    for i in range(0, len(x), batch_size):
        b = x[i:i + batch_size]
        out.append(sess.run(None, {name: b})[0].reshape(len(b), -1))
    return np.concatenate(out)[:, : x.shape[1]]


class GpuFallback:
    """Mixin for ONNX adapters: if a run fails on the GPU, rebuild the session on the CPU, remember it, and
    run again. Subclasses set self.onnx_path, self.model_id and self.sess / self.device_label in load()."""

    def _open(self, device):
        self.sess, self.device_label = session(self.onnx_path, device, self.model_id)

    def _run(self, windows, batch_size):
        try:
            return run(self.sess, windows, batch_size)
        except Exception:
            if self.device_label == "CPU":
                raise
            remember(self.model_id, "cpu")
            self._open("cpu")
            return run(self.sess, windows, batch_size)


# ---------------------------------------------------------------- export (setup time, needs torch)
def freeze(modules):
    """Inference only: no tensor may ask for gradients, including plain tensor attributes some authors keep
    outside nn.Parameter (the tracer refuses to bake those in as constants)."""
    import torch
    for m in modules:
        for sub in m.modules():
            for p in sub.parameters(recurse=False):
                p.requires_grad_(False)
            for k, v in vars(sub).items():
                if isinstance(v, torch.Tensor) and v.requires_grad:
                    setattr(sub, k, v.detach())
        m.eval()


def export_torch(forward, window_len: int, path, n_check: int = 8, seed: int = 0):
    """Export forward(x: (N, L) float32 tensor) -> tensor to ONNX with a dynamic batch axis.
    -> (max |onnx - torch| on random windows in [-1, 1], the test windows)."""
    import torch

    class Wrap(torch.nn.Module):
        def forward(self, x):
            return forward(x)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    g = np.random.default_rng(seed)
    x = np.clip(g.normal(0, 0.5, (n_check, window_len)), -1, 1).astype(np.float32)
    w = Wrap().eval()
    torch.backends.mha.set_fastpath_enabled(False)     # the fused attention kernel has no ONNX form (Yalcin)
    with torch.no_grad():
        ref = w(torch.from_numpy(x)).reshape(n_check, -1).numpy()[:, :window_len]
        for opset in (OPSET, 20):          # 20 only when an operator needs it (affine_grid in Chiu's STN)
            try:
                torch.onnx.export(w, (torch.from_numpy(x[:2]),), str(path), input_names=["ppg"],
                                  output_names=["ecg"], dynamic_axes={"ppg": {0: "n"}, "ecg": {0: "n"}},
                                  opset_version=opset, dynamo=False)
                break
            except Exception as e:  # noqa: BLE001
                if opset == 20 or "opset" not in str(e):
                    raise
    s, _ = session(path, "cpu")
    return float(np.max(np.abs(run(s, x) - ref))), x
