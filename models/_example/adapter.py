"""Example adapter. The app calls build(ctx) once, then adapter.load(), then adapter.infer(windows) per run.

ctx.spec        the parsed model.yaml (ctx.spec.params, ctx.spec.native_fs, ...)
ctx.weights_dir Path to weights/<id>/
ctx.device      'cpu' in v0.1 ('cuda' / 'dml' from v0.2)
ctx.info        dict you may fill with facts shown in the metrics (e.g. training data)

infer() receives float windows of shape (N, window_s * native_fs) and must return the same shape.
Optional hooks (see bench/models/base.py): prepare_record (whole-signal filtering before windowing),
normalize_window, postprocess.
"""
import numpy as np

from bench.models.base import ModelAdapter


class MyModel(ModelAdapter):
    name = "my_model"

    def __init__(self, spec, weights_dir):
        self.native_fs = float(spec.native_fs)
        self.window_s = float(spec.window_s)
        self.weights_dir = weights_dir

    def load(self):
        pass        # e.g. onnxruntime.InferenceSession(str(self.weights_dir / "my_model.onnx"))

    def infer(self, windows):
        return np.zeros_like(windows)


def build(ctx):
    return MyModel(ctx.spec, ctx.weights_dir)
