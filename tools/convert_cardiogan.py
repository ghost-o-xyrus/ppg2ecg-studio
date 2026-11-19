"""CardioGAN: the authors' TF2 checkpoint -> ONNX. Run by setup.py inside a temporary environment with
TensorFlow 2.15 and tf2onnx (removed afterwards); not imported by the app.

    python tools/convert_cardiogan.py --codes REPO/codes --ckpt DIR --out model.onnx --ref ref.npz

The generator is built exactly as in Track 1 (bench.models.cardiogan, dtype float32: the float64 checkpoint
restored, then cast into an identical float32 graph). ref.npz holds test windows and the TensorFlow output
on them, so setup can check the ONNX model against TensorFlow in the app's own environment.
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", required=True)
    a = ap.parse_args()

    import tensorflow as tf
    import tf2onnx

    from bench.models.cardiogan import CardioGAN
    ad = CardioGAN(a.codes, a.ckpt, filter_output=True, dtype="float32")
    ad.load()
    g = np.random.default_rng(0)
    x = np.clip(g.normal(0, 0.5, (8, ad.window_len)), -1, 1).astype(np.float32)
    y = ad.infer(x)
    spec = (tf.TensorSpec([None, ad.window_len], tf.float32, name="ppg"),)
    fn = tf.function(lambda v: ad.model(v, training=False))
    tf2onnx.convert.from_function(fn, input_signature=spec, opset=17, output_path=a.out)
    np.savez(a.ref, x=x, y=y, fp32_vs_fp64=np.float64(ad.float32_error(x.astype(np.float64))))
    print(f"converted: {a.out}")


if __name__ == "__main__":
    main()
