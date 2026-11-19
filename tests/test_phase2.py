"""Pieces added with the learned models: the file-rate model input, the GPU -> CPU fallback of the ONNX
models, the import isolation of the authors' repositories, and every installed model's interface."""
import sys
import types

import numpy as np
import pytest

from studio import config, engine, intake


def test_model_input_is_the_file_rate_signal(synth, tmp_path):
    from conftest import write_csv
    t = np.arange(len(synth.ppg)) / synth.ppg_fs
    p = intake.prepare(write_csv(tmp_path / "r.csv", [t, synth.ppg], ["time", "ppg"]), "ppg")
    ppg128, _, _ = intake.align(p, None)
    x, fs = intake.model_input(p, None, ppg128)
    assert fs == pytest.approx(synth.ppg_fs)
    assert len(x) == pytest.approx(len(ppg128) / config.WORK_FS * fs, abs=1)
    assert np.allclose(x, p.x_in[: len(x)])


def test_model_input_follows_the_time_stamp_alignment(synth, tmp_path):
    from conftest import write_csv
    fs = synth.ppg_fs
    t = np.arange(len(synth.ppg)) / fs
    p = intake.prepare(write_csv(tmp_path / "p.csv", [t, synth.ppg], ["time", "ppg"]), "ppg")
    k = int(5 * fs)                                     # the ECG file starts 5 s later
    e = intake.prepare(write_csv(tmp_path / "e.csv", [t[k:], synth.ecg[k:]], ["time", "ecg"]), "ecg")
    a, b, _ = intake.align(p, e)
    x, xfs = intake.model_input(p, e, a)
    assert len(x) == pytest.approx(len(a) / config.WORK_FS * xfs, abs=1)
    assert np.allclose(x[:100], p.x_in[k:k + 100])      # starts where the aligned 128-Hz PPG starts


class _FailingSession:
    def get_inputs(self):
        return [types.SimpleNamespace(name="ppg")]

    def run(self, *a, **k):
        raise RuntimeError("operation not supported on this GPU")


def test_gpu_failure_falls_back_to_cpu_and_is_remembered(tmp_path, monkeypatch):
    from studio.engine import onnx_run
    monkeypatch.setattr(onnx_run, "DEVICE_FILE", tmp_path / "device.json")
    calls = []

    class M(onnx_run.GpuFallback):
        model_id, onnx_path = "m", "m.onnx"

    m = M()
    m.sess, m.device_label = _FailingSession(), "GPU (DirectML)"

    def cpu_open(device):
        calls.append(device)
        m.sess = types.SimpleNamespace(get_inputs=_FailingSession().get_inputs,
                                       run=lambda out, feed: [feed["ppg"] * 2])
        m.device_label = "CPU"
    m._open = cpu_open
    y = m._run(np.ones((3, 8)), 2)
    assert calls == ["cpu"] and m.device_label == "CPU" and np.allclose(y, 2)
    assert onnx_run.remembered() == {"m": "cpu"}      # setup.py --check reads the label; the next start skips the GPU


def test_repo_imports_keep_clashing_module_names_apart(tmp_path):
    from studio.engine.isolate import repo_imports
    for name, val in (("a", 1), ("b", 2)):
        d = tmp_path / name
        d.mkdir()
        (d / "model.py").write_text(f"VALUE = {val}\n")
    got = []
    for name in ("a", "b"):
        with repo_imports(tmp_path / name, ["model"]):
            sys.path.insert(0, str(tmp_path / name))
            import model
            got.append(model.VALUE)
    assert got == [1, 2] and "model" not in sys.modules


@pytest.mark.parametrize("spec", [s for s in engine.discover() if s.installed and not s.always_installed],
                         ids=lambda s: s.id)
def test_installed_model_interface(spec, synth):
    """Each installed model: float windows in, same shape out, finite, full-length reconstruction."""
    from bench import dsp
    lm = engine.load_model(spec, "cpu")
    assert lm.status == engine.READY, lm.error
    ad = lm.adapter
    w = np.clip(np.random.default_rng(0).normal(0, 0.5, (3, ad.window_len)), -1, 1)
    y = ad.infer(w)
    assert y.shape == w.shape and np.isfinite(y).all()
    x = dsp.resample(synth.ppg[: int(30 * synth.ppg_fs)], synth.ppg_fs, config.WORK_FS)
    r = engine.run_model(lm, x, dsp.detect_ppg_peaks(x, config.WORK_FS))
    assert len(r.pred) == len(x) and np.isfinite(r.pred).all()
