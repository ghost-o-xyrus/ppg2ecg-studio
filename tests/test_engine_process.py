"""The engine process: models load there and report one by one, runs return the same result as in-process,
a crash of the engine fails the pending run and the engine comes back with its models; stochastic models use
the seed they are given."""
import threading
import time

import numpy as np
import pytest

from studio import engine
from studio.engine.process import Engine, EngineCrashed


def _wait(cond, timeout=90.0):
    t = time.time()
    while not cond():
        if time.time() - t > timeout:
            raise AssertionError("timed out")
        time.sleep(0.05)


def _ready(states, ids):
    return all(i in states and states[i].status != engine.LOADING for i in ids)


@pytest.fixture
def synth128(synth):
    from bench import dsp
    from studio import config
    x = dsp.resample(synth.ppg, synth.ppg_fs, config.WORK_FS)
    return x, dsp.detect_ppg_peaks(x, config.WORK_FS)


def test_engine_loads_runs_and_survives_a_crash(synth128):
    ppg, peaks = synth128
    specs = [s for s in engine.discover() if s.id in ("template", "zhu_dct")]
    ids = [s.id for s in specs]
    states, lock, restarted = {}, threading.Lock(), []

    def on_status(st):
        with lock:
            states[st.spec.id] = st

    eng = Engine(on_status, on_restart=lambda: restarted.append(1))
    try:
        eng.load(specs, "cpu")
        _wait(lambda: _ready(states, ids))
        assert states["template"].status == engine.READY
        r = eng.run("template", ppg, peaks).result(60)
        local = engine.run_model(engine.load_model(specs[0]), ppg, peaks)
        assert np.allclose(r.pred, local.pred) and r.model_id == "template"

        states.clear()
        eng.crash_for_test()                            # a model taking the engine down
        f = eng.run("template", ppg, peaks)
        with pytest.raises((EngineCrashed, RuntimeError)):
            f.result(60)
        _wait(lambda: restarted and _ready(states, ids))  # restarted and the models reloaded
        assert eng.run("template", ppg, peaks).result(60).model_id == "template"
    finally:
        eng.close()


def test_unknown_model_is_an_error_not_a_crash(synth128):
    ppg, peaks = synth128
    eng = Engine()
    try:
        with pytest.raises(RuntimeError, match="not ready"):
            eng.run("nope", ppg, peaks).result(60)
        assert eng.restarts == 0
    finally:
        eng.close()


def test_run_order_puts_sampled_models_last():
    specs = engine.discover()
    ids = [s.id for s in specs]
    order = engine.run_order(ids, specs)
    sampled = [s.id for s in specs if s.stochastic]
    assert order[len(order) - len(sampled):] == [i for i in order if i in sampled]


class _Seeded:
    seed = 0

    def __init__(self):
        self.seen = []


def test_seed_reaches_a_sampled_model(monkeypatch):
    """The run's seed must reach the adapter (RDDM and PPGFlowECG seed torch from adapter.seed)."""
    from bench import runner
    spec = engine.ModelSpec("s", "S", "", folder=None, stochastic=True)
    ad = _Seeded()

    def fake_run_record(adapter, rec, **k):
        adapter.seen.append(adapter.seed)
        z = np.zeros(len(rec.ppg))
        return runner.RunOutput(pred=z, true=z, ppg=z, fs=rec.ppg_fs, n_windows=1, infer_seconds=0.0)
    monkeypatch.setattr(runner, "run_record", fake_run_record)
    lm = engine.LoadedModel(spec, engine.READY, ad)
    engine.reconstruct(lm, np.zeros(1280), seed=7)
    engine.reconstruct(lm, np.zeros(1280), seed=3)
    assert ad.seen == [7, 3]
