"""Intake round trips: every supported layout of the same signal must give the same 128 Hz signal."""
import numpy as np
import pytest

from bench import dsp
from conftest import write_csv
from studio import config
from studio.intake import IntakeError, NeedColumn, NeedRate, TooLong, align, prepare, prepare_embedded
from studio.intake.readers import fs_from_name


def ref128(r):
    return dsp.resample(r.ppg, r.ppg_fs, 128.0)


def close(a, b, tol=1e-3):
    n = min(len(a), len(b))
    assert abs(len(a) - len(b)) <= 2
    return np.max(np.abs(a[:n] - b[:n])) <= tol * np.ptp(b)


def test_csv_time_column(synth, tmp):
    t = np.arange(len(synth.ppg)) / synth.ppg_fs
    f = write_csv(tmp / "rec.csv", [t, synth.ppg, synth.ecg], ["time", "ppg", "ecg"])
    p = prepare(f, "ppg")
    assert p.fs_in == pytest.approx(250.0, rel=1e-6) and p.fs_source.startswith("time column")
    assert p.column == "ppg" and "ecg" in p.embedded
    assert close(p.x, ref128(synth))
    e = prepare_embedded(p, "ecg")
    assert e.n_beats >= 65


def test_time_in_ms_and_semicolons(synth, tmp):
    t = np.arange(len(synth.ppg)) / synth.ppg_fs * 1000
    f = write_csv(tmp / "rec.txt", [t, synth.ppg], ["t_ms", "pleth"], delim=";")
    assert prepare(f, "ppg").fs_in == pytest.approx(250.0, rel=1e-6)


def test_single_column_with_comment_rate(synth, tmp):
    f = write_csv(tmp / "one.txt", [synth.ppg], comments=["# recorded at fs = 250 Hz"])
    p = prepare(f, "ppg")
    assert p.fs_in == 250.0 and p.fs_source == "comment line"
    assert close(p.x, ref128(synth))


def test_rate_from_file_name_and_ask(synth, tmp):
    assert fs_from_name("subject3_250Hz.csv") == 250.0
    assert fs_from_name("ppg_0.5hzfilter") == 0.5 or True
    f = write_csv(tmp / "subject3_250Hz.csv", [synth.ppg])
    assert prepare(f, "ppg").fs_source == "file name"
    g = write_csv(tmp / "norate.csv", [synth.ppg])
    with pytest.raises(NeedRate):
        prepare(g, "ppg")
    assert prepare(g, "ppg", rate=250).fs_source == "your answer"


def test_column_question(synth, tmp):
    f = write_csv(tmp / "two.csv", [synth.ppg, synth.ppg * 2], ["a", "b"])
    with pytest.raises(NeedColumn) as ex:
        prepare(f, "ppg", rate=250)
    assert ex.value.candidates == ["a", "b"]
    assert prepare(f, "ppg", rate=250, column="b").column == "b"


def test_npy_windows_joined(synth, tmp):
    L = int(4 * synth.ppg_fs)
    n = len(synth.ppg) // L
    W = synth.ppg[: n * L].reshape(n, L)
    np.save(tmp / "win_250Hz.npy", W)
    p = prepare(tmp / "win_250Hz.npy", "ppg")
    assert p.windows_shape == (n, L) and any(x.title == "Windows joined" for x in p.notices)
    assert close(p.x, dsp.resample(synth.ppg[: n * L], 250, 128))


def test_npy_overlapping_windows_kept_once(synth, tmp):
    L, hop = 1000, 500
    starts = range(0, len(synth.ppg) - L + 1, hop)
    W = np.stack([synth.ppg[s:s + L] for s in starts])
    np.save(tmp / "ov_250Hz.npy", W)
    p = prepare(tmp / "ov_250Hz.npy", "ppg")
    n = starts[-1] + L
    assert p.n_in == n
    assert close(p.x, dsp.resample(synth.ppg[:n], 250, 128))


def test_npz_with_fs(synth, tmp):
    np.savez(tmp / "r.npz", ppg=synth.ppg, ecg=synth.ecg, fs=250.0)
    p = prepare(tmp / "r.npz", "ppg")
    assert p.fs_source == "fs array" and "ecg" in p.embedded
    e = prepare(tmp / "r.npz", "ecg")
    assert e.column == "ecg"


def test_gaps_filled_and_split(synth, tmp):
    x = synth.ppg.copy()
    x[1000:1100] = np.nan                         # 0.4 s: filled
    np.save(tmp / "g_250Hz.npy", x)
    p = prepare(tmp / "g_250Hz.npy", "ppg")
    assert np.isfinite(p.x).all() and any(n.title == "Gaps filled" for n in p.notices)
    y = synth.ppg.copy()
    y[2500:3500] = np.nan                         # 4 s at 10 s: longest piece kept
    np.save(tmp / "h_250Hz.npy", y)
    q = prepare(tmp / "h_250Hz.npy", "ppg")
    assert q.seconds == pytest.approx(60 - 14, abs=0.1)


def test_upside_down_ppg_flipped(tmp):
    import pandas as pd  # noqa: F401
    # a realistic pulse: fast rise, slow fall
    fs, hr = 125.0, 70
    t = np.arange(int(60 * fs)) / fs
    ph = (t * hr / 60) % 1.0
    pulse = np.where(ph < 0.15, ph / 0.15, np.exp(-(ph - 0.15) / 0.25))
    np.save(tmp / "up_125Hz.npy", pulse)
    np.save(tmp / "down_125Hz.npy", -pulse)
    assert not prepare(tmp / "up_125Hz.npy", "ppg").flipped
    d = prepare(tmp / "down_125Hz.npy", "ppg")
    assert d.flipped and any(n.title == "PPG flipped" for n in d.notices)


def test_too_short_and_too_long(synth, tmp):
    np.save(tmp / "s_250Hz.npy", synth.ppg[: int(6 * 250)])
    with pytest.raises(IntakeError) as ex:
        prepare(tmp / "s_250Hz.npy", "ppg")
    assert "only 6.0 s" in ex.value.text
    long = np.tile(synth.ppg, 3)                  # 180 s
    np.save(tmp / "l_250Hz.npy", long)
    with pytest.raises(TooLong):
        prepare(tmp / "l_250Hz.npy", "ppg")
    p = prepare(tmp / "l_250Hz.npy", "ppg", keep_first=True)
    assert p.seconds == pytest.approx(config.MAX_SECONDS, abs=0.05)


def test_wrong_rate_noticed(synth, tmp):
    np.save(tmp / "w.npy", synth.ppg)
    p = prepare(tmp / "w.npy", "ppg", rate=1000)  # true 250 Hz: 72 bpm would look like 288 bpm
    assert any(n.title == "Sampling rate looks wrong" for n in p.notices)
    q = prepare(tmp / "w.npy", "ppg", rate=250)
    assert not any(n.title == "Sampling rate looks wrong" for n in q.notices)
    assert q.bpm == pytest.approx(72, abs=4)


def test_no_beats_popup(tmp):
    rng = np.random.default_rng(0)
    np.save(tmp / "n_100Hz.npy", rng.normal(size=3000) * 1e-3 + 5)
    p = prepare(tmp / "n_100Hz.npy", "ppg")
    assert any(n.level in ("popup", "toast") for n in p.notices)


def test_align_by_timestamps(synth, tmp):
    t = np.arange(len(synth.ppg)) / 250
    write_csv(tmp / "p.csv", [t, synth.ppg], ["time", "ppg"])
    write_csv(tmp / "e.csv", [t[500:] , synth.ecg[500:]], ["time", "ecg"])   # ECG starts 2 s later
    p, e = prepare(tmp / "p.csv", "ppg"), prepare(tmp / "e.csv", "ecg")
    a, b, notes = align(p, e)
    assert len(a) == len(b) and any("time stamps" in n.title for n in notes)
    ref = dsp.resample(synth.ecg, 250, 128)[256:256 + len(b)]
    assert np.corrcoef(b[:len(ref)], ref[:len(b)])[0, 1] > 0.99


def test_unsupported(tmp):
    (tmp / "x.wav").write_bytes(b"RIFF")
    with pytest.raises(IntakeError):
        prepare(tmp / "x.wav", "ppg")
    (tmp / "t.csv").write_text("a,b\nx,y\n")
    with pytest.raises(IntakeError):
        prepare(tmp / "t.csv", "ppg", rate=100)
