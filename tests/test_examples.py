"""Every file in examples/ is read as examples/README.md describes: length, where the rate comes from, the
embedded true ECG, and the notification it is there to show."""
from pathlib import Path

import pytest

from studio import config, intake

EX = Path(__file__).resolve().parents[1] / "examples"

# file -> (seconds after intake, rate, rate source starts with, embedded ECG column or None, expected notices)
CASES = {
    "bidmc_25_full_8min.csv": (120, 125, "time column", "II", {"First part used"}),
    "bidmc_13_ppg_2min.csv": (120, 125, "time column", None, set()),
    "bidmc_45_ppg_ecg_90s.npz": (90, 125, "fs array", "ecg", set()),
    "bidmc_03_ppg_ecg_gap_2min.csv": (120, 125, "time column", "ecg", {"Gaps filled"}),
    "bidmc_41_ppg_ecg_2min.csv": (120, 125, "time column", "ecg", set()),
    "mimic2_ppg_abp_ecg_2min.csv": (120, 125, "time column", "ecg", set()),
    "mimic2_ppg_ecg_75s_250Hz.csv": (75, 250, "time column", "ecg", set()),
    "bidmc_08_ppg_90s_125Hz.npy": (90, 125, "file name", None, set()),
    "bidmc_16_ppg_windows_4s_125Hz.npy": (120, 125, "file name", None, {"Windows joined"}),
    "bidmc_21_ppg_60s_125Hz.txt": (60, 125, "file name", None, set()),
    "bidmc_19_ppg_30s.csv": (30, 125, "comment", None, set()),
    "bidmc_27_ppg_upside_down_60s.csv": (60, 125, "time column", None, {"PPG flipped"}),
    "bidmc_09_ppg_15s.csv": (15, 125, "time column", None, {"Few beats"}),
}


def test_catalogue_is_complete():
    files = {f.name for f in EX.iterdir() if f.suffix != ".md"}
    assert files == set(CASES) | {"bidmc_13_ecg_2min.csv", "bidmc_22_ppg_8s.csv"}
    assert sum(f.stat().st_size for f in EX.iterdir()) < 8e6          # a small catalogue, not a dataset


@pytest.mark.parametrize("name", sorted(CASES))
def test_example_is_read_as_described(name):
    secs, fs, src, emb, notes = CASES[name]
    p = intake.prepare(EX / name, "ppg", keep_first=True)
    assert p.seconds == pytest.approx(secs, abs=0.1)
    assert p.fs_in == pytest.approx(fs) and p.fs_source.startswith(src)
    assert p.n_beats >= 0.5 * secs * p.bpm / 60                    # pulses are found
    assert (list(p.embedded)[:1] == [emb]) if emb else not p.embedded
    assert notes <= {n.title for n in p.notices}
    if emb:
        e = intake.prepare_embedded(p, emb, keep_first=True)
        a, b, _ = intake.align(p, e)
        assert len(a) == len(b) and e.n_beats > 0.5 * secs * 50 / 60


def test_separate_true_ecg_is_aligned_by_time_stamps():
    p = intake.prepare(EX / "bidmc_13_ppg_2min.csv", "ppg")
    e = intake.prepare(EX / "bidmc_13_ecg_2min.csv", "ecg")
    a, b, notes = intake.align(p, e)
    assert "Aligned by time stamps" in {n.title for n in notes}
    assert len(a) == len(b) == pytest.approx(117 * config.WORK_FS, abs=2)


def test_too_short_example_is_refused():
    with pytest.raises(intake.IntakeError) as e:
        intake.prepare(EX / "bidmc_22_ppg_8s.csv", "ppg")
    assert e.value.title == "File too short"
