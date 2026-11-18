# PPG2ECG Studio

A Windows desktop app that turns a PPG recording into an ECG with every installed reconstruction method,
plots the results against the PPG and (optionally) the true ECG, and measures them with the same scorer as
Track 1 of the PPG2ECG thesis.

**Version 0.1** contains the full default workflow (drop a file, convert it, run, plot, measure) with the two
built-in methods: the **template baseline** and **Zhu DCT**. The seven learned models, export, the cache and
large-file mode follow in the next versions ([Roadmap](#roadmap)).

```
┌──────────── Input ────────────┬──────────── Models ───────────┐
│ [PPG signal] [True ECG]    ➕ │ ▶ Run all   Show [10] s from [0] s
│  ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐  │ ☑ ● Template baseline  Reference  0.18 s ▶
│    Drop your PPG file here    │ ☑ ● Zhu DCT  Classical (linear)   0.25 s ▶
│  └ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘  │                                  │
├───────────────────────────────┴──────────────────────────────┬──┤
│ PPG            ∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿         │📄│
│ True ECG       ─╫──╫──╫──╫──╫──╫──╫──╫──╫──╫──╫──╫──         │⇆ │
│ Template       ─╫──╫──╫──╫──╫──╫──╫──╫──╫──╫──╫──╫──  0.18 s │💾│
│ Zhu DCT        ─╫──╫──╫──╫──╫──╫──╫──╫──╫──╫──╫──╫──  0.25 s │⚙ │
└──────────────────────────────────────────────────────────────┴──┘
```

## Contents

- [Install](#install)
- [Use](#use)
- [What the app does with your file](#what-the-app-does-with-your-file)
- [Measurements](#measurements)
- [Command line](#command-line)
- [Check that the numbers match Track 1](#check-that-the-numbers-match-track-1)
- [Setup flags](#setup-flags)
- [Adding a model](#adding-a-model)
- [Repository layout](#repository-layout)
- [Roadmap](#roadmap)
- [References](#references)

## Install

Windows 10 or 11, 64-bit. No administrator rights and no existing Python needed.

1. Download or clone this repository.
2. Double-click **`setup.bat`**. It
   - fetches [uv](https://github.com/astral-sh/uv) into `.tools\uv` and, through it, a private Python 3.11;
   - creates `.venv` and installs the packages in `requirements.txt` (about 250 MB, mostly Qt);
   - installs the models, runs a self-test that measures each model's speed on this computer;
   - adds a **PPG2ECG Studio** shortcut to the Start menu and the desktop.
3. Start the app from the shortcut or with **`run.bat`**.

### Zhu DCT coefficients

Zhu DCT needs 1,300 coefficients fitted on CapnoBase, exactly as in Track 1. Until they are published as a
release asset of this repository, fit them on your computer (about a minute):

```bat
setup.bat --zhu --capnobase C:\path\to\capnobase
```

`PATH` is the folder with the 42 CapnoBase recordings, in any of the forms it is distributed in
(`*_8min.mat`, `*_8min_signal.csv` or the Borealis `*.tab` files; [doi:10.5683/SP2/NLB8IT](https://doi.org/10.5683/SP2/NLB8IT)).
The `.mat` files are MATLAB v7.3 and need `h5py`, which setup installs into `.venv` when needed. The fit
reports whether it reproduces Track 1 (27,661 cycles, delay 0.234375 s). Without these coefficients the app
lists Zhu DCT as *not installed* and runs the template baseline alone.

## Use

1. **Drop a PPG file** on the *PPG signal* tab (or *Browse…*, or *File → Open PPG*, Ctrl+O). CSV, TXT, NPY or
   NPZ, 10 s to 2 min, up to 20 MB, any sampling rate and any window layout. A file card shows its length, the
   rate found, the conversion to 128 Hz and the number of beats.
2. **Optionally drop the true ECG** on the *True ECG* tab. If the PPG file already holds an ECG column (for
   example BIDMC's `II` next to `PLETH`), it is used automatically and a side notification says so; remove it
   with ✕ if you do not want errors measured.
3. **Run all** (Ctrl+R) runs every ticked model on the whole file; ▶ on a row runs one model. Traces appear
   as each model finishes; the row shows its generation time.
4. **Look**: *Show [10] s from [0] s* chooses the plotted window (2–15 s). Scroll to zoom, drag to pan,
   double-click to reset; the status bar reads out time and value under the mouse.
5. **Measure**: 📄 opens the metrics drawer (it opens by itself after Run all). *Model comparison* works
   without a true ECG; *Errors vs true ECG* needs one. Bold marks the best value in each column.

⚙ *Display options*: align each reconstruction to the true ECG (removes its one constant delay, as the
scorer does), show the true R peaks, band-pass 0.5–40 Hz as scored.

### Notifications

| Situation | What you see |
|---|---|
| File shorter than 10 s | Big central pop-up; Run stays disabled for that file |
| File longer than 2 min | Central pop-up offering to use its first 2 minutes |
| No heartbeats found (flat line, noise, wrong column or rate) | Dismissable pop-up; the models can still run |
| Fewer than 20 beats or under 30 s | Side notification: measurements will be rough |
| Sampling rate implies a heart rate outside 36–210 bpm | Side notification with the implied rate |
| PPG upside down | Flipped automatically, side notification |
| Windows joined, gaps filled, clipped samples, true ECG upside down, no true ECG | Side notification |

Any conversion that takes longer than 0.3 s shows a spinner: *Adjusting data to preferred format…*

## What the app does with your file

Every file becomes one continuous 128 Hz signal before a model sees it (`studio/intake/`):

| Step | Script | What happens |
|---|---|---|
| Read | `readers.py` | Delimiter and header detected; non-numeric rows dropped. The signal column is chosen by name (`ppg`, `pleth`, `bvp` for the PPG; `ecg`, `ii` for the ECG, lead II preferred); with several unnamed columns the app asks once. |
| Sampling rate | `fs_adjust.py` | From a time column (seconds or ms; rounded time stamps are fine), a `# fs=125` comment, an `fs` array in an NPZ, `_125Hz` in the file name, the rate remembered for that folder, or else the app asks. The pulse rate implied by that rate must lie at 36–210 bpm. Resampled to 128 Hz with the benchmark's anti-aliased polyphase filter. |
| Window layout | `window_adjust.py` | A 2-D array is rows of windows (any length); rows are joined in order. Sliding windows that overlap are recognised and the overlap kept once. NaN gaps up to 1 s are interpolated; at longer gaps the longest continuous part is used. |
| Checks | `checks.py` | Length limits, flat line, clipping, PPG polarity (an upright pulse rises faster than it falls), pulses present and periodic, true-ECG R peaks and polarity. |
| Align | `pipeline.py` | PPG and true ECG start together (or by their time stamps when both have them) and are trimmed to their overlap. |

Each model then cuts its own windows from the 128 Hz signal (template and Zhu DCT: 10 s, overlap 0.5, tapered
stitching), exactly as `bench.runner.run_record` did in Track 1, and the whole file is always reconstructed;
the plot length only chooses what is drawn.

## Measurements

*Errors vs true ECG* is `bench.metrics.evaluate` with the Track 1 settings (10-s windows, Fréchet distance on at
most 20 windows); each value is the mean over 10-s windows, plus record-level values for R peaks and RMSSD.

| Column | Meaning |
|---|---|
| Heart-rate error | Mean absolute HR error per 10-s window (bpm) |
| R-peak F1 (raw / delay-corrected) | R peaks within 150 ms of the true ones; delay-corrected after removing the record's one constant delay (the Track 1 headline) |
| Delay | That constant delay (ms) |
| R-peak jitter | Inter-quartile range of R-peak timing errors after the delay (ms) |
| Waveform r (raw / best lag) | Pearson r per window, unshifted / at the best shift within ±300 ms |
| Beat shape r | Mean correlation of matched beats, each centred on its own R peak (the Track 1 morphology metric) |
| RMSE, PRD, Fréchet distance | Waveform distances with both signals scaled to [-1, 1] per window |
| RMSSD error | Difference in beat-to-beat variability (ms) |

*Model comparison* needs no true ECG: heart rate of the reconstruction and of the PPG pulses, beats drawn and
pulses found, their agreement (F1 after removing the constant pulse delay), RMSSD of both, flags (flat output,
missing or extra beats), generation time, real-time factor and device. Generation time is the wall-clock time of
the model's own run (windowing, inference, stitching), not loading and not the shared conversion.

## Command line

The same intake, models and scorer without the window:

```bat
.venv\Scripts\python -m studio.cli recording.csv --ecg same
.venv\Scripts\python -m studio.cli ppg.npy --rate 125 --ecg ecg.csv --models template,zhu_dct --out results
```

`--ecg same` takes the true ECG from the PPG file; `--full` lifts the 2-minute limit; `--out` writes each
reconstruction as a CSV (time, ECG at 128 Hz).

## Check that the numbers match Track 1

The tests cover the intake (every supported layout of the same signal must give the same 128 Hz signal), the
engine, the window (driven off-screen) and, with BIDMC, the Track 1 numbers:

```bat
set BIDMC_DIR=C:\path\to\bidmc-ppg-and-respiration-dataset-1.0.0\bidmc_csv
run_tests.bat
```

`tests/test_track1.py` runs the app's own intake and engine on the whole 8-min BIDMC recordings 01, 05 and 27
and compares heart-rate error, delay-corrected R-peak F1, beat-shape r, waveform r, Fréchet distance and RMSSD
error with `results/track1/subjects_cons.csv` of the thesis (agreement to about 10⁻³; Track 1 stored
predictions in float16 before scoring). Zhu DCT is included once its CapnoBase coefficients are installed.
By hand:

```bat
.venv\Scripts\python -m studio.cli %BIDMC_DIR%\bidmc_01_Signals.csv --ecg same --full
```

| BIDMC 01, template baseline | Heart-rate error | R-peak F1 (dc) | Beat shape r | Waveform r (lag) | Fréchet | RMSSD error |
|---|---|---|---|---|---|---|
| Track 1 | 1.207 | 0.988 | 0.920 | 0.786 | 1.643 | 56.5 |
| PPG2ECG Studio 0.1 | 1.21 | 0.988 | 0.920 | 0.786 | 1.643 | 56.5 |

In the window, BIDMC files are 8 min long, so the app offers their first 2 minutes; those numbers differ from
the whole-record values above, as they should.

## Setup flags

`setup.bat` passes its arguments to `setup.py` (`--all` when double-clicked). After the first run,
`.venv\Scripts\python setup.py …` works too.

| Flag | Effect |
|---|---|
| `--all` | All methods available in this version |
| `--zhu`, `--cardiogan`, `--ppgflowecg`, `--rddm`, `--clepgan`, `--chiu`, `--p2e`, `--yalcin` | Only these methods (the template baseline is always installed). The learned models install from v0.2. |
| `--capnobase PATH` | Fit Zhu DCT (and refit the template delay) from CapnoBase instead of downloading |
| `--offline FOLDER` | Install weight files downloaded by hand |
| `--list` | Available and installed models, sizes, measured speed |
| `--remove ID` | Uninstall a model |
| `--check` | Re-run the self-test and speed measurement |
| `--device auto\|gpu\|cpu` | Override the automatic device choice |
| `--shortcut` / `--no-shortcut` | Create the shortcuts / skip them |

## Adding a model

A model is a folder `models/<id>/` with `model.yaml` (manifest: name, family, sampling rate, window length,
overlap, weight files with checksums, citation, licence) and `adapter.py` with `build(ctx)` returning an adapter
that has `load()` and `infer(windows)`: the interface every built-in method uses (`bench/models/base.py`).
Copy `models/_example/` to start; weight files go in `weights/<id>/`. In v0.1 any such folder is picked up at
start; `setup.py --add-model` with its automatic checks (interface, speed, determinism, memory, citation and
licence) arrives in v0.4.

## Repository layout

```
setup.bat  setup.py  run.bat  run_tests.bat  requirements.txt
studio/
  app.py            start the window (python -m studio)
  cli.py            command-line run
  config.py         limits, rates, paths, per-user settings (%LOCALAPPDATA%\PPG2ECG Studio)
  intake/           readers.py, fs_adjust.py, window_adjust.py, checks.py, pipeline.py
  engine/           model registry, device probe, reconstruction (bench.runner)
  scoring/          Track 1 scorer (bench.metrics) + PPG-referenced measures
  ui/               main window, input and model panels, plots, metrics drawer, notifications
bench/              trimmed copy of the thesis benchmark: dsp, metrics, runner, record, the two baselines,
                    BIDMC / CapnoBase readers, beat pairing (kept identical so scores match Track 1)
models/             template/, zhu_dct/, _example/   (model.yaml + adapter.py each)
weights/            installed weights (not in Git)
tools/              fit_baselines.py (Zhu DCT and template delay from CapnoBase)
tests/              intake, engine, window, Track 1 reproduction
```

No data and no weights are kept in Git.

## Roadmap

| Version | Adds |
|---|---|
| 0.1 | Default workflow with the template baseline and Zhu DCT: intake and conversion, window, plots, metrics drawer, notifications, setup with flags |
| 0.2 | The seven learned models (CardioGAN, PPGFlowECG, RDDM, CLEP-GAN, Chiu, P2E-WGAN, Yalcin) through ONNX Runtime (DirectML / CUDA / CPU) and PyTorch, in a separate engine process; device probe; downloads with checksums |
| 0.3 | Export (PNG / PDF + TXT, one PDF, reconstructed ECG files in the input's format), Save / Save as, cache of earlier entries |
| 0.4 | Large-file mode (up to 2 h, preview, time left, Abort), compare with a previous entry, resident engine, `--add-model` checks |

## References

- Zhu Q, Tian X, Wong C-W, Wu M. ECG reconstruction via PPG: a pilot study. *IEEE EMBS International Conference on
  Biomedical and Health Informatics (BHI)*, 2019. [arXiv:1904.10481](https://arxiv.org/abs/1904.10481)
- Pimentel MAF, Johnson AEW, Charlton PH, et al. Toward a robust estimation of respiratory rate from pulse
  oximeters. *IEEE Transactions on Biomedical Engineering* 64(8):1914–1923, 2017. (BIDMC dataset,
  [PhysioNet](https://physionet.org/content/bidmc/1.0.0/))
- Goldberger AL, Amaral LAN, Glass L, et al. PhysioBank, PhysioToolkit, and PhysioNet. *Circulation*
  101(23):e215–e220, 2000.
- Karlen W, Raman S, Ansermino JM, Dumont GA. Multiparameter respiratory rate estimation from the
  photoplethysmogram. *IEEE Transactions on Biomedical Engineering* 60(7):1946–1953, 2013. (CapnoBase)
- Pan J, Tompkins WJ. A real-time QRS detection algorithm. *IEEE Transactions on Biomedical Engineering*
  32(3):230–236, 1985.
- Elgendi M, Norton I, Brearley M, Abbott D, Schuurmans D. Systolic peak detection in acceleration
  photoplethysmograms measured from emergency responders in tropical conditions. *PLoS ONE* 8(10):e76585, 2013.

Third-party software and licences: [THIRD_PARTY.md](THIRD_PARTY.md). The licence of this repository has not
been chosen yet.
