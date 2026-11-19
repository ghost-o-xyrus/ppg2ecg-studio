# Example recordings

Fifteen short real recordings for trying the app: drop any of them on the *PPG signal* tab. Most contain the
true ECG as well, which the app picks up by itself. Together they cover heart rates from 66 to 125 bpm, every
file layout the app reads, and each notification it can show.

| File | Length | Layout | What it shows |
|---|---|---|---|
| `bidmc_25_full_8min.csv` | 8 min | The original BIDMC file (time, RESP, PLETH, V, AVR, II) | Long-file pop-up (*Use the first 2 minutes*); PPG from `PLETH`, true ECG from `II`; 125 bpm |
| `bidmc_13_ppg_2min.csv` + `bidmc_13_ecg_2min.csv` | 2 min | Two files with time columns; the ECG file starts 3 s later | A separate true ECG, aligned by the time stamps; 66 bpm |
| `bidmc_45_ppg_ecg_90s.npz` | 90 s | NumPy archive with `ppg`, `ecg` and `fs` | Rate from the archive; 66 bpm |
| `bidmc_03_ppg_ecg_gap_2min.csv` | 2 min | Time in milliseconds, PPG and ECG | A 0.6-s gap of missing values is filled; the true ECG is recorded upside down; 77 bpm |
| `bidmc_41_ppg_ecg_2min.csv` | 2 min | Time, PPG and ECG | A hard case: about 40 % of the heartbeats leave no clear pulse in the PPG (113 pulses against 189 R peaks) |
| `mimic2_ppg_abp_ecg_2min.csv` | 2 min | Time, PPG, arterial pressure and ECG | A second source (MIMIC-II); the pressure column is ignored; 117 bpm |
| `mimic2_ppg_ecg_75s_250Hz.csv` | 75 s | Time, PPG and ECG at 250 Hz | Conversion from 250 Hz; 114 bpm |
| `bidmc_08_ppg_90s_125Hz.npy` | 90 s | NumPy array, PPG only | Rate from the file name; no true ECG; 99 bpm |
| `bidmc_16_ppg_windows_4s_125Hz.npy` | 2 min | 30 rows of 4-s windows | Windows joined into one signal; 110 bpm |
| `bidmc_21_ppg_60s_125Hz.txt` | 60 s | One bare column of numbers | Rate from the file name; 121 bpm |
| `bidmc_19_ppg_30s.csv` | 30 s | A `# fs=125` comment line, then a `ppg` column | Rate from the comment; 73 bpm |
| `bidmc_27_ppg_upside_down_60s.csv` | 60 s | Time and PPG, sign flipped | *PPG flipped* notification; 88 bpm |
| `bidmc_09_ppg_15s.csv` | 15 s | Time and PPG | *Few beats* notification (19 beats) |
| `bidmc_22_ppg_8s.csv` | 8 s | Time and PPG | *File too short* pop-up (the minimum is 10 s) |

All recordings are at 125 Hz except the 250-Hz file. `tools/make_examples.py` rebuilds this folder from the two
datasets and documents every cut; `tests/test_examples.py` checks that each file is read as described above.

## Sources and licences

The files are excerpts of two public datasets, redistributed under their licences with attribution:

- **BIDMC PPG and Respiration Dataset** (files `bidmc_*`): Pimentel MAF, Johnson AEW, Charlton PH et al. Toward a robust estimation of respiratory rate from pulse oximeters.
  *IEEE Transactions on Biomedical Engineering* 64(8):1914–1923, 2017. Data from
  [PhysioNet](https://physionet.org/content/bidmc/1.0.0/) (Goldberger AL et al., *Circulation* 101(23):e215–e220,
  2000), licensed under the Open Data Commons Attribution License v1.0.
- **Cuff-Less Blood Pressure Estimation** (files `mimic2_*`, records 1 and 2 of `Part_1.mat`): Kachuee M,
  Kiani MM, Mohammadzade H, Shabany M. [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/340/cuff+less+blood+pressure+estimation),
  2015, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); the recordings come from the
  MIMIC-II waveform database.

Changes made: cut to the lengths above and rewritten in the layouts above. In addition,
`bidmc_27_ppg_upside_down_60s.csv` has its PPG sign flipped, `bidmc_03_ppg_ecg_gap_2min.csv` has 0.6 s of PPG
(50.0–50.6 s) replaced by missing values, and `mimic2_ppg_ecg_75s_250Hz.csv` was resampled from 125 Hz to
250 Hz. All other values are as published.
