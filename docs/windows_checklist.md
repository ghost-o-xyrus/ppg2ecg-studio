# Windows check for v0.2 (about 20 minutes, plus the downloads)

Use a BIDMC CSV (`bidmc_01_Signals.csv`) and any short file of your own.

| # | Do | Expect |
|---|---|---|
| 1 | Double-click `setup.bat` | Every model is downloaded or reported with a clear reason; the ONNX exports print a difference of about 1e-6; self-test lists nine models `ok` with their device and speed; ends with "Done"; shortcuts on desktop and Start menu |
| 2 | `setup.bat --zhu --capnobase <CapnoBase folder>` | "Matches Track 1 (27,661 cycles …)"; `Zhu DCT ok` in the self-test |
| 3 | `run_tests.bat` with `set BIDMC_DIR=…\bidmc_csv` first | All tests pass (the Track 1 test takes about 6 minutes) |
| 4 | Start the app from the shortcut | Window opens in 1–2 s; the dots turn green one by one (template and Zhu DCT first, RDDM and PPGFlowECG last); Run all is disabled with tooltip "Drop a PPG file first" |
| 5 | Drop `bidmc_01_Signals.csv` on the PPG tab | Spinner "Adjusting data to preferred format…"; pop-up "Long file" → *Use the first 2 minutes*; file card: 125 Hz (time column) → 128 Hz; True ECG tab fills itself from column II with a side notification |
| 6 | Run all | Traces appear model by model, fastest first; generation times on the rows and plots; metrics drawer opens with nine rows; both tabs filled, best values bold |
| 7 | Show 5 s from 30 s; scroll, drag, double-click on the plot | Window follows; double-click returns to 30–35 s; status bar reads time and value |
| 8 | ⚙ → untick each option | Traces shift / R-peak lines disappear / filtering changes, no error |
| 9 | ✕ on the True ECG card, Run all | No ECG row; side notification "No true ECG given"; Errors tab shows the message |
| 10 | Drop a file shorter than 10 s | Big central pop-up "File too short"; the previous file stays |
| 11 | Drop a one-column CSV without a rate | Rate question; answer 125; next file from the same folder is not asked again |
| 12 | `.venv\Scripts\python -m studio.cli <bidmc_01> --ecg same --full` | Template row: 1.21 / 0.988 / 0.920; CLEP-GAN row: 0.78 / 0.991 / 0.965 (heart-rate error, R-peak F1 dc, beat shape r) |
| 13 | `setup.bat --list` | Nine models, flags, sizes, device and "10 s of PPG in" for each |
