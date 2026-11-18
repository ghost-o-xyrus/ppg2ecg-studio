# Windows check for v0.1 (about 10 minutes)

Use a BIDMC CSV (`bidmc_01_Signals.csv`) and any short file of your own.

| # | Do | Expect |
|---|---|---|
| 1 | Double-click `setup.bat` | Ends with "Done"; self-test shows `Template baseline ok`; shortcuts on desktop and Start menu |
| 2 | `setup.bat --zhu --capnobase <CapnoBase folder>` | "Matches Track 1 (27,661 cycles …)"; `Zhu DCT ok` in the self-test |
| 3 | `run_tests.bat` with `set BIDMC_DIR=…\bidmc_csv` first | All tests pass (none skipped except as noted) |
| 4 | Start the app from the shortcut | Window opens in 1–2 s; both model dots turn green; Run all is disabled with tooltip "Drop a PPG file first" |
| 5 | Drop `bidmc_01_Signals.csv` on the PPG tab | Spinner "Adjusting data to preferred format…"; pop-up "Long file" → *Use the first 2 minutes*; file card: 125 Hz (time column) → 128 Hz; True ECG tab fills itself from column II with a side notification |
| 6 | Run all | Both traces appear, generation times on the rows and plots; metrics drawer opens; both tabs filled, best values bold |
| 7 | Show 5 s from 30 s; scroll, drag, double-click on the plot | Window follows; double-click returns to 30–35 s; status bar reads time and value |
| 8 | ⚙ → untick each option | Traces shift / R-peak lines disappear / filtering changes, no error |
| 9 | ✕ on the True ECG card, Run all | No ECG row; side notification "No true ECG given"; Errors tab shows the message |
| 10 | Drop a file shorter than 10 s | Big central pop-up "File too short"; the previous file stays |
| 11 | Drop a one-column CSV without a rate | Rate question; answer 125; next file from the same folder is not asked again |
| 12 | `.venv\Scripts\python -m studio.cli <bidmc_01> --ecg same --full` | Template row: 1.21 / 0.988 / 0.920 (heart-rate error, R-peak F1 dc, beat shape r) |
