# Third-party software

Installed by `setup.bat` into the private `.venv`; nothing below is copied into this repository.

| Package | Use | Licence |
|---|---|---|
| [PySide6](https://doc.qt.io/qtforpython-6/) (Qt for Python) | Window, widgets, threads | LGPL-3.0 (used unmodified, as a separate library) |
| [pyqtgraph](https://www.pyqtgraph.org/) | Live stacked plots | MIT |
| [NumPy](https://numpy.org/) | Arrays | BSD-3-Clause |
| [SciPy](https://scipy.org/) | Filters, resampling, DCT | BSD-3-Clause |
| [pandas](https://pandas.pydata.org/) | Reading CSV/TXT | BSD-3-Clause |
| [PyYAML](https://pyyaml.org/) | Model manifests | MIT |
| [pytest](https://pytest.org/) | Tests | MIT |
| [uv](https://github.com/astral-sh/uv) | Fetches Python 3.11 and the packages during setup | MIT or Apache-2.0 |
| [h5py](https://www.h5py.org/) (only for `--capnobase` with `.mat` files) | Reads MATLAB v7.3 files | BSD-3-Clause |

## Methods

| Method | Source | Weights |
|---|---|---|
| Template baseline | This work (PPG2ECG thesis, Track 1) | One number (pulse arrival time, 0.3828125 s), fitted on CapnoBase |
| Zhu DCT | Re-implemented from Zhu et al., 2019 (arXiv:1904.10481) | 1,300 coefficients, fitted by us on CapnoBase (`tools/fit_baselines.py`) |

The learned models of v0.2 are fetched from their authors' own links at setup and are never re-hosted; their
licences will be listed here before they are added.

## Data

No dataset is included or redistributed. The tests read BIDMC from a folder you point them to; the Zhu DCT fit
reads CapnoBase from a folder you point it to.
