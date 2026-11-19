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
| [PyTorch](https://pytorch.org/) (CPU build) | RDDM, PPGFlowECG; ONNX export at setup | BSD-3-Clause |
| [ONNX Runtime](https://onnxruntime.ai/) (DirectML build) | CardioGAN, Chiu, CLEP-GAN, P2E-WGAN, Yalcin | MIT |
| [ONNX](https://onnx.ai/) | Model files written at setup | Apache-2.0 |
| [einops](https://github.com/arogozhnikov/einops), [tqdm](https://github.com/tqdm/tqdm) | Imported by the PPGFlowECG code | MIT; MPL-2.0 and MIT |
| [pytest](https://pytest.org/) | Tests | MIT |
| [uv](https://github.com/astral-sh/uv) | Fetches Python 3.11 and the packages during setup | MIT or Apache-2.0 |
| [h5py](https://www.h5py.org/) (only for `--capnobase` with `.mat` files) | Reads MATLAB v7.3 files | BSD-3-Clause |

Used during setup only, in a temporary environment that is deleted afterwards:

| Package | Use | Licence |
|---|---|---|
| [TensorFlow](https://www.tensorflow.org/) 2.15 | Restores the CardioGAN checkpoint | Apache-2.0 |
| [tf2onnx](https://github.com/onnx/tensorflow-onnx) | Converts CardioGAN to ONNX | Apache-2.0 |
| [gdown](https://github.com/wkentaro/gdown) | Downloads the RDDM weights from Google Drive | MIT |

## Methods

The authors' code is downloaded from their repositories at the commit listed below (the commit Track 1 used)
into `weights/<id>/repo/`; their released weights come from their own links. Neither is re-hosted. Our own
weights (Zhu DCT and the four models we trained with the authors' code) are assets of this repository's
GitHub Release.

| Method | Authors' code (commit) | Code licence | Weights |
|---|---|---|---|
| Template baseline | This work (PPG2ECG thesis, Track 1) | this repository | One number (pulse arrival time, 0.3828125 s), fitted on CapnoBase |
| Zhu DCT | Re-implemented from Zhu et al., 2019 (arXiv:1904.10481) | this repository | 1,300 coefficients, fitted by us on CapnoBase (`tools/fit_baselines.py`); Release asset |
| CardioGAN | [pritamqu/ppg2ecg-cardiogan](https://github.com/pritamqu/ppg2ecg-cardiogan) (`d7239b0`) | CC BY-NC 4.0 (non-commercial) | Authors' [GitHub release](https://github.com/pritamqu/ppg2ecg-cardiogan/releases/tag/model_weights), same licence; converted to ONNX on your computer |
| PPGFlowECG | [PKUDigitalHealth/PPGFlowECG](https://github.com/PKUDigitalHealth/PPGFlowECG) (`56b2cd2`) | none stated in the repository or on the weights page | Authors' [Hugging Face page](https://huggingface.co/XiaochengFang/PPGFlowECG); reduced to the inference parts on your computer |
| RDDM | [DebadityaQU/RDDM](https://github.com/DebadityaQU/RDDM) (`7d53488`) | MIT | Authors' [Google Drive folder](https://drive.google.com/drive/folders/1Z7JQ5VdTrekx4lbARJNIUiR5D-4Kz7wg), linked from their README; no separate licence stated |
| CLEP-GAN | [Mathematics-Analytics-Data-Science-Lab/CLEP-GAN](https://github.com/Mathematics-Analytics-Data-Science-Lab/CLEP-GAN) (`9b35d05`) | GPL-3.0 | Trained by us with the authors' code (Kaggle job k91); Release asset, fp16 |
| Chiu | [james77777778/ppg2ecg-pytorch](https://github.com/james77777778/ppg2ecg-pytorch) (`af1bd25`) | MIT | Trained by us with the authors' code (Kaggle job k90); Release asset |
| P2E-WGAN | [khuongav/P2E-WGAN-ecg-ppg-reconstruction](https://github.com/khuongav/P2E-WGAN-ecg-ppg-reconstruction) (`7a2b4e0`) | MIT | Trained by us with the authors' code on MIMIC-II (Kaggle job k92m); Release asset |
| Yalcin | [m1237/self-supervised-ppg-to-ecg](https://github.com/m1237/self-supervised-ppg-to-ecg) (`99cf07a`) | MIT | Trained by us with the authors' code (Kaggle job k93); Release asset |

The full commits and the SHA-256 of every weight file are in `models/<id>/model.yaml`.

## Data

No dataset is included or redistributed. The tests read BIDMC from a folder you point them to; the Zhu DCT fit
reads CapnoBase from a folder you point it to.
