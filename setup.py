"""PPG2ECG Studio setup: install models and their weights, list them, check them.

    python setup.py --all                      everything available in this version (what setup.bat does)
    python setup.py --zhu                      only these methods (the template baseline is always installed)
    python setup.py --zhu --capnobase PATH     fit Zhu DCT locally from CapnoBase instead of downloading it
    python setup.py --offline FOLDER           install weight files downloaded by hand
    python setup.py --list                     available and installed models, sizes, measured speed
    python setup.py --remove zhu_dct           uninstall a model and free its space
    python setup.py --check                    re-run the self-test and speed measurement
    python setup.py --device auto|gpu|cpu      override the automatic device choice
    python setup.py --shortcut                 (re)create the Start-menu and desktop shortcuts

Our weights (Zhu DCT, Chiu, CLEP-GAN, P2E-WGAN, Yalcin) come from this repository's GitHub Release; the
authors' released weights (CardioGAN, PPGFlowECG, RDDM) come from the authors' own links, and every
author's code from their repository at a pinned commit. Nothing of theirs is re-hosted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("TQDM_DISABLE", "1")          # PPGFlowECG's sampler would print a progress bar
sys.path.insert(0, str(ROOT))

from studio import config, engine  # noqa: E402

# flag -> model id, for every method of the thesis
FLAGS = {flag: mid for mid, flag in config.SETUP_FLAGS.items()}
SPEED_FILE = config.WEIGHTS_DIR / "speed.json"


def say(msg=""):
    print(msg, flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def specs_by_id():
    return {s.id: s for s in engine.discover()}


# ---------------------------------------------------------------- installing weights
def _verify(path, expected):
    if expected and sha256(path) != expected:
        raise ValueError(f"{path.name}: checksum does not match the manifest")


def _download(url, dest, retries=3):
    tmp = dest.with_suffix(dest.suffix + ".part")
    for k in range(retries):
        try:
            have = tmp.stat().st_size if tmp.exists() else 0
            req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
            with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "ab" if have else "wb") as f:
                if have and r.status != 206:          # server ignored the range: start over
                    f.seek(0)
                    f.truncate()
                shutil.copyfileobj(r, f, 1 << 20)
            tmp.replace(dest)
            return
        except Exception as e:
            say(f"    download failed ({e}); retry {k + 1}/{retries}")
            time.sleep(2 * (k + 1))
    raise RuntimeError(f"could not download {url}")


def ensure_h5py():
    try:
        import h5py  # noqa: F401
        return True
    except ImportError:
        pass
    say("  CapnoBase .mat files need h5py; installing it into this environment ...")
    cmds = [[sys.executable, "-m", "pip", "install", "h5py"]]
    uv = shutil.which("uv") or str(ROOT / ".tools" / "uv" / "uv.exe")
    if os.path.exists(uv):
        cmds.insert(0, [uv, "pip", "install", "--python", sys.executable, "h5py"])
    for c in cmds:
        if subprocess.run(c).returncode == 0:
            return True
    return False


def _capnobase_needs_h5py(root):
    """True when some subject exists only as a .mat file (the CSV/TAB copies are read first)."""
    root = Path(root)
    tables = {f.name.split("_")[0] for pat in ("*_8min_signal.csv", "*_8min_signal.tab") for f in root.rglob(pat)}
    return any(f.name.split("_")[0] not in tables for f in root.rglob("*_8min.mat"))


def _uv():
    for c in (shutil.which("uv"), ROOT / ".tools" / "uv" / "uv.exe", ROOT / ".tools" / "uv" / "bin" / "uv.exe"):
        if c and os.path.exists(c):
            return str(c)
    return None


@contextmanager
def temp_env(packages, python="3.11"):
    """A throw-away Python environment (with its own package cache) for one-off setup tools, such as
    TensorFlow for the CardioGAN conversion; everything is deleted afterwards."""
    uv = _uv()
    if uv is None:
        raise RuntimeError("uv is needed for this step (run setup.bat once)")
    d = Path(tempfile.mkdtemp(prefix="ppg2ecg-setup-"))
    env = dict(os.environ, UV_CACHE_DIR=str(d / "cache"))
    try:
        subprocess.run([uv, "venv", "-q", "--python", python, str(d / "venv")], env=env, check=True)
        py = d / "venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        subprocess.run([uv, "pip", "install", "-q", "--python", str(py), *packages], env=env, check=True)
        yield py
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _github_zip(url, commit):
    slug = url.rstrip("/").removesuffix(".git").split("github.com/", 1)[1]
    return f"https://codeload.github.com/{slug}/zip/{commit}"


def fetch_repo(spec, dest=None):
    """The authors' code at the pinned commit -> weights/<id>/repo (a GitHub source archive, no git needed)."""
    r = spec.repo
    dest = Path(dest or spec.weights_dir / "repo")
    stamp = dest / ".commit"
    if stamp.exists() and stamp.read_text().strip() == r["commit"]:
        return dest
    say(f"    authors' code: {r['url']} at {r['commit'][:10]}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    z = dest.parent / (dest.name + ".zip")
    _download(_github_zip(r["url"], r["commit"]), z)
    tmp = dest.parent / (dest.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(z) as f:
        f.extractall(tmp)
    top = [p for p in tmp.iterdir() if p.is_dir()]
    shutil.rmtree(dest, ignore_errors=True)
    shutil.move(str(top[0] if len(top) == 1 else tmp), str(dest))
    shutil.rmtree(tmp, ignore_errors=True)
    z.unlink(missing_ok=True)
    stamp.write_text(r["commit"] + "\n")
    return dest


def _flag(model_id):
    return next((f for f, m in FLAGS.items() if m == model_id), model_id)


def _gdrive_folder(spec, entries):
    """Files from a shared Google Drive folder (RDDM). Drive rate-limits automated downloads: retry with
    back-off, then print the link for --offline."""
    url = entries[0]["url"].replace("gdrive-folder://", "https://drive.google.com/drive/folders/")
    need = [w["file"] for w in entries if not (spec.weights_dir / w["file"]).exists()]
    if not need:
        return True
    say(f"    downloading {', '.join(need)} from the authors' Google Drive folder ...")
    try:
        with temp_env(["gdown==5.2.0"]) as py:
            for k in range(3):
                out = Path(tempfile.mkdtemp(prefix="ppg2ecg-drive-"))
                try:
                    subprocess.run([str(py), "-m", "gdown", "--folder", url, "-O", str(out), "--remaining-ok"])
                    for f in need:
                        hit = next(out.rglob(f), None)
                        if hit is not None:
                            shutil.move(str(hit), str(spec.weights_dir / f))
                finally:
                    shutil.rmtree(out, ignore_errors=True)
                need = [f for f in need if not (spec.weights_dir / f).exists()]
                if not need:
                    return True
                say(f"    Google Drive did not deliver everything (retry {k + 1}/3) ...")
                time.sleep(20 * (k + 1))
    except Exception as e:  # noqa: BLE001
        say(f"    {e}")
    say(f"  {spec.name}: Google Drive refused the download. Download these files by hand from\n"
        f"    {url}\n    ({', '.join(need)}) into one folder and run:  setup.py --{_flag(spec.id)} --offline FOLDER")
    return False


def _total_ram_gb():
    if sys.platform != "win32":
        return None
    import ctypes

    class MS(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
            (n, ctypes.c_ulonglong) for n in ("total", "avail", "tpage", "apage", "tvirt", "avirt", "aext")]
    m = MS()
    m.len = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.total / 2 ** 30


def _room_for(spec, force):
    """Disk and RAM check before a large model (PPGFlowECG: 3.8 GB download, about 5 GB RAM while running)."""
    disk = float(spec.raw.get("setup_disk_gb", 0))
    ram = float(spec.raw.get("min_ram_gb", 0))
    config.WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(config.WEIGHTS_DIR).free / 2 ** 30
    total = _total_ram_gb()
    problems = []
    if disk and free < disk:
        problems.append(f"needs {disk:.0f} GB of free disk during setup, {free:.1f} GB free")
    if ram >= 2 and total is not None and total < ram:
        problems.append(f"needs about {ram:.0f} GB of RAM, this computer has {total:.1f} GB")
    if problems and not force:
        say(f"  {spec.name}: skipped - " + "; ".join(problems) + ". Add --force to install it anyway.")
        return False
    return True


def _hook(spec, offline):
    """The model's own setup step (adapter.py: setup(ctx, say)) once its files are in place."""
    mod = engine.registry._import_adapter(spec)
    if hasattr(mod, "setup"):
        ctx = SimpleNamespace(spec=spec, weights_dir=spec.weights_dir, device="cpu", info={}, offline=offline,
                              temp_env=temp_env, download=_download, root=ROOT)
        mod.setup(ctx, say)


def _size(d):
    return sum(f.stat().st_size for f in Path(d).rglob("*") if f.is_file()) if Path(d).exists() else 0


def install(spec, offline=None, capnobase=None, force=False):
    """-> True when the model's weights are in place."""
    if spec.always_installed or not spec.weights:
        say(f"  {spec.name}: built in")
        return True
    if spec.id != "zhu_dct":
        if spec.installed and not force:
            say(f"  {spec.name}: installed")
            return True
        if not _room_for(spec, force):
            return False
        say(f"  {spec.name}:")
    spec.weights_dir.mkdir(parents=True, exist_ok=True)
    if spec.id == "zhu_dct" and capnobase:
        if not _capnobase_needs_h5py(capnobase) or ensure_h5py():
            from tools import fit_baselines
            say(f"  {spec.name}: fitting on CapnoBase in {capnobase} (about a minute) ...")
            try:
                fit_baselines.main(["--capnobase", str(capnobase), "--out", str(config.WEIGHTS_DIR)])
            except (Exception, SystemExit) as e:
                say(f"  {spec.name}: fitting failed: {e}")
                return False
        return not spec.missing_weights()
    drive = [w for w in spec.weights if str(w.get("url", "")).startswith("gdrive-folder://")]
    for w in spec.weights:
        dest = spec.weights_dir / w["file"]
        if dest.exists():
            try:
                _verify(dest, w.get("sha256"))
                continue
            except ValueError as e:
                say(f"  {spec.name}: {e}; fetching it again")
                dest.unlink()
        if offline:
            hits = list(Path(offline).rglob(w["file"]))
            if hits:
                shutil.copy2(hits[0], dest)
                _verify(dest, w.get("sha256"))
                say(f"  {spec.name}: {w['file']} copied from {hits[0].parent}")
                continue
        url = str(w.get("url", ""))
        if url.startswith("gdrive-folder://"):
            continue
        if url.startswith("release://"):
            url = config.RELEASE_URL.rstrip("/") + "/" + url[len("release://"):]
        if "YOUR_GITHUB_USERNAME" in url or not url:
            say(f"  {spec.name}: no download location is configured yet for {w['file']}.")
            if "fit" in w:
                say(f"    Instead: {w['fit']}")
            return False
        say(f"  {spec.name}: downloading {w['file']} ...")
        try:
            _download(url, dest)
            _verify(dest, w.get("sha256"))
        except Exception as e:
            say(f"  {spec.name}: {e}")
            if "fit" in w:
                say(f"    Instead: {w['fit']}")
            say(f"    Or download it by hand from {url} and run  setup.py --{_flag(spec.id)} --offline FOLDER")
            return False
    if drive and not _gdrive_folder(spec, drive):
        return False
    if spec.id == "zhu_dct":
        return True
    try:
        for w in spec.weights:
            f = spec.weights_dir / w["file"]
            if w.get("sha256") and f.exists():
                _verify(f, w["sha256"])
            if w.get("unzip") and f.exists():
                with zipfile.ZipFile(f) as z:
                    z.extractall(spec.weights_dir)
                f.unlink()
        if spec.repo:
            fetch_repo(spec)
        _hook(spec, offline)
    except Exception as e:  # noqa: BLE001 - one model failing must not stop the others
        msg = str(e).strip().splitlines()[0][:300] if str(e).strip() else ""
        say(f"  {spec.name}: {type(e).__name__}: {msg}")
        return False
    missing = spec.missing_weights()
    if missing:
        say(f"  {spec.name}: still missing {', '.join(missing)}")
        return False
    say(f"    installed ({_size(spec.weights_dir) / 1e6:.0f} MB)")
    return True


# ---------------------------------------------------------------- self-test and speed
GPU_TOL = 1e-3          # max |GPU - CPU| output accepted before a model is kept on the CPU


def _timed(lm, ppg, peaks):
    engine.run_model(lm, ppg, peaks)                  # warm-up (first-run allocations, graph optimisation)
    return engine.run_model(lm, ppg, peaks)


def _pick_device(s, ppg, peaks, setting):
    """ONNX models: run on the GPU and on the CPU; keep the GPU only if it works, agrees with the CPU within
    GPU_TOL and is faster. The choice is remembered (weights/device.json). -> (LoadedModel, Result, note)."""
    from studio.engine import device as devmod
    from studio.engine import onnx_run
    cpu = engine.load_model(s, "cpu")
    if cpu.status != engine.READY:
        return cpu, None, ""
    rc = _timed(cpu, ppg, peaks)
    if s.runtime != "onnx" or devmod.probe(setting) != "gpu":
        return cpu, rc, ""
    onnx_run.remember(s.id, "auto")
    gpu = engine.load_model(s, "gpu")
    if gpu.status != engine.READY or gpu.device == "CPU":
        onnx_run.remember(s.id, "cpu")
        return cpu, rc, "GPU refused the model; CPU"
    try:
        rg = _timed(gpu, ppg, peaks)
    except Exception as e:  # noqa: BLE001
        onnx_run.remember(s.id, "cpu")
        return cpu, rc, f"GPU failed ({type(e).__name__}); CPU"
    if getattr(gpu.adapter, "device_label", "CPU") == "CPU":       # a GPU run failed and fell back to the CPU
        onnx_run.remember(s.id, "cpu")
        return cpu, rc, "an operation failed on the GPU; CPU"
    diff = float(np.max(np.abs(rg.pred - rc.pred)))
    if diff > GPU_TOL:
        onnx_run.remember(s.id, "cpu")
        return cpu, rc, f"GPU output differs by {diff:.1e}; CPU"
    if rg.gen_s > rc.gen_s:
        onnx_run.remember(s.id, "cpu")
        return cpu, rc, f"CPU faster than GPU ({rc.gen_s:.3f} s vs {rg.gen_s:.3f} s)"
    onnx_run.remember(s.id, "gpu")
    return gpu, rg, f"GPU {rg.gen_s:.3f} s vs CPU {rc.gen_s:.3f} s, max diff {diff:.0e}"


def check(specs, setting="auto"):
    """Self-test and speed of every installed model on a 30-s synthetic PPG; picks each ONNX model's device."""
    from bench import dsp
    from bench.datasets import make_synthetic
    r = make_synthetic(subject="check", duration=30.0, fs=128.0, hr=72.0, seed=3)
    ppg = dsp.resample(r.ppg, r.ppg_fs, config.WORK_FS)
    peaks = dsp.detect_ppg_peaks(ppg, config.WORK_FS)
    speed = {}
    ok_all = True
    for s in specs:
        if not s.installed:
            say(f"  {s.name:<20} not installed")
            continue
        try:
            lm, res, note = _pick_device(s, ppg, peaks, setting)
        except Exception as e:  # noqa: BLE001
            say(f"  {s.name:<20} FAIL  {type(e).__name__}: {e}")
            ok_all = False
            continue
        if res is None:
            say(f"  {s.name:<20} {lm.status}  {lm.error[:200]}")
            ok_all = False
            continue
        good = len(res.pred) == len(ppg) and bool(np.isfinite(res.pred).all())
        speed[s.id] = dict(rt_factor=res.rt_factor, load_s=lm.load_seconds, device=lm.device,
                           beats=res.vs_ppg["beats"], pulses=res.vs_ppg["pulses"])
        say(f"  {s.name:<20} {'ok  ' if good else 'FAIL'} {lm.device:<15} load {lm.load_seconds:5.2f} s   "
            f"10 s of PPG in {res.rt_factor * 10:6.3f} s   beats {res.vs_ppg['beats']}/{res.vs_ppg['pulses']}"
            + (f"   ({note})" if note else ""))
        ok_all &= good
    config.WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    old = json.loads(SPEED_FILE.read_text()) if SPEED_FILE.exists() else {}
    old.update(speed)
    SPEED_FILE.write_text(json.dumps(old, indent=1))
    return ok_all


def list_models():
    speed = json.loads(SPEED_FILE.read_text()) if SPEED_FILE.exists() else {}
    say(f"  {'id':<12} {'name':<14} {'flag':<14} {'installed':<10} {'size':>8}  {'device':<15} {'10 s of PPG in':>14}")
    for s in specs_by_id().values():
        sp = speed.get(s.id, {})
        rt = sp.get("rt_factor")
        flag = "--" + config.SETUP_FLAGS[s.id] if s.id in config.SETUP_FLAGS else "(built in)"
        say(f"  {s.id:<12} {s.name:<14} {flag:<14} {'yes' if s.installed else 'no':<10} "
            f"{_size(s.weights_dir) / 1e6:>5.0f} MB  {sp.get('device', '-') if s.installed else '-':<15} "
            f"{(f'{rt * 10:.3f} s') if rt and s.installed else '-':>14}")


# ---------------------------------------------------------------- shortcuts
def make_shortcuts():
    if sys.platform != "win32":
        say("  shortcuts are created on Windows only")
        return
    pyw = ROOT / ".venv" / "Scripts" / "pythonw.exe"
    if not pyw.exists():
        pyw = Path(sys.executable).with_name("pythonw.exe")
    ps = r"""
$s = New-Object -ComObject WScript.Shell
foreach ($dir in @([Environment]::GetFolderPath('Programs'), [Environment]::GetFolderPath('Desktop'))) {
  $l = $s.CreateShortcut((Join-Path $dir 'PPG2ECG Studio.lnk'))
  $l.TargetPath = '%s'
  $l.Arguments = '-m studio'
  $l.WorkingDirectory = '%s'
  $l.Description = 'PPG2ECG Studio'
  $l.Save()
}
""" % (str(pyw).replace("'", "''"), str(ROOT).replace("'", "''"))
    r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps])
    say("  Start-menu and desktop shortcuts created" if r.returncode == 0 else "  could not create shortcuts")


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(prog="setup.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="all methods (what setup.bat does)")
    for flag in FLAGS:
        ap.add_argument(f"--{flag}", action="store_true", help=f"install {FLAGS[flag]}")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--remove", metavar="ID")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--device", choices=["auto", "gpu", "cpu"])
    ap.add_argument("--offline", metavar="FOLDER")
    ap.add_argument("--capnobase", metavar="PATH", help="fit Zhu DCT from CapnoBase instead of downloading it")
    ap.add_argument("--add-model", metavar="FOLDER")
    ap.add_argument("--force", action="store_true", help="reinstall, and install even when disk or RAM look short")
    ap.add_argument("--shortcut", action="store_true")
    ap.add_argument("--no-shortcut", action="store_true")
    a = ap.parse_args(argv)
    specs = specs_by_id()
    settings = config.Settings()

    if a.device:
        settings.set("device", a.device)
        say(f"Device set to {a.device}. The ONNX models (CardioGAN, Chiu, CLEP-GAN, P2E-WGAN, Yalcin) follow it; "
            "RDDM and PPGFlowECG always run on the CPU.")
    if a.list:
        list_models()
        return 0
    if a.remove:
        s = specs.get(a.remove) or specs.get(FLAGS.get(a.remove, ""))
        if s is None:
            sys.exit(f"unknown model '{a.remove}'")
        if s.always_installed:
            sys.exit(f"{s.name} is built in and cannot be removed")
        if s.weights_dir.exists():
            shutil.rmtree(s.weights_dir)
        say(f"Removed {s.name}.")
        return 0
    if a.add_model:
        say("--add-model (with its automatic checks) arrives in a later version. For now, copy the folder to "
            "models\\<id>\\ (see models\\_example) and put its weights in weights\\<id>\\.")
        return 1

    wanted = [mid for flag, mid in FLAGS.items() if a.all or getattr(a, flag)]
    if a.capnobase and "zhu_dct" not in wanted:
        wanted.append("zhu_dct")
    ok = True
    not_installed = []
    if wanted or a.offline:
        say("Installing models")
        install(specs["template"])
        for mid in wanted or [s.id for s in specs.values()]:
            if mid in specs and not install(specs[mid], a.offline, a.capnobase, a.force):
                not_installed.append(specs[mid].name)
        say()
    if wanted or a.check or a.offline:
        say("Self-test and speed on this computer (30 s synthetic PPG)")
        ok &= check(list(engine.discover()), settings.get("device", "auto"))
        say()
    if (a.all and not a.no_shortcut) or a.shortcut:
        make_shortcuts()
    if not (wanted or a.check or a.offline or a.shortcut or a.device):
        ap.print_help()
        return 0
    list_models()
    if not_installed:
        say(f"\nNot installed this time: {', '.join(not_installed)} (see the messages above). The app works "
            "without them; run setup again with the same flag later.")
    if not ok:
        say("\nThe self-test FAILED for a model above.")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
