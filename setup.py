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

v0.1 ships the two NumPy methods (template baseline, Zhu DCT). The flags of the seven learned models are
accepted now and install them from v0.2 on.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from studio import config, engine  # noqa: E402

# flag -> model id, for every method of the thesis (v0.1 installs the first two)
FLAGS = {"zhu": "zhu_dct", "cardiogan": "cardiogan", "ppgflowecg": "ppgflowecg", "rddm": "rddm",
         "clepgan": "clep_gan", "chiu": "chiu2020", "p2e": "p2e_wgan", "yalcin": "yalcin2026"}
LATER = {"cardiogan", "ppgflowecg", "rddm", "clep_gan", "chiu2020", "p2e_wgan", "yalcin2026"}
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


def install(spec, offline=None, capnobase=None):
    """-> True when the model's weights are in place."""
    if spec.always_installed or not spec.weights:
        say(f"  {spec.name}: built in")
        return True
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
            say(f"    Or download it by hand from {url} and run  setup.py --offline FOLDER")
            return False
    return True


# ---------------------------------------------------------------- self-test and speed
def check(specs, device="cpu"):
    from bench import dsp
    from bench.datasets import make_synthetic
    r = make_synthetic(subject="check", duration=30.0, fs=128.0, hr=72.0, seed=3)
    ppg = dsp.resample(r.ppg, r.ppg_fs, config.WORK_FS)
    peaks = dsp.detect_ppg_peaks(ppg, config.WORK_FS)
    speed = {}
    ok_all = True
    for s in specs:
        lm = engine.load_model(s, device)
        if lm.status != engine.READY:
            say(f"  {s.name:<20} {lm.status}  {lm.error}")
            ok_all &= lm.status == engine.MISSING
            continue
        res = engine.run_model(lm, ppg, peaks)
        good = len(res.pred) == len(ppg) and bool(np.isfinite(res.pred).all())
        speed[s.id] = dict(rt_factor=res.rt_factor, load_s=lm.load_seconds, device=lm.device,
                           beats=res.vs_ppg["beats"], pulses=res.vs_ppg["pulses"])
        say(f"  {s.name:<20} {'ok  ' if good else 'FAIL'} load {lm.load_seconds:5.2f} s   "
            f"10 s of PPG in {res.rt_factor * 10:6.3f} s   beats {res.vs_ppg['beats']}/{res.vs_ppg['pulses']}")
        ok_all &= good
    config.WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    old = json.loads(SPEED_FILE.read_text()) if SPEED_FILE.exists() else {}
    old.update(speed)
    SPEED_FILE.write_text(json.dumps(old, indent=1))
    return ok_all


def list_models():
    speed = json.loads(SPEED_FILE.read_text()) if SPEED_FILE.exists() else {}
    say(f"  {'id':<12} {'name':<20} {'installed':<10} {'size':>9}  {'10 s of PPG in':>14}")
    specs = specs_by_id()
    for s in specs.values():
        size = sum(f.stat().st_size for f in s.weights_dir.rglob("*") if f.is_file()) if s.weights_dir.exists() else 0
        sp = speed.get(s.id, {}).get("rt_factor")
        say(f"  {s.id:<12} {s.name:<20} {'yes' if s.installed else 'no':<10} {size / 1e3:>7.0f} kB  "
            f"{(f'{sp * 10:.3f} s') if sp else '-':>14}")
    for flag, mid in FLAGS.items():
        if mid not in specs:
            say(f"  {mid:<12} {'(--' + flag + ')':<20} {'later':<10}")


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
    ap.add_argument("--shortcut", action="store_true")
    ap.add_argument("--no-shortcut", action="store_true")
    a = ap.parse_args(argv)
    specs = specs_by_id()
    settings = config.Settings()

    if a.device:
        settings.set("device", a.device)
        say(f"Device set to {a.device} (v0.1 runs the NumPy methods on the CPU in any case).")
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
            if mid in LATER:
                say(f"  {mid}: arrives in v0.2 (skipped)")
                continue
            if mid in specs and not install(specs[mid], a.offline, a.capnobase):
                not_installed.append(specs[mid].name)
        say()
    if wanted or a.check or a.offline:
        say("Self-test and speed on this computer (30 s synthetic PPG)")
        ok &= check(list(engine.discover()))
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
