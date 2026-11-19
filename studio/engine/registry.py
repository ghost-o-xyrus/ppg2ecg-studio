"""Model registry: discovers models/<id>/model.yaml + adapter.py, checks their weights, loads them.

The contract a model folder fulfils (the same one the built-in models use):

    models/<id>/model.yaml   manifest (id, name, family, native_fs, window_s, overlap, weights ...)
    models/<id>/adapter.py   build(ctx) -> bench.models.base.ModelAdapter (load(), infer(windows) ...)
    weights/<id>/...         the files listed under `weights:` in the manifest

Folders whose name starts with '_' (the example) are skipped.
"""
from __future__ import annotations

import importlib.util
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from .. import config

READY, LOADING, FAILED, MISSING = "ready", "loading", "failed", "not installed"


@dataclass
class ModelSpec:
    id: str
    name: str
    family: str
    folder: Path
    version: str = "1.0"
    order: int = 50
    native_fs: float = 128.0
    window_s: float = 4.0
    overlap: float = 0.5
    stochastic: bool = False
    runtime: str = "numpy"
    always_installed: bool = False
    citation: str = ""
    licence: str = ""
    description: str = ""
    params: dict = field(default_factory=dict)
    weights: list = field(default_factory=list)
    requires: list = field(default_factory=list)     # files in weights/<id>/ that mean "installed"
    repo: dict = field(default_factory=dict)         # authors' code: {url, commit}, fetched into weights/<id>/repo
    raw: dict = field(default_factory=dict)

    @property
    def weights_dir(self) -> Path:
        return config.WEIGHTS_DIR / self.id

    def missing_weights(self) -> list[str]:
        need = list(self.requires) or [w["file"] for w in self.weights]
        if self.repo:
            need.append("repo")
        return [f for f in need if not (self.weights_dir / f).exists()]

    @property
    def installed(self) -> bool:
        return self.always_installed or not self.missing_weights()


def read_spec(folder: Path) -> ModelSpec:
    m = yaml.safe_load((folder / "model.yaml").read_text(encoding="utf-8")) or {}
    for k in ("id", "name"):
        if not m.get(k):
            raise ValueError(f"{folder / 'model.yaml'}: '{k}' is missing")
    return ModelSpec(id=str(m["id"]), name=str(m["name"]), family=str(m.get("family", "")), folder=folder,
                     version=str(m.get("version", "1.0")), order=int(m.get("order", 50)),
                     native_fs=float(m.get("native_fs", 128)), window_s=float(m.get("window_s", 4)),
                     overlap=float(m.get("overlap", 0.5)), stochastic=bool(m.get("stochastic", False)),
                     runtime=str(m.get("runtime", "numpy")), always_installed=bool(m.get("always_installed", False)),
                     citation=str(m.get("citation", "")).strip(), licence=str(m.get("licence", "")),
                     description=str(m.get("description", "")).strip(), params=dict(m.get("params") or {}),
                     weights=list(m.get("weights") or []), requires=[str(f) for f in m.get("requires") or []],
                     repo=dict(m.get("repo") or {}), raw=m)


def discover(models_dir: Path = config.MODELS_DIR) -> list[ModelSpec]:
    specs, seen = [], set()
    for f in sorted(Path(models_dir).glob("*/model.yaml")):
        if f.parent.name.startswith(("_", ".")):
            continue
        try:
            s = read_spec(f.parent)
        except Exception as e:  # a broken manifest must not stop the app
            print(f"[registry] skipped {f.parent.name}: {e}", file=sys.stderr)
            continue
        if s.id in seen:
            continue
        seen.add(s.id)
        specs.append(s)
    return sorted(specs, key=lambda s: (s.order, s.name.lower()))


_IMPORT_LOCK = threading.Lock()     # models load on parallel threads; their imports must not interleave


def _import_adapter(spec: ModelSpec):
    with _IMPORT_LOCK:
        return _import_adapter_locked(spec)


def _import_adapter_locked(spec: ModelSpec):
    path = spec.folder / "adapter.py"
    mod_name = f"studio_models.{spec.id}"
    imp = importlib.util.spec_from_file_location(mod_name, path)
    if imp is None or imp.loader is None:
        raise ImportError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(imp)
    sys.modules[mod_name] = mod
    imp.loader.exec_module(mod)
    if not hasattr(mod, "build"):
        raise AttributeError(f"{path} has no build(ctx) function")
    return mod


@dataclass
class LoadedModel:
    spec: ModelSpec
    status: str = LOADING
    adapter: Any = None
    device: str = "CPU"
    error: str = ""
    load_seconds: float = 0.0
    info: dict = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


def load_model(spec: ModelSpec, device: str = "cpu") -> LoadedModel:
    lm = LoadedModel(spec)
    if not spec.installed:
        lm.status = MISSING
        lm.error = "weights not installed: " + ", ".join(spec.missing_weights())
        return lm
    t0 = time.perf_counter()
    try:
        mod = _import_adapter(spec)
        ctx = SimpleNamespace(spec=spec, weights_dir=spec.weights_dir, device=device, info=lm.info)
        ad = mod.build(ctx)
        if spec.runtime == "torch":     # authors' code imported by plain module names: one load at a time
            with _IMPORT_LOCK:
                ad.load()
        else:
            ad.load()
        lm.adapter, lm.status = ad, READY
        lm.device = lm.info.get("device") or "CPU"     # ONNX models report the provider they got
    except Exception as e:
        lm.status, lm.error = FAILED, f"{type(e).__name__}: {e}"
        traceback.print_exc()
    lm.load_seconds = time.perf_counter() - t0
    return lm
