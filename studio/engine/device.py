"""Device probe: NVIDIA (CUDA) -> any GPU through DirectML -> CPU.

The ONNX models (CardioGAN, Chiu, CLEP-GAN, P2E-WGAN, Yalcin) run on the GPU when ONNX Runtime has a GPU
provider; a model that fails there falls back to the CPU (engine.onnx_run). RDDM and PPGFlowECG run in
PyTorch on the CPU, the template baseline and Zhu DCT in NumPy. The probe reads installed package metadata
and the registry only, so the window starts without loading ONNX Runtime.
"""
from __future__ import annotations

import os
import platform
import sys
from functools import lru_cache
from importlib import metadata

VIRTUAL = ("parsec", "basic display", "basic render", "remote", "virtual", "idd", "mirage", "citrix", "vmware")


@lru_cache(maxsize=1)
def cpu_name() -> str:
    name = ""
    if sys.platform == "win32":
        try:   # the registry answers instantly (a WMI query takes seconds)
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                name = str(winreg.QueryValueEx(k, "ProcessorNameString")[0])
        except OSError:
            name = ""
    elif os.path.exists("/proc/cpuinfo"):
        try:
            for ln in open("/proc/cpuinfo", encoding="utf-8", errors="ignore"):
                if ln.lower().startswith("model name"):
                    name = ln.split(":", 1)[1].strip()
                    break
        except OSError:
            pass
    return " ".join((name or platform.processor() or "CPU").split())


@lru_cache(maxsize=1)
def gpu_names() -> tuple:
    """Display adapters from the registry, virtual ones (remote desktop, basic display) left out."""
    out = []
    if sys.platform != "win32":
        return ()
    try:
        import winreg
        base = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base) as k:
            for i in range(winreg.QueryInfoKey(k)[0]):
                sub = winreg.EnumKey(k, i)
                if not sub.isdigit():
                    continue
                try:
                    with winreg.OpenKey(k, sub) as s:
                        n = str(winreg.QueryValueEx(s, "DriverDesc")[0])
                except OSError:
                    continue
                if n and not any(v in n.lower() for v in VIRTUAL) and n not in out:
                    out.append(n)
    except OSError:
        pass
    return tuple(out)


def _has(dist):
    try:
        metadata.version(dist)
        return True
    except metadata.PackageNotFoundError:
        return False


@lru_cache(maxsize=1)
def gpu_provider() -> str | None:
    """'CUDA' or 'DirectML' when ONNX Runtime can use a GPU here, else None."""
    if _has("onnxruntime-gpu") and any("nvidia" in g.lower() for g in gpu_names()):
        return "CUDA"
    if _has("onnxruntime-directml") and gpu_names():
        return "DirectML"
    return None


def probe(setting: str = "auto") -> str:
    """'gpu' or 'cpu' for the ONNX models; setting is the user's override (auto | gpu | cpu)."""
    if setting == "cpu":
        return "cpu"
    return "gpu" if gpu_provider() else "cpu"


def describe(device: str = "cpu") -> str:
    if device == "gpu" and gpu_provider():
        names = [g for g in gpu_names() if "nvidia" in g.lower()] if gpu_provider() == "CUDA" else list(gpu_names())
        return f"GPU: {' '.join(names[0].replace('(R)', '').replace('(TM)', '').split())} via {gpu_provider()}"
    return f"CPU: {cpu_name()}"
