"""Device probe. v0.1 runs the two NumPy methods on the CPU; the GPU order (CUDA -> DirectML -> CPU) for the
ONNX / PyTorch models arrives with them in v0.2."""
from __future__ import annotations

import os
import platform
import sys
from functools import lru_cache


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


def describe(device: str = "cpu") -> str:
    return f"CPU: {cpu_name()}"


def probe() -> str:
    """Best device for the models installed in this version."""
    return "cpu"
