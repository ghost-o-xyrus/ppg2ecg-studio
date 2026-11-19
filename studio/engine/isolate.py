"""Load an author's repository whose top-level module names clash with another's.

RDDM and PPGFlowECG both have a top-level `model` module (and PPGFlowECG a `utils` package). Their adapters
put the repository first on sys.path and import by those plain names, so whichever loads second would get
the other's modules. While one of them loads, the clashing names are taken out of sys.modules and put back
afterwards; the objects built during load keep their own modules. The registry loads these models one at a
time (runtime: torch), so the swap never overlaps.
"""
from __future__ import annotations

import sys
from contextlib import contextmanager


@contextmanager
def repo_imports(repo_dir, names):
    names = set(names)

    def mine(k):
        return k.split(".", 1)[0] in names

    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if mine(k)}
    path_before = list(sys.path)
    try:
        yield
    finally:
        for k in [k for k in sys.modules if mine(k)]:
            del sys.modules[k]
        sys.modules.update(saved)
        sys.path[:] = path_before
