"""
pipeline_utils.py
Shared helpers for the docking pipeline.

Centralizes the environment sanitization (previously duplicated across
main_pipeline.py, docking_engine.py and protein_preparation.py) and the
CPU-count resolution so behavior stays consistent everywhere.
"""

import os
import multiprocessing

# Environment variables that leak conda/venv libraries into shelled-out
# native tools (OpenBabel, prepare_receptor, ChimeraX) and must be stripped.
_POLLUTING_ENV_VARS = ["BABEL_LIBDIR", "BABEL_DATADIR", "LD_LIBRARY_PATH", "PYTHONPATH"]


def clean_subprocess_env() -> dict:
    """Returns a copy of the current environment with virtualenv/OpenBabel
    polluting variables removed and PATH pinned to system locations first,
    so external binaries resolve system libraries rather than conda/venv ones."""
    env = os.environ.copy()
    for var in _POLLUTING_ENV_VARS:
        env.pop(var, None)
    env["PATH"] = "/usr/bin:/usr/local/bin:/bin:" + env.get("PATH", "")
    return env


def resolve_cpu_count(requested=None) -> int:
    """Resolves how many CPU cores docking should use.

    Decoupled from Vina 'exhaustiveness' (search depth) which is a different
    concept. If `requested` is a positive int it is clamped to the system core
    count; otherwise all available cores are used.
    """
    system_cpus = multiprocessing.cpu_count()
    try:
        requested = int(requested) if requested is not None else 0
    except (TypeError, ValueError):
        requested = 0
    if requested > 0:
        return max(1, min(system_cpus, requested))
    return max(1, system_cpus)
