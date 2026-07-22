"""Deploy-configured storage roots + copy helpers — NO runtime detection.

Two storage locations, declared *by whoever deploys* (via ``.env`` or the
environment); the code just reads them and copies between them. Which one is
fast/slow/persistent is the deployer's configuration, never something probed at
runtime.

* ``ZENNIT_PERSIST_ROOT`` — durable storage that survives pod bounces (this
  deployment: the NFS/GPFS workspace). Default: ``<repo>/data``.
* ``ZENNIT_SCRATCH_ROOT`` — fast, ephemeral scratch, wiped on bounce (this
  deployment: node-local overlay). Default: ``~/.cache/zennit-crp``.

Usage pattern for an expensive, regenerable cache (e.g. a FeatureVisualization
index): build it under :data:`SCRATCH_ROOT` (fast, and it avoids the small-file
wedge that many tiny writes cause on network storage), then :func:`sync` /
:func:`persist` it to :data:`PERSIST_ROOT` so it outlives the pod; on startup
:func:`hydrate` refills scratch from the persistent mirror so nothing is
recomputed after a bounce. Durable *results* (parquet, checkpoints, figures,
web-referenced data) are written straight under the persistent root / repo and
never left only on scratch.

See ``.env.example`` for the two knobs.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Union

REPO_ROOT = Path(__file__).resolve().parents[1]

_PathLike = Union[str, Path]


def _load_dotenv(path: Path) -> None:
    """Minimal ``.env`` loader (no dependency): ``KEY=VALUE`` lines, ``#``
    comments, optional surrounding quotes. Only sets keys *absent* from the
    environment, so an explicit ``export`` always wins. A missing or malformed
    file is ignored — defaults then apply."""
    try:
        text = path.read_text()
    except OSError:
        return
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


_load_dotenv(REPO_ROOT / ".env")

# The two declared roots (env-configured; repo-relative defaults keep single-box
# / dev runs working out of the box).
PERSIST_ROOT = Path(os.environ.get("ZENNIT_PERSIST_ROOT", str(REPO_ROOT / "data")))
SCRATCH_ROOT = Path(os.environ.get("ZENNIT_SCRATCH_ROOT", str(Path.home() / ".cache" / "zennit-crp")))


def _nonempty(p: Path) -> bool:
    return p.exists() and any(p.rglob("*"))


def scratch(subrel: _PathLike) -> Path:
    """Return (and create) the scratch working path for ``subrel``."""
    p = SCRATCH_ROOT / subrel
    p.mkdir(parents=True, exist_ok=True)
    return p


def persistent(subrel: _PathLike) -> Path:
    """Return the persistent (durable) path for ``subrel`` (not created)."""
    return PERSIST_ROOT / subrel


def sync(src: _PathLike, dst: _PathLike) -> Path:
    """Copy directory tree ``src`` → ``dst`` (merge, overwrite). No-op if ``src``
    is missing/empty. Returns ``dst``. This is a plain copy — no fstype logic."""
    src, dst = Path(src), Path(dst)
    if _nonempty(src):
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst, dirs_exist_ok=True)
    return dst


def hydrate(subrel: _PathLike) -> Path:
    """Ensure the scratch copy of ``subrel`` exists, refilling it from the
    persistent mirror when scratch is empty but the mirror has content (e.g. after
    a bounce wiped scratch). Returns the scratch path (the working location)."""
    dst = SCRATCH_ROOT / subrel
    if not _nonempty(dst):
        sync(PERSIST_ROOT / subrel, dst)
    dst.mkdir(parents=True, exist_ok=True)
    return dst


def persist(subrel: _PathLike) -> Path:
    """Copy the scratch build of ``subrel`` to the persistent mirror so it
    survives a bounce. Returns the persistent path. No-op if scratch is empty."""
    return sync(SCRATCH_ROOT / subrel, PERSIST_ROOT / subrel)
