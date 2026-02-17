from __future__ import annotations

import os
from pathlib import Path


_DEFAULT_DATASET_CANDIDATES = (
    "~/Xyk/datasets",
    "~/Xyk/Datasets",
    "~/Datasets",
)


def resolve_datasets_root() -> Path:
    """Resolve datasets root from override or known local workspace candidates."""
    override = os.environ.get("INSTINCT_DATASETS_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve()

    for candidate in _DEFAULT_DATASET_CANDIDATES:
        candidate_path = Path(candidate).expanduser()
        if candidate_path.exists() and candidate_path.is_dir():
            return candidate_path.resolve()

    # Keep a deterministic fallback even when no candidate exists yet.
    return Path(_DEFAULT_DATASET_CANDIDATES[1]).expanduser().resolve()
