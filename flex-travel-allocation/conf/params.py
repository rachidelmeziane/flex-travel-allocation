"""Load tunable analysis parameters from conf/params.yml.

Used by the simulation / demo scripts so they never hardcode magic numbers.
Core functions in ``src`` keep their own safe defaults; the scripts pass these.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_PARAMS_FILE = Path(__file__).resolve().parent / "params.yml"


@lru_cache(maxsize=1)
def load_params(path: str | Path | None = None) -> dict[str, Any]:
    """Read and cache conf/params.yml (override ``path`` in tests)."""
    with open(Path(path) if path else _PARAMS_FILE, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


# Module-level convenience so scripts can do: from conf.params import PARAMS
PARAMS: dict[str, Any] = load_params()
