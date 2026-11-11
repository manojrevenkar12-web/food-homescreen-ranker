"""Settings. Everything that decides reads config/platform.toml; the truth lives in config/world.toml."""

from __future__ import annotations

import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


@lru_cache
def platform() -> dict[str, Any]:
    with open(ROOT / "config" / "platform.toml", "rb") as f:
        return tomllib.load(f)
