# Configuration for the modeling analyses (sale -> activation -> travel).
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

_MODELING_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _MODELING_DIR.parent
_DATA_DIR = _REPO_ROOT.parent / "data"


@dataclass(frozen=True)
class Margins:
    # Per-unit margins as a fraction of the sale price, by outcome state (sale-price basis).
    # breakage: never activated -> keep full price at zero service cost.
    # idle: activated but never travelled -> keep price at ~no variable cost.
    # used: activated and travelled -> price minus variable service cost.
    breakage_fraction: float = 1.00
    idle_fraction: float = 0.98
    used_fraction: float = 0.72


@dataclass(frozen=True)
class ModelingConfig:
    data_dir: Path = _DATA_DIR
    output_dir: Path = _MODELING_DIR / "output"
    sales_file: str = "sales_flex_dummy.parquet"
    activations_file: str = "activations_flex_dummy.parquet"
    articles_file: str = "article_code_flex.parquet"
    as_of_date: dt.date | None = None       # None -> max observed activation date + 1 day
    seed: int = 42
    bootstrap_draws: int = 1000
    postpaid_margin_fraction: float = 0.62   # margin per sale under today's (only-sale) model
    margins: Margins = field(default_factory=Margins)
