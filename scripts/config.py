"""Central configuration for preprocessing and modeling experiments."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration bundle describing a forecasting experiment."""

    name: str
    lookback: int
    forecast_horizon: int
    step: int = 1
    max_sequences: Optional[int] = None
    sample_strategy: str = "all"
    train_ratio: float = 0.7
    val_ratio: float = 0.15

    @property
    def test_ratio(self) -> float:
        return 1.0 - self.train_ratio - self.val_ratio


DEFAULT_DATA_DIR: Path = Path("data")
DEFAULT_OUTPUT_DIR: Path = Path("artifacts")


EXPERIMENT_CONFIGS: Dict[str, ExperimentConfig] = {
    "1h": ExperimentConfig(name="1h", lookback=32, forecast_horizon=4, max_sequences=50000),
    "1d": ExperimentConfig(
        name="1d",
        lookback=96,
        forecast_horizon=96,
        step=2,
        max_sequences=40000,
        sample_strategy="stride",
    ),
    "1w": ExperimentConfig(
        name="1w",
        lookback=672,
        forecast_horizon=672,
        step=6,
        max_sequences=30000,
        sample_strategy="stride",
    ),
}


__all__ = [
    "ExperimentConfig",
    "DEFAULT_DATA_DIR",
    "DEFAULT_OUTPUT_DIR",
    "EXPERIMENT_CONFIGS",
]
