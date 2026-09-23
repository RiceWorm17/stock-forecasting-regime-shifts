"""Locked data schema and deterministic configuration loading."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping

import yaml


CANONICAL_COLUMNS: tuple[str, ...] = (
    "date",
    "asset",
    "open",
    "high",
    "low",
    "close",
    "volume",
)
PRICE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close")
NUMERIC_COLUMNS: tuple[str, ...] = (*PRICE_COLUMNS, "volume")
TARGET_ASSETS: tuple[str, ...] = ("AAPL", "MSFT", "GOOGL", "NVDA")
MARKET_REFERENCES: tuple[str, ...] = ("SPY",)
LOCKED_START_DATE = date(2015, 1, 1)
LOCKED_END_DATE = date(2025, 12, 31)
LOCKED_FREQUENCY = "daily"
LOCKED_TIMEZONE = "America/New_York"
DEFAULT_DATA_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "data.yaml"


class ConfigurationError(ValueError):
    """Raised when configuration differs from the frozen experiment contract."""


@dataclass(frozen=True)
class DataConfig:
    """Validated, immutable representation of ``configs/data.yaml``."""

    schema_version: int
    provider: str
    assets: tuple[str, ...]
    market_reference: tuple[str, ...]
    start_date: date
    end_date: date
    frequency: str
    timezone: str
    auto_adjust: bool

    @property
    def all_assets(self) -> tuple[str, ...]:
        return self.assets + self.market_reference


def _as_date(value: Any, field_name: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ConfigurationError(
                f"{field_name} must be an ISO date (YYYY-MM-DD), got {value!r}."
            ) from exc
    raise ConfigurationError(
        f"{field_name} must be an ISO date string, got {type(value).__name__}."
    )


def _as_ticker_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigurationError(f"{field_name} must be a non-empty YAML list.")
    tickers = tuple(str(item).strip().upper() for item in value)
    if any(not ticker for ticker in tickers):
        raise ConfigurationError(f"{field_name} contains an empty ticker.")
    if len(set(tickers)) != len(tickers):
        raise ConfigurationError(f"{field_name} contains duplicate tickers: {tickers!r}.")
    return tickers


def load_data_config(path: str | Path | None = None) -> DataConfig:
    """Load and strictly validate the frozen data configuration."""

    config_path = Path(path) if path is not None else DEFAULT_DATA_CONFIG_PATH
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigurationError(f"Unable to read data config {config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in data config {config_path}: {exc}") from exc

    if not isinstance(payload, Mapping):
        raise ConfigurationError("Data config must contain a YAML mapping.")

    required = {
        "schema_version",
        "provider",
        "assets",
        "market_reference",
        "start_date",
        "end_date",
        "frequency",
        "timezone",
        "auto_adjust",
    }
    missing = required.difference(payload)
    unknown = set(payload).difference(required)
    if missing:
        raise ConfigurationError(f"Data config is missing keys: {sorted(missing)}.")
    if unknown:
        raise ConfigurationError(f"Data config contains unsupported keys: {sorted(unknown)}.")

    assets = _as_ticker_tuple(payload["assets"], "assets")
    references = _as_ticker_tuple(payload["market_reference"], "market_reference")
    start_date = _as_date(payload["start_date"], "start_date")
    end_date = _as_date(payload["end_date"], "end_date")

    config = DataConfig(
        schema_version=int(payload["schema_version"]),
        provider=str(payload["provider"]).strip().lower(),
        assets=assets,
        market_reference=references,
        start_date=start_date,
        end_date=end_date,
        frequency=str(payload["frequency"]).strip().lower(),
        timezone=str(payload["timezone"]).strip(),
        auto_adjust=payload["auto_adjust"],
    )

    violations: list[str] = []
    if config.schema_version != 1:
        violations.append("schema_version must be 1")
    if config.provider != "yfinance":
        violations.append("provider must be yfinance")
    if config.assets != TARGET_ASSETS:
        violations.append(f"assets must be {list(TARGET_ASSETS)!r} in locked order")
    if config.market_reference != MARKET_REFERENCES:
        violations.append(f"market_reference must be {list(MARKET_REFERENCES)!r}")
    if set(config.assets).intersection(config.market_reference):
        violations.append("target assets and market reference must be distinct")
    if config.start_date != LOCKED_START_DATE:
        violations.append(f"start_date must be {LOCKED_START_DATE.isoformat()}")
    if config.end_date != LOCKED_END_DATE:
        violations.append(f"end_date must be {LOCKED_END_DATE.isoformat()}")
    if config.frequency != LOCKED_FREQUENCY:
        violations.append(f"frequency must be {LOCKED_FREQUENCY}")
    if config.timezone != LOCKED_TIMEZONE:
        violations.append(f"timezone must be {LOCKED_TIMEZONE}")
    if not isinstance(config.auto_adjust, bool) or config.auto_adjust is not True:
        violations.append("auto_adjust must be the boolean true")

    if violations:
        raise ConfigurationError("Data config violates the frozen contract: " + "; ".join(violations))
    return config


def asset_role(asset: str) -> str:
    """Return the locked role for an asset, rejecting unknown tickers."""

    normalized = str(asset).strip().upper()
    if normalized in TARGET_ASSETS:
        return "target"
    if normalized in MARKET_REFERENCES:
        return "market_reference"
    raise ConfigurationError(f"Unknown asset {asset!r}; asset is outside the locked universe.")

