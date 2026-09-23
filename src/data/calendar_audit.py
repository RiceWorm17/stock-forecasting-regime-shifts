"""US exchange-session audits for the frozen daily market-data snapshot."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd

from src.data.schema import DataConfig, load_data_config
from src.data.validation import validate_canonical_data


class CalendarAuditError(ValueError):
    """Raised when a calendar audit is unusable or detects anomalies on demand."""


@dataclass(frozen=True)
class AssetCalendarResult:
    """Observed-versus-exchange-session comparison for one asset."""

    observed_count: int
    expected_count: int
    missing_expected_sessions: tuple[str, ...]
    unexpected_weekday_dates: tuple[str, ...]

    @property
    def is_clean(self) -> bool:
        return not self.missing_expected_sessions and not self.unexpected_weekday_dates

    def to_dict(self) -> dict[str, Any]:
        return {
            "observed_count": self.observed_count,
            "expected_count": self.expected_count,
            "missing_expected_sessions": list(self.missing_expected_sessions),
            "unexpected_weekday_dates": list(self.unexpected_weekday_dates),
            "is_clean": self.is_clean,
        }


@dataclass(frozen=True)
class TargetSpyCalendarResult:
    """Direct observed-date comparison between one target and SPY."""

    dates_in_spy_not_target: tuple[str, ...]
    dates_in_target_not_spy: tuple[str, ...]

    @property
    def is_clean(self) -> bool:
        return not self.dates_in_spy_not_target and not self.dates_in_target_not_spy

    def to_dict(self) -> dict[str, Any]:
        return {
            "dates_in_spy_not_target": list(self.dates_in_spy_not_target),
            "dates_in_target_not_spy": list(self.dates_in_target_not_spy),
            "is_clean": self.is_clean,
        }


@dataclass(frozen=True)
class CalendarAudit:
    """Structured, serializable exchange-calendar audit results."""

    calendar_name: str
    requested_start: str
    requested_end: str
    expected_session_count: int
    expected_first_session: str
    expected_last_session: str
    by_asset: dict[str, AssetCalendarResult]
    target_vs_spy: dict[str, TargetSpyCalendarResult]

    @property
    def is_clean(self) -> bool:
        return all(result.is_clean for result in self.by_asset.values()) and all(
            result.is_clean for result in self.target_vs_spy.values()
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "calendar_name": self.calendar_name,
            "requested_start": self.requested_start,
            "requested_end": self.requested_end,
            "expected_session_count": self.expected_session_count,
            "expected_first_session": self.expected_first_session,
            "expected_last_session": self.expected_last_session,
            "by_asset": {
                asset: result.to_dict() for asset, result in self.by_asset.items()
            },
            "target_vs_spy": {
                asset: result.to_dict()
                for asset, result in self.target_vs_spy.items()
            },
            "is_clean": self.is_clean,
        }


SessionProvider = Callable[[date, date], pd.DatetimeIndex]


def xnys_sessions(start: date, end: date) -> pd.DatetimeIndex:
    """Return official XNYS sessions from the ``exchange_calendars`` package."""

    try:
        import exchange_calendars as xcals
    except ImportError as exc:
        raise RuntimeError(
            "exchange_calendars is required for the XNYS market-session audit."
        ) from exc

    try:
        sessions = pd.DatetimeIndex(
            xcals.get_calendar("XNYS").sessions_in_range(
                pd.Timestamp(start), pd.Timestamp(end)
            )
        )
    except Exception as exc:  # library errors vary across supported versions
        raise CalendarAuditError(
            f"Unable to obtain XNYS sessions for {start} through {end}: {exc}"
        ) from exc

    if sessions.tz is not None:
        sessions = sessions.tz_localize(None)
    return sessions.normalize()


def _normalize_sessions(sessions: pd.DatetimeIndex) -> pd.DatetimeIndex:
    normalized = pd.DatetimeIndex(sessions)
    if normalized.tz is not None:
        normalized = normalized.tz_localize(None)
    normalized = normalized.normalize().sort_values().unique()
    if len(normalized) == 0:
        raise CalendarAuditError("Exchange calendar returned no sessions.")
    return pd.DatetimeIndex(normalized)


def _iso_dates(values: set[pd.Timestamp]) -> tuple[str, ...]:
    return tuple(value.date().isoformat() for value in sorted(values))


def audit_exchange_calendar(
    frame: pd.DataFrame,
    *,
    config: DataConfig | None = None,
    session_provider: SessionProvider | None = None,
    calendar_name: str = "XNYS",
) -> CalendarAudit:
    """Compare every observed asset date with the locked US exchange calendar.

    The default provider is the reputable ``exchange_calendars`` XNYS calendar.
    Dependency injection exists only to make deterministic unit tests possible.
    """

    active_config = config or load_data_config()
    validate_canonical_data(
        frame,
        config=active_config,
        require_sorted=True,
        require_all_assets=True,
    )

    provider = session_provider or xnys_sessions
    expected_index = _normalize_sessions(
        provider(active_config.start_date, active_config.end_date)
    )
    locked_start = pd.Timestamp(active_config.start_date)
    locked_end = pd.Timestamp(active_config.end_date)
    if expected_index.min() < locked_start or expected_index.max() > locked_end:
        raise CalendarAuditError(
            "Exchange session provider returned dates outside the locked requested range."
        )
    expected = set(expected_index)

    working = frame.loc[:, ["date", "asset"]].copy()
    working["date"] = pd.to_datetime(working["date"], errors="raise").dt.normalize()
    observed_by_asset = {
        asset: set(working.loc[working["asset"] == asset, "date"])
        for asset in active_config.all_assets
    }

    by_asset: dict[str, AssetCalendarResult] = {}
    for asset in active_config.all_assets:
        observed = observed_by_asset[asset]
        by_asset[asset] = AssetCalendarResult(
            observed_count=len(observed),
            expected_count=len(expected),
            missing_expected_sessions=_iso_dates(expected.difference(observed)),
            unexpected_weekday_dates=_iso_dates(observed.difference(expected)),
        )

    if len(active_config.market_reference) != 1:
        raise CalendarAuditError(
            "Calendar comparison requires exactly one locked market reference."
        )
    reference = active_config.market_reference[0]
    spy_dates = observed_by_asset[reference]
    target_vs_spy: dict[str, TargetSpyCalendarResult] = {}
    for asset in active_config.assets:
        target_dates = observed_by_asset[asset]
        target_vs_spy[asset] = TargetSpyCalendarResult(
            dates_in_spy_not_target=_iso_dates(spy_dates.difference(target_dates)),
            dates_in_target_not_spy=_iso_dates(target_dates.difference(spy_dates)),
        )

    return CalendarAudit(
        calendar_name=calendar_name,
        requested_start=active_config.start_date.isoformat(),
        requested_end=active_config.end_date.isoformat(),
        expected_session_count=len(expected_index),
        expected_first_session=expected_index.min().date().isoformat(),
        expected_last_session=expected_index.max().date().isoformat(),
        by_asset=by_asset,
        target_vs_spy=target_vs_spy,
    )


def assert_calendar_clean(audit: CalendarAudit) -> None:
    """Fail loudly when any exchange or target-versus-SPY discrepancy exists."""

    if audit.is_clean:
        return

    summaries: list[str] = []
    for asset, result in audit.by_asset.items():
        if not result.is_clean:
            summaries.append(
                f"{asset}: {len(result.missing_expected_sessions)} missing exchange "
                f"sessions, {len(result.unexpected_weekday_dates)} unexpected dates"
            )
    for asset, result in audit.target_vs_spy.items():
        if not result.is_clean:
            summaries.append(
                f"{asset} vs SPY: {len(result.dates_in_spy_not_target)} dates only in "
                f"SPY, {len(result.dates_in_target_not_spy)} dates only in target"
            )
    raise CalendarAuditError("Calendar audit detected anomalies: " + "; ".join(summaries))
