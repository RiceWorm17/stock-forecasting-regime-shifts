"""Strict target-date walk-forward registry with an F1 access guard."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml


DEFAULT_WALK_FORWARD_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "walk_forward.yaml"
)
DEVELOPMENT_FOLD_IDS: tuple[str, ...] = ("D1", "D2", "D3", "D4", "D5")
FINAL_FOLD_ID = "F1"

EXPECTED_FOLDS: dict[str, dict[str, tuple[str, str] | str]] = {
    "D1": {
        "train": ("2015-01-01", "2018-12-31"),
        "validation": ("2019-01-01", "2019-12-31"),
        "test": ("2020-01-01", "2020-12-31"),
        "test_role": "development",
    },
    "D2": {
        "train": ("2015-01-01", "2019-12-31"),
        "validation": ("2020-01-01", "2020-12-31"),
        "test": ("2021-01-01", "2021-12-31"),
        "test_role": "development",
    },
    "D3": {
        "train": ("2015-01-01", "2020-12-31"),
        "validation": ("2021-01-01", "2021-12-31"),
        "test": ("2022-01-01", "2022-12-31"),
        "test_role": "development",
    },
    "D4": {
        "train": ("2015-01-01", "2021-12-31"),
        "validation": ("2022-01-01", "2022-12-31"),
        "test": ("2023-01-01", "2023-12-31"),
        "test_role": "development",
    },
    "D5": {
        "train": ("2015-01-01", "2022-12-31"),
        "validation": ("2023-01-01", "2023-12-31"),
        "test": ("2024-01-01", "2024-12-31"),
        "test_role": "development",
    },
    "F1": {
        "train": ("2015-01-01", "2023-12-31"),
        "validation": ("2024-01-01", "2024-12-31"),
        "test": ("2025-01-01", "2025-12-31"),
        "test_role": "final_untouched",
    },
}


class FoldConfigurationError(ValueError):
    """Raised when the registry differs from the frozen fold contract."""


class FinalTestAccessError(PermissionError):
    """Raised when F1 test rows are requested without explicit authorization."""


class FinalTestAccessWarning(UserWarning):
    """Warning emitted whenever guarded F1 test rows are explicitly requested."""


@dataclass(frozen=True)
class DateRange:
    start: date
    end: date

    def contains(self, values: pd.Series) -> pd.Series:
        timestamps = pd.to_datetime(values, errors="raise")
        return timestamps.between(pd.Timestamp(self.start), pd.Timestamp(self.end), inclusive="both")


@dataclass(frozen=True)
class FoldSpec:
    fold_id: str
    train: DateRange
    validation: DateRange
    test: DateRange
    test_role: str

    @property
    def is_final(self) -> bool:
        return self.test_role == "final_untouched"


@dataclass(frozen=True)
class FoldRegistry:
    schema_version: int
    partition_key: str
    folds: tuple[FoldSpec, ...]
    default_development_folds: tuple[str, ...]
    final_fold: str

    def get(self, fold_id: str) -> FoldSpec:
        for fold in self.folds:
            if fold.fold_id == fold_id:
                return fold
        raise KeyError(f"Unknown fold {fold_id!r}.")

    def default_fold_ids(self, *, include_final: bool = False) -> tuple[str, ...]:
        if include_final:
            return self.default_development_folds + (self.final_fold,)
        return self.default_development_folds


@dataclass(frozen=True)
class FoldPartitions:
    fold: FoldSpec
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def _as_date(value: Any, context: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise FoldConfigurationError(f"{context} is not an ISO date: {value!r}.") from exc
    raise FoldConfigurationError(f"{context} must be an ISO date string.")


def _load_range(value: Any, context: str) -> DateRange:
    if not isinstance(value, Mapping) or set(value) != {"start", "end"}:
        raise FoldConfigurationError(f"{context} must contain exactly start and end.")
    result = DateRange(
        start=_as_date(value["start"], f"{context}.start"),
        end=_as_date(value["end"], f"{context}.end"),
    )
    if result.start > result.end:
        raise FoldConfigurationError(f"{context} starts after it ends.")
    return result


def load_fold_registry(path: str | Path | None = None) -> FoldRegistry:
    """Load and verify the exact folds frozen in experiment specification 1.1."""

    config_path = Path(path) if path is not None else DEFAULT_WALK_FORWARD_CONFIG_PATH
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise FoldConfigurationError(f"Unable to read fold config {config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise FoldConfigurationError(f"Invalid fold YAML in {config_path}: {exc}") from exc

    if not isinstance(payload, Mapping):
        raise FoldConfigurationError("Walk-forward config must contain a YAML mapping.")
    expected_top_keys = {
        "schema_version",
        "partition_key",
        "default_development_folds",
        "final_fold",
        "folds",
    }
    if set(payload) != expected_top_keys:
        raise FoldConfigurationError(
            "Walk-forward config keys must be exactly " f"{sorted(expected_top_keys)}."
        )
    if payload["schema_version"] != 1:
        raise FoldConfigurationError("Walk-forward schema_version must be 1.")
    if payload["partition_key"] != "target_date":
        raise FoldConfigurationError("Fold partition_key must be target_date.")
    if tuple(payload["default_development_folds"]) != DEVELOPMENT_FOLD_IDS:
        raise FoldConfigurationError(
            f"Default development folds must be {DEVELOPMENT_FOLD_IDS}."
        )
    if payload["final_fold"] != FINAL_FOLD_ID:
        raise FoldConfigurationError(f"Final fold must be {FINAL_FOLD_ID}.")

    raw_folds = payload["folds"]
    if not isinstance(raw_folds, Mapping) or tuple(raw_folds) != tuple(EXPECTED_FOLDS):
        raise FoldConfigurationError(
            f"Fold IDs and order must be exactly {tuple(EXPECTED_FOLDS)}."
        )

    folds: list[FoldSpec] = []
    for fold_id, expected in EXPECTED_FOLDS.items():
        raw_fold = raw_folds[fold_id]
        if not isinstance(raw_fold, Mapping) or set(raw_fold) != {
            "train",
            "validation",
            "test",
            "test_role",
        }:
            raise FoldConfigurationError(
                f"Fold {fold_id} must contain train, validation, test, and test_role."
            )
        fold = FoldSpec(
            fold_id=fold_id,
            train=_load_range(raw_fold["train"], f"{fold_id}.train"),
            validation=_load_range(raw_fold["validation"], f"{fold_id}.validation"),
            test=_load_range(raw_fold["test"], f"{fold_id}.test"),
            test_role=str(raw_fold["test_role"]),
        )
        if not (fold.train.end < fold.validation.start <= fold.validation.end < fold.test.start):
            raise FoldConfigurationError(
                f"Fold {fold_id} must satisfy train < validation < test."
            )

        for partition_name in ("train", "validation", "test"):
            expected_start, expected_end = expected[partition_name]  # type: ignore[misc]
            actual_range = getattr(fold, partition_name)
            if actual_range != DateRange(
                date.fromisoformat(expected_start), date.fromisoformat(expected_end)
            ):
                raise FoldConfigurationError(
                    f"Fold {fold_id} {partition_name} differs from the frozen contract."
                )
        if fold.test_role != expected["test_role"]:
            raise FoldConfigurationError(
                f"Fold {fold_id} test_role must be {expected['test_role']!r}."
            )
        folds.append(fold)

    return FoldRegistry(
        schema_version=1,
        partition_key="target_date",
        folds=tuple(folds),
        default_development_folds=DEVELOPMENT_FOLD_IDS,
        final_fold=FINAL_FOLD_ID,
    )


def _validated_target_dates(samples: pd.DataFrame, partition_key: str) -> pd.Series:
    if not isinstance(samples, pd.DataFrame):
        raise TypeError(f"samples must be a pandas DataFrame, got {type(samples).__name__}.")
    if partition_key not in samples.columns:
        raise ValueError(f"Samples are missing partition key {partition_key!r}.")
    try:
        target_dates = pd.to_datetime(samples[partition_key], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Samples contain invalid target dates: {exc}") from exc
    if target_dates.isna().any():
        raise ValueError("Samples contain missing target dates.")

    identity = ["asset", "origin_date", "target_date"]
    if all(column in samples.columns for column in identity):
        if samples.duplicated(identity, keep=False).any():
            raise ValueError("Samples contain duplicate supervised-sample identities.")
    return target_dates


def split_samples_for_fold(
    samples: pd.DataFrame,
    fold_id: str,
    *,
    registry: FoldRegistry | None = None,
    include_test: bool = True,
    allow_final_test: bool = False,
) -> FoldPartitions:
    """Partition samples by target date, guarding F1 test access by default."""

    active_registry = registry or load_fold_registry()
    fold = active_registry.get(fold_id)
    target_dates = _validated_target_dates(samples, active_registry.partition_key)

    if fold.is_final and include_test and not allow_final_test:
        raise FinalTestAccessError(
            "F1 test rows are FINAL / UNTOUCHED. Pass allow_final_test=True explicitly "
            "only for the authorized final evaluation."
        )
    if fold.is_final and include_test and allow_final_test:
        warnings.warn(
            "Accessing F1 FINAL / UNTOUCHED test rows.",
            FinalTestAccessWarning,
            stacklevel=2,
        )

    train_mask = fold.train.contains(target_dates)
    validation_mask = fold.validation.contains(target_dates)
    test_mask = fold.test.contains(target_dates) if include_test else pd.Series(
        False, index=samples.index
    )

    overlap = (
        train_mask.astype("int8")
        + validation_mask.astype("int8")
        + test_mask.astype("int8")
    ) > 1
    if overlap.any():
        raise AssertionError(f"Fold {fold_id} has overlapping partitions.")

    normalized = samples.copy()
    normalized[active_registry.partition_key] = target_dates
    return FoldPartitions(
        fold=fold,
        train=normalized.loc[train_mask].copy(),
        validation=normalized.loc[validation_mask].copy(),
        test=normalized.loc[test_mask].copy(),
    )


def get_partition(
    samples: pd.DataFrame,
    fold_id: str,
    partition: str,
    *,
    registry: FoldRegistry | None = None,
    allow_final_test: bool = False,
) -> pd.DataFrame:
    """Return one named partition while preserving the F1 test guard."""

    if partition not in {"train", "validation", "test"}:
        raise ValueError("partition must be train, validation, or test.")
    partitions = split_samples_for_fold(
        samples,
        fold_id,
        registry=registry,
        include_test=partition == "test",
        allow_final_test=allow_final_test,
    )
    return getattr(partitions, partition)

