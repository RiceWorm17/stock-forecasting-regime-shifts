from __future__ import annotations

from datetime import timedelta

import pandas as pd
import pytest

from src.evaluation.walk_forward import (
    DEVELOPMENT_FOLD_IDS,
    EXPECTED_FOLDS,
    FINAL_FOLD_ID,
    FinalTestAccessError,
    FinalTestAccessWarning,
    get_partition,
    load_fold_registry,
    split_samples_for_fold,
)


def _sample_frame(target_dates: list[str]) -> pd.DataFrame:
    parsed = pd.to_datetime(target_dates)
    return pd.DataFrame(
        {
            "asset": "AAPL",
            "origin_date": parsed - pd.offsets.BDay(1),
            "target_date": parsed,
            "target_log_return": 0.0,
        }
    )


def test_registry_matches_every_frozen_boundary_exactly():
    registry = load_fold_registry()

    assert tuple(fold.fold_id for fold in registry.folds) == (*DEVELOPMENT_FOLD_IDS, FINAL_FOLD_ID)
    assert registry.partition_key == "target_date"
    for fold_id, expected in EXPECTED_FOLDS.items():
        fold = registry.get(fold_id)
        for partition in ("train", "validation", "test"):
            start, end = expected[partition]
            actual = getattr(fold, partition)
            assert actual.start.isoformat() == start
            assert actual.end.isoformat() == end
        assert fold.test_role == expected["test_role"]
        assert fold.train.end < fold.validation.start
        assert fold.validation.end < fold.test.start


@pytest.mark.parametrize("fold_id", DEVELOPMENT_FOLD_IDS)
def test_development_fold_partition_boundaries_and_no_overlap(fold_id):
    registry = load_fold_registry()
    fold = registry.get(fold_id)
    dates = [
        fold.train.start.isoformat(),
        fold.train.end.isoformat(),
        fold.validation.start.isoformat(),
        fold.validation.end.isoformat(),
        fold.test.start.isoformat(),
        fold.test.end.isoformat(),
    ]
    samples = _sample_frame(dates)

    partitions = split_samples_for_fold(samples, fold_id, registry=registry)

    assert set(partitions.train["target_date"]) == set(pd.to_datetime(dates[:2]))
    assert set(partitions.validation["target_date"]) == set(pd.to_datetime(dates[2:4]))
    assert set(partitions.test["target_date"]) == set(pd.to_datetime(dates[4:]))
    train_ids = set(partitions.train.index)
    validation_ids = set(partitions.validation.index)
    test_ids = set(partitions.test.index)
    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(test_ids)


def test_partition_membership_uses_target_date_at_year_boundary():
    samples = pd.DataFrame(
        {
            "asset": ["AAPL"],
            "origin_date": pd.to_datetime(["2019-12-31"]),
            "target_date": pd.to_datetime(["2020-01-02"]),
            "target_log_return": [0.01],
        }
    )

    partitions = split_samples_for_fold(samples, "D1")

    assert partitions.train.empty
    assert partitions.validation.empty
    assert len(partitions.test) == 1
    assert partitions.test.iloc[0]["origin_date"] == pd.Timestamp("2019-12-31")


def test_default_fold_workflow_excludes_final_fold():
    registry = load_fold_registry()
    assert registry.default_fold_ids() == DEVELOPMENT_FOLD_IDS
    assert "F1" not in registry.default_fold_ids()
    assert registry.default_fold_ids(include_final=True) == (*DEVELOPMENT_FOLD_IDS, "F1")


def test_f1_test_access_is_disabled_by_default():
    samples = _sample_frame(["2023-06-01", "2024-06-03", "2025-06-02"])

    with pytest.raises(FinalTestAccessError, match="FINAL / UNTOUCHED"):
        split_samples_for_fold(samples, "F1")
    with pytest.raises(FinalTestAccessError, match="allow_final_test=True"):
        get_partition(samples, "F1", "test")


def test_f1_training_and_validation_are_available_without_test_access():
    samples = _sample_frame(["2023-06-01", "2024-06-03", "2025-06-02"])

    partitions = split_samples_for_fold(samples, "F1", include_test=False)

    assert len(partitions.train) == 1
    assert len(partitions.validation) == 1
    assert partitions.test.empty


def test_explicit_f1_access_warns_and_returns_only_f1_test_rows():
    samples = _sample_frame(["2023-06-01", "2024-06-03", "2025-06-02"])

    with pytest.warns(FinalTestAccessWarning, match="FINAL / UNTOUCHED"):
        test_rows = get_partition(samples, "F1", "test", allow_final_test=True)

    assert test_rows["target_date"].tolist() == [pd.Timestamp("2025-06-02")]


def test_duplicate_supervised_identity_is_rejected():
    samples = _sample_frame(["2020-06-01"])
    samples = pd.concat([samples, samples], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate supervised-sample"):
        split_samples_for_fold(samples, "D1")

