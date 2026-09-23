"""Frozen LightGBM 4.x contract and estimator-construction primitives.

This module deliberately contains no experiment runner.  It validates the
Phase 2D preregistration, maps its native LightGBM names to the sklearn wrapper,
and exposes small primitives used by the Phase 2E evaluation layer.  LightGBM
is imported lazily so contract tests do not require the optional dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.metadata
from numbers import Integral, Real
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import yaml


DEFAULT_MODEL_SEARCH_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "model_search.yaml"
)

SPEC_VERSION = "1.1"
RAW_DATASET_SHA256 = "55f74f058493102387f1c2d0a848bd673a3cfa0473d617dfdb34aea8a94f7736"
PROCESSED_DATASET_SHA256 = (
    "9352da6b434011f9844ef0acc1813ea33dce39152712752bba21a3005aabf7be"
)
TARGET_COLUMN = "target_log_return"
TARGET_ASSETS: tuple[str, ...] = ("AAPL", "MSFT", "GOOGL", "NVDA")
FROZEN_FEATURE_COLUMNS: tuple[str, ...] = (
    "asset_log_return_1d",
    "asset_log_return_lag_1",
    "asset_return_mean_5",
    "asset_return_std_5",
    "asset_volume_mean_5",
    "asset_volume_std_5",
    "asset_return_mean_21",
    "asset_return_std_21",
    "asset_volume_mean_21",
    "asset_volume_std_21",
    "spy_log_return_1d",
    "spy_log_return_lag_1",
    "spy_return_mean_5",
    "spy_volatility_5",
    "spy_return_mean_21",
    "spy_volatility_21",
    "spy_trend_63",
)
CANONICAL_PREDICTION_FIELDS: tuple[str, ...] = (
    "run_id",
    "spec_version",
    "data_version",
    "model",
    "model_config_id",
    "seed",
    "fold",
    "partition",
    "asset",
    "origin_date",
    "target_date",
    "actual_log_return",
    "predicted_log_return",
    "actual_direction",
    "predicted_direction",
    "trend_regime",
    "volatility_regime",
    "transition_regime",
)
REGIME_PLACEHOLDER = "not_labeled_pre_regime_analysis"


class LightGBMContractError(ValueError):
    """Raised when the machine-readable preregistration has drifted."""


class LightGBMDependencyError(RuntimeError):
    """Raised when an unavailable or incompatible LightGBM runtime is requested."""


@dataclass(frozen=True)
class LightGBMCandidate:
    """One frozen structural candidate in the LightGBM search grid."""

    config_id: str
    num_leaves: int
    min_child_samples: int

    @property
    def simplicity_key(self) -> tuple[int, int, str]:
        """Return the preregistered lower-compute ordering key."""

        return (self.num_leaves, -self.min_child_samples, self.config_id)


FROZEN_CANDIDATES: tuple[LightGBMCandidate, ...] = (
    LightGBMCandidate("LGBM_01", num_leaves=15, min_child_samples=20),
    LightGBMCandidate("LGBM_02", num_leaves=31, min_child_samples=20),
    LightGBMCandidate("LGBM_03", num_leaves=15, min_child_samples=50),
    LightGBMCandidate("LGBM_04", num_leaves=31, min_child_samples=50),
)


@dataclass(frozen=True)
class LightGBMContract:
    """Validated LightGBM subset of the frozen Phase 2D contract."""

    schema_version: int
    spec_version: str
    raw_sha256: str
    processed_sha256: str
    target_column: str
    assets: tuple[str, ...]
    feature_columns: tuple[str, ...]
    fixed_parameters: Mapping[str, Any]
    early_stopping_rounds: int
    candidates: tuple[LightGBMCandidate, ...]
    tie_tolerance: float
    canonical_prediction_fields: tuple[str, ...]
    regime_placeholder: str
    max_candidate_fits: int
    max_winning_refits: int

    def candidate(self, config_id: str) -> LightGBMCandidate:
        """Return one frozen candidate, rejecting unknown identifiers."""

        for candidate in self.candidates:
            if candidate.config_id == config_id:
                return candidate
        raise LightGBMContractError(f"Unknown LightGBM config_id {config_id!r}.")

    @property
    def seed(self) -> int:
        return int(self.fixed_parameters["random_seed"])

    @property
    def max_estimators(self) -> int:
        return int(self.fixed_parameters["max_estimators"])

    @property
    def metric(self) -> str:
        return str(self.fixed_parameters["metric"])


@dataclass(frozen=True)
class LightGBMRuntime:
    """Lazily resolved LightGBM 4.x runtime components."""

    module: Any
    version: str
    estimator_class: type[Any]


EstimatorFactory = Callable[..., Any]


_EXPECTED_TOP_LEVEL_KEYS = {
    "schema_version",
    "artifact_type",
    "status",
    "frozen_on",
    "experiment_specification",
    "data_contract",
    "prediction_task",
    "assets",
    "features",
    "walk_forward",
    "candidate_selection",
    "lightgbm",
    "lstm",
    "resource_budget",
    "metrics",
    "prediction_artifact_contract",
    "regime_analysis",
    "patchtst",
    "dependency_plan",
    "execution_environment",
    "change_control",
}

_EXPECTED_FIXED_PARAMETERS: dict[str, Any] = {
    "objective": "regression_l1",
    "metric": "l1",
    "learning_rate": 0.03,
    "max_estimators": 1000,
    "feature_fraction": 1.0,
    "bagging_fraction": 1.0,
    "bagging_freq": 0,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "deterministic": True,
    "force_col_wise": True,
    "n_jobs": 1,
    "random_seed": 1729,
}

_EXPECTED_DEVELOPMENT_FOLDS: dict[str, dict[str, dict[str, str]]] = {
    "D1": {
        "train": {"start": "2015-01-01", "end": "2018-12-31"},
        "validation": {"start": "2019-01-01", "end": "2019-12-31"},
        "test": {"start": "2020-01-01", "end": "2020-12-31"},
    },
    "D2": {
        "train": {"start": "2015-01-01", "end": "2019-12-31"},
        "validation": {"start": "2020-01-01", "end": "2020-12-31"},
        "test": {"start": "2021-01-01", "end": "2021-12-31"},
    },
    "D3": {
        "train": {"start": "2015-01-01", "end": "2020-12-31"},
        "validation": {"start": "2021-01-01", "end": "2021-12-31"},
        "test": {"start": "2022-01-01", "end": "2022-12-31"},
    },
    "D4": {
        "train": {"start": "2015-01-01", "end": "2021-12-31"},
        "validation": {"start": "2022-01-01", "end": "2022-12-31"},
        "test": {"start": "2023-01-01", "end": "2023-12-31"},
    },
    "D5": {
        "train": {"start": "2015-01-01", "end": "2022-12-31"},
        "validation": {"start": "2023-01-01", "end": "2023-12-31"},
        "test": {"start": "2024-01-01", "end": "2024-12-31"},
    },
}

_EXPECTED_FINAL_FOLD: dict[str, Any] = {
    "fold": "F1",
    "train": {"start": "2015-01-01", "end": "2023-12-31"},
    "validation": {"start": "2024-01-01", "end": "2024-12-31"},
    "test": {"start": "2025-01-01", "end": "2025-12-31"},
    "status": "LOCKED_UNTOUCHED",
    "execute_during_model_development": False,
    "inspect_2025_performance_during_model_development": False,
    "generate_2025_learned_predictions_during_model_development": False,
    "allow_2025_metrics_or_performance_artifacts_during_model_development": False,
    "fit_or_select_f1_candidates_during_d1_d5_development": False,
    "use_2024_as_f1_validation_during_d1_d5_development": False,
    "reuse_d5_winner_for_f1": False,
    "use_d1_d5_aggregate_tests_to_select_f1": False,
    "unlock_only_under_experiment_specification_final_test_procedure": True,
}


def _as_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LightGBMContractError(f"{path} must be a YAML mapping.")
    return value


def _at(payload: Mapping[str, Any], path: str) -> Any:
    value: Any = payload
    traversed: list[str] = []
    for part in path.split("."):
        traversed.append(part)
        mapping = _as_mapping(value, ".".join(traversed[:-1]) or "root")
        if part not in mapping:
            raise LightGBMContractError(f"Missing frozen config field {path!r}.")
        value = mapping[part]
    return value


def _expect(payload: Mapping[str, Any], path: str, expected: Any) -> None:
    actual = _at(payload, path)
    if actual != expected:
        raise LightGBMContractError(
            f"Frozen config field {path!r} is {actual!r}; expected {expected!r}."
        )


def _load_yaml_mapping(path: Path) -> Mapping[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise LightGBMContractError(f"Unable to read model config {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise LightGBMContractError(f"Invalid model-search YAML in {path}: {exc}") from exc
    return _as_mapping(payload, "root")


def load_lightgbm_contract(
    path: str | Path | None = None,
) -> LightGBMContract:
    """Load and strictly validate the frozen LightGBM implementation contract."""

    config_path = Path(path) if path is not None else DEFAULT_MODEL_SEARCH_CONFIG_PATH
    payload = _load_yaml_mapping(config_path)
    if set(payload) != _EXPECTED_TOP_LEVEL_KEYS:
        missing = sorted(_EXPECTED_TOP_LEVEL_KEYS.difference(payload))
        extra = sorted(set(payload).difference(_EXPECTED_TOP_LEVEL_KEYS))
        raise LightGBMContractError(
            f"Model-search top-level keys drifted; missing={missing}, extra={extra}."
        )

    _expect(payload, "schema_version", 1)
    _expect(payload, "artifact_type", "learned_model_preregistration")
    _expect(payload, "status", "FROZEN_BEFORE_LEARNED_MODEL_TRAINING")
    _expect(payload, "experiment_specification.version", SPEC_VERSION)
    _expect(payload, "experiment_specification.path", "docs/EXPERIMENT_SPEC.md")
    _expect(payload, "data_contract.raw_dataset_sha256", RAW_DATASET_SHA256)
    _expect(payload, "data_contract.processed_dataset_sha256", PROCESSED_DATASET_SHA256)
    _expect(payload, "data_contract.existing_feature_values_mutable", False)
    _expect(payload, "data_contract.target_definition_mutable", False)

    _expect(payload, "prediction_task.primary_task", "regression")
    _expect(payload, "prediction_task.target_column", TARGET_COLUMN)
    _expect(payload, "prediction_task.forecast_horizon_observed_sessions", 1)
    _expect(payload, "prediction_task.raw_price_prediction", "prohibited")
    _expect(
        payload,
        "prediction_task.secondary_direction.rule",
        "1 if predicted_log_return > 0 else 0",
    )
    _expect(payload, "prediction_task.secondary_direction.separate_classifier", "prohibited")

    _expect(payload, "assets.target_assets", list(TARGET_ASSETS))
    _expect(payload, "assets.market_reference", "SPY")
    _expect(payload, "assets.market_reference_is_prediction_target", False)
    _expect(
        payload,
        "assets.modeling_policy.fitted_models",
        "one_separate_regression_model_per_target_asset",
    )
    _expect(payload, "assets.modeling_policy.pooled_multi_asset_model", "prohibited")
    _expect(payload, "assets.modeling_policy.candidate_space_shared_across_assets", True)

    _expect(payload, "features.count", len(FROZEN_FEATURE_COLUMNS))
    _expect(payload, "features.columns", list(FROZEN_FEATURE_COLUMNS))
    _expect(payload, "features.additional_predictive_features_allowed", False)
    _expect(payload, "features.regime_labels_are_predictive_features", False)
    _expect(payload, "features.eligibility.filter_column", "core_evaluation_eligible")
    _expect(payload, "features.eligibility.required_filter_value", True)
    _expect(payload, "features.eligibility.require_all_features_finite", True)
    _expect(
        payload,
        "features.eligibility.headline_test_keys_must_equal_phase_2c_canonical_keys",
        True,
    )
    _expect(payload, "features.eligibility.mismatch_action", "fail_hard")

    _expect(payload, "walk_forward.partition_key", "target_date")
    _expect(payload, "walk_forward.candidate_space_scope", "global_and_frozen")
    _expect(payload, "walk_forward.candidate_scores_scope", "fold_local")
    _expect(payload, "walk_forward.winner_scope", "model_family_and_fold")
    _expect(payload, "walk_forward.winner_shared_across_assets_within_fold", True)
    _expect(payload, "walk_forward.fitted_asset_models_remain_separate", True)
    _expect(payload, "walk_forward.later_folds_may_influence_earlier_selection", False)
    _expect(payload, "walk_forward.update_model_parameters_inside_test_year", False)
    _expect(payload, "walk_forward.development_folds", _EXPECTED_DEVELOPMENT_FOLDS)
    _expect(payload, "walk_forward.final_test", _EXPECTED_FINAL_FOLD)

    _expect(payload, "candidate_selection.unit", "model_family_and_fold")
    _expect(
        payload,
        "candidate_selection.primary_statistic",
        "equal_weight_macro_validation_mae",
    )
    _expect(
        payload,
        "candidate_selection.asset_weights",
        {"AAPL": 0.25, "MSFT": 0.25, "GOOGL": 0.25, "NVDA": 0.25},
    )
    _expect(payload, "candidate_selection.optimization_direction", "minimize")
    _expect(payload, "candidate_selection.tie_breaking.tolerance", 0.00001)
    _expect(
        payload,
        "candidate_selection.tie_breaking.family_simplicity_rules.lightgbm",
        "fewer_num_leaves_then_larger_min_child_samples",
    )
    _expect(
        payload,
        "candidate_selection.tie_breaking.family_simplicity_order.lightgbm",
        ["LGBM_03", "LGBM_01", "LGBM_04", "LGBM_02"],
    )

    lightgbm = _as_mapping(_at(payload, "lightgbm"), "lightgbm")
    expected_lightgbm_keys = {
        "required",
        "framework",
        "framework_major_version",
        "estimator",
        "prediction_artifact_model_value",
        "task",
        "models_per_asset",
        "input_scaling",
        "fixed_parameters",
        "implementation_mapping",
        "early_stopping",
        "candidates",
        "candidate_count",
        "candidate_fit",
        "winning_candidate_refit",
        "artifact_seed",
    }
    if set(lightgbm) != expected_lightgbm_keys:
        raise LightGBMContractError("LightGBM section keys differ from the frozen contract.")
    _expect(payload, "lightgbm.required", True)
    _expect(payload, "lightgbm.framework", "LightGBM")
    _expect(payload, "lightgbm.framework_major_version", 4)
    _expect(payload, "lightgbm.estimator", "LGBMRegressor")
    _expect(payload, "lightgbm.prediction_artifact_model_value", "lightgbm")
    _expect(payload, "lightgbm.task", "regression")
    _expect(payload, "lightgbm.models_per_asset", 1)
    _expect(payload, "lightgbm.input_scaling", "none")
    _expect(payload, "lightgbm.fixed_parameters", _EXPECTED_FIXED_PARAMETERS)
    _expect(payload, "lightgbm.early_stopping.enabled_for_candidate_fit", True)
    _expect(payload, "lightgbm.early_stopping.validation_partition_only", True)
    _expect(payload, "lightgbm.early_stopping.rounds", 50)
    _expect(payload, "lightgbm.candidate_count", len(FROZEN_CANDIDATES))
    _expect(payload, "lightgbm.artifact_seed", 1729)

    raw_candidates = _at(payload, "lightgbm.candidates")
    expected_candidates = [
        {
            "config_id": candidate.config_id,
            "num_leaves": candidate.num_leaves,
            "min_child_samples": candidate.min_child_samples,
        }
        for candidate in FROZEN_CANDIDATES
    ]
    if raw_candidates != expected_candidates:
        raise LightGBMContractError(
            f"LightGBM candidates differ from the frozen grid: {raw_candidates!r}."
        )

    _expect(payload, "resource_budget.lightgbm.candidates", 4)
    _expect(payload, "resource_budget.lightgbm.development_folds", 5)
    _expect(payload, "resource_budget.lightgbm.assets", 4)
    _expect(payload, "resource_budget.lightgbm.seeds_per_fit", 1)
    _expect(payload, "resource_budget.lightgbm.maximum_candidate_fits", 80)
    _expect(payload, "resource_budget.lightgbm.winning_refits", 20)

    _expect(
        payload,
        "prediction_artifact_contract.canonical_field_count",
        len(CANONICAL_PREDICTION_FIELDS),
    )
    _expect(
        payload,
        "prediction_artifact_contract.canonical_fields",
        list(CANONICAL_PREDICTION_FIELDS),
    )
    _expect(payload, "prediction_artifact_contract.seed_values.lightgbm", [1729])
    _expect(
        payload,
        "prediction_artifact_contract.regime_placeholders.value",
        REGIME_PLACEHOLDER,
    )
    _expect(payload, "prediction_artifact_contract.regime_placeholders.use_for_training", False)
    _expect(
        payload,
        "prediction_artifact_contract.regime_placeholders.use_for_candidate_selection",
        False,
    )
    _expect(payload, "regime_analysis.inspect_candidate_regime_performance", False)
    _expect(payload, "patchtst.status", "CLOSED")
    _expect(payload, "dependency_plan.install_during_phase_2d", False)
    _expect(payload, "change_control.add_candidates_after_training_starts", "prohibited")

    return LightGBMContract(
        schema_version=1,
        spec_version=SPEC_VERSION,
        raw_sha256=RAW_DATASET_SHA256,
        processed_sha256=PROCESSED_DATASET_SHA256,
        target_column=TARGET_COLUMN,
        assets=TARGET_ASSETS,
        feature_columns=FROZEN_FEATURE_COLUMNS,
        fixed_parameters=MappingProxyType(dict(_EXPECTED_FIXED_PARAMETERS)),
        early_stopping_rounds=50,
        candidates=FROZEN_CANDIDATES,
        tie_tolerance=0.00001,
        canonical_prediction_fields=CANONICAL_PREDICTION_FIELDS,
        regime_placeholder=REGIME_PLACEHOLDER,
        max_candidate_fits=80,
        max_winning_refits=20,
    )


def load_lightgbm_runtime() -> LightGBMRuntime:
    """Import LightGBM lazily and require an exact 4.x runtime family."""

    try:
        module = importlib.import_module("lightgbm")
    except (ImportError, OSError) as exc:
        raise LightGBMDependencyError(
            "LightGBM is unavailable. Install the frozen Phase 2E dependency "
            "range lightgbm>=4,<5 before executing the benchmark."
        ) from exc

    try:
        version = importlib.metadata.version("lightgbm")
    except importlib.metadata.PackageNotFoundError:
        version = str(getattr(module, "__version__", ""))
    match = re.fullmatch(r"(\d+)(?:\.\d+){1,3}(?:[-+].*)?", version)
    if match is None or int(match.group(1)) != 4:
        raise LightGBMDependencyError(
            f"The frozen benchmark requires LightGBM 4.x; found {version!r}."
        )

    estimator_class = getattr(module, "LGBMRegressor", None)
    if not isinstance(estimator_class, type):
        raise LightGBMDependencyError("LightGBM runtime has no LGBMRegressor class.")
    for callback_name in ("early_stopping", "record_evaluation"):
        if not callable(getattr(module, callback_name, None)):
            raise LightGBMDependencyError(
                f"LightGBM 4.x runtime has no callable {callback_name!r}."
            )
    return LightGBMRuntime(module=module, version=version, estimator_class=estimator_class)


def _validated_candidate(
    contract: LightGBMContract, candidate: LightGBMCandidate
) -> LightGBMCandidate:
    if not isinstance(contract, LightGBMContract):
        raise TypeError("contract must be a LightGBMContract.")
    if not isinstance(candidate, LightGBMCandidate):
        raise TypeError("candidate must be a LightGBMCandidate.")
    frozen = contract.candidate(candidate.config_id)
    if candidate != frozen:
        raise LightGBMContractError(
            f"Candidate {candidate.config_id!r} values differ from the frozen grid."
        )
    return frozen


def resolve_estimator_parameters(
    contract: LightGBMContract,
    candidate: LightGBMCandidate,
    *,
    n_estimators: int | None = None,
) -> dict[str, Any]:
    """Map frozen names to one non-duplicated sklearn-wrapper parameter set."""

    frozen_candidate = _validated_candidate(contract, candidate)
    rounds = contract.max_estimators if n_estimators is None else n_estimators
    if isinstance(rounds, bool) or not isinstance(rounds, Integral):
        raise ValueError("n_estimators must be an integer.")
    rounds = int(rounds)
    if not 1 <= rounds <= contract.max_estimators:
        raise ValueError(
            f"n_estimators must be between 1 and {contract.max_estimators}, got {rounds}."
        )

    fixed = contract.fixed_parameters
    return {
        "objective": fixed["objective"],
        "learning_rate": fixed["learning_rate"],
        "n_estimators": rounds,
        "num_leaves": frozen_candidate.num_leaves,
        "min_child_samples": frozen_candidate.min_child_samples,
        "colsample_bytree": fixed["feature_fraction"],
        "subsample": fixed["bagging_fraction"],
        "subsample_freq": fixed["bagging_freq"],
        "reg_lambda": fixed["lambda_l2"],
        "verbosity": fixed["verbosity"],
        "deterministic": fixed["deterministic"],
        "force_col_wise": fixed["force_col_wise"],
        "n_jobs": fixed["n_jobs"],
        "random_state": fixed["random_seed"],
    }


def build_lgbm_regressor(
    contract: LightGBMContract,
    candidate: LightGBMCandidate,
    *,
    n_estimators: int | None = None,
    estimator_factory: EstimatorFactory | None = None,
) -> Any:
    """Create a fresh estimator without fitting it.

    ``estimator_factory`` is called with keyword parameters and permits unit
    testing without importing or installing LightGBM.
    """

    parameters = resolve_estimator_parameters(
        contract, candidate, n_estimators=n_estimators
    )
    factory = estimator_factory
    if factory is None:
        factory = load_lightgbm_runtime().estimator_class
    if not callable(factory):
        raise TypeError("estimator_factory must be callable.")
    try:
        estimator = factory(**parameters)
    except Exception as exc:
        raise LightGBMDependencyError(
            f"Unable to construct LGBMRegressor with frozen parameters: {exc}"
        ) from exc
    if not callable(getattr(estimator, "fit", None)) or not callable(
        getattr(estimator, "predict", None)
    ):
        raise LightGBMDependencyError(
            "Estimator factory must return an object with callable fit and predict methods."
        )
    return estimator


def build_candidate_callbacks(
    contract: LightGBMContract,
    *,
    lightgbm_module: Any | None = None,
) -> tuple[dict[str, dict[str, list[float]]], list[Any]]:
    """Build the exact LightGBM 4.x validation callbacks and their history sink."""

    if not isinstance(contract, LightGBMContract):
        raise TypeError("contract must be a LightGBMContract.")
    module = lightgbm_module
    if module is None:
        module = load_lightgbm_runtime().module
    record_evaluation = getattr(module, "record_evaluation", None)
    early_stopping = getattr(module, "early_stopping", None)
    if not callable(record_evaluation) or not callable(early_stopping):
        raise LightGBMDependencyError(
            "Callback module must provide callable record_evaluation and early_stopping."
        )

    history: dict[str, dict[str, list[float]]] = {}
    callbacks = [
        record_evaluation(history),
        early_stopping(
            stopping_rounds=contract.early_stopping_rounds,
            first_metric_only=True,
            verbose=False,
            min_delta=0.0,
        ),
    ]
    if any(not callable(callback) for callback in callbacks):
        raise LightGBMDependencyError("LightGBM callback builders returned a non-callable.")
    return history, callbacks


def validate_best_iteration(
    estimator: Any,
    history: Mapping[str, Mapping[str, Sequence[Real]]],
    *,
    max_estimators: int,
    dataset_name: str = "validation",
    metric_name: str = "l1",
) -> int:
    """Validate LightGBM's one-based best iteration against recorded validation L1."""

    if isinstance(max_estimators, bool) or not isinstance(max_estimators, Integral):
        raise TypeError("max_estimators must be an integer.")
    maximum = int(max_estimators)
    if maximum < 1:
        raise ValueError("max_estimators must be positive.")
    if not isinstance(history, Mapping) or set(history) != {dataset_name}:
        raise LightGBMContractError(
            f"Evaluation history must contain only dataset {dataset_name!r}."
        )
    metrics = history[dataset_name]
    if not isinstance(metrics, Mapping) or set(metrics) != {metric_name}:
        raise LightGBMContractError(
            f"Evaluation history for {dataset_name!r} must contain only {metric_name!r}."
        )
    values = metrics[metric_name]
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise LightGBMContractError("Validation metric history must be a numeric sequence.")
    if not values or len(values) > maximum:
        raise LightGBMContractError(
            f"Validation history length must be in [1, {maximum}], got {len(values)}."
        )
    try:
        numeric = np.asarray(values, dtype="float64")
    except (TypeError, ValueError) as exc:
        raise LightGBMContractError("Validation metric history is not numeric.") from exc
    if numeric.ndim != 1 or not np.isfinite(numeric).all():
        raise LightGBMContractError(
            "Validation metric history must be one-dimensional and finite."
        )

    reported = getattr(estimator, "best_iteration_", None)
    if isinstance(reported, bool) or not isinstance(reported, Integral):
        raise LightGBMContractError(
            "Fitted candidate must expose an integer sklearn best_iteration_."
        )
    best_iteration = int(reported)
    if not 1 <= best_iteration <= maximum:
        raise LightGBMContractError(
            f"best_iteration_ must be in [1, {maximum}], got {best_iteration}."
        )

    expected = int(np.argmin(numeric)) + 1
    if best_iteration != expected:
        raise LightGBMContractError(
            "LightGBM best_iteration_ disagrees with the first minimum in recorded "
            f"validation {metric_name}: reported={best_iteration}, expected={expected}."
        )
    return best_iteration


__all__ = [
    "CANONICAL_PREDICTION_FIELDS",
    "DEFAULT_MODEL_SEARCH_CONFIG_PATH",
    "FROZEN_CANDIDATES",
    "FROZEN_FEATURE_COLUMNS",
    "LightGBMCandidate",
    "LightGBMContract",
    "LightGBMContractError",
    "LightGBMDependencyError",
    "LightGBMRuntime",
    "PROCESSED_DATASET_SHA256",
    "RAW_DATASET_SHA256",
    "REGIME_PLACEHOLDER",
    "SPEC_VERSION",
    "TARGET_ASSETS",
    "TARGET_COLUMN",
    "build_candidate_callbacks",
    "build_lgbm_regressor",
    "load_lightgbm_contract",
    "load_lightgbm_runtime",
    "resolve_estimator_parameters",
    "validate_best_iteration",
]
