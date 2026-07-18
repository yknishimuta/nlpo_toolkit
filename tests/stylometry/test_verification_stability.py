from __future__ import annotations

import io
import json
import math
import random
from pathlib import Path

import pytest

from nlpo_toolkit.corpus_analysis import cli
from nlpo_toolkit.stylometry.authorship import build_work_profiles
from nlpo_toolkit.stylometry.errors import StylometryError
from nlpo_toolkit.stylometry.evaluation_models import (
    LabeledFeatureDataset,
    LabeledFeatureObservation,
)
from nlpo_toolkit.stylometry.stability_engine import (
    evaluate_verification_stability,
    summarize_decisions,
    summarize_distribution,
)
from nlpo_toolkit.stylometry.stability_models import (
    ResamplingAxis,
    ResamplingIntervalSettings,
    VerificationStabilitySettings,
    VerificationStabilityStatus,
)
from nlpo_toolkit.stylometry.stability_resampling import (
    bootstrap_work_profiles,
    derive_iteration_seed,
    subsample_feature_names,
    subsample_reference_works,
)
from nlpo_toolkit.stylometry.verification_models import (
    VerificationDecision,
    VerificationThresholdSettings,
)
from nlpo_toolkit.stylometry.verification import evaluate_verification


def _dataset() -> LabeledFeatureDataset:
    rows = []
    specifications = (
        ("A1", "A", (0.0, 1.0, 2.0)),
        ("A2", "A", (1.0, 2.0, 2.5)),
        ("A3", "A", (2.0, 1.5, 3.0)),
        ("A4", "A", (3.0, 3.0, 1.0)),
        ("A5", "A", (4.0, 2.0, 4.0)),
        ("B1", "B", (8.0, 7.0, 5.0)),
        ("B2", "B", (9.0, 5.0, 7.0)),
        ("C1", "C", (10.0, 8.0, 6.0)),
        ("C2", "C", (11.0, 6.0, 8.0)),
        ("Q", "unknown", (2.0, 2.0, 2.0)),
    )
    for work, author, values in specifications:
        for index, offset in enumerate((0.0, 0.25)):
            rows.append(
                LabeledFeatureObservation(
                    f"{work}_{index}",
                    author,
                    work,
                    tuple(value + offset for value in values),
                )
            )
    return LabeledFeatureDataset(("f1", "f2", "f3"), tuple(rows))


def _settings(seed: int = 7) -> VerificationStabilitySettings:
    return VerificationStabilitySettings(
        (ResamplingAxis.WORKS, ResamplingAxis.SAMPLES, ResamplingAxis.FEATURES),
        iterations=12,
        seed=seed,
        work_fraction=0.6,
        feature_fraction=0.67,
    )


def test_same_seed_is_reproducible_and_different_seed_changes_replicates() -> None:
    first = evaluate_verification_stability(
        _dataset(),
        candidate_author="A",
        query_work="Q",
        verification_thresholds=VerificationThresholdSettings(),
        settings=_settings(17),
    )
    second = evaluate_verification_stability(
        _dataset(),
        candidate_author="A",
        query_work="Q",
        verification_thresholds=VerificationThresholdSettings(),
        settings=_settings(17),
    )
    different = evaluate_verification_stability(
        _dataset(),
        candidate_author="A",
        query_work="Q",
        verification_thresholds=VerificationThresholdSettings(),
        settings=_settings(18),
    )
    assert first == second
    assert first.replicates != different.replicates
    assert first.successful_iterations == 12
    assert (
        sum(
            (
                first.decision_stability.accept_count,
                first.decision_stability.inconclusive_count,
                first.decision_stability.reject_count,
            )
        )
        == 12
    )
    assert sum(item.nearest_count for item in first.nearest_background_frequency) == 12
    ordinary = evaluate_verification(
        _dataset().feature_names,
        build_work_profiles(_dataset()),
        candidate_author="A",
        query_work="Q",
        settings=VerificationThresholdSettings(),
    )
    assert first.base_result == ordinary


def test_resampling_primitives_are_stratified_ordered_and_work_local() -> None:
    profiles = build_work_profiles(_dataset())
    candidate = tuple(item for item in profiles if item.author == "A")
    chosen = subsample_reference_works(
        candidate, fraction=0.5, minimum=3, rng=random.Random(2)
    )
    assert len(chosen) == 3
    assert tuple(item.work_id for item in chosen) == tuple(
        sorted(item.work_id for item in chosen)
    )
    bootstrapped = bootstrap_work_profiles(
        _dataset(), included_work_ids=frozenset(("A1",)), rng=random.Random(1)
    )
    assert len(bootstrapped) == 1
    assert bootstrapped[0].work_id == "A1"
    assert len(bootstrapped[0].observation_ids) == 2
    selected = subsample_feature_names(
        ("a", "b", "c", "d"), fraction=0.5, rng=random.Random(3)
    )
    assert len(selected) == 2
    assert selected == tuple(name for name in ("a", "b", "c", "d") if name in selected)
    assert derive_iteration_seed(1, 2, 3) == derive_iteration_seed(1, 2, 3)


def test_distribution_and_modal_tie_are_defined() -> None:
    summary = summarize_distribution((1.0, 2.0, 3.0, 4.0), 0.25, 0.75)
    assert summary.mean == pytest.approx(2.5)
    assert summary.median == pytest.approx(2.5)
    assert summary.sample_standard_deviation == pytest.approx(math.sqrt(5 / 3))
    decisions = summarize_decisions(
        (VerificationDecision.ACCEPT, VerificationDecision.INCONCLUSIVE),
        base_decision=VerificationDecision.ACCEPT,
        stability_threshold=0.5,
    )
    assert decisions.modal_decision is VerificationDecision.INCONCLUSIVE
    assert decisions.status is VerificationStabilityStatus.STABLE


def test_cli_outputs_deterministic_json_and_tables(tmp_path: Path) -> None:
    features = tmp_path / "features.csv"
    metadata = tmp_path / "metadata.csv"
    dataset = _dataset()
    features.write_text(
        "sample_id,f1,f2,f3\n"
        + "".join(
            f"{item.identifier},{item.values[0]},{item.values[1]},{item.values[2]}\n"
            for item in dataset.observations
        ),
        encoding="utf-8",
    )
    metadata.write_text(
        "sample_id,author,work\n"
        + "".join(
            f"{item.identifier},{item.author},{item.work_id}\n"
            for item in dataset.observations
        ),
        encoding="utf-8",
    )
    replicates = tmp_path / "replicates.csv"
    stability = tmp_path / "features.csv.out"
    stdout = io.StringIO()
    stderr = io.StringIO()
    arguments = [
        "stylometry",
        "verify-stability",
        "--features",
        str(features),
        "--metadata",
        str(metadata),
        "--id-column",
        "sample_id",
        "--feature-prefix",
        "f",
        "--candidate-author",
        "A",
        "--query-work",
        "Q",
        "--resample-axis",
        "works",
        "--resample-axis",
        "features",
        "--work-fraction",
        "0.6",
        "--feature-fraction",
        "0.67",
        "--iterations",
        "5",
        "--seed",
        "9",
        "--replicates-out",
        str(replicates),
        "--feature-stability-out",
        str(stability),
    ]
    assert cli.main(arguments, stdout=stdout, stderr=stderr) == 0
    value = json.loads(stdout.getvalue())
    assert value["resampling"]["successful_iterations"] == 5
    assert value["method"] == "candidate_authorship_verification_stability"
    assert (
        replicates.read_text(encoding="utf-8")
        .splitlines()[0]
        .startswith("iteration,attempt")
    )
    assert (
        stability.read_text(encoding="utf-8")
        .splitlines()[0]
        .startswith("feature,selected_count")
    )
    assert "Base verification decision:" in stderr.getvalue()


def test_duplicate_axes_and_unchangeable_settings_are_rejected(tmp_path: Path) -> None:
    stderr = io.StringIO()
    assert (
        cli.main(
            [
                "stylometry",
                "verify-stability",
                "--features",
                str(tmp_path / "x"),
                "--metadata",
                str(tmp_path / "y"),
                "--feature-prefix",
                "f",
                "--candidate-author",
                "A",
                "--query-work",
                "Q",
                "--resample-axis",
                "works",
                "--resample-axis",
                "works",
            ],
            stdout=io.StringIO(),
            stderr=stderr,
        )
        == 1
    )
    assert "unique" in stderr.getvalue()

    with pytest.raises(SystemExit) as help_exit:
        cli.main(["stylometry", "verify-stability", "--help"])
    assert help_exit.value.code == 0


def test_input_output_path_collision_is_rejected_before_writing(tmp_path: Path) -> None:
    input_path = tmp_path / "features.csv"
    stderr = io.StringIO()
    code = cli.main(
        [
            "stylometry",
            "verify-stability",
            "--features",
            str(input_path),
            "--metadata",
            str(tmp_path / "metadata.csv"),
            "--feature-prefix",
            "f",
            "--candidate-author",
            "A",
            "--query-work",
            "Q",
            "--resample-axis",
            "features",
            "--out",
            str(input_path),
        ],
        stdout=io.StringIO(),
        stderr=stderr,
    )
    assert code == 1
    assert "all be different" in stderr.getvalue()
    assert not input_path.exists()


def test_interval_validation() -> None:
    with pytest.raises(StylometryError):
        ResamplingIntervalSettings(0.6, 0.9)


def test_zero_variance_attempt_is_retried_and_max_attempts_is_enforced(
    monkeypatch,
) -> None:
    from nlpo_toolkit.stylometry import stability_engine

    original = stability_engine.evaluate_verification
    calls = 0

    def reject_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise StylometryError(
                "all selected features have zero variance in verification reference works"
            )
        return original(*args, **kwargs)

    monkeypatch.setattr(stability_engine, "evaluate_verification", reject_once)
    result = evaluate_verification_stability(
        _dataset(),
        candidate_author="A",
        query_work="Q",
        verification_thresholds=VerificationThresholdSettings(),
        settings=VerificationStabilitySettings(
            (ResamplingAxis.FEATURES,),
            iterations=2,
            max_attempts=4,
            feature_fraction=0.66,
        ),
    )
    assert result.attempted_iterations == 3
    assert result.rejected_attempts == 1
    assert result.rejected_attempt_reasons[0].count == 1

    calls = 0

    def reject_always_after_base(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise StylometryError(
                "all selected features have zero variance in verification reference works"
            )
        return original(*args, **kwargs)

    monkeypatch.setattr(
        stability_engine, "evaluate_verification", reject_always_after_base
    )
    with pytest.raises(StylometryError, match="max attempts"):
        evaluate_verification_stability(
            _dataset(),
            candidate_author="A",
            query_work="Q",
            verification_thresholds=VerificationThresholdSettings(),
            settings=VerificationStabilitySettings(
                (ResamplingAxis.FEATURES,),
                iterations=1,
                max_attempts=1,
                feature_fraction=0.66,
            ),
        )
