from __future__ import annotations

from collections import Counter
from dataclasses import FrozenInstanceError
import io
import math
from pathlib import Path

import pytest

from nlpo_toolkit.corpus_analysis import cli
from nlpo_toolkit.corpus_analysis.analysis_policy import AnalysisExtractionPolicy
from nlpo_toolkit.corpus_analysis.analysis_records import NLPAnalysisRecord
from nlpo_toolkit.corpus_analysis.corpus import PreparedCorpus
from nlpo_toolkit.corpus_analysis.features.engine import build_feature_matrix
from nlpo_toolkit.corpus_analysis.features.errors import FeatureError
from nlpo_toolkit.corpus_analysis.features.lexical_diversity import (
    compute_hdd,
    compute_lexical_diversity_features,
    compute_mattr,
    compute_msttr,
    compute_mtld,
)
from nlpo_toolkit.corpus_analysis.features.models import (
    FeatureCommandResult,
    FeatureOptions,
    FeatureRequest,
    FeatureSamplingOptions,
    LexicalDiversityOptions,
)
from nlpo_toolkit.corpus_analysis.cli.output import write_feature_result
from nlpo_toolkit.nlp.contracts import NLPDocument, NLPSentence, NLPToken


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"window_size": 0}, "window_size"),
        ({"window_size": -1}, "window_size"),
        ({"window_size": True}, "window_size"),
        ({"hdd_sample_size": 0}, "hdd_sample_size"),
        ({"hdd_sample_size": True}, "hdd_sample_size"),
        ({"mtld_threshold": 0.0}, "mtld_threshold"),
        ({"mtld_threshold": 1.0}, "mtld_threshold"),
        ({"mtld_threshold": -0.1}, "mtld_threshold"),
        ({"mtld_threshold": 1.1}, "mtld_threshold"),
        ({"mtld_threshold": True}, "mtld_threshold"),
        ({"mtld_threshold": math.nan}, "mtld_threshold"),
        ({"mtld_threshold": math.inf}, "mtld_threshold"),
        ({"mtld_threshold": -math.inf}, "mtld_threshold"),
    ),
)
def test_options_validate_strictly(kwargs, message: str) -> None:
    with pytest.raises(FeatureError, match=message):
        LexicalDiversityOptions(**kwargs)


def test_options_are_frozen_and_preserve_valid_values() -> None:
    options = LexicalDiversityOptions(50, 0.5, 20)
    assert (options.window_size, options.mtld_threshold, options.hdd_sample_size) == (
        50,
        0.5,
        20,
    )
    with pytest.raises(FrozenInstanceError):
        options.window_size = 10  # type: ignore[misc]


@pytest.mark.parametrize(
    ("values", "window_size", "expected"),
    (
        ((), 2, 0.0),
        (("a",), 2, 1.0),
        (("a", "a"), 2, 0.5),
        (("a", "a", "b"), 2, 0.75),
        (("a", "a", "a", "a"), 2, 0.5),
        (("a", "b", "c", "d"), 2, 1.0),
        (("a", "b", "a", "c"), 3, pytest.approx(5 / 6)),
    ),
)
def test_mattr(values, window_size: int, expected) -> None:
    source = list(values)
    assert compute_mattr(source, window_size=window_size) == expected
    assert source == list(values)


@pytest.mark.parametrize(
    ("values", "segment_size", "expected"),
    (
        ((), 2, 0.0),
        (("a",), 2, 1.0),
        (("a", "a"), 2, 0.5),
        (("a", "a", "b", "c", "c"), 2, 0.75),
        (("a", "b", "c", "d", "a"), 2, 1.0),
    ),
)
def test_msttr(values, segment_size: int, expected: float) -> None:
    assert compute_msttr(values, segment_size=segment_size) == expected


def test_mtld_special_cases_and_bidirectional_symmetry() -> None:
    assert compute_mtld((), threshold=0.72) == 0.0
    assert compute_mtld(("a",), threshold=0.72) == 1.0
    assert compute_mtld(("a", "b", "c", "d"), threshold=0.72) == 4.0
    repeated = compute_mtld(("a", "a", "a", "a"), threshold=0.72)
    assert repeated > 0.0 and math.isfinite(repeated)
    values = ("a", "b", "a", "c", "c", "d")
    assert compute_mtld(values, threshold=0.72) == pytest.approx(
        compute_mtld(tuple(reversed(values)), threshold=0.72)
    )


def test_hdd_examples_bounds_and_short_sequence_fallback() -> None:
    assert compute_hdd((), sample_size=42) == 0.0
    assert compute_hdd(("a",), sample_size=42) == 1.0
    assert compute_hdd(("a", "b", "c"), sample_size=42) == 1.0
    assert compute_hdd(("a", "a", "a", "a"), sample_size=4) == 0.25
    assert compute_hdd(("a", "a", "b", "b"), sample_size=2) == pytest.approx(5 / 6)
    value = compute_hdd(("a", "a", "b", "c"), sample_size=2)
    assert 0.0 <= value <= 1.0


def _record(token: str, lemma: str | None) -> NLPAnalysisRecord:
    return NLPAnalysisRecord(
        chunk_index=0,
        sentence_index=0,
        token_index=0,
        global_token_index=0,
        char_start_in_chunk=None,
        char_end_in_chunk=None,
        char_start_in_text=None,
        char_end_in_text=None,
        sentence="",
        token=token,
        lemma=lemma,
        upos="NOUN",
    )


def test_feature_mapping_builds_token_and_lemma_sequences_once_in_fixed_order() -> None:
    records = (
        _record(" A ", "same"),
        _record("B", "same"),
        _record("a", None),
    )
    row = compute_lexical_diversity_features(
        records, options=LexicalDiversityOptions(window_size=2, hdd_sample_size=2)
    )

    assert tuple(row) == (
        "mattr_token",
        "mattr_lemma",
        "msttr_token",
        "msttr_lemma",
        "mtld_token",
        "mtld_lemma",
        "hdd_token",
        "hdd_lemma",
    )
    assert row["mattr_token"] != row["mattr_lemma"]
    assert all(isinstance(value, float) for value in row.values())


def test_empty_records_produce_eight_zero_values() -> None:
    row = compute_lexical_diversity_features((), options=LexicalDiversityOptions())
    assert len(row) == 8
    assert set(row.values()) == {0.0}


class TokenNLP:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, text: str) -> NLPDocument:
        self.calls += 1
        return NLPDocument(
            (
                NLPSentence(
                    tuple(NLPToken(value, value, "NOUN") for value in text.split()),
                    text,
                ),
            ),
            text,
        )


def _prepared(text: str) -> PreparedCorpus:
    return PreparedCorpus("g", (Path("a.txt"),), text, text, Counter())


def test_engine_adds_family_only_when_enabled_and_works_without_basic() -> None:
    backend = TokenNLP()
    disabled = build_feature_matrix(
        corpora=(_prepared("a a b"),),
        nlp=backend,
        extraction_policy=AnalysisExtractionPolicy(),
        options=FeatureOptions(include_upos=False),
    )[0]
    enabled = build_feature_matrix(
        corpora=(_prepared("a a b"),),
        nlp=backend,
        extraction_policy=AnalysisExtractionPolicy(),
        options=FeatureOptions(
            include_basic=False,
            include_upos=False,
            lexical_diversity=LexicalDiversityOptions(window_size=2),
        ),
    )[0]

    assert "mattr_token" not in disabled
    assert tuple(enabled) == (
        "group",
        "mattr_token",
        "mattr_lemma",
        "msttr_token",
        "msttr_lemma",
        "mtld_token",
        "mtld_lemma",
        "hdd_token",
        "hdd_lemma",
    )
    assert backend.calls == 2


def test_fixed_samples_compute_independent_diversity_without_repeating_nlp() -> None:
    backend = TokenNLP()
    rows = build_feature_matrix(
        corpora=(_prepared("a a b c"),),
        nlp=backend,
        extraction_policy=AnalysisExtractionPolicy(),
        options=FeatureOptions(
            include_basic=False,
            include_upos=False,
            sampling=FeatureSamplingOptions(2),
            lexical_diversity=LexicalDiversityOptions(window_size=100),
        ),
    )

    assert backend.calls == 1
    assert [row["mattr_token"] for row in rows] == [0.5, 1.0]
    assert [row["msttr_token"] for row in rows] == [0.5, 1.0]


def test_field_selection_does_not_change_lexical_diversity() -> None:
    options = LexicalDiversityOptions(window_size=2, hdd_sample_size=2)
    rows = tuple(
        build_feature_matrix(
            corpora=(_prepared("a a b"),),
            nlp=TokenNLP(),
            extraction_policy=AnalysisExtractionPolicy(),
            options=FeatureOptions(
                field=field,
                mfw=1,
                lexical_diversity=options,
            ),
        )[0]
        for field in ("lemma", "token")
    )
    columns = (
        "mattr_token",
        "mattr_lemma",
        "msttr_token",
        "msttr_lemma",
        "mtld_token",
        "mtld_lemma",
        "hdd_token",
        "hdd_lemma",
    )

    assert tuple(rows[0][column] for column in columns) == tuple(
        rows[1][column] for column in columns
    )


def test_csv_and_tsv_preserve_the_same_lexical_diversity_column_order() -> None:
    row = build_feature_matrix(
        corpora=(_prepared("a a b"),),
        nlp=TokenNLP(),
        extraction_policy=AnalysisExtractionPolicy(),
        options=FeatureOptions(
            include_basic=False,
            include_upos=False,
            lexical_diversity=LexicalDiversityOptions(window_size=2),
        ),
    )[0]
    csv_stream = io.StringIO()
    tsv_stream = io.StringIO()

    write_feature_result(
        FeatureCommandResult((row,)), stream=csv_stream, output_format="csv"
    )
    write_feature_result(
        FeatureCommandResult((row,)), stream=tsv_stream, output_format="tsv"
    )

    assert csv_stream.getvalue().splitlines()[0].split(",") == list(row)
    assert tsv_stream.getvalue().splitlines()[0].split("\t") == list(row)


def test_cli_enables_and_composes_lexical_diversity_options(monkeypatch) -> None:
    import nlpo_toolkit.corpus_analysis.cli.features as feature_cli

    requests: list[FeatureRequest] = []

    def execute(request, *, dependencies):
        requests.append(request)
        return FeatureCommandResult(())

    monkeypatch.setattr(feature_cli, "execute_feature_command", execute)
    monkeypatch.setattr(
        feature_cli, "default_feature_command_dependencies", lambda: object()
    )
    assert (
        cli.main(
            [
                "features",
                "--lexdiv-window",
                "50",
                "--mtld-threshold",
                "0.5",
                "--hdd-sample-size",
                "20",
            ],
            stdout=io.StringIO(),
            stderr=io.StringIO(),
        )
        == 0
    )
    assert requests[0].lexical_diversity == LexicalDiversityOptions(50, 0.5, 20)


@pytest.mark.parametrize(
    ("argument", "value", "expected"),
    (
        ("--lexical-diversity", None, LexicalDiversityOptions()),
        ("--lexdiv-window", "25", LexicalDiversityOptions(window_size=25)),
        (
            "--mtld-threshold",
            "0.5",
            LexicalDiversityOptions(mtld_threshold=0.5),
        ),
        (
            "--hdd-sample-size",
            "20",
            LexicalDiversityOptions(hdd_sample_size=20),
        ),
    ),
)
def test_each_cli_argument_enables_lexical_diversity(
    monkeypatch, argument: str, value: str | None, expected: LexicalDiversityOptions
) -> None:
    import nlpo_toolkit.corpus_analysis.cli.features as feature_cli

    requests: list[FeatureRequest] = []

    def execute(request, *, dependencies):
        requests.append(request)
        return FeatureCommandResult(())

    monkeypatch.setattr(feature_cli, "execute_feature_command", execute)
    monkeypatch.setattr(
        feature_cli, "default_feature_command_dependencies", lambda: object()
    )
    argv = ["features", argument]
    if value is not None:
        argv.append(value)

    assert cli.main(argv, stdout=io.StringIO(), stderr=io.StringIO()) == 0
    assert requests[0].lexical_diversity == expected


def test_cli_invalid_lexical_diversity_value_returns_one() -> None:
    stderr = io.StringIO()
    assert (
        cli.main(
            ["features", "--lexdiv-window", "0"],
            stdout=io.StringIO(),
            stderr=stderr,
        )
        == 1
    )
    assert "window_size must be a positive integer" in stderr.getvalue()
