from __future__ import annotations

from collections.abc import Iterator
from typing import Literal

from .errors import FeatureError
from .models import (
    AnalyzedFeatureCorpus,
    FeatureSampleMetadata,
    FeatureSamplingOptions,
)


def iter_feature_window_ranges(
    token_count: int,
    *,
    options: FeatureSamplingOptions,
) -> Iterator[tuple[int, int, Literal["full", "partial"]]]:
    if not options.enabled:
        return
    window = options.window_tokens
    step = options.effective_step_tokens
    if window is None or step is None:
        raise AssertionError("enabled sampling must resolve window and step")
    start = 0
    while start < token_count:
        end = start + window
        if end <= token_count:
            yield start, end, "full"
        elif options.include_partial:
            yield start, token_count, "partial"
            return
        else:
            return
        start += step


def _sample_char_count(
    corpus: AnalyzedFeatureCorpus, raw_start: int, raw_end: int
) -> int:
    raw_span = corpus.raw_records[raw_start:raw_end]
    first_start = raw_span[0].char_start_in_text
    last_end = raw_span[-1].char_end_in_text
    if first_start is not None and last_end is not None and last_end >= first_start:
        return last_end - first_start
    return sum(len(record.token) for record in raw_span)


def _sample_text(
    corpus: AnalyzedFeatureCorpus, raw_start: int, raw_end: int
) -> str | None:
    raw_span = corpus.raw_records[raw_start:raw_end]
    first_start = raw_span[0].char_start_in_text
    last_end = raw_span[-1].char_end_in_text
    if (
        first_start is None
        or last_end is None
        or first_start < 0
        or last_end < first_start
        or last_end > len(corpus.source.prepared_text)
    ):
        return None
    return corpus.source.prepared_text[first_start:last_end]


def _lexical_raw_indices(corpus: AnalyzedFeatureCorpus) -> tuple[int, ...]:
    indices: list[int] = []
    raw_offset = 0
    for lexical_record in corpus.lexical_records:
        while (
            raw_offset < len(corpus.raw_records)
            and corpus.raw_records[raw_offset] is not lexical_record
        ):
            raw_offset += 1
        if raw_offset >= len(corpus.raw_records):
            raise ValueError("lexical record is not present in raw record order")
        indices.append(raw_offset)
        raw_offset += 1
    return tuple(indices)


def sample_feature_corpus(
    corpus: AnalyzedFeatureCorpus,
    *,
    options: FeatureSamplingOptions,
) -> tuple[AnalyzedFeatureCorpus, ...]:
    if not options.enabled:
        return (corpus,)
    if len(corpus.source.files) != 1:
        raise FeatureError(
            "fixed-token sampling requires one source file per prepared corpus; "
            "use --group-by-file or grouping.mode: per_file"
        )
    samples: list[AnalyzedFeatureCorpus] = []
    lexical_raw_indices = _lexical_raw_indices(corpus)
    for sample_index, (start, end, kind) in enumerate(
        iter_feature_window_ranges(len(corpus.lexical_records), options=options),
        start=1,
    ):
        eligible_indices = lexical_raw_indices[start:end]
        raw_start = eligible_indices[0]
        raw_end = eligible_indices[-1] + 1
        raw_span = corpus.raw_records[raw_start:raw_end]
        samples.append(
            AnalyzedFeatureCorpus(
                source=corpus.source,
                raw_records=raw_span,
                lexical_records=corpus.lexical_records[start:end],
                char_count=_sample_char_count(corpus, raw_start, raw_end),
                text=_sample_text(corpus, raw_start, raw_end),
                sample=FeatureSampleMetadata(
                    source_file=str(corpus.source.files[0]),
                    sample_id=f"{corpus.source.label}__sample_{sample_index:04d}",
                    sample_index=sample_index,
                    start_token=start,
                    end_token=end,
                    kind=kind,
                ),
            )
        )
    return tuple(samples)
