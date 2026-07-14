from __future__ import annotations

from collections import Counter

from nlpo_toolkit.corpus_analysis.analysis_execution import consume_analysis_records
from nlpo_toolkit.corpus_analysis.analysis_records import AnalysisOptions, NLPAnalysisRecord


class _WriterSpy:
    def __init__(self) -> None:
        self.records = []

    def write(self, record) -> None:
        self.records.append(record)

    def consider(self, record) -> None:
        self.records.append(record)


def _record(token: str, upos: str, index: int) -> NLPAnalysisRecord:
    return NLPAnalysisRecord(
        chunk_index=0,
        sentence_index=0,
        token_index=index,
        global_token_index=index,
        char_start_in_chunk=None,
        char_end_in_chunk=None,
        char_start_in_text=None,
        char_end_in_text=None,
        sentence="rosa et",
        token=token,
        lemma=token,
        upos=upos,
    )


def test_consume_records_iterates_once_and_writes_every_evaluated_record() -> None:
    iterations = 0

    def records():
        nonlocal iterations
        iterations += 1
        yield _record("rosa", "NOUN", 0)
        yield _record("et", "CCONJ", 1)

    artifact = _WriterSpy()
    trace = _WriterSpy()
    result = consume_analysis_records(
        records=records(),
        options=AnalysisOptions(
            group="text",
            source_files=(),
            use_lemma=True,
            upos_targets=frozenset({"NOUN"}),
        ),
        artifact_writer=artifact,  # type: ignore[arg-type]
        trace_writer=trace,  # type: ignore[arg-type]
    )

    assert iterations == 1
    assert result.counter == Counter({"rosa": 1})
    assert result.record_count == 2
    assert len(artifact.records) == 2
    assert len(trace.records) == 2
