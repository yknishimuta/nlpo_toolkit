from __future__ import annotations

from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from typing import Iterable, Iterator, Mapping
from nlpo_toolkit.backends import NLPBackendInfo

from .analysis_cache import (
    AnalysisCacheGroupResult,
    AnalysisCacheRunStats,
    AnalysisFingerprint,
    build_analysis_cache_key,
    get_or_compute_analysis_records,
    prepared_text_sha256,
)
from .analysis_policy import AnalysisExtractionPolicy
from .analysis_records import (
    AnalysisOptions,
    NLPAnalysisRecord,
    evaluate_analysis_record,
    iter_nlp_analysis_records_from_text,
)
from .config import TraceConfig
from .corpus import PreparedCorpus, prepare_corpora, sanitize_label
from .diagnostic_trace import DiagnosticTraceWriter
from .outputs import (
    FrequencyOutputPaths,
    build_frequency_output_paths,
    write_frequency_csv,
)
from .runner_types import (
    AnalysisResults,
    DictCheckOutput,
    GroupAnalysisResult,
    RunContext,
    RunnerDependencies,
)
from .token_artifact import (
    TokenArtifactMetadata,
    TokenArtifactWriter,
    token_artifact_metadata_path,
)


def _trace_base_path(
    trace_cfg: TraceConfig,
    out_dir: Path,
    project_root: Path,
) -> Path:
    if trace_cfg.path:
        trace_path = Path(str(trace_cfg.path))
        if not trace_path.is_absolute():
            trace_path = (project_root / trace_path).resolve()
    else:
        trace_path = out_dir / "trace.tsv"
    return trace_path


def trace_path_for_work_item(
    *,
    base_path: Path,
    label: str,
    work_item_count: int,
    force_label_suffix: bool = False,
) -> Path:
    if work_item_count <= 1 and not force_label_suffix:
        return base_path
    safe_label = sanitize_label(label)
    suffix = base_path.suffix or ".tsv"
    stem = base_path.stem or "trace"
    return base_path.with_name(f"{stem}_{safe_label}{suffix}")


def build_trace_paths(
    *,
    base_path: Path,
    labels: list[str],
) -> dict[str, Path]:
    counts: dict[str, int] = {}
    paths: dict[str, Path] = {}
    work_item_count = len(labels)
    for label in labels:
        safe_label = sanitize_label(label)
        counts[safe_label] = counts.get(safe_label, 0) + 1
        effective_label = safe_label if counts[safe_label] == 1 else f"{safe_label}_{counts[safe_label]}"
        paths[label] = trace_path_for_work_item(
            base_path=base_path,
            label=effective_label,
            work_item_count=work_item_count,
        )
    return paths


def build_token_artifact_paths(
    *,
    base_path: Path,
    labels: list[str],
) -> dict[str, Path]:
    return build_trace_paths(base_path=base_path, labels=labels)


def _token_artifact_base_path(
    path_value: str,
    project_root: Path,
) -> Path:
    path = Path(str(path_value))
    if not path.is_absolute():
        path = (project_root / path).resolve()
    if not path.suffix:
        path = path.with_suffix(".tsv")
    return path


def _validate_trace_artifact_paths(
    *,
    trace_paths: Mapping[str, Path],
    token_artifact_paths: Mapping[str, Path],
) -> None:
    trace_resolved = {Path(path).resolve() for path in trace_paths.values()}
    for path in token_artifact_paths.values():
        resolved = Path(path).resolve()
        if resolved in trace_resolved:
            raise ValueError(
                f"Token artifact path and diagnostic trace path must be different: {resolved}"
            )


def apply_lemma_normalization(
    counter: Counter[str],
    normalization_map: Mapping[str, str],
) -> Counter[str]:
    normalized: Counter[str] = Counter()
    for lemma, count in counter.items():
        normalized[normalization_map.get(lemma, lemma)] += count
    return normalized


def split_known_unknown(
    counter: Counter[str],
    known: Iterable[str],
) -> tuple[Counter[str], Counter[str]]:
    known_set = set(known)
    known_counter = Counter({w: n for (w, n) in counter.items() if w in known_set})
    unknown_counter = Counter({w: n for (w, n) in counter.items() if w not in known_set})
    return known_counter, unknown_counter


def _load_lemma_normalization_map(context: RunContext) -> Mapping[str, str] | None:
    plan = context.plan
    norm_map_rel_path = plan.config.dictcheck.lemma_normalize
    if not norm_map_rel_path:
        return None
    norm_map_path = Path(str(norm_map_rel_path))
    if not norm_map_path.is_absolute():
        norm_map_path = (plan.project_root / norm_map_path).resolve()
    if not norm_map_path.exists():
        return None
    from .dictcheck import load_lemma_normalize_map

    return load_lemma_normalize_map(norm_map_path)


def _text_for_counting(context: RunContext, corpus: PreparedCorpus) -> str:
    if context.splitter_nlp is None:
        return corpus.prepared_text
    doc = context.splitter_nlp(corpus.prepared_text)
    joined = "\n".join([s.text for s in getattr(doc, "sentences", [])])
    if not joined.strip():
        return corpus.prepared_text
    return joined


def _token_artifact_metadata(
    *,
    context: RunContext,
    corpus: PreparedCorpus,
    path: Path,
) -> TokenArtifactMetadata:
    plan = context.plan
    return TokenArtifactMetadata(
        group=corpus.label,
        source_files=tuple(str(file) for file in corpus.files),
        analysis_unit=plan.analysis_unit,
        upos_targets=tuple(sorted(plan.config.filters.upos_targets)),
        nlp=context.backend_info.to_dict(),
        filters={
            "min_token_length": plan.config.filters.min_token_length,
            "drop_roman_numerals": plan.config.filters.drop_roman_numerals,
        },
        artifact_path=str(path.resolve()),
    )


def _analysis_cache_dir(context: RunContext) -> Path:
    plan = context.plan
    cache_dir = Path(plan.config.analysis_cache.directory)
    if not cache_dir.is_absolute():
        cache_dir = plan.project_root / cache_dir
    return cache_dir.resolve()


def build_analysis_fingerprint(
    *,
    backend_info: NLPBackendInfo,
    policy: AnalysisExtractionPolicy,
) -> AnalysisFingerprint:
    return AnalysisFingerprint(
        backend=backend_info.name,
        language=backend_info.language,
        model=backend_info.model,
        package=backend_info.package,
        processors=policy.processors,
        chunk_size=policy.chunk_chars,
        chunk_strategy=policy.chunk_strategy,
        device=backend_info.device,
    )


def build_analysis_options(
    *,
    context: RunContext,
    corpus: PreparedCorpus,
) -> AnalysisOptions:
    plan = context.plan
    return AnalysisOptions(
        group=corpus.label,
        source_files=tuple(corpus.files),
        use_lemma=plan.use_lemma,
        upos_targets=frozenset(plan.config.filters.upos_targets),
        min_token_length=plan.config.filters.min_token_length,
        drop_roman_numerals=plan.config.filters.drop_roman_numerals,
        roman_exceptions=context.roman_exceptions,
    )


def obtain_analysis_records(
    *,
    context: RunContext,
    corpus: PreparedCorpus,
    text: str,
    analysis_cache_stats: AnalysisCacheRunStats | None = None,
) -> tuple[Iterator[NLPAnalysisRecord], str, str]:
    policy = context.extraction_policy
    fingerprint = build_analysis_fingerprint(
        backend_info=context.backend_info,
        policy=policy,
    )
    text_hash = prepared_text_sha256(text)
    cache_key = build_analysis_cache_key(
        prepared_text_sha256=text_hash,
        fingerprint=fingerprint,
    )

    if context.plan.config.analysis_cache.enabled:
        raw_records, cache_status, _payload_path, _metadata_path = get_or_compute_analysis_records(
            cache_dir=_analysis_cache_dir(context),
            cache_key=cache_key,
            prepared_text_sha256=text_hash,
            prepared_text_length=len(text),
            fingerprint=fingerprint,
            compute_records=lambda: iter_nlp_analysis_records_from_text(
                text=text,
                nlp=context.nlp,
                policy=policy,
            ),
            lock_timeout_sec=context.plan.config.analysis_cache.lock_timeout_sec,
        )
        return raw_records, cache_status, cache_key

    return (
        iter_nlp_analysis_records_from_text(
            text=text,
            nlp=context.nlp,
            policy=policy,
        ),
        "disabled",
        cache_key,
    )


def consume_analysis_records(
    *,
    records: Iterable[NLPAnalysisRecord],
    options: AnalysisOptions,
    artifact_writer: TokenArtifactWriter | None,
    trace_writer: DiagnosticTraceWriter | None,
) -> tuple[Counter[str], int]:
    counter: Counter[str] = Counter()
    record_count = 0

    for raw_record in records:
        record_count += 1
        record = evaluate_analysis_record(raw_record, options=options)
        if artifact_writer is not None:
            artifact_writer.write(record)
        if trace_writer is not None:
            trace_writer.consider(record)
        if record.included and record.analysis_key:
            counter[record.analysis_key] += 1

    return counter, record_count


def _count_corpus_records(
    *,
    context: RunContext,
    corpus: PreparedCorpus,
    text: str,
    token_artifact_path: Path | None,
    trace_path: Path | None,
    analysis_cache_stats: AnalysisCacheRunStats | None = None,
) -> tuple[Counter[str], Mapping[str, object] | None, tuple[Path, ...]]:
    artifact_meta: Mapping[str, object] | None = None
    generated: list[Path] = []

    with ExitStack() as stack:
        artifact_writer: TokenArtifactWriter | None = None
        if token_artifact_path is not None:
            artifact_writer = stack.enter_context(
                TokenArtifactWriter(
                    token_artifact_path,
                    metadata=_token_artifact_metadata(
                        context=context,
                        corpus=corpus,
                        path=token_artifact_path,
                    ),
                )
            )

        trace_writer: DiagnosticTraceWriter | None = None
        if trace_path is not None:
            trace_writer = stack.enter_context(
                DiagnosticTraceWriter(
                    trace_path,
                    max_rows=int(context.plan.config.trace.max_rows or 0),
                    only_keys=context.plan.config.trace.only_keys,
                    write_truncation_marker=context.plan.config.trace.write_truncation_marker,
                )
            )

        raw_records, cache_status, cache_key = obtain_analysis_records(
            context=context,
            corpus=corpus,
            text=text,
            analysis_cache_stats=analysis_cache_stats,
        )
        counter, record_count = consume_analysis_records(
            records=raw_records,
            options=build_analysis_options(context=context, corpus=corpus),
            artifact_writer=artifact_writer,
            trace_writer=trace_writer,
        )

        if analysis_cache_stats is not None:
            if cache_status == "hit":
                analysis_cache_stats.hits += 1
                analysis_cache_stats.records_read += record_count
            elif cache_status == "miss":
                analysis_cache_stats.misses += 1
                analysis_cache_stats.objects_written += 1
                analysis_cache_stats.records_written += record_count
            analysis_cache_stats.groups.append(
                AnalysisCacheGroupResult(
                    group=corpus.label,
                    status=cache_status,
                    cache_key=cache_key,
                    record_count=record_count,
                )
            )

    if token_artifact_path is not None:
        metadata_path = token_artifact_metadata_path(token_artifact_path)
        generated.extend((token_artifact_path, metadata_path))
        artifact_writer_meta = artifact_writer.final_metadata if artifact_writer is not None else None
        if artifact_writer_meta is not None:
            artifact_meta = {
                "group": corpus.label,
                "path": str(token_artifact_path.resolve()),
                "metadata_path": str(metadata_path.resolve()),
                "schema_version": artifact_writer_meta.schema_version,
                "row_count": artifact_writer_meta.row_count,
                "included_row_count": artifact_writer_meta.included_row_count,
                "complete": artifact_writer_meta.complete,
                "sha256": artifact_writer_meta.sha256,
            }
    return counter, artifact_meta, tuple(generated)


def write_dictcheck_outputs(
    *,
    context: RunContext,
    label: str,
    counter: Counter[str],
    frequency_paths: FrequencyOutputPaths | None = None,
) -> DictCheckOutput | None:
    plan = context.plan
    wordlist = plan.config.dictcheck.wordlist
    if plan.config.dictcheck.enabled and not wordlist:
        raise ValueError(
            f"dictcheck.wordlist is required when dictcheck.enabled=true (analysis_unit={plan.analysis_unit})"
        )
    if not plan.config.dictcheck.enabled:
        return None

    wl_path = Path(str(wordlist))
    if not wl_path.is_absolute():
        wl_path = (plan.project_root / wl_path).resolve()
    known_words = (
        x.strip()
        for x in wl_path.read_text(encoding="utf-8").splitlines()
        if x.strip()
    )
    known_counter, unknown_counter = split_known_unknown(counter, known_words)

    paths = frequency_paths or build_frequency_output_paths(plan.out_dir, label)
    write_frequency_csv(paths.known, known_counter, header=plan.csv_header)
    write_frequency_csv(paths.unknown, unknown_counter, header=plan.csv_header)
    return DictCheckOutput(
        known=known_counter,
        unknown=unknown_counter,
        generated_outputs=(paths.known, paths.unknown),
    )


def analyze_one_corpus(
    *,
    context: RunContext,
    dependencies: RunnerDependencies,
    corpus: PreparedCorpus,
    trace_paths: Mapping[str, Path],
    lemma_normalization_map: Mapping[str, str] | None,
    token_artifact_paths: Mapping[str, Path] | None = None,
    analysis_cache_stats: AnalysisCacheRunStats | None = None,
) -> GroupAnalysisResult:
    plan = context.plan
    label = corpus.label
    generated_outputs: list[Path] = []
    token_generated_outputs: tuple[Path, ...] = ()
    token_artifact_meta: Mapping[str, object] | None = None
    if plan.config.ref_tags.enabled:
        ref_tags_path = plan.out_dir / f"ref_tags_{label}.csv"
        write_frequency_csv(ref_tags_path, corpus.ref_tag_counts, header=("tag", "count"))
        generated_outputs.append(ref_tags_path)

    text = _text_for_counting(context, corpus)
    counter, token_artifact_meta, token_generated_outputs = _count_corpus_records(
        context=context,
        corpus=corpus,
        text=text,
        token_artifact_path=(token_artifact_paths or {}).get(label),
        trace_path=trace_paths.get(label) if plan.config.trace.enabled else None,
        analysis_cache_stats=analysis_cache_stats,
    )
    if lemma_normalization_map is not None:
        counter = apply_lemma_normalization(counter, lemma_normalization_map)

    frequency_paths = build_frequency_output_paths(plan.out_dir, label)
    write_frequency_csv(frequency_paths.base, counter, header=plan.csv_header)
    generated_outputs.append(frequency_paths.base)

    dictcheck_output = write_dictcheck_outputs(
        context=context,
        label=label,
        counter=counter,
        frequency_paths=frequency_paths,
    )
    if dictcheck_output is not None:
        generated_outputs.extend(dictcheck_output.generated_outputs)
    generated_outputs.extend(token_generated_outputs)

    return GroupAnalysisResult(
        label=label,
        files=tuple(corpus.files),
        counter=counter.copy(),
        ref_tag_counts=corpus.ref_tag_counts,
        generated_outputs=tuple(generated_outputs),
        token_artifact=token_artifact_meta,
    )


def analyze_corpora(
    context: RunContext,
    dependencies: RunnerDependencies,
) -> AnalysisResults:
    plan = context.plan
    prepared_corpora = prepare_corpora(
        work_items=plan.work_items,
        config=plan.config,
        project_root=plan.project_root,
    )
    trace_paths: dict[str, Path] = {}
    if plan.config.trace.enabled:
        trace_paths = build_trace_paths(
            base_path=_trace_base_path(plan.config.trace, plan.out_dir, plan.project_root),
            labels=[corpus.label for corpus in prepared_corpora],
        )
    token_artifact_paths: dict[str, Path] = {}
    if plan.config.artifacts.tokens.enabled:
        token_artifact_paths = build_token_artifact_paths(
            base_path=_token_artifact_base_path(
                plan.config.artifacts.tokens.path,
                plan.project_root,
            ),
            labels=[corpus.label for corpus in prepared_corpora],
        )
        _validate_trace_artifact_paths(
            trace_paths=trace_paths,
            token_artifact_paths=token_artifact_paths,
        )
    lemma_normalization_map = _load_lemma_normalization_map(context)
    analysis_cache_stats = AnalysisCacheRunStats(
        enabled=plan.config.analysis_cache.enabled,
        directory=str(_analysis_cache_dir(context)),
    )

    groups: list[GroupAnalysisResult] = []
    generated_outputs: list[Path] = []
    token_artifacts: list[Mapping[str, object]] = []
    for corpus in prepared_corpora:
        result = analyze_one_corpus(
            context=context,
            dependencies=dependencies,
            corpus=corpus,
            trace_paths=trace_paths,
            token_artifact_paths=token_artifact_paths,
            lemma_normalization_map=lemma_normalization_map,
            analysis_cache_stats=analysis_cache_stats,
        )
        groups.append(result)
        generated_outputs.extend(result.generated_outputs)
        if result.token_artifact is not None:
            token_artifacts.append(result.token_artifact)

    return AnalysisResults(
        groups=tuple(groups),
        counters_by_group={result.label: result.counter.copy() for result in groups},
        files_by_group={result.label: result.files for result in groups},
        ref_tags_by_group={result.label: result.ref_tag_counts for result in groups},
        trace_paths=trace_paths,
        generated_outputs=tuple(generated_outputs),
        token_artifacts=tuple(token_artifacts),
        analysis_cache=analysis_cache_stats.to_dict(),
    )
