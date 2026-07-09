from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, List, Mapping, Optional, Sequence, Tuple

from nlpo_toolkit.backends import (
    BuiltNLPBackend,
    NLPBackendInfo,
    create_nlp_backend,
    render_backend_info,
)
from .config import AppConfig, TraceConfig, ensure_app_config
from .corpus import (
    CorpusWorkItem,
    PreparedCorpus,
    prepare_corpora,
    resolve_corpus_work_items,
    resolve_project_path,
    run_preprocess_if_needed,
    sanitize_label,
)
from .comparison import (
    comparison_csv_name,
    comparison_result_meta,
    parse_comparison_specs,
    run_comparisons,
    write_comparison_csv,
    write_group_comparisons_json,
)
from .outputs import (
    FrequencyOutputPaths,
    build_frequency_output_paths,
    build_run_meta,
    collect_runtime_environment,
    write_frequency_csv,
    write_run_meta,
)
from .partition_validation import (
    parse_partition_specs,
    partition_result_meta,
    partition_result_summary,
    sanitize_partition_name,
    validate_partitions,
    write_partition_validation_csv,
    write_partition_validation_json,
)
from .analysis_cache import (
    AnalysisCacheGroupResult,
    AnalysisCacheRunStats,
    AnalysisFingerprint,
    build_analysis_cache_key,
    get_or_compute_analysis_records,
    prepared_text_sha256,
)
from .token_artifact import (
    AnalysisOptions,
    DiagnosticTraceWriter,
    TokenArtifactMetadata,
    TokenArtifactWriter,
    evaluate_analysis_record,
    iter_nlp_analysis_records_from_text,
    token_artifact_metadata_path,
)
from nlpo_toolkit.nlp import load_roman_exceptions


@dataclass(frozen=True)
class RunnerDependencies:
    load_config: Callable[[Path], AppConfig | Mapping[str, object]]
    clean_module: Any
    count_group: Callable[..., Counter]
    render_stanza_package_table: Callable[..., List[str]]
    build_pipeline: Callable[[str, str, bool], Tuple[Any, str]] | None = None
    backend_factory: Callable[[Any], BuiltNLPBackend] | None = None
    build_sentence_splitter: Callable[..., Any] | None = None


@dataclass(frozen=True)
class RunContext:
    project_root: Path
    config_path: Path
    config: AppConfig
    out_dir: Path
    grouping_mode: str
    per_file: bool
    auto_mode: bool
    auto_group_name: str
    analysis_unit: str
    use_lemma: bool
    csv_header: tuple[str, str]
    partition_specs: tuple[Any, ...]
    comparison_specs: tuple[Any, ...]
    work_items: tuple[CorpusWorkItem, ...]
    group_files: Mapping[str, tuple[Path, ...]]
    nlp: Any
    backend_info: NLPBackendInfo
    stanza_package: Any
    splitter_nlp: Any | None
    roman_exceptions: frozenset[str]
    cleaned_dir: Path | None


@dataclass(frozen=True)
class GroupAnalysisResult:
    label: str
    files: tuple[Path, ...]
    counter: Counter[str]
    ref_tag_counts: Counter[str]
    generated_outputs: tuple[Path, ...]
    token_artifact: Mapping[str, object] | None = None


@dataclass(frozen=True)
class AnalysisResults:
    groups: tuple[GroupAnalysisResult, ...]
    counters_by_group: Mapping[str, Counter[str]]
    files_by_group: Mapping[str, tuple[Path, ...]]
    ref_tags_by_group: Mapping[str, Counter[str]]
    trace_paths: Mapping[str, Path]
    generated_outputs: tuple[Path, ...]
    token_artifacts: tuple[Mapping[str, object], ...] = ()
    analysis_cache: Mapping[str, object] | None = None


@dataclass(frozen=True)
class DictCheckOutput:
    known: Counter[str]
    unknown: Counter[str]
    generated_outputs: tuple[Path, ...]


@dataclass(frozen=True)
class PartitionRunResult:
    results: tuple[Any, ...]
    summaries: tuple[Mapping[str, object], ...]
    metadata: tuple[Mapping[str, object], ...]
    generated_outputs: tuple[Path, ...]
    exit_code: int


@dataclass(frozen=True)
class ComparisonRunResult:
    results: tuple[Any, ...]
    metadata: tuple[Mapping[str, object], ...]
    generated_outputs: tuple[Path, ...]


def _trace_base_path(
    trace_cfg: TraceConfig,
    out_dir: Path,
    project_root: Path,
) -> Path:
    if trace_cfg.path:
        raw_path = trace_cfg.path
        trace_path = Path(str(raw_path))
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


def build_token_artifact_paths(
    *,
    base_path: Path,
    labels: list[str],
) -> dict[str, Path]:
    return build_trace_paths(base_path=base_path, labels=labels)


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


def _resolve_analysis_unit(config: AppConfig) -> tuple[str, bool, tuple[str, str]]:
    """
    Returns:
      - unit: "lemma" | "surface"
      - use_lemma: bool (to pass into count_group_fn)
      - csv_header: (col1, col2)
    """
    unit = config.analysis_unit
    use_lemma = (unit == "lemma")

    # default headers
    if unit == "lemma":
        header = ("lemma", "count")
    else:
        header = ("word", "frequency")

    # optional override: csv_header: ["...", "..."]
    if config.csv_header is not None:
        header = config.csv_header

    return unit, use_lemma, header

def _format_normalization_kv(norm: object) -> str:
    if hasattr(norm, "__dataclass_fields__"):
        norm_dict = asdict(norm)
    elif isinstance(norm, dict):
        norm_dict = norm
    else:
        return "(none)"

    keys_first = ["enabled", "casefold", "uv", "ij", "diacritics"]
    parts: list[str] = []

    for k in keys_first:
        if k in norm_dict:
            parts.append(f"{k}={norm_dict[k]}")

    lig = norm_dict.get("ligatures")
    if isinstance(lig, dict) and lig:
        lig_s = ",".join(f"{a}→{b}" for a, b in sorted(lig.items(), key=lambda x: x[0]))
        parts.append(f"ligatures={lig_s}")

    for k in sorted(norm_dict.keys()):
        if k in keys_first or k == "ligatures":
            continue
        parts.append(f"{k}={norm_dict[k]}")

    return " ".join(parts)


def _resolve_run_paths(
    *,
    project_root: Path | None,
    script_dir: Path | None,
    config_path: Path,
) -> tuple[Path, Path]:
    if project_root is None:
        if script_dir is None:
            raise TypeError("project_root is required")
        project_root = script_dir

    resolved_root = Path(project_root).resolve()
    resolved_config = Path(config_path)
    if not resolved_config.is_absolute():
        resolved_config = (resolved_root / resolved_config).resolve()
    if not resolved_config.exists():
        raise FileNotFoundError(f"Config file not found: {resolved_config}")
    return resolved_root, resolved_config


def _resolve_grouping_flags(
    *,
    config: AppConfig,
    group_by_file: bool | None,
    auto_single_cleaned: bool,
) -> tuple[str, bool, bool]:
    grouping_mode = config.grouping.mode
    auto_mode = bool(auto_single_cleaned) or grouping_mode == "auto_single_cleaned"
    per_file = (bool(group_by_file) or grouping_mode == "per_file") and not auto_mode
    return grouping_mode, per_file, auto_mode


def _resolve_out_dir(config: AppConfig, project_root: Path) -> Path:
    out_dir = Path(config.out_dir)
    if not out_dir.is_absolute():
        out_dir = (project_root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _initialize_nlp(
    *,
    config: AppConfig,
    dependencies: RunnerDependencies,
) -> tuple[Any, NLPBackendInfo, Any]:
    language = config.nlp.language
    stanza_package = config.nlp.stanza_package
    cpu_only = config.nlp.cpu_only

    if dependencies.backend_factory is not None:
        built_backend = dependencies.backend_factory(config.nlp)
        return built_backend.backend, built_backend.info, built_backend.info.package

    if dependencies.build_pipeline is not None:
        nlp, package = dependencies.build_pipeline(language, stanza_package, cpu_only)
        return (
            nlp,
            NLPBackendInfo(
                name="stanza",
                language=language,
                package=package,
                use_gpu=not cpu_only,
            ),
            package,
        )

    built_backend = create_nlp_backend(config.nlp)
    return built_backend.backend, built_backend.info, built_backend.info.package


def _initialize_sentence_splitter(
    *,
    config: AppConfig,
    package: Any,
    dependencies: RunnerDependencies,
) -> Any | None:
    if dependencies.build_sentence_splitter is None:
        return None
    try:
        return dependencies.build_sentence_splitter(
            config.nlp.language,
            stanza_package=package,
            cpu_only=config.nlp.cpu_only,
        )
    except Exception:
        return None


def _load_roman_exceptions(config: AppConfig, project_root: Path) -> frozenset[str]:
    roman_exceptions_file = config.filters.roman_exceptions_file
    if not roman_exceptions_file:
        return frozenset()
    path = resolve_project_path(project_root, roman_exceptions_file)
    return load_roman_exceptions(path)


def _validate_specs_against_grouping(
    *,
    partition_specs: Sequence[Any],
    comparison_specs: Sequence[Any],
    per_file: bool,
) -> None:
    if partition_specs and per_file:
        raise ValueError("validations.partitions cannot be used with --group-by-file or grouping.mode: per_file")
    if comparison_specs and per_file:
        raise ValueError("comparisons cannot be used with grouping.mode=per_file")


def _validate_partition_group_references(
    *,
    partition_specs: Sequence[Any],
    group_files: Mapping[str, Sequence[Path]],
) -> None:
    for spec in partition_specs:
        for name in (spec.whole, *spec.parts):
            if not group_files.get(name):
                raise ValueError(f"Partition {spec.name} references empty group: {name}")


def _parse_and_validate_specs(
    *,
    config: AppConfig,
    per_file: bool,
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    partition_specs = tuple(parse_partition_specs(config))
    comparison_specs = tuple(parse_comparison_specs(config))
    _validate_specs_against_grouping(
        partition_specs=partition_specs,
        comparison_specs=comparison_specs,
        per_file=per_file,
    )
    return partition_specs, comparison_specs


def _resolve_and_validate_work_items(
    *,
    config: AppConfig,
    project_root: Path,
    cleaned_dir: Path | None,
    group_by_file: bool | None,
    auto_single_cleaned: bool,
    error_on_empty_group: bool,
    partition_specs: Sequence[Any],
) -> tuple[tuple[CorpusWorkItem, ...], Mapping[str, tuple[Path, ...]]]:
    resolved = resolve_corpus_work_items(
        config=config,
        project_root=project_root,
        cleaned_dir=cleaned_dir,
        group_by_file=bool(group_by_file),
        auto_single_cleaned=auto_single_cleaned,
        error_on_empty_group=error_on_empty_group,
    )
    _validate_partition_group_references(
        partition_specs=partition_specs,
        group_files=resolved.group_files,
    )
    return tuple(resolved.work_items), resolved.group_files


def prepare_run_context(
    *,
    project_root: Path | None,
    script_dir: Path | None,
    config_path: Path,
    group_by_file: bool | None,
    auto_single_cleaned: bool,
    error_on_empty_group: bool,
    dependencies: RunnerDependencies,
) -> RunContext:
    resolved_root, resolved_config = _resolve_run_paths(
        project_root=project_root,
        script_dir=script_dir,
        config_path=config_path,
    )
    config = ensure_app_config(dependencies.load_config(resolved_config))
    if not config.groups:
        raise ValueError("config.groups must be a non-empty mapping")

    grouping_mode, per_file, auto_mode = _resolve_grouping_flags(
        config=config,
        group_by_file=group_by_file,
        auto_single_cleaned=auto_single_cleaned,
    )
    partition_specs, comparison_specs = _parse_and_validate_specs(
        config=config,
        per_file=per_file,
    )

    cleaned_dir = run_preprocess_if_needed(
        config=config,
        project_root=resolved_root,
        clean_mod=dependencies.clean_module,
    )
    out_dir = _resolve_out_dir(config, resolved_root)
    analysis_unit, use_lemma, csv_header = _resolve_analysis_unit(config)
    nlp, backend_info, package = _initialize_nlp(
        config=config,
        dependencies=dependencies,
    )
    splitter_nlp = _initialize_sentence_splitter(
        config=config,
        package=package,
        dependencies=dependencies,
    )
    roman_exceptions = _load_roman_exceptions(config, resolved_root)
    work_items, group_files = _resolve_and_validate_work_items(
        config=config,
        project_root=resolved_root,
        cleaned_dir=cleaned_dir,
        group_by_file=bool(group_by_file),
        auto_single_cleaned=auto_single_cleaned,
        error_on_empty_group=error_on_empty_group,
        partition_specs=partition_specs,
    )

    return RunContext(
        project_root=resolved_root,
        config_path=resolved_config,
        config=config,
        out_dir=out_dir,
        grouping_mode=grouping_mode,
        per_file=per_file,
        auto_mode=auto_mode,
        auto_group_name=config.grouping.auto_group_name,
        analysis_unit=analysis_unit,
        use_lemma=use_lemma,
        csv_header=csv_header,
        partition_specs=partition_specs,
        comparison_specs=comparison_specs,
        work_items=work_items,
        group_files=group_files,
        nlp=nlp,
        backend_info=backend_info,
        stanza_package=config.nlp.stanza_package,
        splitter_nlp=splitter_nlp,
        roman_exceptions=roman_exceptions,
        cleaned_dir=cleaned_dir,
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
    norm_map_rel_path = context.config.dictcheck.lemma_normalize
    if not norm_map_rel_path:
        return None
    norm_map_path = Path(str(norm_map_rel_path))
    if not norm_map_path.is_absolute():
        norm_map_path = (context.project_root / norm_map_path).resolve()
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


def _trace_kwargs_for_label(context: RunContext, trace_paths: Mapping[str, Path], label: str) -> dict[str, Any]:
    if not context.config.trace.enabled:
        return {}
    return {
        "trace_tsv": trace_paths[label],
        "trace_max_rows": int(context.config.trace.max_rows or 0),
        "trace_only_keys": set(context.config.trace.only_keys),
        "trace_write_truncation_marker": context.config.trace.write_truncation_marker,
    }


def _token_artifact_metadata(
    *,
    context: RunContext,
    corpus: PreparedCorpus,
    path: Path,
) -> TokenArtifactMetadata:
    return TokenArtifactMetadata(
        group=corpus.label,
        source_files=tuple(str(file) for file in corpus.files),
        analysis_unit=context.analysis_unit,
        upos_targets=tuple(sorted(context.config.filters.upos_targets)),
        nlp=context.backend_info.to_dict(),
        filters={
            "min_token_length": context.config.filters.min_token_length,
            "drop_roman_numerals": context.config.filters.drop_roman_numerals,
        },
        artifact_path=str(path.resolve()),
    )


def _analysis_cache_dir(context: RunContext) -> Path:
    cache_dir = Path(context.config.analysis_cache.directory)
    if not cache_dir.is_absolute():
        cache_dir = context.project_root / cache_dir
    return cache_dir.resolve()


def _analysis_fingerprint(context: RunContext) -> AnalysisFingerprint:
    package = context.backend_info.package
    package_value: object | None = package
    return AnalysisFingerprint(
        backend=context.backend_info.name,
        language=context.backend_info.language,
        model=context.backend_info.model,
        package=package_value,
        processors=("tokenize", "mwt", "pos", "lemma"),
        chunk_size=200_000,
        chunk_strategy="char_whitespace",
        device=context.backend_info.device,
    )


def _count_with_token_records(
    *,
    context: RunContext,
    corpus: PreparedCorpus,
    text: str,
    token_artifact_path: Path | None,
    trace_path: Path | None,
    analysis_cache_stats: AnalysisCacheRunStats | None = None,
) -> tuple[Counter[str], Mapping[str, object] | None, tuple[Path, ...]]:
    counter: Counter[str] = Counter()
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
                    max_rows=int(context.config.trace.max_rows or 0),
                    only_keys=context.config.trace.only_keys,
                    write_truncation_marker=context.config.trace.write_truncation_marker,
                )
            )

        fingerprint = _analysis_fingerprint(context)
        text_hash = prepared_text_sha256(text)
        cache_key = build_analysis_cache_key(
            prepared_text_sha256=text_hash,
            fingerprint=fingerprint,
        )

        if context.config.analysis_cache.enabled:
            raw_records, cache_status, _payload_path, _metadata_path = get_or_compute_analysis_records(
                cache_dir=_analysis_cache_dir(context),
                cache_key=cache_key,
                prepared_text_sha256=text_hash,
                prepared_text_length=len(text),
                fingerprint=fingerprint,
                compute_records=lambda: iter_nlp_analysis_records_from_text(
                    text=text,
                    nlp=context.nlp,
                    chunk_chars=200_000,
                ),
                lock_timeout_sec=context.config.analysis_cache.lock_timeout_sec,
            )
        else:
            cache_status = "disabled"
            raw_records = iter_nlp_analysis_records_from_text(
                text=text,
                nlp=context.nlp,
                chunk_chars=200_000,
            )

        options = AnalysisOptions(
            group=corpus.label,
            source_files=tuple(corpus.files),
            use_lemma=context.use_lemma,
            upos_targets=frozenset(context.config.filters.upos_targets),
            min_token_length=context.config.filters.min_token_length,
            drop_roman_numerals=context.config.filters.drop_roman_numerals,
            roman_exceptions=context.roman_exceptions,
        )
        record_count = 0
        for raw_record in raw_records:
            record_count += 1
            record = evaluate_analysis_record(raw_record, options=options)
            if artifact_writer is not None:
                artifact_writer.write(record)
            if trace_writer is not None:
                trace_writer.consider(record)
            if record.included and record.analysis_key:
                counter[record.analysis_key] += 1

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
    wordlist = context.config.dictcheck.wordlist
    if context.config.dictcheck.enabled and not wordlist:
        raise ValueError(
            f"dictcheck.wordlist is required when dictcheck.enabled=true (analysis_unit={context.analysis_unit})"
        )
    if not context.config.dictcheck.enabled:
        return None

    wl_path = Path(str(wordlist))
    if not wl_path.is_absolute():
        wl_path = (context.project_root / wl_path).resolve()
    known_words = (
        x.strip()
        for x in wl_path.read_text(encoding="utf-8").splitlines()
        if x.strip()
    )
    known_counter, unknown_counter = split_known_unknown(counter, known_words)

    paths = frequency_paths or build_frequency_output_paths(context.out_dir, label)
    known_path = paths.known
    write_frequency_csv(known_path, known_counter, header=context.csv_header)
    unknown_path = paths.unknown
    write_frequency_csv(unknown_path, unknown_counter, header=context.csv_header)
    return DictCheckOutput(
        known=known_counter,
        unknown=unknown_counter,
        generated_outputs=(known_path, unknown_path),
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
    label = corpus.label
    generated_outputs: list[Path] = []
    token_generated_outputs: tuple[Path, ...] = ()
    token_artifact_meta: Mapping[str, object] | None = None
    if context.config.ref_tags.enabled:
        ref_tags_path = context.out_dir / f"ref_tags_{label}.csv"
        write_frequency_csv(ref_tags_path, corpus.ref_tag_counts, header=("tag", "count"))
        generated_outputs.append(ref_tags_path)

    text = _text_for_counting(context, corpus)
    if context.config.artifacts.tokens.enabled or context.config.analysis_cache.enabled:
        counter, token_artifact_meta, token_generated_outputs = _count_with_token_records(
            context=context,
            corpus=corpus,
            text=text,
            token_artifact_path=(token_artifact_paths or {}).get(label),
            trace_path=trace_paths.get(label) if context.config.trace.enabled else None,
            analysis_cache_stats=analysis_cache_stats,
        )
    else:
        counter = dependencies.count_group(
            text,
            context.nlp,
            use_lemma=context.use_lemma,
            upos_targets=context.config.filters.upos_targets,
            min_token_length=context.config.filters.min_token_length,
            drop_roman_numerals=context.config.filters.drop_roman_numerals,
            roman_exceptions=context.roman_exceptions,
            label=label,
            **_trace_kwargs_for_label(context, trace_paths, label),
        )
    if lemma_normalization_map is not None:
        counter = apply_lemma_normalization(counter, lemma_normalization_map)

    frequency_paths = build_frequency_output_paths(context.out_dir, label)
    base_csv_path = frequency_paths.base
    write_frequency_csv(base_csv_path, counter, header=context.csv_header)
    generated_outputs.append(base_csv_path)

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
    prepared_corpora = prepare_corpora(
        work_items=context.work_items,
        config=context.config,
        project_root=context.project_root,
    )
    trace_paths: dict[str, Path] = {}
    if context.config.trace.enabled:
        trace_paths = build_trace_paths(
            base_path=_trace_base_path(context.config.trace, context.out_dir, context.project_root),
            labels=[corpus.label for corpus in prepared_corpora],
        )
    token_artifact_paths: dict[str, Path] = {}
    if context.config.artifacts.tokens.enabled:
        token_artifact_paths = build_token_artifact_paths(
            base_path=_token_artifact_base_path(
                context.config.artifacts.tokens.path,
                context.project_root,
            ),
            labels=[corpus.label for corpus in prepared_corpora],
        )
        _validate_trace_artifact_paths(
            trace_paths=trace_paths,
            token_artifact_paths=token_artifact_paths,
        )
    lemma_normalization_map = _load_lemma_normalization_map(context)
    analysis_cache_stats = AnalysisCacheRunStats(
        enabled=context.config.analysis_cache.enabled,
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


def execute_partition_validations(
    *,
    context: RunContext,
    analysis: AnalysisResults,
) -> PartitionRunResult:
    partition_results = validate_partitions(context.partition_specs, analysis.counters_by_group)
    summaries: list[Mapping[str, object]] = []
    metadata: list[Mapping[str, object]] = []
    generated_outputs: list[Path] = []
    exit_code = 0

    for spec, result in zip(context.partition_specs, partition_results):
        csv_name = f"partition_validation_{sanitize_partition_name(spec.name)}.csv"
        csv_path = context.out_dir / csv_name
        write_partition_validation_csv(csv_path, result)
        generated_outputs.append(csv_path)
        summaries.append(partition_result_summary(spec, result, csv_name=csv_name))
        metadata.append(partition_result_meta(spec, result))

        if not result.exact_match:
            level = "ERROR" if spec.on_mismatch == "error" else "WARN"
            print(
                f"[{level}] partition {spec.name} mismatch: "
                f"token_delta={result.token_delta} mismatched_items={result.mismatched_items}",
                file=sys.stderr,
            )
            if spec.on_mismatch == "error":
                exit_code = 1

    if context.partition_specs:
        partition_json_path = context.out_dir / "partition_validation.json"
        write_partition_validation_json(partition_json_path, summaries)
        generated_outputs.append(partition_json_path)

    return PartitionRunResult(
        results=tuple(partition_results),
        summaries=tuple(summaries),
        metadata=tuple(metadata),
        generated_outputs=tuple(generated_outputs),
        exit_code=exit_code,
    )


def execute_group_comparisons(
    *,
    context: RunContext,
    analysis: AnalysisResults,
) -> ComparisonRunResult:
    comparison_results = run_comparisons(
        specs=context.comparison_specs,
        counters=analysis.counters_by_group,
        analysis_unit=context.analysis_unit,
    )
    generated_outputs: list[Path] = []
    metadata: list[Mapping[str, object]] = []
    for result in comparison_results:
        csv_name = comparison_csv_name(result.spec)
        csv_path = context.out_dir / csv_name
        write_comparison_csv(csv_path, result)
        generated_outputs.append(csv_path)
        metadata.append(comparison_result_meta(result, csv_name=csv_name))

    if comparison_results:
        comparison_json_path = context.out_dir / "group_comparisons.json"
        write_group_comparisons_json(comparison_json_path, comparison_results)
        generated_outputs.append(comparison_json_path)

    return ComparisonRunResult(
        results=tuple(comparison_results),
        metadata=tuple(metadata),
        generated_outputs=tuple(generated_outputs),
    )


def build_summary_lines(
    *,
    context: RunContext,
    analysis: AnalysisResults,
    partitions: PartitionRunResult,
    comparisons: ComparisonRunResult,
    dependencies: RunnerDependencies,
) -> list[str]:
    lines: list[str] = [
        "# Summary",
        "",
        f"language: {context.config.nlp.language}",
        f"stanza_package: {context.config.nlp.stanza_package}",
        f"nlp_backend: {context.backend_info.name}",
        f"analysis_unit: {context.analysis_unit}",
        f"normalization: {_format_normalization_kv(context.config.normalization)}",
        "",
    ]
    if context.backend_info.name == "stanza":
        lines.extend(
            dependencies.render_stanza_package_table(
                context.nlp,
                context.config.nlp.stanza_package,
            )
        )
    else:
        lines.extend(render_backend_info(context.backend_info))
    lines.append("")

    if context.config.ref_tags.enabled:
        for label, counts in analysis.ref_tags_by_group.items():
            lines.append(
                f"- group={label} ref_tag_types={len(counts)} ref_tag_tokens={sum(counts.values())}"
            )

    if analysis.token_artifacts:
        lines.extend(["", "# Token artifacts", ""])
        for artifact in analysis.token_artifacts:
            lines.append(
                f"- token_artifact={artifact.get('group', '')} "
                f"path={artifact.get('path', '')} "
                f"rows={artifact.get('row_count', 0)} "
                f"included={artifact.get('included_row_count', 0)} "
                f"schema={artifact.get('schema_version', '')}"
            )

    if analysis.analysis_cache:
        cache_meta = analysis.analysis_cache
        lines.extend(["", "# Analysis cache", ""])
        lines.append(
            "analysis_cache "
            f"enabled={cache_meta.get('enabled', False)} "
            f"hits={cache_meta.get('hits', 0)} "
            f"misses={cache_meta.get('misses', 0)} "
            f"records_read={cache_meta.get('records_read', 0)} "
            f"records_written={cache_meta.get('records_written', 0)}"
        )

    if partitions.results:
        lines.extend(["", "# Partition validation", ""])
        for spec, result in zip(context.partition_specs, partitions.results):
            if result.exact_match:
                lines.append(
                    f"- name={result.name} status=OK whole={result.whole} "
                    f"parts={','.join(result.parts)} "
                    f"target_tokens={result.whole_target_tokens} "
                    f"parts_target_tokens={result.parts_target_tokens} "
                    f"mismatched_items={result.mismatched_items}"
                )
            else:
                status = "ERROR" if spec.on_mismatch == "error" else "WARN"
                lines.append(
                    f"- name={result.name} status={status} whole={result.whole} "
                    f"parts={','.join(result.parts)} "
                    f"target_tokens={result.whole_target_tokens} "
                    f"parts_target_tokens={result.parts_target_tokens} "
                    f"token_delta={result.token_delta} "
                    f"mismatched_items={result.mismatched_items}"
                )

    if comparisons.results:
        lines.extend(["", "# Group comparisons", ""])
        for result in comparisons.results:
            spec = result.spec
            lines.append(
                f"- name={spec.name} group_a={spec.group_a} group_b={spec.group_b} "
                f"analysis_unit={result.analysis_unit} "
                f"group_a_tokens={result.group_a_tokens} "
                f"group_b_tokens={result.group_b_tokens} "
                f"items={result.rows_after_filter} scale={spec.scale} "
                f"zero_correction={spec.zero_correction}"
            )
    return lines


def write_summary(path: Path, lines: Sequence[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def merge_generated_outputs(*groups: Iterable[Path]) -> tuple[Path, ...]:
    seen: set[Path] = set()
    merged: list[Path] = []
    for group in groups:
        for path in group:
            resolved = Path(path)
            if resolved in seen:
                continue
            seen.add(resolved)
            merged.append(resolved)
    return tuple(merged)


def build_final_run_metadata(
    *,
    context: RunContext,
    analysis: AnalysisResults,
    partitions: PartitionRunResult,
    comparisons: ComparisonRunResult,
    generated_outputs: Sequence[Path],
) -> dict[str, object]:
    meta = build_run_meta(
        groups_files={
            label: [str(path) for path in files]
            for label, files in analysis.files_by_group.items()
        },
        hash_inputs=False,
    )
    meta["analysis_unit"] = context.analysis_unit
    meta["nlp"] = context.backend_info.to_dict()
    if context.auto_mode:
        meta["grouping"] = {
            "mode": "auto_single_cleaned",
            "auto_group_name": context.auto_group_name,
        }
    else:
        meta["grouping"] = {"mode": "per_file" if context.per_file else "groups"}
    meta["environment"] = collect_runtime_environment(context.project_root)

    norm_dict = asdict(context.config.normalization)
    norm_canon = json.dumps(norm_dict, ensure_ascii=False, sort_keys=True)
    meta["normalization"] = norm_dict
    meta["normalization_hash_sha256"] = hashlib.sha256(norm_canon.encode("utf-8")).hexdigest()
    meta["partition_validations"] = list(partitions.metadata)
    meta["group_comparisons"] = list(comparisons.metadata)
    meta["trace"] = {
        "enabled": context.config.trace.enabled,
        "files": {
            label: str(path.resolve())
            for label, path in analysis.trace_paths.items()
        },
    }
    meta["token_artifacts"] = list(analysis.token_artifacts)
    meta["analysis_cache"] = dict(analysis.analysis_cache or {})
    meta["generated_outputs"] = [str(path.resolve()) for path in generated_outputs]
    return meta

def run(
    *,
    project_root: Path | None = None,
    script_dir: Path | None = None,
    config_path: Path,
    group_by_file: Optional[bool] = None,
    load_config_fn: Callable[[Path], AppConfig | Mapping[str, object]],
    clean_mod: Any,
    build_pipeline_fn: Callable[[str, str, bool], Tuple[Any, str]] | None = None,
    backend_factory: Callable[[Any], BuiltNLPBackend] | None = None,
    build_sentence_splitter_fn: Optional[Callable[..., Any]] = None,
    count_group_fn: Callable[..., Counter] | None = None,
    render_stanza_package_table_fn: Callable[..., List[str]] | None = None,
    error_on_empty_group: bool = False,
    auto_single_cleaned: bool = False,
) -> int:
    """Core runner. Dependencies are injectable for CLI and tests."""
    if count_group_fn is None:
        raise TypeError("count_group_fn is required")

    dependencies = RunnerDependencies(
        load_config=load_config_fn,
        clean_module=clean_mod,
        build_pipeline=build_pipeline_fn,
        backend_factory=backend_factory,
        build_sentence_splitter=build_sentence_splitter_fn,
        count_group=count_group_fn,
        render_stanza_package_table=render_stanza_package_table_fn
        or (lambda *_args, **_kwargs: []),
    )

    context = prepare_run_context(
        project_root=project_root,
        script_dir=script_dir,
        config_path=config_path,
        group_by_file=group_by_file,
        auto_single_cleaned=auto_single_cleaned,
        error_on_empty_group=error_on_empty_group,
        dependencies=dependencies,
    )
    analysis = analyze_corpora(context, dependencies)
    partitions = execute_partition_validations(
        context=context,
        analysis=analysis,
    )
    comparisons = execute_group_comparisons(
        context=context,
        analysis=analysis,
    )

    summary_path = write_summary(
        context.out_dir / "summary.txt",
        build_summary_lines(
            context=context,
            analysis=analysis,
            partitions=partitions,
            comparisons=comparisons,
            dependencies=dependencies,
        ),
    )
    run_meta_path = context.out_dir / "run_meta.json"
    generated_outputs = merge_generated_outputs(
        analysis.generated_outputs,
        partitions.generated_outputs,
        comparisons.generated_outputs,
        (summary_path, run_meta_path),
    )
    write_run_meta(
        build_final_run_metadata(
            context=context,
            analysis=analysis,
            partitions=partitions,
            comparisons=comparisons,
            generated_outputs=generated_outputs,
        ),
        context.out_dir,
    )
    return partitions.exit_code
