from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, List, Mapping, Optional, Tuple

from .config import AppConfig, TraceConfig, ensure_app_config
from .corpus import (
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
from nlpo_toolkit.nlp import load_roman_exceptions


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

def run(
    *,
    project_root: Path | None = None,
    script_dir: Path | None = None,
    config_path: Path,
    group_by_file: Optional[bool] = None,
    load_config_fn: Callable[[Path], AppConfig | Mapping[str, object]],
    clean_mod: Any,
    build_pipeline_fn: Callable[[str, str, bool], Tuple[Any, str]],
    build_sentence_splitter_fn: Optional[Callable[..., Any]],
    count_group_fn: Callable[..., Counter],
    render_stanza_package_table_fn: Callable[..., List[str]],
    error_on_empty_group: bool = False,
    auto_single_cleaned: bool = False,
) -> int:
    """
    Core runner. Dependencies are injectable so tests can monkeypatch:
      - load_config_fn
      - clean_mod.main
      - build_pipeline_fn
      - build_sentence_splitter_fn
      - count_group_fn
      - render_stanza_package_table_fn
    """

    if project_root is None:
        if script_dir is None:
            raise TypeError("project_root is required")
        project_root = script_dir

    project_root = Path(project_root).resolve()
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = (project_root / config_path).resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = ensure_app_config(load_config_fn(config_path))
    grouping_mode = config.grouping.mode
    auto_mode = bool(auto_single_cleaned) or grouping_mode == "auto_single_cleaned"
    per_file = (bool(group_by_file) or grouping_mode == "per_file") and not auto_mode
    partition_specs = parse_partition_specs(config)
    comparison_specs = parse_comparison_specs(config)
    if partition_specs and per_file:
        raise ValueError("validations.partitions cannot be used with --group-by-file or grouping.mode: per_file")
    if comparison_specs and per_file:
        raise ValueError("comparisons cannot be used with grouping.mode=per_file")

    # preprocess (optional)
    cleaned_dir = run_preprocess_if_needed(config=config, project_root=project_root, clean_mod=clean_mod)

    # output directory
    out_dir = Path(config.out_dir)
    if not out_dir.is_absolute():
        out_dir = (project_root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # language settings
    language = config.nlp.language
    stanza_package = config.nlp.stanza_package
    cpu_only = config.nlp.cpu_only

    # analysis unit (lemma / surface)
    unit, use_lemma, csv_header = _resolve_analysis_unit(config)

    # build NLP
    nlp, package = build_pipeline_fn(language, stanza_package, cpu_only)

    # sentence splitter is optional
    splitter_nlp = None
    if build_sentence_splitter_fn is not None:
        try:
            splitter_nlp = build_sentence_splitter_fn(
                language,
                stanza_package=package,
                cpu_only=cpu_only,
            )
        except Exception:
            splitter_nlp = None

    # ref_tags setting is global (summary/meta needs it)
    ref_enabled = config.ref_tags.enabled

    min_token_length = config.filters.min_token_length
    drop_roman_numerals = config.filters.drop_roman_numerals
    roman_exceptions_file = config.filters.roman_exceptions_file
    roman_exceptions = frozenset()
    if roman_exceptions_file:
        roman_exceptions_file = resolve_project_path(project_root, roman_exceptions_file)
        roman_exceptions = load_roman_exceptions(roman_exceptions_file)

    if not config.groups:
        raise ValueError("config.groups must be a non-empty mapping")

    group_ref_tags: dict[str, Counter] = {}
    groups_files: dict[str, list[str]] = {}
    auto_group_name = config.grouping.auto_group_name
    resolved = resolve_corpus_work_items(
        config=config,
        project_root=project_root,
        cleaned_dir=cleaned_dir,
        group_by_file=bool(group_by_file),
        auto_single_cleaned=auto_single_cleaned,
        error_on_empty_group=error_on_empty_group,
    )
    group_files = resolved.group_files
    if partition_specs:
        for spec in partition_specs:
            for name in (spec.whole, *spec.parts):
                if not group_files.get(name):
                    raise ValueError(
                        f"Partition {spec.name} references empty group: {name}"
                    )

    generated_outputs: List[Path] = []
    group_counters: dict[str, Counter] = {}
    prepared_corpora = prepare_corpora(
        work_items=resolved.work_items,
        config=config,
        project_root=project_root,
    )
    trace_paths: dict[str, Path] = {}
    if config.trace.enabled:
        trace_paths = build_trace_paths(
            base_path=_trace_base_path(config.trace, out_dir, project_root),
            labels=[corpus.label for corpus in prepared_corpora],
        )

    for corpus in prepared_corpora:
        gname = corpus.label
        groups_files[gname] = [str(p) for p in corpus.files]

        if splitter_nlp is not None:
            doc = splitter_nlp(corpus.prepared_text)
            joined = "\n".join([s.text for s in getattr(doc, "sentences", [])])
            if not joined.strip():
                joined = corpus.prepared_text
        else:
            joined = corpus.prepared_text

        if ref_enabled:
            group_ref_tags[gname] = corpus.ref_tag_counts

            # ref_tags csv (per group)
            write_frequency_csv(
                out_dir / f"ref_tags_{gname}.csv",
                corpus.ref_tag_counts,
                header=("tag", "count"),
            )
            generated_outputs.append(out_dir / f"ref_tags_{gname}.csv")
        
        # ---- trace (optional) ----
        trace_kwargs: dict[str, Any] = {}

        if config.trace.enabled:
            trace_path = trace_paths[gname]

            trace_kwargs = {
                "trace_tsv": trace_path,
                "trace_max_rows": int(config.trace.max_rows or 0),
                "trace_only_keys": set(config.trace.only_keys),
                "trace_write_truncation_marker": config.trace.write_truncation_marker,
            }

        c = count_group_fn(
            joined,
            nlp,
            use_lemma=use_lemma,
            min_token_length=min_token_length,
            drop_roman_numerals=drop_roman_numerals,
            roman_exceptions=roman_exceptions,
            label=gname,
            **trace_kwargs,
        )
        
        norm_map_rel_path = config.dictcheck.lemma_normalize
        
        if norm_map_rel_path:
            norm_map_path = Path(str(norm_map_rel_path))
            if not norm_map_path.is_absolute():
                norm_map_path = (project_root / norm_map_path).resolve()
            
            if norm_map_path.exists():
                from .dictcheck import load_lemma_normalize_map
                lemma_map = load_lemma_normalize_map(norm_map_path)
                
                new_c = Counter()
                for lemma, count in c.items():
                    target_lemma = lemma_map.get(lemma, lemma)
                    new_c[target_lemma] += count
                c = new_c

        group_counters[gname] = c.copy()

        # base csv
        base = f"noun_frequency_{gname}"
        base_csv_path = out_dir / f"{base}.csv"
        write_frequency_csv(base_csv_path, c, header=csv_header)
        generated_outputs.append(base_csv_path)

        # dictcheck
        wordlist = config.dictcheck.wordlist

        if config.dictcheck.enabled and not wordlist:
            raise ValueError(
                f"dictcheck.wordlist is required when dictcheck.enabled=true (analysis_unit={unit})"
            )

        if config.dictcheck.enabled:
            wl_path = Path(str(wordlist))
            if not wl_path.is_absolute():
                wl_path = (project_root / wl_path).resolve()

            known = set(
                x.strip()
                for x in wl_path.read_text(encoding="utf-8").splitlines()
                if x.strip()
            )

            known_c = Counter({w: n for (w, n) in c.items() if w in known})
            unknown_c = Counter({w: n for (w, n) in c.items() if w not in known})

            known_path = out_dir / f"noun_frequency_{gname}.known.csv"
            write_frequency_csv(
                known_path,
                known_c,
                header=csv_header,
            )
            generated_outputs.append(known_path)
            unknown_path = out_dir / f"noun_frequency_{gname}.unknown.csv"
            write_frequency_csv(
                unknown_path,
                unknown_c,
                header=csv_header,
            )
            generated_outputs.append(unknown_path)

    partition_results = validate_partitions(partition_specs, group_counters)
    partition_summaries: List[dict[str, Any]] = []
    partition_meta: List[dict[str, Any]] = []
    partition_exit_code = 0

    for spec, result in zip(partition_specs, partition_results):
        csv_name = f"partition_validation_{sanitize_partition_name(spec.name)}.csv"
        csv_path = out_dir / csv_name
        write_partition_validation_csv(csv_path, result)
        generated_outputs.append(csv_path)

        partition_summaries.append(
            partition_result_summary(spec, result, csv_name=csv_name)
        )
        partition_meta.append(partition_result_meta(spec, result))

        if not result.exact_match:
            level = "ERROR" if spec.on_mismatch == "error" else "WARN"
            print(
                f"[{level}] partition {spec.name} mismatch: "
                f"token_delta={result.token_delta} mismatched_items={result.mismatched_items}",
                file=sys.stderr,
            )
            if spec.on_mismatch == "error":
                partition_exit_code = 1

    if partition_specs:
        partition_json_path = out_dir / "partition_validation.json"
        write_partition_validation_json(partition_json_path, partition_summaries)
        generated_outputs.append(partition_json_path)

    comparison_results = run_comparisons(
        specs=comparison_specs,
        counters=group_counters,
        analysis_unit=unit,
    )
    comparison_meta: List[dict[str, Any]] = []

    for result in comparison_results:
        csv_name = comparison_csv_name(result.spec)
        csv_path = out_dir / csv_name
        write_comparison_csv(csv_path, result)
        generated_outputs.append(csv_path)
        comparison_meta.append(comparison_result_meta(result, csv_name=csv_name))

    if comparison_results:
        comparison_json_path = out_dir / "group_comparisons.json"
        write_group_comparisons_json(comparison_json_path, comparison_results)
        generated_outputs.append(comparison_json_path)

    # ---- summary.txt ----
    summary_lines: List[str] = []
    summary_lines.append("# Summary")
    summary_lines.append("")
    summary_lines.append(f"language: {language}")
    summary_lines.append(f"stanza_package: {stanza_package}")
    summary_lines.append(f"analysis_unit: {unit}")

    # normalization policy (human-readable, stable)
    norm = config.normalization
    summary_lines.append(f"normalization: {_format_normalization_kv(norm)}")

    summary_lines.append("")
    summary_lines.extend(render_stanza_package_table_fn(nlp, stanza_package))
    summary_lines.append("")

    if ref_enabled:
        for gn, rc in group_ref_tags.items():
            summary_lines.append(
                f"- group={gn} ref_tag_types={len(rc)} ref_tag_tokens={sum(rc.values())}"
            )

    if partition_results:
        summary_lines.append("")
        summary_lines.append("# Partition validation")
        summary_lines.append("")
        for spec, result in zip(partition_specs, partition_results):
            if result.exact_match:
                summary_lines.append(
                    f"- name={result.name} status=OK whole={result.whole} "
                    f"parts={','.join(result.parts)} "
                    f"target_tokens={result.whole_target_tokens} "
                    f"parts_target_tokens={result.parts_target_tokens} "
                    f"mismatched_items={result.mismatched_items}"
                )
            else:
                status = "ERROR" if spec.on_mismatch == "error" else "WARN"
                summary_lines.append(
                    f"- name={result.name} status={status} whole={result.whole} "
                    f"parts={','.join(result.parts)} "
                    f"target_tokens={result.whole_target_tokens} "
                    f"parts_target_tokens={result.parts_target_tokens} "
                    f"token_delta={result.token_delta} "
                    f"mismatched_items={result.mismatched_items}"
                )

    if comparison_results:
        summary_lines.append("")
        summary_lines.append("# Group comparisons")
        summary_lines.append("")
        for result in comparison_results:
            spec = result.spec
            summary_lines.append(
                f"- name={spec.name} group_a={spec.group_a} group_b={spec.group_b} "
                f"analysis_unit={result.analysis_unit} "
                f"group_a_tokens={result.group_a_tokens} "
                f"group_b_tokens={result.group_b_tokens} "
                f"items={result.rows_after_filter} scale={spec.scale} "
                f"zero_correction={spec.zero_correction}"
            )

    summary_path = out_dir / "summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    generated_outputs.append(summary_path)

    # ---- run_meta.json ----
    run_meta_path = out_dir / "run_meta.json"
    generated_outputs.append(run_meta_path)
    meta = build_run_meta(
        groups_files=groups_files,
        hash_inputs=False,
    )

    meta["analysis_unit"] = unit
    if auto_mode:
        meta["grouping"] = {
            "mode": "auto_single_cleaned",
            "auto_group_name": auto_group_name,
        }
    else:
        meta["grouping"] = {"mode": "per_file" if per_file else "groups"}
    meta["environment"] = collect_runtime_environment(project_root)

    norm_dict = asdict(norm)
    norm_canon = json.dumps(norm_dict, ensure_ascii=False, sort_keys=True)
    meta["normalization"] = norm_dict
    meta["normalization_hash_sha256"] = hashlib.sha256(norm_canon.encode("utf-8")).hexdigest()
    meta["partition_validations"] = partition_meta
    meta["group_comparisons"] = comparison_meta
    meta["trace"] = {
        "enabled": config.trace.enabled,
        "files": {
            label: str(path.resolve())
            for label, path in trace_paths.items()
        },
    }
    meta["generated_outputs"] = [str(p.resolve()) for p in generated_outputs]

    write_run_meta(meta, out_dir)

    return partition_exit_code
