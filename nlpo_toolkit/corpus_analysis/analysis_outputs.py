from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import nlpo_toolkit.nlp as nlp

from .config import TraceConfig
from .corpus import PreparedCorpus, sanitize_label
from . import dictcheck
from .outputs import build_frequency_output_paths, write_frequency_csv
from .run_plan import AnalysisPlan
from .runner_types import DictCheckOutput
from .token_artifact import token_artifact_metadata_path

__all__ = [
    "AnalysisOutputPlan",
    "GroupOutputPaths",
    "GroupOutputResult",
    "KnownUnknownCounters",
    "apply_lemma_normalization",
    "build_analysis_output_plan",
    "build_labeled_output_paths",
    "load_configured_known_words",
    "load_configured_lemma_normalization",
    "split_known_unknown",
    "write_dictcheck_outputs",
    "write_group_analysis_outputs",
]


@dataclass(frozen=True)
class GroupOutputPaths:
    trace: Path | None
    token_artifact: Path | None


@dataclass(frozen=True)
class AnalysisOutputPlan:
    by_group: Mapping[str, GroupOutputPaths]

    def for_group(self, label: str) -> GroupOutputPaths:
        return self.by_group[label]

    @property
    def trace_paths(self) -> dict[str, Path]:
        return {
            label: paths.trace
            for label, paths in self.by_group.items()
            if paths.trace is not None
        }


@dataclass(frozen=True)
class KnownUnknownCounters:
    known: Counter[str]
    unknown: Counter[str]


@dataclass(frozen=True)
class GroupOutputResult:
    counter: Counter[str]
    generated_outputs: tuple[Path, ...]


def _trace_base_path(trace: TraceConfig, out_dir: Path, project_root: Path) -> Path:
    if trace.path:
        path = Path(str(trace.path))
        return (project_root / path).resolve() if not path.is_absolute() else path
    return out_dir / "trace.tsv"


def _token_artifact_base_path(path_value: str, project_root: Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = (project_root / path).resolve()
    return path if path.suffix else path.with_suffix(".tsv")


def build_labeled_output_paths(
    *, base_path: Path, labels: Sequence[str]
) -> dict[str, Path]:
    counts: dict[str, int] = {}
    paths: dict[str, Path] = {}
    for label in labels:
        safe = sanitize_label(label)
        counts[safe] = counts.get(safe, 0) + 1
        effective = safe if counts[safe] == 1 else f"{safe}_{counts[safe]}"
        if len(labels) <= 1:
            paths[label] = base_path
        else:
            suffix = base_path.suffix or ".tsv"
            stem = base_path.stem or "trace"
            paths[label] = base_path.with_name(f"{stem}_{effective}{suffix}")
    return paths


def _validate_output_paths(plan: AnalysisOutputPlan) -> None:
    owners: dict[Path, str] = {}
    for label, paths in plan.by_group.items():
        candidates = (("trace", paths.trace), ("token artifact", paths.token_artifact))
        for kind, path in candidates:
            if path is None:
                continue
            resolved = path.resolve()
            previous = owners.get(resolved)
            if previous is not None:
                if {previous.split(" ", 1)[0], kind} == {"trace", "token artifact"}:
                    raise ValueError(
                        "Token artifact path and diagnostic trace path must be "
                        f"different: {resolved}"
                    )
                raise ValueError(f"Analysis output path collision: {previous} and {kind} {label}: {resolved}")
            owners[resolved] = f"{kind} {label}"
            if kind == "token artifact":
                metadata = token_artifact_metadata_path(path).resolve()
                previous = owners.get(metadata)
                if previous is not None:
                    raise ValueError(f"Analysis output path collision: {previous} and token artifact metadata {label}: {metadata}")
                owners[metadata] = f"token artifact metadata {label}"


def build_analysis_output_plan(
    *, plan: AnalysisPlan, corpora: Sequence[PreparedCorpus]
) -> AnalysisOutputPlan:
    labels = [corpus.label for corpus in corpora]
    traces = (
        build_labeled_output_paths(
            base_path=_trace_base_path(plan.config.trace, plan.out_dir, plan.project_root),
            labels=labels,
        )
        if plan.config.trace.enabled
        else {}
    )
    artifacts = (
        build_labeled_output_paths(
            base_path=_token_artifact_base_path(
                plan.config.artifacts.tokens.path, plan.project_root
            ),
            labels=labels,
        )
        if plan.config.artifacts.tokens.enabled
        else {}
    )
    result = AnalysisOutputPlan(
        {
            label: GroupOutputPaths(traces.get(label), artifacts.get(label))
            for label in labels
        }
    )
    _validate_output_paths(result)
    return result


def apply_lemma_normalization(
    counter: Mapping[str, int], normalization_map: Mapping[str, str]
) -> Counter[str]:
    normalized: Counter[str] = Counter()
    for lemma, count in counter.items():
        normalized[normalization_map.get(lemma, lemma)] += count
    return normalized


def split_known_unknown(
    counter: Mapping[str, int], known_words: Iterable[str]
) -> KnownUnknownCounters:
    known_set = set(known_words)
    return KnownUnknownCounters(
        known=Counter({word: count for word, count in counter.items() if word in known_set}),
        unknown=Counter({word: count for word, count in counter.items() if word not in known_set}),
    )


def load_configured_lemma_normalization(plan: AnalysisPlan) -> Mapping[str, str] | None:
    path = plan.config_files.path("dictcheck.lemma_normalize")
    return dictcheck.load_lemma_normalize_map(path) if path is not None else None


def load_configured_known_words(plan: AnalysisPlan) -> frozenset[str] | None:
    path = plan.config_files.path("dictcheck.wordlist")
    if plan.config.dictcheck.enabled and path is None:
        raise ValueError(
            "dictcheck.wordlist is required when dictcheck.enabled=true "
            f"(analysis_unit={plan.analysis_unit})"
        )
    return frozenset(nlp.load_vocab(path)) if plan.config.dictcheck.enabled else None


def write_dictcheck_outputs(
    *,
    plan: AnalysisPlan,
    label: str,
    counter: Mapping[str, int],
    known_words: Iterable[str] | None,
) -> DictCheckOutput | None:
    if not plan.config.dictcheck.enabled:
        return None
    assert known_words is not None
    split = split_known_unknown(counter, known_words)
    paths = build_frequency_output_paths(plan.out_dir, label)
    write_frequency_csv(paths.known, split.known, header=plan.csv_header)
    write_frequency_csv(paths.unknown, split.unknown, header=plan.csv_header)
    return DictCheckOutput(split.known, split.unknown, (paths.known, paths.unknown))


def write_group_analysis_outputs(
    *,
    plan: AnalysisPlan,
    corpus: PreparedCorpus,
    counter: Counter[str],
    normalization_map: Mapping[str, str] | None,
    known_words: Iterable[str] | None,
    token_generated_outputs: tuple[Path, ...],
) -> GroupOutputResult:
    result_counter = (
        apply_lemma_normalization(counter, normalization_map)
        if normalization_map is not None
        else counter
    )
    generated: list[Path] = []
    if plan.config.ref_tags.enabled:
        ref_tags_path = plan.out_dir / f"ref_tags_{corpus.label}.csv"
        write_frequency_csv(ref_tags_path, corpus.ref_tag_counts, header=("tag", "count"))
        generated.append(ref_tags_path)
    frequency_paths = build_frequency_output_paths(plan.out_dir, corpus.label)
    write_frequency_csv(frequency_paths.base, result_counter, header=plan.csv_header)
    generated.append(frequency_paths.base)
    dictcheck = write_dictcheck_outputs(
        plan=plan,
        label=corpus.label,
        counter=result_counter,
        known_words=known_words,
    )
    if dictcheck is not None:
        generated.extend(dictcheck.generated_outputs)
    generated.extend(token_generated_outputs)
    return GroupOutputResult(result_counter, tuple(generated))
