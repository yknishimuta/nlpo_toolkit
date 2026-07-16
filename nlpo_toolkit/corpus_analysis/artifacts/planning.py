from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence

from nlpo_toolkit.comparison.configured import sanitize_comparison_name

from ..corpus import PreparedCorpus, sanitize_label
from ..partition_validation import sanitize_partition_name
from ..run_plan import ResolvedAnalysisPlan
from .models import ArtifactKind, ArtifactPlan, PlannedArtifact


def _configured_path(value: str, project_root: Path, *, default_suffix: str = "") -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    if default_suffix and not path.suffix:
        path = path.with_suffix(default_suffix)
    return path.resolve()


def _labeled_paths(base: Path, labels: Sequence[str]) -> dict[str, Path]:
    counts: dict[str, int] = {}
    result: dict[str, Path] = {}
    for label in labels:
        safe = sanitize_label(label)
        counts[safe] = counts.get(safe, 0) + 1
        effective = safe if counts[safe] == 1 else f"{safe}_{counts[safe]}"
        if len(labels) == 1:
            result[label] = base
        else:
            result[label] = base.with_name(
                f"{base.stem or 'artifact'}_{effective}{base.suffix or '.tsv'}"
            )
    return result


def build_count_artifact_plan(*, plan: ResolvedAnalysisPlan,
                              corpora: Sequence[PreparedCorpus]) -> ArtifactPlan:
    config = plan.config
    out_dir = plan.out_dir.resolve()
    labels = [corpus.label for corpus in corpora]
    artifacts: list[PlannedArtifact] = []
    trace_paths: dict[str, Path] = {}
    if config.trace.enabled:
        trace_base = (_configured_path(str(config.trace.path), plan.project_root)
                      if config.trace.path else out_dir / "trace.tsv")
        trace_paths = _labeled_paths(trace_base, labels)
    token_paths: dict[str, Path] = {}
    if config.artifacts.tokens.enabled:
        token_base = _configured_path(config.artifacts.tokens.path, plan.project_root,
                                      default_suffix=".tsv")
        token_paths = _labeled_paths(token_base, labels)

    for label in labels:
        frequency = out_dir / f"frequency_{label}.csv"
        artifacts.append(PlannedArtifact(ArtifactKind.FREQUENCY, frequency, group=label))
        if config.dictcheck.enabled:
            artifacts.extend((
                PlannedArtifact(ArtifactKind.DICTCHECK_KNOWN,
                                frequency.with_name(f"{frequency.stem}.known.csv"), group=label),
                PlannedArtifact(ArtifactKind.DICTCHECK_UNKNOWN,
                                frequency.with_name(f"{frequency.stem}.unknown.csv"), group=label),
            ))
        if config.ref_tags.enabled:
            artifacts.append(PlannedArtifact(ArtifactKind.REFERENCE_TAGS,
                                             out_dir / f"ref_tags_{label}.csv", group=label))
        if label in trace_paths:
            artifacts.append(PlannedArtifact(ArtifactKind.DIAGNOSTIC_TRACE,
                                             trace_paths[label], group=label))
        if label in token_paths:
            token = token_paths[label]
            artifacts.extend((
                PlannedArtifact(ArtifactKind.TOKEN_ARTIFACT, token, group=label),
                PlannedArtifact(ArtifactKind.TOKEN_ARTIFACT_METADATA,
                                token.with_name(f"{token.stem}.meta.json"), group=label),
            ))

    for spec in plan.partition_specs:
        artifacts.append(PlannedArtifact(
            ArtifactKind.PARTITION_VALIDATION_CSV,
            out_dir / f"partition_validation_{sanitize_partition_name(spec.name)}.csv",
            name=spec.name,
        ))
    if plan.partition_specs:
        artifacts.append(PlannedArtifact(ArtifactKind.PARTITION_VALIDATION_JSON,
                                         out_dir / "partition_validation.json"))
    for spec in plan.comparison_specs:
        artifacts.append(PlannedArtifact(
            ArtifactKind.GROUP_COMPARISON_CSV,
            out_dir / f"group_comparison_{sanitize_comparison_name(spec.name)}.csv",
            name=spec.name,
        ))
    if plan.comparison_specs:
        artifacts.append(PlannedArtifact(ArtifactKind.GROUP_COMPARISONS_JSON,
                                         out_dir / "group_comparisons.json"))
    artifacts.extend((
        PlannedArtifact(ArtifactKind.SUMMARY, out_dir / "summary.txt"),
        PlannedArtifact(ArtifactKind.RUN_METADATA, out_dir / "run_meta.json"),
    ))
    return ArtifactPlan(tuple(artifacts))
