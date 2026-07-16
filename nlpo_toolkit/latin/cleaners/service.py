from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from nlpo_toolkit.cleaner_contracts import (
    CleanedFileResult,
    CleanerApplicationError,
    CleanerExecutionRequest,
    CleanerExecutionResult,
)

from .errors import (
    CleanerExecutionError,
    CleanerInputReadError,
    CleanerOutputPlanError,
    CleanerOutputWriteError,
    CleanerTemplateError,
)
from .events import write_ref_events
from .models import CleanerProgram, RefEvent
from .pipeline import clean_document, load_cleaner_program


@dataclass(frozen=True)
class _CleanerFilePlan:
    input_path: Path
    output_path: Path
    doc_id: str


def _validate_parent(path: Path) -> None:
    parent = path.parent
    while parent != parent.parent:
        if parent.exists():
            if not parent.is_dir():
                raise CleanerOutputPlanError(
                    f"Cleaner output parent is not a directory: {parent}; output={path}"
                )
            return
        parent = parent.parent


def _template_name(template: str, source: Path, index: int) -> str:
    try:
        name = template.format(
            index=index, stem=source.stem, ext=source.suffix.lstrip(".")
        )
    except Exception as exc:
        raise CleanerTemplateError(
            f"Invalid output_filename_template={template!r} for input {source}: {exc}"
        ) from exc
    if not name or not name.strip():
        raise CleanerTemplateError(
            f"Output filename template produced an empty name: {template!r}; input={source}"
        )
    candidate = Path(name)
    if candidate.is_absolute():
        raise CleanerTemplateError(
            f"Output filename template produced an absolute path: {template!r}; input={source}"
        )
    return name


def _build_file_plans(request: CleanerExecutionRequest) -> tuple[_CleanerFilePlan, ...]:
    inspection = request.inspection
    config = inspection.config
    sources = tuple(inspection.input_files)
    directory_mode = config.input_path.is_dir()
    if directory_mode and config.output_path.exists() and not config.output_path.is_dir():
        raise CleanerOutputPlanError(
            f"Directory cleaner input requires a directory output: "
            f"input={config.input_path}; output={config.output_path}"
        )
    plans: list[_CleanerFilePlan] = []
    output_root = config.output_path.resolve()
    for index, source in enumerate(sources, 1):
        source = source.resolve()
        if directory_mode:
            name = (
                _template_name(config.output_filename_template, source, index)
                if config.output_filename_template is not None
                else source.name
            )
            output = (output_root / name).resolve()
            if not output.is_relative_to(output_root):
                raise CleanerTemplateError(
                    f"Output filename template escapes configured directory: "
                    f"{config.output_filename_template!r}; input={source}; output={output}"
                )
        else:
            output = (
                (output_root / source.name).resolve()
                if output_root.is_dir()
                else output_root
            )
        prefix = config.doc_id_prefix
        doc_id = f"{prefix}:{source.stem}" if prefix else source.stem
        plans.append(_CleanerFilePlan(source, output, doc_id))

    owners: dict[str, _CleanerFilePlan] = {}
    for plan in plans:
        key = str(plan.output_path).casefold()
        previous = owners.get(key)
        if previous is not None:
            raise CleanerOutputPlanError(
                f"Cleaner output path collision: {previous.input_path} and "
                f"{plan.input_path} use {plan.output_path}"
            )
        owners[key] = plan
        if plan.output_path == plan.input_path:
            raise CleanerOutputPlanError(
                f"Cleaner input and output paths must differ: {plan.input_path}"
            )
        if plan.output_path.exists() and plan.output_path.is_dir():
            raise CleanerOutputPlanError(
                f"Cleaner output file is an existing directory: {plan.output_path}"
            )
        _validate_parent(plan.output_path)
    if config.ref_tsv_path is not None:
        ref_path = config.ref_tsv_path.resolve()
        if ref_path.exists() and ref_path.is_dir():
            raise CleanerOutputPlanError(
                f"Cleaner reference event output is an existing directory: {ref_path}"
            )
        for plan in plans:
            if str(ref_path).casefold() == str(plan.output_path).casefold():
                raise CleanerOutputPlanError(
                    f"Cleaner output path collision: cleaned output and reference "
                    f"events use {ref_path}"
                )
            if ref_path == plan.input_path:
                raise CleanerOutputPlanError(
                    f"Cleaner input and reference event paths must differ: {ref_path}"
                )
        _validate_parent(ref_path)
    return tuple(plans)


def _execute_file(
    plan: _CleanerFilePlan, *, program: CleanerProgram
) -> tuple[CleanedFileResult, tuple[RefEvent, ...]]:
    try:
        raw = plan.input_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise CleanerInputReadError(
            f"Failed to read cleaner input as UTF-8: {plan.input_path}: {exc}"
        ) from exc
    try:
        cleaned = clean_document(
            raw,
            profile=program.profile,
            rules=program.rules,
            lexicon_map=program.lexicon_map,
            doc_id=plan.doc_id,
            snippet_chars=program.snippet_chars,
        )
    except CleanerApplicationError:
        raise
    except Exception as exc:
        raise CleanerExecutionError(
            f"Cleaner pipeline failed for input {plan.input_path}: {exc}"
        ) from exc
    try:
        plan.output_path.parent.mkdir(parents=True, exist_ok=True)
        plan.output_path.write_text(cleaned.text, encoding="utf-8")
    except OSError as exc:
        raise CleanerOutputWriteError(
            f"Failed to write cleaner output: {plan.output_path}: {exc}"
        ) from exc
    return (
        CleanedFileResult(
            plan.input_path, plan.output_path, plan.doc_id, len(cleaned.events)
        ),
        cleaned.events,
    )


def execute_cleaner(request: CleanerExecutionRequest) -> CleanerExecutionResult:
    inspection = request.inspection
    config = inspection.config
    plans = _build_file_plans(request)
    try:
        program = load_cleaner_program(
            kind=config.kind,
            rules_path=config.rules_path,
            lexicon_map_path=config.lexicon_map_path,
        )
    except CleanerApplicationError:
        raise
    except Exception as exc:
        raise CleanerExecutionError(
            f"Failed to build cleaner program for {config.source_path}: {exc}"
        ) from exc

    results: list[CleanedFileResult] = []
    events: list[RefEvent] = []
    for plan in plans:
        result, file_events = _execute_file(plan, program=program)
        results.append(result)
        events.extend(file_events)
    if config.ref_tsv_path is not None:
        try:
            write_ref_events(config.ref_tsv_path, events)
        except OSError as exc:
            raise CleanerOutputWriteError(
                f"Failed to write cleaner reference events: {config.ref_tsv_path}: {exc}"
            ) from exc
    return CleanerExecutionResult(
        config_path=config.source_path,
        kind=config.kind,
        configured_output_path=config.output_path,
        files=tuple(results),
        ref_tsv_path=config.ref_tsv_path,
    )
