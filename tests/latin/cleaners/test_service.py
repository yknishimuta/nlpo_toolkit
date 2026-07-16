from pathlib import Path
from types import MappingProxyType

import pytest

import nlpo_toolkit.latin.cleaners.service as service
from nlpo_toolkit.cleaner_contracts import (
    CleanerConfig,
    CleanerConfigInspection,
    CleanerExecutionRequest,
)
from nlpo_toolkit.latin.cleaners.errors import (
    CleanerExecutionError,
    CleanerInputReadError,
    CleanerOutputPlanError,
    CleanerOutputWriteError,
    CleanerTemplateError,
)
from nlpo_toolkit.latin.cleaners.models import (
    CleanerProfile,
    CleanerProgram,
    CleaningResult,
    RefEvent,
    RuleReference,
    RuleSet,
)


def _program(kind="scholastic_text") -> CleanerProgram:
    profile = CleanerProfile(kind, Path("rules.yml"), lambda text: tuple(text.splitlines()), lambda line: line)
    return CleanerProgram(profile, RuleSet(), MappingProxyType({}))


def _inspection(
    tmp_path: Path,
    *,
    directory: bool = False,
    template: str | None = None,
    output: Path | None = None,
    ref_tsv: Path | None = None,
) -> CleanerConfigInspection:
    source = tmp_path / ("input" if directory else "input.txt")
    if directory:
        source.mkdir()
        (source / "b.txt").write_text("b", encoding="utf-8")
        (source / "a.txt").write_text("a", encoding="utf-8")
        files = tuple(sorted(source.glob("*.txt")))
    else:
        source.write_text("input", encoding="utf-8")
        files = (source,)
    config_path = tmp_path / "cleaner.yml"
    config = CleanerConfig(
        config_path.resolve(),
        "scholastic_text",
        source.resolve(),
        (output or tmp_path / "cleaned").resolve(),
        ref_tsv_path=ref_tsv,
        output_filename_template=template,
        doc_id_prefix="DOC",
    )
    return CleanerConfigInspection(config, tuple(path.resolve() for path in files), ())


def test_single_file_uses_inspection_and_returns_typed_result(tmp_path, monkeypatch) -> None:
    inspection = _inspection(tmp_path, output=tmp_path / "result.txt")
    loads = []
    monkeypatch.setattr(service, "load_cleaner_program", lambda **kwargs: loads.append(kwargs) or _program())
    monkeypatch.setattr(service, "clean_document", lambda raw, **kwargs: CleaningResult(raw.upper(), ()))

    result = service.execute_cleaner(CleanerExecutionRequest(inspection))

    assert len(loads) == 1
    assert result.config_path == inspection.config.source_path
    assert result.configured_output_path == inspection.config.output_path
    assert result.output_files == (tmp_path / "result.txt",)
    assert result.files[0].doc_id == "DOC:input"
    assert (tmp_path / "result.txt").read_text(encoding="utf-8") == "INPUT"


def test_directory_template_is_always_respected_and_program_built_once(tmp_path, monkeypatch) -> None:
    inspection = _inspection(tmp_path, directory=True, template="cleaned_{index:03d}.txt")
    loads = []
    docs = []
    monkeypatch.setattr(service, "load_cleaner_program", lambda **kwargs: loads.append(kwargs) or _program())

    def clean(raw, **kwargs):
        docs.append(kwargs["doc_id"])
        return CleaningResult(raw.upper(), ())

    monkeypatch.setattr(service, "clean_document", clean)
    result = service.execute_cleaner(CleanerExecutionRequest(inspection))

    assert len(loads) == 1
    assert [path.name for path in result.output_files] == ["cleaned_001.txt", "cleaned_002.txt"]
    assert docs == ["DOC:a", "DOC:b"]
    assert not (tmp_path / "cleaned/a.cleaned.txt").exists()


@pytest.mark.parametrize("template", ["same.txt", "../outside.txt", "/absolute.txt", "{missing}.txt", ""])
def test_invalid_or_colliding_directory_plan_fails_before_cleaning(tmp_path, monkeypatch, template) -> None:
    inspection = _inspection(tmp_path, directory=True, template=template)
    called = False

    def clean(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(service, "clean_document", clean)
    expected = CleanerOutputPlanError if template == "same.txt" else CleanerTemplateError
    with pytest.raises(expected):
        service.execute_cleaner(CleanerExecutionRequest(inspection))
    assert not called
    assert not inspection.config.output_path.exists()


def test_input_and_output_same_path_is_rejected_before_program_load(tmp_path, monkeypatch) -> None:
    source = tmp_path / "input.txt"
    inspection = _inspection(tmp_path, output=source)
    monkeypatch.setattr(service, "load_cleaner_program", lambda **kwargs: pytest.fail("program must not load"))
    with pytest.raises(CleanerOutputPlanError, match="must differ"):
        service.execute_cleaner(CleanerExecutionRequest(inspection))


def test_directory_output_that_is_a_file_is_rejected_without_partial_output(tmp_path) -> None:
    output = tmp_path / "occupied"
    output.write_text("existing", encoding="utf-8")
    inspection = _inspection(tmp_path, directory=True, output=output)
    with pytest.raises(CleanerOutputPlanError, match="directory output"):
        service.execute_cleaner(CleanerExecutionRequest(inspection))
    assert output.read_text(encoding="utf-8") == "existing"


def test_events_are_aggregated_and_replaced_for_each_run(tmp_path, monkeypatch) -> None:
    event_path = tmp_path / "events.tsv"
    inspection = _inspection(tmp_path, directory=True, ref_tsv=event_path)
    monkeypatch.setattr(service, "load_cleaner_program", lambda **kwargs: _program())

    def clean(raw, **kwargs):
        event = RefEvent(kwargs["doc_id"], "scholastic_text", "rule", "substitute", 1, 1, RuleReference(), raw)
        return CleaningResult(raw, (event,))

    monkeypatch.setattr(service, "clean_document", clean)
    result = service.execute_cleaner(CleanerExecutionRequest(inspection))
    assert result.reference_event_count == 2
    assert len(event_path.read_text(encoding="utf-8").splitlines()) == 3

    single = CleanerConfigInspection(inspection.config, inspection.input_files[:1], ())
    result = service.execute_cleaner(CleanerExecutionRequest(single))
    assert result.reference_event_count == 1
    assert len(event_path.read_text(encoding="utf-8").splitlines()) == 2


def test_invalid_utf8_is_a_typed_read_error(tmp_path, monkeypatch) -> None:
    inspection = _inspection(tmp_path)
    inspection.input_files[0].write_bytes(b"\xff")
    monkeypatch.setattr(service, "load_cleaner_program", lambda **kwargs: _program())
    with pytest.raises(CleanerInputReadError) as caught:
        service.execute_cleaner(CleanerExecutionRequest(inspection))
    assert isinstance(caught.value.__cause__, UnicodeDecodeError)


def test_pipeline_failure_is_typed_and_preserves_cause(tmp_path, monkeypatch) -> None:
    inspection = _inspection(tmp_path)
    failure = LookupError("pipeline")
    monkeypatch.setattr(service, "load_cleaner_program", lambda **kwargs: _program())
    monkeypatch.setattr(service, "clean_document", lambda *args, **kwargs: (_ for _ in ()).throw(failure))
    with pytest.raises(CleanerExecutionError) as caught:
        service.execute_cleaner(CleanerExecutionRequest(inspection))
    assert caught.value.__cause__ is failure


def test_event_write_failure_is_typed_and_preserves_cause(tmp_path, monkeypatch) -> None:
    inspection = _inspection(tmp_path, ref_tsv=tmp_path / "events.tsv")
    failure = OSError("disk full")
    monkeypatch.setattr(service, "load_cleaner_program", lambda **kwargs: _program())
    monkeypatch.setattr(service, "clean_document", lambda raw, **kwargs: CleaningResult(raw, ()))
    monkeypatch.setattr(service, "write_ref_events", lambda path, events: (_ for _ in ()).throw(failure))
    with pytest.raises(CleanerOutputWriteError) as caught:
        service.execute_cleaner(CleanerExecutionRequest(inspection))
    assert caught.value.__cause__ is failure
