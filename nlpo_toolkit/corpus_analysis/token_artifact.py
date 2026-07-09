from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Collection, Iterable, Iterator, Mapping, Sequence

from nlpo_toolkit.nlp import (
    effective_roman_exceptions,
    iter_char_chunks,
    resolve_roman_exceptions,
    should_drop_roman_numeral,
)


TOKEN_ARTIFACT_SCHEMA_NAME = "nlpo-token-artifact"
TOKEN_ARTIFACT_SCHEMA_VERSION = 1

TOKEN_ARTIFACT_COLUMNS = (
    "group",
    "source_file",
    "section",
    "chunk_index",
    "sentence_index",
    "token_index",
    "global_token_index",
    "char_start_in_chunk",
    "char_end_in_chunk",
    "char_start_in_text",
    "char_end_in_text",
    "sentence",
    "token",
    "lemma",
    "upos",
    "analysis_key",
    "included",
    "exclusion_reason",
    "ref_tag",
)

LEGACY_TRACE_COLUMNS = (
    "label",
    "chunk",
    "sent_idx",
    "token_idx",
    "token_char_start_in_chunk",
    "token_char_start_in_text",
    "sentence",
    "token",
    "lemma",
    "upos",
    "ref_tag",
    "global_row",
)


class TokenArtifactError(ValueError):
    pass


@dataclass(frozen=True)
class NLPAnalysisRecord:
    chunk_index: int
    sentence_index: int
    token_index: int
    global_token_index: int
    char_start_in_chunk: int | None
    char_end_in_chunk: int | None
    char_start_in_text: int | None
    char_end_in_text: int | None
    sentence: str
    token: str
    lemma: str | None
    upos: str | None


@dataclass(frozen=True)
class TokenRecord:
    group: str
    source_file: str | None
    chunk_index: int
    sentence_index: int
    token_index: int
    global_token_index: int
    char_start_in_chunk: int | None
    char_end_in_chunk: int | None
    char_start_in_text: int | None
    char_end_in_text: int | None
    sentence: str
    token: str
    lemma: str | None
    upos: str | None
    analysis_key: str | None
    included: bool
    exclusion_reason: str | None
    ref_tag: str | None
    section: str | None = None


@dataclass(frozen=True)
class AnalysisOptions:
    group: str
    source_files: tuple[Path, ...]
    use_lemma: bool
    upos_targets: frozenset[str]
    min_token_length: int = 0
    drop_roman_numerals: bool = False
    roman_exceptions: frozenset[str] = frozenset()
    ref_tag_detector: Callable[[str], str] | None = None
    ref_tag_counter: Counter[str] | None = None


@dataclass(frozen=True)
class TokenArtifactMetadata:
    schema: str = TOKEN_ARTIFACT_SCHEMA_NAME
    schema_version: int = TOKEN_ARTIFACT_SCHEMA_VERSION
    format: str = "tsv"
    encoding: str = "utf-8"
    delimiter: str = "\t"
    complete: bool = True
    row_count: int = 0
    included_row_count: int = 0
    excluded_row_count: int = 0
    group: str = ""
    source_files: tuple[str, ...] = ()
    analysis_unit: str = ""
    upos_targets: tuple[str, ...] = ()
    nlp: Mapping[str, object] | None = None
    filters: Mapping[str, object] | None = None
    artifact_path: str = ""
    sha256: str = ""
    size_bytes: int = 0

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["source_files"] = list(self.source_files)
        data["upos_targets"] = list(self.upos_targets)
        data["nlp"] = dict(self.nlp or {})
        data["filters"] = dict(self.filters or {})
        return data


def token_artifact_metadata_path(tsv_path: Path) -> Path:
    return tsv_path.with_name(f"{tsv_path.stem}.meta.json")


def _tmp_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.tmp")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_int(value: str, *, path: Path, line_number: int, column: str) -> int | None:
    if value == "":
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise TokenArtifactError(
            f"Invalid integer in column {column} at {path}:{line_number}"
        ) from exc


def _required_int(value: str, *, path: Path, line_number: int, column: str) -> int:
    parsed = _optional_int(value, path=path, line_number=line_number, column=column)
    if parsed is None:
        raise TokenArtifactError(
            f"Missing integer in column {column} at {path}:{line_number}"
        )
    return parsed


def _parse_bool(value: str, *, path: Path, line_number: int, column: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise TokenArtifactError(
        f"Invalid boolean in column {column} at {path}:{line_number}"
    )


def _record_to_row(record: TokenRecord) -> dict[str, str]:
    def opt(value: object) -> str:
        return "" if value is None else str(value)

    return {
        "group": record.group,
        "source_file": opt(record.source_file),
        "section": opt(record.section),
        "chunk_index": str(record.chunk_index),
        "sentence_index": str(record.sentence_index),
        "token_index": str(record.token_index),
        "global_token_index": str(record.global_token_index),
        "char_start_in_chunk": opt(record.char_start_in_chunk),
        "char_end_in_chunk": opt(record.char_end_in_chunk),
        "char_start_in_text": opt(record.char_start_in_text),
        "char_end_in_text": opt(record.char_end_in_text),
        "sentence": record.sentence,
        "token": record.token,
        "lemma": opt(record.lemma),
        "upos": opt(record.upos),
        "analysis_key": opt(record.analysis_key),
        "included": "true" if record.included else "false",
        "exclusion_reason": opt(record.exclusion_reason),
        "ref_tag": opt(record.ref_tag),
    }


def _row_to_record(row: Mapping[str, str], *, path: Path, line_number: int) -> TokenRecord:
    return TokenRecord(
        group=row.get("group", ""),
        source_file=row.get("source_file") or None,
        section=row.get("section") or None,
        chunk_index=_required_int(row.get("chunk_index", ""), path=path, line_number=line_number, column="chunk_index"),
        sentence_index=_required_int(row.get("sentence_index", ""), path=path, line_number=line_number, column="sentence_index"),
        token_index=_required_int(row.get("token_index", ""), path=path, line_number=line_number, column="token_index"),
        global_token_index=_required_int(row.get("global_token_index", ""), path=path, line_number=line_number, column="global_token_index"),
        char_start_in_chunk=_optional_int(row.get("char_start_in_chunk", ""), path=path, line_number=line_number, column="char_start_in_chunk"),
        char_end_in_chunk=_optional_int(row.get("char_end_in_chunk", ""), path=path, line_number=line_number, column="char_end_in_chunk"),
        char_start_in_text=_optional_int(row.get("char_start_in_text", ""), path=path, line_number=line_number, column="char_start_in_text"),
        char_end_in_text=_optional_int(row.get("char_end_in_text", ""), path=path, line_number=line_number, column="char_end_in_text"),
        sentence=row.get("sentence", ""),
        token=row.get("token", ""),
        lemma=row.get("lemma") or None,
        upos=row.get("upos") or None,
        analysis_key=row.get("analysis_key") or None,
        included=_parse_bool(row.get("included", ""), path=path, line_number=line_number, column="included"),
        exclusion_reason=row.get("exclusion_reason") or None,
        ref_tag=row.get("ref_tag") or None,
    )


class TokenArtifactWriter:
    def __init__(self, path: Path, *, metadata: TokenArtifactMetadata) -> None:
        self.path = Path(path)
        self.metadata_path = token_artifact_metadata_path(self.path)
        self._metadata = metadata
        self._tmp = _tmp_path(self.path)
        self._meta_tmp = _tmp_path(self.metadata_path)
        self._file: Any = None
        self._writer: csv.DictWriter[str] | None = None
        self.row_count = 0
        self.included_row_count = 0
        self.excluded_row_count = 0
        self.final_metadata: TokenArtifactMetadata | None = None

    def __enter__(self) -> "TokenArtifactWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._tmp.open("w", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=TOKEN_ARTIFACT_COLUMNS,
            delimiter="\t",
            lineterminator="\n",
        )
        self._writer.writeheader()
        return self

    def write(self, record: TokenRecord) -> None:
        if self._writer is None:
            raise TokenArtifactError("TokenArtifactWriter is not open")
        self._writer.writerow(_record_to_row(record))
        self.row_count += 1
        if record.included:
            self.included_row_count += 1
        else:
            self.excluded_row_count += 1

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        try:
            if self._file is not None:
                self._file.close()
            if exc_type is not None:
                self._cleanup_temp()
                return

            self._tmp.replace(self.path)
            metadata = replace(
                self._metadata,
                complete=True,
                row_count=self.row_count,
                included_row_count=self.included_row_count,
                excluded_row_count=self.excluded_row_count,
                artifact_path=str(self.path),
                sha256=_sha256_file(self.path),
                size_bytes=self.path.stat().st_size,
            )
            self._meta_tmp.write_text(
                json.dumps(metadata.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self._meta_tmp.replace(self.metadata_path)
            self.final_metadata = metadata
        except Exception:
            self._cleanup_temp()
            raise

    def _cleanup_temp(self) -> None:
        for path in (self._tmp, self._meta_tmp):
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def read_token_artifact_metadata(path: Path) -> TokenArtifactMetadata:
    metadata_path = token_artifact_metadata_path(path)
    if not metadata_path.exists():
        raise TokenArtifactError(f"Token artifact metadata was not found: {metadata_path}")
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise TokenArtifactError(f"Failed to read token artifact metadata: {metadata_path}") from exc

    if data.get("schema") != TOKEN_ARTIFACT_SCHEMA_NAME:
        raise TokenArtifactError(f"Unsupported token artifact schema: {metadata_path}")
    if data.get("schema_version") != TOKEN_ARTIFACT_SCHEMA_VERSION:
        raise TokenArtifactError(
            f"Unsupported token artifact schema version {data.get('schema_version')}: {path}"
        )

    return TokenArtifactMetadata(
        complete=bool(data.get("complete", False)),
        row_count=int(data.get("row_count", 0)),
        included_row_count=int(data.get("included_row_count", 0)),
        excluded_row_count=int(data.get("excluded_row_count", 0)),
        group=str(data.get("group", "")),
        source_files=tuple(str(item) for item in data.get("source_files", [])),
        analysis_unit=str(data.get("analysis_unit", "")),
        upos_targets=tuple(str(item) for item in data.get("upos_targets", [])),
        nlp=data.get("nlp") if isinstance(data.get("nlp"), dict) else {},
        filters=data.get("filters") if isinstance(data.get("filters"), dict) else {},
        artifact_path=str(data.get("artifact_path", "")),
        sha256=str(data.get("sha256", "")),
        size_bytes=int(data.get("size_bytes", 0)),
    )


def read_token_records(
    path: Path,
    *,
    require_complete: bool = True,
    verify_hash: bool = False,
) -> Iterator[TokenRecord]:
    path = Path(path)
    if not path.exists():
        raise TokenArtifactError(f"Token artifact not found: {path}")
    metadata = read_token_artifact_metadata(path)
    if require_complete and not metadata.complete:
        raise TokenArtifactError(f"Token artifact is incomplete: {path}")
    if verify_hash and metadata.sha256 and _sha256_file(path) != metadata.sha256:
        raise TokenArtifactError(f"Token artifact hash mismatch: {path}")

    count = 0
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if tuple(reader.fieldnames or ()) != TOKEN_ARTIFACT_COLUMNS:
            raise TokenArtifactError(f"Token artifact header does not match schema: {path}")
        for line_number, row in enumerate(reader, start=2):
            count += 1
            yield _row_to_record(row, path=path, line_number=line_number)
    if count != metadata.row_count:
        raise TokenArtifactError(
            f"Token artifact row_count mismatch: {path} expected {metadata.row_count}, found {count}"
        )


def validate_token_artifact(path: Path) -> TokenArtifactMetadata:
    metadata = read_token_artifact_metadata(path)
    for _record in read_token_records(path, verify_hash=True):
        pass
    return metadata


def _legacy_int(value: str | None, default: int = 0) -> int:
    try:
        return int(value or "")
    except ValueError:
        return default


def read_legacy_trace_records(path: Path) -> Iterator[TokenRecord]:
    path = Path(path)
    if not path.exists():
        raise TokenArtifactError(f"Trace not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for idx, row in enumerate(reader):
            token = row.get("token", "")
            if token == "(trace stopped; counting continues)" or row.get("upos") == "TRACE_TRUNCATED":
                continue
            group = row.get("group") or row.get("label") or ""
            key = row.get("lemma") or row.get("token") or ""
            source_file = row.get("source_file") or row.get("file") or None
            start_in_chunk = row.get("token_char_start_in_chunk") or row.get("char_start_in_chunk")
            start_in_text = row.get("token_char_start_in_text") or row.get("char_start_in_text")
            yield TokenRecord(
                group=group,
                source_file=source_file,
                section=None,
                chunk_index=_legacy_int(row.get("chunk") or row.get("chunk_index")),
                sentence_index=_legacy_int(row.get("sent_idx") or row.get("sentence_index")),
                token_index=_legacy_int(row.get("token_idx") or row.get("token_index")),
                global_token_index=_legacy_int(row.get("global_row") or row.get("global_token_index"), idx + 1),
                char_start_in_chunk=_legacy_int(start_in_chunk) if start_in_chunk not in (None, "") else None,
                char_end_in_chunk=None,
                char_start_in_text=_legacy_int(start_in_text) if start_in_text not in (None, "") else None,
                char_end_in_text=None,
                sentence=row.get("sentence", ""),
                token=token,
                lemma=row.get("lemma") or None,
                upos=row.get("upos") or None,
                analysis_key=key.strip().lower() or None,
                included=True,
                exclusion_reason=None,
                ref_tag=row.get("ref_tag") or None,
            )


def read_token_rows(path: Path, *, allow_legacy_trace: bool = True) -> Iterator[TokenRecord]:
    if token_artifact_metadata_path(path).exists():
        yield from read_token_records(path)
        return
    if allow_legacy_trace:
        yield from read_legacy_trace_records(path)
        return
    raise TokenArtifactError(f"Token artifact metadata was not found: {token_artifact_metadata_path(path)}")


def _sentence_text(sent: Any) -> str:
    text = getattr(sent, "text", None)
    if text:
        return str(text)
    return " ".join(str(getattr(token, "text", "")) for token in getattr(sent, "tokens", []) or [])


def _token_key_from_values(token: str, lemma: str | None, *, use_lemma: bool) -> str | None:
    selected = lemma if (use_lemma and lemma) else token
    if selected is None:
        return None
    key = str(selected).strip().lower()
    return key or None


def iter_nlp_analysis_records(
    *,
    document: Any,
    chunk_index: int,
    chunk_start_in_text: int,
    global_token_start: int,
) -> Iterator[NLPAnalysisRecord]:
    global_index = global_token_start
    for sentence_index, sent in enumerate(getattr(document, "sentences", []) or []):
        sent_text = _sentence_text(sent)
        for token_index, token in enumerate(getattr(sent, "tokens", []) or []):
            token_text = str(getattr(token, "text", "") or "")
            lemma = getattr(token, "lemma", None)
            upos = getattr(token, "upos", None)
            start = getattr(token, "start_char", None)
            end = getattr(token, "end_char", None)
            yield NLPAnalysisRecord(
                chunk_index=chunk_index,
                sentence_index=sentence_index,
                token_index=token_index,
                global_token_index=global_index,
                char_start_in_chunk=start,
                char_end_in_chunk=end,
                char_start_in_text=chunk_start_in_text + start if start is not None else None,
                char_end_in_text=chunk_start_in_text + end if end is not None else None,
                sentence=sent_text,
                token=token_text,
                lemma=str(lemma) if lemma is not None else None,
                upos=str(upos) if upos is not None else None,
            )
            global_index += 1


def iter_nlp_analysis_records_from_text(
    *,
    text: str,
    nlp: Callable[[str], Any],
    chunk_chars: int = 200_000,
) -> Iterator[NLPAnalysisRecord]:
    if not text:
        return

    global_index = 0
    chunk_base_offset = 0

    for chunk_index, chunk in enumerate(iter_char_chunks(text, chunk_chars=chunk_chars)):
        doc = nlp(chunk)
        emitted = 0
        for record in iter_nlp_analysis_records(
            document=doc,
            chunk_index=chunk_index,
            chunk_start_in_text=chunk_base_offset,
            global_token_start=global_index,
        ):
            emitted += 1
            yield record
        global_index += emitted
        chunk_base_offset += len(chunk)


def evaluate_analysis_record(
    record: NLPAnalysisRecord,
    *,
    options: AnalysisOptions,
) -> TokenRecord:
    source_file = str(options.source_files[0]) if len(options.source_files) == 1 else None
    key = _token_key_from_values(record.token, record.lemma, use_lemma=options.use_lemma)
    effective_exceptions = effective_roman_exceptions(
        use_lemma=options.use_lemma,
        configured_exceptions=options.roman_exceptions,
    )
    ref_tag = ""
    exclusion_reason: str | None = None
    if key is None:
        exclusion_reason = "missing_key"
    elif record.upos not in options.upos_targets:
        exclusion_reason = "upos_not_targeted"
    elif len(key) < options.min_token_length:
        exclusion_reason = "too_short"
    elif should_drop_roman_numeral(
        key,
        drop_roman_numerals=options.drop_roman_numerals,
        effective_exceptions=effective_exceptions,
    ):
        exclusion_reason = "roman_numeral"
    elif options.ref_tag_detector is not None:
        ref_tag = options.ref_tag_detector(key)
        if ref_tag:
            if options.ref_tag_counter is not None:
                options.ref_tag_counter[ref_tag] += 1
            exclusion_reason = "reference_tag"

    return TokenRecord(
        group=options.group,
        source_file=source_file,
        section=None,
        chunk_index=record.chunk_index,
        sentence_index=record.sentence_index,
        token_index=record.token_index,
        global_token_index=record.global_token_index,
        char_start_in_chunk=record.char_start_in_chunk,
        char_end_in_chunk=record.char_end_in_chunk,
        char_start_in_text=record.char_start_in_text,
        char_end_in_text=record.char_end_in_text,
        sentence=record.sentence,
        token=record.token,
        lemma=record.lemma,
        upos=record.upos,
        analysis_key=key,
        included=exclusion_reason is None,
        exclusion_reason=exclusion_reason,
        ref_tag=ref_tag or None,
    )


def iter_token_records(
    *,
    text: str,
    nlp: Callable[[str], Any],
    group: str,
    source_files: Sequence[Path],
    use_lemma: bool,
    upos_targets: Collection[str],
    min_token_length: int = 0,
    drop_roman_numerals: bool = False,
    roman_exceptions: Collection[str] | None = None,
    ref_tag_detector: Callable[[str], str] | None = None,
    ref_tag_counter: Counter[str] | None = None,
    chunk_chars: int = 200_000,
) -> Iterator[TokenRecord]:
    options = AnalysisOptions(
        group=group,
        source_files=tuple(source_files),
        use_lemma=use_lemma,
        upos_targets=frozenset(upos_targets),
        min_token_length=min_token_length,
        drop_roman_numerals=drop_roman_numerals,
        roman_exceptions=resolve_roman_exceptions(roman_exceptions=roman_exceptions),
        ref_tag_detector=ref_tag_detector,
        ref_tag_counter=ref_tag_counter,
    )
    for record in iter_nlp_analysis_records_from_text(
        text=text,
        nlp=nlp,
        chunk_chars=chunk_chars,
    ):
        yield evaluate_analysis_record(record, options=options)


def counter_from_token_records(records: Iterable[TokenRecord]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for record in records:
        if record.included and record.analysis_key:
            counter[record.analysis_key] += 1
    return counter


class DiagnosticTraceWriter:
    def __init__(
        self,
        path: Path,
        *,
        max_rows: int = 0,
        only_keys: Collection[str] | None = None,
        write_truncation_marker: bool = True,
    ) -> None:
        self.path = Path(path)
        self.max_rows = max_rows
        self.only_keys = {str(key).strip().lower() for key in (only_keys or ()) if str(key).strip()}
        self.write_truncation_marker = write_truncation_marker
        self._file: Any = None
        self._writer: csv.writer[Any] | None = None
        self._written = 0
        self._truncated = False

    def __enter__(self) -> "DiagnosticTraceWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8", newline="")
        self._writer = csv.writer(self._file, delimiter="\t")
        self._writer.writerow(LEGACY_TRACE_COLUMNS)
        return self

    def consider(self, record: TokenRecord) -> None:
        if self._writer is None or self._truncated or not record.included:
            return
        key = record.analysis_key or ""
        if self.only_keys and key not in self.only_keys:
            return
        if self.max_rows > 0 and self._written >= self.max_rows:
            if self.write_truncation_marker:
                self._writer.writerow(
                    [
                        record.group,
                        record.chunk_index,
                        record.sentence_index,
                        record.token_index,
                        record.char_start_in_chunk if record.char_start_in_chunk is not None else "",
                        record.char_start_in_text if record.char_start_in_text is not None else "",
                        record.sentence,
                        "(trace stopped; counting continues)",
                        "",
                        "TRACE_TRUNCATED",
                        "",
                        self._written + 1,
                    ]
                )
            self._truncated = True
            return
        self._writer.writerow(
            [
                record.group,
                record.chunk_index,
                record.sentence_index,
                record.token_index,
                record.char_start_in_chunk if record.char_start_in_chunk is not None else "",
                record.char_start_in_text if record.char_start_in_text is not None else "",
                record.sentence,
                record.token,
                record.lemma or "",
                record.upos or "",
                record.ref_tag or "",
                self._written + 1,
            ]
        )
        self._written += 1

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self._file is not None:
            self._file.close()
