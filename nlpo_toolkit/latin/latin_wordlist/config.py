from __future__ import annotations

from pathlib import Path

from pydantic import Field, StrictInt, ValidationError, field_validator, model_validator

from nlpo_toolkit.config_model import ConfigModel
from nlpo_toolkit.configuration.yaml_loader import YamlLoadError, load_yaml_mapping

from .errors import LatinWordlistConfigError
from .models import (
    LatinWordlistBuildRequest,
    WordlistFilterPolicy,
    WordlistTokenizationPolicy,
)


def _validate_path_input(value: object) -> object:
    if isinstance(value, str) and not value.strip():
        raise ValueError("path must not be empty")
    return value


class WordlistInputsConfig(ConfigModel):
    conllu_dir: Path = Path("input/treebank_latin")
    latin_text_dir: Path = Path("input/latin_texts")
    extra_wordlists: tuple[Path, ...] = ()

    @field_validator("conllu_dir", "latin_text_dir", mode="before")
    @classmethod
    def validate_directory_path(cls, value: object) -> object:
        return _validate_path_input(value)

    @field_validator("extra_wordlists", mode="before")
    @classmethod
    def validate_extra_paths(cls, value: object) -> object:
        if isinstance(value, (list, tuple)):
            for item in value:
                _validate_path_input(item)
        return value

    @model_validator(mode="after")
    def reject_duplicate_extra_paths(self) -> WordlistInputsConfig:
        if len(set(self.extra_wordlists)) != len(self.extra_wordlists):
            raise ValueError("extra_wordlists contains duplicate paths")
        return self


class WordlistOutputConfig(ConfigModel):
    latin_wordlist_out: Path = Path("output/latin_words.txt")

    @field_validator("latin_wordlist_out", mode="before")
    @classmethod
    def validate_output_path(cls, value: object) -> object:
        return _validate_path_input(value)


class WordlistFiltersConfig(ConfigModel):
    min_length: StrictInt = Field(default=2, ge=1)
    min_form_freq: StrictInt = Field(default=2, ge=1)
    min_text_freq: StrictInt = Field(default=3, ge=1)


class WordlistTokenizationConfig(ConfigModel):
    extra_punct: str = "“”‘’«»…—–-­"


class LatinWordlistConfig(ConfigModel):
    inputs: WordlistInputsConfig = WordlistInputsConfig()
    output: WordlistOutputConfig = WordlistOutputConfig()
    filters: WordlistFiltersConfig = WordlistFiltersConfig()
    tokenize: WordlistTokenizationConfig = WordlistTokenizationConfig()


def _resolve_from(base: Path, path: Path) -> Path:
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def load_wordlist_build_request(config_path: Path) -> LatinWordlistBuildRequest:
    source_path = config_path.expanduser().resolve()
    try:
        raw = load_yaml_mapping(source_path)
        config = LatinWordlistConfig.model_validate(raw)
    except (YamlLoadError, ValidationError) as exc:
        raise LatinWordlistConfigError(f"Invalid wordlist config {source_path}: {exc}") from exc

    base = source_path.parent
    latin_text_dir = _resolve_from(base, config.inputs.latin_text_dir)
    output_path = _resolve_from(base, config.output.latin_wordlist_out)
    extra_wordlists = tuple(
        _resolve_from(base, path) for path in config.inputs.extra_wordlists
    )
    if len(set(extra_wordlists)) != len(extra_wordlists):
        raise LatinWordlistConfigError(
            f"Invalid wordlist config {source_path}: extra_wordlists resolves to "
            "duplicate paths"
        )
    if output_path.suffix.lower() == ".txt" and output_path.is_relative_to(latin_text_dir):
        raise LatinWordlistConfigError(
            f"Invalid wordlist config {source_path}: output path {output_path} is "
            f"inside text input directory {latin_text_dir} and would be collected as input"
        )

    return LatinWordlistBuildRequest(
        config_path=source_path,
        conllu_dir=_resolve_from(base, config.inputs.conllu_dir),
        latin_text_dir=latin_text_dir,
        extra_wordlists=extra_wordlists,
        output_path=output_path,
        filters=WordlistFilterPolicy(
            min_length=config.filters.min_length,
            min_form_freq=config.filters.min_form_freq,
            min_text_freq=config.filters.min_text_freq,
        ),
        tokenization=WordlistTokenizationPolicy(
            extra_punct=config.tokenize.extra_punct
        ),
    )
