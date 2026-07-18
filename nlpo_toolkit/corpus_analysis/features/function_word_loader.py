from __future__ import annotations

from pathlib import Path

from .errors import FeatureError
from .filtering import normalize_feature_value
from .models import FunctionWordVocabulary


def load_function_word_vocabulary(path: Path) -> FunctionWordVocabulary:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FeatureError(f"function-word file not found: {path}") from exc
    except UnicodeDecodeError as exc:
        raise FeatureError(f"function-word file is not valid UTF-8: {path}") from exc
    except OSError as exc:
        raise FeatureError(f"could not read function-word file {path}: {exc}") from exc

    terms: list[str] = []
    first_lines: dict[str, int] = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "\t" in raw_line or any(character.isspace() for character in stripped):
            raise FeatureError(
                "function-word term must be a single token "
                f"at line {line_number}: {stripped!r}"
            )
        term = normalize_feature_value(stripped)
        previous_line = first_lines.get(term)
        if previous_line is not None:
            raise FeatureError(
                f"duplicate function word {term!r} at lines "
                f"{previous_line} and {line_number}"
            )
        first_lines[term] = line_number
        terms.append(term)
    return FunctionWordVocabulary(tuple(terms))
