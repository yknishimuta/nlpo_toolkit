from __future__ import annotations

import os
import tempfile
from pathlib import Path

from .errors import LatinWordlistPublicationError
from .models import WordlistPublication


def publish_wordlist(publication: WordlistPublication) -> None:
    output_path = publication.output_path
    temporary_path: Path | None = None
    try:
        if output_path.is_dir():
            raise IsADirectoryError(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            if publication.entries:
                stream.write("\n".join(publication.entries) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, output_path)
        temporary_path = None
    except (OSError, UnicodeError) as exc:
        raise LatinWordlistPublicationError(
            f"Failed to publish Latin wordlist to {output_path}: {exc}"
        ) from exc
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
