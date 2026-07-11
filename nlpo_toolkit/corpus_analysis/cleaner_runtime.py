"""Lazy loading and execution boundary for cleaner preprocessing."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol


class CleanerError(RuntimeError):
    """Base error for cleaner loading or execution."""


class CleanerUnavailableError(CleanerError):
    """The configured cleaner could not be loaded or invoked."""


class CleanerExecutionError(CleanerError):
    """The cleaner raised an exception or returned a failure status."""


class CleanerRunner(Protocol):
    def main(self, argv: list[str] | None = None) -> int | None: ...


CleanerLoader = Callable[[], CleanerRunner]


def load_default_cleaner() -> CleanerRunner:
    """Import the bundled cleaner only when preprocessing requests it."""
    try:
        from nlpo_toolkit.latin.cleaners import run_clean_corpus
    except ImportError as exc:
        raise CleanerUnavailableError(
            "Cleaner preprocessing was requested, but "
            "nlpo_toolkit.latin.cleaners.run_clean_corpus could not be imported."
        ) from exc
    return run_clean_corpus


def run_cleaner(
    *,
    config_path: Path,
    cleaner: CleanerRunner | None = None,
    cleaner_loader: CleanerLoader = load_default_cleaner,
) -> None:
    """Run a cleaner and reject missing entry points and failure statuses."""
    active_cleaner = cleaner if cleaner is not None else cleaner_loader()
    main = getattr(active_cleaner, "main", None)
    if not callable(main):
        raise CleanerUnavailableError(
            "Cleaner runner must provide a callable 'main(argv)' method."
        )

    try:
        status = main([str(config_path)])
    except CleanerError:
        raise
    except Exception as exc:
        raise CleanerExecutionError(
            f"Cleaner preprocessing failed for: {config_path}: {exc}"
        ) from exc

    if isinstance(status, bool) or (status is not None and status != 0):
        raise CleanerExecutionError(
            "Cleaner preprocessing failed with exit code "
            f"{status}: {config_path}"
        )
