from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from nlpo_toolkit.backends import BuiltNLPBackend, NLPBackendInfo
from nlpo_toolkit.corpus_analysis.config import AppConfig, ensure_app_config
from nlpo_toolkit.corpus_analysis.analysis_policy import (
    AnalysisExtractionPolicy,
    DEFAULT_ANALYSIS_EXTRACTION_POLICY,
)
from nlpo_toolkit.corpus_analysis.dependencies import (
    AnalysisDependencies,
    BackendFactory,
    CorpusPlanningDependencies,
    CountCommandDependencies,
    RunnerDependencies,
)
from nlpo_toolkit.corpus_analysis.archive import create_run_archive
from nlpo_toolkit.models import NLPDocument, NLPSentence, NLPToken
from nlpo_toolkit.latin.cleaners.config_loader import inspect_cleaner_config


TokenSpec = tuple[str, str | None, str]


@dataclass
class FakeNLPBackend:
    tokens: Sequence[TokenSpec] | None = None
    per_text: dict[str, Sequence[TokenSpec]] = field(default_factory=dict)
    calls: list[str] = field(default_factory=list)

    def __call__(self, text: str) -> NLPDocument:
        self.calls.append(text)
        specs = self.per_text.get(text, self.tokens)
        if specs is None:
            specs = tuple(
                (match.group(0), match.group(0).lower(), "NOUN")
                for match in re.finditer(r"\S+", text)
            )
        tokens: list[NLPToken] = []
        search_from = 0
        for surface, lemma, upos in specs:
            start = text.find(surface, search_from)
            if start < 0:
                start = None
                end = None
            else:
                end = start + len(surface)
                search_from = end
            tokens.append(
                NLPToken(
                    text=surface,
                    lemma=lemma,
                    upos=upos,
                    start_char=start,
                    end_char=end,
                )
            )
        return NLPDocument(
            sentences=[NLPSentence(tokens=tokens, text=text)],
            text=text,
        )


def fake_backend_factory(
    tokens: Iterable[TokenSpec] | None = None,
    *,
    backend: FakeNLPBackend | None = None,
):
    fake = backend or FakeNLPBackend(tokens=tuple(tokens) if tokens is not None else None)

    def factory(config):
        return BuiltNLPBackend(
            backend=fake,
            info=NLPBackendInfo(
                name="fake",
                language=getattr(config, "language", "la"),
                package="fake",
                use_gpu=False,
            ),
        )

    return factory


def runner_dependencies(
    load_config,
    backend_factory: BackendFactory,
    *,
    cleaner=None,
    extraction_policy: AnalysisExtractionPolicy = DEFAULT_ANALYSIS_EXTRACTION_POLICY,
) -> RunnerDependencies:
    def canonical_loader(path) -> AppConfig:
        return ensure_app_config(load_config(path))

    def cleaner_loader():
        if cleaner is None:
            raise AssertionError("cleaner loader must not be called")
        return cleaner

    return RunnerDependencies(
        planning=CorpusPlanningDependencies(
            load_config=canonical_loader,
            cleaner_loader=cleaner_loader,
            cleaner_inspector=inspect_cleaner_config,
        ),
        analysis=AnalysisDependencies(
            backend_factory=backend_factory,
            extraction_policy=extraction_policy,
        ),
    )


def count_command_dependencies(
    load_config,
    backend_factory: BackendFactory,
    *,
    cleaner=None,
    extraction_policy: AnalysisExtractionPolicy = DEFAULT_ANALYSIS_EXTRACTION_POLICY,
) -> CountCommandDependencies:
    from nlpo_toolkit.corpus_analysis.runner import run

    return CountCommandDependencies(
        runner=runner_dependencies(
            load_config,
            backend_factory,
            cleaner=cleaner,
            extraction_policy=extraction_policy,
        ),
        run_analysis=run,
        archive_creator=create_run_archive,
    )
