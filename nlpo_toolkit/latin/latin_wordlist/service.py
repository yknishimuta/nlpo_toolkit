from __future__ import annotations

from .engine import merge_wordlist_candidates, select_frequent_forms
from .models import (
    LatinWordlistBuildRequest,
    LatinWordlistBuildResult,
    WordlistBuildStatistics,
    WordlistPublication,
)
from .ports import LatinWordlistDependencies


def execute_latin_wordlist_build(
    request: LatinWordlistBuildRequest,
    *,
    dependencies: LatinWordlistDependencies,
) -> LatinWordlistBuildResult:
    conllu, conllu_notices = dependencies.collect_conllu(
        directory=request.conllu_dir, min_length=request.filters.min_length
    )
    text, text_notices = dependencies.collect_text(
        directory=request.latin_text_dir,
        policy=request.tokenization,
        min_length=request.filters.min_length,
    )
    extras = []
    notices = [*conllu_notices, *text_notices]
    for path in request.extra_wordlists:
        extra, extra_notices = dependencies.collect_extra_wordlist(path=path)
        notices.extend(extra_notices)
        if extra is not None:
            extras.append(extra)

    entries = merge_wordlist_candidates(
        conllu=conllu,
        text=text,
        extras=tuple(extras),
        filters=request.filters,
    )
    dependencies.publish(WordlistPublication(request.output_path, entries))
    conllu_forms = select_frequent_forms(
        conllu.form_counts, minimum_frequency=request.filters.min_form_freq
    )
    text_forms = select_frequent_forms(
        text.form_counts, minimum_frequency=request.filters.min_text_freq
    )
    statistics = WordlistBuildStatistics(
        conllu_file_count=len(conllu.files),
        conllu_lemma_count=len(conllu.lemmas),
        conllu_form_count=len(conllu_forms),
        text_file_count=len(text.files),
        text_form_count=len(text_forms),
        extra_wordlist_counts={extra.path: len(extra.entries) for extra in extras},
        merged_word_count=len(entries),
    )
    return LatinWordlistBuildResult(
        output_path=request.output_path,
        word_count=len(entries),
        statistics=statistics,
        notices=tuple(notices),
    )
