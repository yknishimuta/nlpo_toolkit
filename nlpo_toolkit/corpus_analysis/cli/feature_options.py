from __future__ import annotations

import argparse
from pathlib import Path

from ..features.errors import FeatureError
from ..features.models import (
    CharacterNgramMode,
    CharacterNgramOptions,
    FeatureRequest,
    FeatureSamplingOptions,
    FunctionWordSource,
    LexicalDiversityOptions,
    MorphologyOptions,
    UposNgramOptions,
)
from ..requests import CorpusPreparationRequest


def add_feature_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--field", choices=("lemma", "token"), default="lemma")
    parser.add_argument("--mfw", type=int, default=0)
    parser.add_argument("--window-tokens", type=int, default=None)
    parser.add_argument("--step-tokens", type=int, default=None)
    parser.add_argument("--include-partial-window", action="store_true")
    parser.add_argument(
        "--include-upos", dest="include_upos", action="store_true", default=True
    )
    parser.add_argument("--no-upos", dest="include_upos", action="store_false")
    parser.add_argument(
        "--include-basic", dest="include_basic", action="store_true", default=True
    )
    parser.add_argument("--no-basic", dest="include_basic", action="store_false")
    parser.add_argument("--lexical-diversity", action="store_true")
    parser.add_argument("--lexdiv-window", type=int, default=None)
    parser.add_argument("--mtld-threshold", type=float, default=None)
    parser.add_argument("--hdd-sample-size", type=int, default=None)
    parser.add_argument("--function-words", type=Path, default=None)
    parser.add_argument(
        "--function-word-field", choices=("lemma", "token"), default=None
    )
    parser.add_argument("--char-ngram-size", type=int, action="append", default=[])
    parser.add_argument("--char-ngram-top", type=int, default=None)
    parser.add_argument(
        "--char-ngram-mode",
        choices=tuple(mode.value for mode in CharacterNgramMode),
        action="append",
        default=[],
    )
    parser.add_argument("--upos-ngram-size", type=int, action="append", default=[])
    parser.add_argument("--upos-ngram-top", type=int, default=None)
    parser.add_argument("--morphology", action="store_true")
    parser.add_argument("--morph-attribute", action="append", default=[])
    parser.add_argument("--morph-bundle-top", type=int, default=None)


def build_feature_request(
    args: argparse.Namespace, *, corpus: CorpusPreparationRequest
) -> FeatureRequest:
    if args.function_word_field is not None and args.function_words is None:
        raise FeatureError("--function-word-field requires --function-words")
    if args.char_ngram_top is not None and not args.char_ngram_size:
        raise FeatureError("--char-ngram-top requires --char-ngram-size")
    if args.char_ngram_mode and not args.char_ngram_size:
        raise FeatureError("--char-ngram-mode requires --char-ngram-size")
    if len(args.char_ngram_mode) != len(set(args.char_ngram_mode)):
        duplicate = next(
            mode for mode in args.char_ngram_mode if args.char_ngram_mode.count(mode) > 1
        )
        raise FeatureError(f"duplicate --char-ngram-mode: {duplicate}")
    if args.upos_ngram_top is not None and not args.upos_ngram_size:
        raise FeatureError("--upos-ngram-top requires --upos-ngram-size")
    lexical_diversity = None
    if args.lexical_diversity or any(
        value is not None
        for value in (args.lexdiv_window, args.mtld_threshold, args.hdd_sample_size)
    ):
        lexical_diversity = LexicalDiversityOptions(
            window_size=args.lexdiv_window if args.lexdiv_window is not None else 100,
            mtld_threshold=args.mtld_threshold
            if args.mtld_threshold is not None
            else 0.72,
            hdd_sample_size=args.hdd_sample_size
            if args.hdd_sample_size is not None
            else 42,
        )
    return FeatureRequest(
        corpus=corpus,
        field=args.field,
        mfw=args.mfw,
        include_upos=bool(args.include_upos),
        include_basic=bool(args.include_basic),
        sampling=FeatureSamplingOptions(
            args.window_tokens, args.step_tokens, args.include_partial_window
        ),
        lexical_diversity=lexical_diversity,
        function_words=(
            FunctionWordSource(args.function_words, args.function_word_field or "lemma")
            if args.function_words is not None
            else None
        ),
        character_ngrams=(
            CharacterNgramOptions(
                tuple(args.char_ngram_size),
                args.char_ngram_top or 500,
                tuple(CharacterNgramMode(mode) for mode in args.char_ngram_mode)
                or (CharacterNgramMode.FULL,),
            )
            if args.char_ngram_size
            else None
        ),
        upos_ngrams=(
            UposNgramOptions(tuple(args.upos_ngram_size), args.upos_ngram_top or 100)
            if args.upos_ngram_size
            else None
        ),
        morphology=(
            MorphologyOptions(
                enabled=True,
                attributes=tuple(args.morph_attribute),
                bundle_top=args.morph_bundle_top,
            )
            if args.morphology
            or args.morph_attribute
            or args.morph_bundle_top is not None
            else None
        ),
    )
