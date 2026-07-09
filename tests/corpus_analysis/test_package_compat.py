from __future__ import annotations


def test_canonical_package_imports() -> None:
    import nlpo_toolkit.corpus_analysis
    import nlpo_toolkit.corpus_analysis.cli
    import nlpo_toolkit.corpus_analysis.config
    import nlpo_toolkit.corpus_analysis.runner


def test_old_package_imports_remain_compatible() -> None:
    import nlpo_toolkit.count_vocabula
    import nlpo_toolkit.count_vocabula.cli
    import nlpo_toolkit.count_vocabula.config
    import nlpo_toolkit.count_vocabula.runner


def test_old_config_symbols_alias_canonical_symbols() -> None:
    from nlpo_toolkit.corpus_analysis.config import AppConfig as NewAppConfig
    from nlpo_toolkit.count_vocabula.config import AppConfig as OldAppConfig

    assert OldAppConfig is NewAppConfig


def test_old_public_symbols_alias_canonical_symbols() -> None:
    from nlpo_toolkit.corpus_analysis.cli import main as new_main
    from nlpo_toolkit.corpus_analysis.corpus import PreparedCorpus as NewPreparedCorpus
    from nlpo_toolkit.corpus_analysis.features import (
        FeatureError as NewFeatureError,
        FeatureOptions as NewFeatureOptions,
    )
    from nlpo_toolkit.corpus_analysis.runner import run as new_run
    from nlpo_toolkit.count_vocabula.cli import main as old_main
    from nlpo_toolkit.count_vocabula.corpus import PreparedCorpus as OldPreparedCorpus
    from nlpo_toolkit.count_vocabula.features import (
        FeatureError as OldFeatureError,
        FeatureOptions as OldFeatureOptions,
    )
    from nlpo_toolkit.count_vocabula.runner import run as old_run

    assert old_main is new_main
    assert old_run is new_run
    assert OldPreparedCorpus is NewPreparedCorpus
    assert OldFeatureOptions is NewFeatureOptions
    assert OldFeatureError is NewFeatureError
