from pathlib import Path

from nlpo_toolkit.latin.cleaners.lexicon import load_lexicon_map
from nlpo_toolkit.latin.cleaners.pipeline import clean_document
from nlpo_toolkit.latin.cleaners.registry import get_cleaner_profile
from nlpo_toolkit.latin.cleaners.rule_loader import load_rule_set


def test_pipeline_applies_lexicon_map_after_corpus_cleaning(tmp_path: Path) -> None:
    lexicon = tmp_path / "lexicon.tsv"
    lexicon.write_text("ipsus\tipse\n", encoding="utf-8")
    result = clean_document(
        "meta\n#####\nipsus ipsorum\n",
        profile=get_cleaner_profile("corpus_corporum"),
        rules=load_rule_set(get_cleaner_profile("corpus_corporum").default_rules_path),
        lexicon_map=load_lexicon_map(lexicon),
    )
    assert result.text == "ipse ipsorum\n"
