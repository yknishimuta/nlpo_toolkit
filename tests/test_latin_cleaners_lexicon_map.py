from __future__ import annotations

from pathlib import Path

from nlpo_toolkit.latin.cleaners import cleaners


def test_clean_text_applies_lexicon_map_tsv_word_boundary(tmp_path: Path):
    """
    Unit test for the 3rd-layer lexicon map:
    - TSV mapping is loaded
    - mapping applies only on word boundaries (no partial replacement)
    - clean_text(kind="corpus_corporum") uses lexicon_map_path
    """
    lex = tmp_path / "lexicon_map.tsv"
    lex.write_text(
        "# from\tto\n"
        "ipsus\tipse\n",
        encoding="utf-8",
    )

    # NOTE: corpus_corporum cleaner skips metadata until a line of 5+ '#'
    text = "meta line\n#####\nipsus ipsorum\n"

    out = cleaners.clean_text(
        text,
        kind="corpus_corporum",
        rules_path=None,  # make sure None is accepted (default fallback)
        lexicon_map_path=lex,
        ref_tsv=None,
        doc_id="TEST",
    )

    # 'ipsus' should change, 'ipsust' should not (word boundary)
    assert "ipse" in out
    assert "ipsus" not in out
    assert "ipsorum" in out