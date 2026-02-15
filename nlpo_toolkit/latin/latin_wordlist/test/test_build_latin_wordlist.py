from pathlib import Path
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import build_latin_wordlist as mod


def test_build_latin_wordlist_small_corpus(tmp_path, monkeypatch):
    """
    End-to-end style test for build_latin_wordlist.build().

    This test:
      - creates a tiny artificial corpus under tmp_path
      - rewires the module-level paths (CONLLU_DIR, LATIN_TEXT_DIR, EXTRA_WORDLISTS,
        LATIN_WORDLIST_OUT) to point to that corpus
      - calls build()
      - verifies that the output wordlist file is created and contains
        expected words from each source:
          * lemmas from CoNLL-U
          * forms from CoNLL-U (with frequency filter)
          * frequent forms from raw text
          * extra wordlist entries
    """


    # Prepare a small fake CoNLL-U treebank
    conllu_dir = tmp_path / "input" / "treebank_latin"
    conllu_dir.mkdir(parents=True, exist_ok=True)

    conllu_file = conllu_dir / "sample.conllu"
    # CoNLL-U columns (we only care about FORM = col[1], LEMMA = col[2]):
    # Here:
    #   - puella appears twice (so it passes MIN_FORM_FREQ=2)
    #   - rosa appears once (fails MIN_FORM_FREQ=2, but its lemma is still collected)
    conllu_content = "\n".join([
        "# sent 1",
        "1\tpuella\tpuella\tNOUN\t_\t_\t0\troot\t_\t_",
        "2\trosam\trosa\tNOUN\t_\t_\t1\tobj\t_\t_",
        "",
        "# sent 2",
        "1\tpuella\tpuella\tNOUN\t_\t_\t0\troot\t_\t_",
        "2\tamat\tamo\tVERB\t_\t_\t1\tobj\t_\t_",
        "",
    ])
    conllu_file.write_text(conllu_content, encoding="utf-8")

    # Prepare a small Latin text corpus
    latin_text_dir = tmp_path / "input" / "latin_texts"
    latin_text_dir.mkdir(parents=True, exist_ok=True)

    text_file = latin_text_dir / "text1.txt"
    # deus appears 3 times -> passes MIN_TEXT_FREQ=3 by default
    text_file.write_text(
        "Deus deus deus amat puellam.\nRosa pulchra est.\n",
        encoding="utf-8",
    )

    # Prepare an extra wordlist
    extra_dir = tmp_path / "input"
    extra_dir.mkdir(parents=True, exist_ok=True)
    extra_wordlist = extra_dir / "perseus_lemmas.txt"
    extra_wordlist.write_text(
        "# comment line\n"
        "homo\n"
        "bonus\n",
        encoding="utf-8",
    )

    out_path = tmp_path / "output" / "latin_words_test.txt"
    cfg_path = tmp_path / "latin_wordlist.yml"
    cfg_path.write_text(
        textwrap.dedent(f"""\
    inputs:
        conllu_dir: "{conllu_dir}"
        latin_text_dir: "{latin_text_dir}"
        extra_wordlists:
            - "{extra_wordlist}"

    output:
        latin_wordlist_out: "{out_path}"

    filters:
        min_length: 2
        min_form_freq: 2
        min_text_freq: 3

    tokenize:
        extra_punct: ""
    """),
        encoding="utf-8",
    )

    cfg = mod.load_config(cfg_path)
    rc = mod.build(cfg)
    assert rc == 0

    # Verify the output
    assert out_path.is_file(), f"Expected output file not found: {out_path}"

    vocab = {
        line.strip()
        for line in out_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }

    # From lemmas in CoNLL-U:
    #   puella, rosa, amo
    assert "puella" in vocab
    assert "rosa" in vocab
    assert "amo" in vocab

    # From forms in CoNLL-U with MIN_FORM_FREQ=2:
    #   puella appears twice as FORM -> should be included via collect_forms_from_conllu
    #   (it is already checked above, but we ensure at least it is present)
    assert "puella" in vocab

    # From text corpus with MIN_TEXT_FREQ=3:
    #   deus appears 3 times -> should be included
    assert "deus" in vocab

    # From extra wordlist:
    assert "homo" in vocab
    assert "bonus" in vocab

