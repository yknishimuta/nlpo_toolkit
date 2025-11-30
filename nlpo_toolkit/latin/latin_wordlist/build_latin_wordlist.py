#!/usr/bin/env python3
"""
build_latin_wordlist.py
-----------------------

Build a general-purpose Latin wordlist covering Classical to Medieval Latin.
No command-line arguments; all settings are controlled by the variables at the
top of this file.

Sources:
    1. Treebank files in CoNLL-U format
    2. Latin text corpora (.txt)
    3. Optional existing wordlists

Output:
    Writes a vocabulary file (one lemma/form per line)
    to LATIN_WORDLIST_OUT.
"""

from __future__ import annotations
from pathlib import Path
from collections import Counter
import string

# ==========================================
# ====== CONFIGURATION ======
# ==========================================

# --- inputs ---
CONLLU_DIR = Path("input/treebank_latin")
LATIN_TEXT_DIR = Path("input/latin_texts")
EXTRA_WORDLISTS = [
    Path("input/perseus_lemmas.txt"),
]

# --- output ---
LATIN_WORDLIST_OUT = Path("output/latin_words.txt")

# --- parameter for filtering ---
MIN_LENGTH = 2 
MIN_FORM_FREQ = 2 
MIN_TEXT_FREQ = 3 

# Punctuation removal table
PUNCT_TABLE = str.maketrans(
    {ch: " " for ch in (string.punctuation + "“”‘’«»…—–-­")}
)


def tokenize(text: str, min_length: int = MIN_LENGTH):
    """ Simple tokenizer: lowercase alphabetic sequences, punctuation removed. """
    t = text.translate(PUNCT_TABLE)
    for w in t.split():
        w = w.lower()
        if len(w) >= min_length and w.isalpha():
            yield w


def collect_lemmas_from_conllu(dir: Path):
    lemmas = set()
    if not dir.is_dir():
        print(f"[WARN] conllu dir not found: {dir}")
        return lemmas

    for path in dir.rglob("*.conllu"):
        try:
            with path.open(encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    cols = line.split("\t")
                    if len(cols) < 3:
                        continue
                    lemma = cols[2].strip().lower()
                    if len(lemma) >= MIN_LENGTH and lemma.isalpha():
                        lemmas.add(lemma)
        except Exception as e:
            print(f"[WARN] cannot read {path}: {e}")
    return lemmas


def collect_forms_from_conllu(dir: Path):
    """ Collect word forms (FORM column) from CoNLL-U files
    and apply minimum frequency filtering. """
    freq = Counter()
    if not dir.is_dir():
        return set()

    for path in dir.rglob("*.conllu"):
        try:
            with path.open(encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    cols = line.split("\t")
                    if len(cols) < 2:
                        continue
                    form = cols[1].strip().lower()
                    if len(form) >= MIN_LENGTH and form.isalpha():
                        freq[form] += 1
        except Exception as e:
            print(f"[WARN] cannot read {path}: {e}")

    return {w for w, c in freq.items() if c >= MIN_FORM_FREQ}


def collect_from_text_dir(dir: Path):
    """ Extract frequently occurring word forms from raw Latin text files. """
    freq = Counter()
    if not dir.is_dir():
        return set()

    for path in dir.rglob("*.txt"):
        try:
            txt = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"[WARN] cannot read {path}: {e}")
            continue
        for w in tokenize(txt):
            freq[w] += 1

    return {w for w, c in freq.items() if c >= MIN_TEXT_FREQ}


def load_wordlist(path: Path):
    """ Load additional wordlists (one word per line). """
    if not path.is_file():
        print(f"[WARN] extra wordlist missing: {path}")
        return set()
    return {
        line.strip().lower()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }


def build():
    vocab = set()

    # Lemmas from treebank
    lemmas = collect_lemmas_from_conllu(CONLLU_DIR)
    print(f"[INFO] lemmas from conllu : {len(lemmas):,}")
    vocab |= lemmas

    # Word forms from treebank with frequency filtering
    forms = collect_forms_from_conllu(CONLLU_DIR)
    print(f"[INFO] forms from conllu  : {len(forms):,}")
    vocab |= forms

    # Frequent forms from Latin raw text
    txt_forms = collect_from_text_dir(LATIN_TEXT_DIR)
    print(f"[INFO] forms from txt     : {len(txt_forms):,}")
    vocab |= txt_forms

    # Additional external lists
    for p in EXTRA_WORDLISTS:
        ws = load_wordlist(p)
        print(f"[INFO] extra wordlist {p}: {len(ws):,}")
        vocab |= ws

    # Write final merged vocabulary
    LATIN_WORDLIST_OUT.parent.mkdir(parents=True, exist_ok=True)
    LATIN_WORDLIST_OUT.write_text(
        "\n".join(sorted(vocab)) + "\n",
        encoding="utf-8",
    )
    print(f"[OK] wrote {len(vocab):,} words -> {LATIN_WORDLIST_OUT}")


if __name__ == "__main__":
    build()

