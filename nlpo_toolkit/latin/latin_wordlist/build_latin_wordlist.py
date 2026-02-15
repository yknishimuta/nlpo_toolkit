#!/usr/bin/env python3
"""
build_latin_wordlist.py (YAML-configured)
----------------------------------------

Build a general-purpose Latin wordlist covering Classical to Medieval Latin.

Sources:
    1. Treebank files in CoNLL-U format
    2. Latin text corpora (.txt)
    3. Optional existing wordlists

Output:
    Writes a vocabulary file (one lemma/form per line)
"""

from __future__ import annotations

import argparse
import string
from pathlib import Path
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import yaml  # pip install pyyaml


# ----------------------------
# Config model
# ----------------------------

@dataclass(frozen=True)
class InputsCfg:
    conllu_dir: Path
    latin_text_dir: Path
    extra_wordlists: list[Path]

@dataclass(frozen=True)
class OutputCfg:
    latin_wordlist_out: Path

@dataclass(frozen=True)
class FiltersCfg:
    min_length: int
    min_form_freq: int
    min_text_freq: int

@dataclass(frozen=True)
class TokenizeCfg:
    extra_punct: str

@dataclass(frozen=True)
class AppCfg:
    inputs: InputsCfg
    output: OutputCfg
    filters: FiltersCfg
    tokenize: TokenizeCfg

def load_config(cfg_path: Path) -> AppCfg:
    """Load YAML config and resolve relative paths relative to config file dir."""
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    project_root = Path(__file__).resolve().parent
    base = project_root

    def rp(x: str) -> Path:
        px = Path(x)
        return (base / px).resolve() if not px.is_absolute() else px

    inputs = raw.get("inputs", {})
    output = raw.get("output", {})
    filters = raw.get("filters", {})
    tok = raw.get("tokenize", {})

    conllu_dir = rp(inputs.get("conllu_dir", "input/treebank_latin"))
    latin_text_dir = rp(inputs.get("latin_text_dir", "input/latin_texts"))
    extra_wordlists = [rp(p) for p in inputs.get("extra_wordlists", [])]

    latin_wordlist_out = rp(output.get("latin_wordlist_out", "output/latin_words.txt"))

    min_length = int(filters.get("min_length", 2))
    min_form_freq = int(filters.get("min_form_freq", 2))
    min_text_freq = int(filters.get("min_text_freq", 3))

    extra_punct = str(tok.get("extra_punct", "“”‘’«»…—–-­"))

    return AppCfg(
        inputs=InputsCfg(
            conllu_dir=conllu_dir,
            latin_text_dir=latin_text_dir,
            extra_wordlists=extra_wordlists,
        ),
        output=OutputCfg(latin_wordlist_out=latin_wordlist_out),
        filters=FiltersCfg(
            min_length=min_length,
            min_form_freq=min_form_freq,
            min_text_freq=min_text_freq,
        ),
        tokenize=TokenizeCfg(extra_punct=extra_punct),
    )


def make_punct_table(extra_punct: str) -> dict[int, str]:
    return str.maketrans({ch: " " for ch in (string.punctuation + extra_punct)})


def tokenize(text: str, punct_table: dict[int, str], min_length: int) -> Iterable[str]:
    """Simple tokenizer: lowercase alphabetic sequences, punctuation removed."""
    t = text.translate(punct_table)
    for w in t.split():
        w = w.lower()
        if len(w) >= min_length and w.isalpha():
            yield w


def collect_lemmas_from_conllu(dir: Path, min_length: int) -> set[str]:
    lemmas: set[str] = set()
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
                    if len(lemma) >= min_length and lemma.isalpha():
                        lemmas.add(lemma)
        except Exception as e:
            print(f"[WARN] cannot read {path}: {e}")
    return lemmas


def collect_forms_from_conllu(dir: Path, min_length: int, min_form_freq: int) -> set[str]:
    """Collect word forms (FORM column) from CoNLL-U files and apply min frequency."""
    freq: Counter[str] = Counter()
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
                    if len(form) >= min_length and form.isalpha():
                        freq[form] += 1
        except Exception as e:
            print(f"[WARN] cannot read {path}: {e}")

    return {w for w, c in freq.items() if c >= min_form_freq}


def collect_from_text_dir(dir: Path, punct_table: dict[int, str], min_length: int, min_text_freq: int) -> set[str]:
    """Extract frequently occurring word forms from raw Latin text files."""
    freq: Counter[str] = Counter()
    if not dir.is_dir():
        return set()

    for path in dir.rglob("*.txt"):
        try:
            txt = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"[WARN] cannot read {path}: {e}")
            continue
        for w in tokenize(txt, punct_table, min_length):
            freq[w] += 1

    return {w for w, c in freq.items() if c >= min_text_freq}


def load_wordlist(path: Path) -> set[str]:
    """Load additional wordlists (one word per line)."""
    if not path.is_file():
        print(f"[WARN] extra wordlist missing: {path}")
        return set()
    out: set[str] = set()
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        s = s.lower()
        if s.isalpha():  # keep consistent with your pipeline; relax if you want
            out.add(s)
    return out


def build(cfg: AppCfg) -> int:
    vocab: set[str] = set()

    punct_table = make_punct_table(cfg.tokenize.extra_punct)
    f = cfg.filters
    inp = cfg.inputs

    # Lemmas from treebank
    lemmas = collect_lemmas_from_conllu(inp.conllu_dir, f.min_length)
    print(f"[INFO] lemmas from conllu : {len(lemmas):,}")
    vocab |= lemmas

    # Word forms from treebank with frequency filtering
    forms = collect_forms_from_conllu(inp.conllu_dir, f.min_length, f.min_form_freq)
    print(f"[INFO] forms from conllu  : {len(forms):,}")
    vocab |= forms

    # Frequent forms from Latin raw text
    txt_forms = collect_from_text_dir(inp.latin_text_dir, punct_table, f.min_length, f.min_text_freq)
    print(f"[INFO] forms from txt     : {len(txt_forms):,}")
    vocab |= txt_forms

    # Additional external lists
    for p in inp.extra_wordlists:
        ws = load_wordlist(p)
        print(f"[INFO] extra wordlist {p}: {len(ws):,}")
        vocab |= ws

    # Write final merged vocabulary
    out_path = cfg.output.latin_wordlist_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(sorted(vocab)) + "\n", encoding="utf-8")
    print(f"[OK] wrote {len(vocab):,} words -> {out_path}")
    return 0


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=Path,
        default=Path("config/latin_wordlist.yml"),
        help="Path to YAML config file",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    return build(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
