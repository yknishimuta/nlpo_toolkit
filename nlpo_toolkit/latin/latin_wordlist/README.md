# Latin Wordlist Builder

This tool builds a **Latin wordlist** suitable for Classical and Medieval Latin NLP tasks.
 It merges vocabulary from multiple sources — CoNLL-U treebanks, plain-text corpora, and optional external wordlists — and outputs a clean, deduplicated list of Latin word forms and lemmas.

The script requires **no command-line arguments**.
 All configuration is handled by variables at the top of the script.

## Usage

## Usage

### 1. Place your input files

Before running the script, **make sure your input corpora exist under the `input/` directory**:

```
input/
    treebank_latin/
        your_file1.conllu
        your_file2.conllu
        ...
    latin_texts/
        corpus1.txt
        corpus2.txt
        ...
    perseus_lemmas.txt   # optional
```

- **CoNLL-U files (`\*.conllu`)** go in:
   `input/treebank_latin/`
- **Plain Latin text files (`\*.txt`)** go in:
   `input/latin_texts/`
- **Additional custom wordlists** go in:
   `input/` and are added via the `EXTRA_WORDLISTS` list at the top of the script.

If a directory is missing, the script will warn you but continue processing.

------

### 2. Run the script

```
python build_latin_wordlist.py
```

## Configuration

At the top of the script, the following variables can be edited:

### Input sources

```
CONLLU_DIR = Path("input/treebank_latin")
LATIN_TEXT_DIR = Path("input/latin_texts")
EXTRA_WORDLISTS = [
    Path("input/perseus_lemmas.txt"),
]
```

### Output file

```
LATIN_WORDLIST_OUT = Path("output/latin_words.txt")
```

### Filtering parameters

```
MIN_LENGTH = 2        # minimum token length
MIN_FORM_FREQ = 2     # minimum frequency for forms in treebanks
MIN_TEXT_FREQ = 3     # minimum frequency in raw text corpora
```