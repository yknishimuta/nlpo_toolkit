# nlpo_toolkit

`nlpo_toolkit` is a Python toolkit for cleaning and analyzing Latin corpora.
The `nlpo` command supports lemmatized vocabulary counts, corpus comparisons,
concordances, n-grams, feature extraction, and stylometric analysis.
It can use Stanza's Latin NLP model and stores results in reusable CSV/TSV/JSON
files.

## Main commands

| Command | Purpose |
| --- | --- |
| `nlpo count` | Count surface forms or lemmas by corpus/group; write frequency tables |
| `nlpo features` | Extract lexical, grammatical, and stylometric feature tables |
| `nlpo compare` | Compare already-generated frequency CSV files |
| `nlpo concordance` | Build KWIC concordances from a token artifact |
| `nlpo ngram` | Count token or lemma n-grams |
| `nlpo stylometry ...` | Compute distances, evaluate attribution, and verify candidate authorship |
| `nlpo cache clear` | Clear the configured analysis cache |

See [Usage](docs/usage.md) for examples of the commands and
[Specification](docs/specification.md) for their interpretation and limitations.

## Environment

- Python and package dependencies as declared in `pyproject.toml`.
- For NLP-backed runs: [Stanza](https://stanfordnlp.github.io/stanza/) and the
  Latin `perseus` model (downloaded separately for local installation).
- Optional: Docker with Docker Compose, which avoids repeated local Stanza setup.
- Inputs and configuration files are UTF-8.

**Tested versions:** the original project README does not identify verified
Python, operating-system, Stanza, or Docker versions. Check `pyproject.toml`
and your project CI/tests for actual support; do not treat this list as a tested
version matrix.

## Installation

From the repository root:

```bash
python -m pip install .
python -c "import stanza; stanza.download('la', package='perseus')"
```

For an editable development installation:

```bash
python -m pip install -e ".[dev]"
```

Check the installed commands with `nlpo --help` and
`nlpo <command> --help` (for example, `nlpo count --help`).

## Quick start: vocabulary count

1. Put one or more UTF-8 Latin texts in `input/`, for example
   `input/text1.txt` and `input/text2.txt`.
2. Create `config/groups.config.yml` with the following contents:

   ```yaml
   groups:
     text:
       files:
         - input/*.txt

   out_dir: output
   nlp:
     backend: stanza
     language: la
     stanza_package: perseus
     cpu_only: true
   analysis_unit: lemma
   ```

3. Check the configuration and file matches before running NLP:

   ```bash
   nlpo count --dry-run --project-root . --config config/groups.config.yml
   ```

4. Run the count:

   ```bash
   nlpo count --project-root . --config config/groups.config.yml
   ```

Typical output files:

```text
output/
  frequency_text.csv
  summary.txt
  run_meta.json
```

Use `--group-by-file` if you want one frequency table for **each** input file
rather than one table for the combined `text` group:

```bash
nlpo count --project-root . --config config/groups.config.yml --group-by-file
```

Configuration paths are resolved relative to `--project-root`; with no
`--config`, the default is `<project-root>/config/groups.config.yml`.
The input-file glob above is relative to the project root.

## Common workflows

Extract a feature matrix (one row per file):

```bash
nlpo features --project-root . --config config/groups.config.yml \
  --group-by-file --out output/features.csv
```

Compare two existing frequency tables (no new NLP run):

```bash
nlpo compare --inputs output/frequency_a.csv output/frequency_b.csv \
  --labels a b --out output/comparison.csv
```

Build a surface-token 2-gram table directly from the configured text files
(without starting the NLP backend):

```bash
nlpo ngram --project-root . --config config/groups.config.yml \
  --field token --n 2 --out output/ngram_text.tsv
```

Concordance and lemma n-grams require a complete **token artifact** generated
by `nlpo count`. Enable it in the YAML config, rerun the count, then use it:

```yaml
artifacts:
  tokens:
    enabled: true
    path: output/tokens.tsv
```

```bash
nlpo concordance --tokens output/tokens.tsv --keys vir --field lemma --window 5
nlpo ngram --tokens output/tokens.tsv --n 2 --field lemma --out output/ngram_2.tsv
```

For stylometry, see [Usage](docs/usage.md) for feature selection, Burrows's
Delta, LOWO evaluation, and candidate-author verification examples.

## Docker

Build once and run the same configured count in a container:

```bash
docker compose build
docker compose run --rm nlpo count --project-root /workspace \
  --config config/groups.config.yml
```

The repository is mounted at `/workspace`; Stanza resources are cached in a
named volume at `/opt/stanza_resources`. Avoid `docker compose down -v` unless
you intentionally want to delete that volume and its models.

## Documentation

- [Usage](docs/usage.md): installation, command workflows, common options,
  outputs, archives, and Docker.
- [Configuration](docs/configuration.md): YAML keys, examples, validation, and
  configuration-file roles.
- [Specification](docs/specification.md): calculation semantics, file contracts,
  evaluation protocols, caveats, and reproducibility.
- [Architecture](docs/architecture.md): component roles, data flow, ports,
  sessions, and publication boundaries.
- [Original README archive](docs/legacy-readme.md): preserved full text of the
  former README for reference; not the maintained entry point.

## License

Licensed under the MIT License. 
