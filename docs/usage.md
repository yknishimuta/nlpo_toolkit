# Usage

This guide describes how to run `nlpo_toolkit` from the repository root.
Paths in the examples are relative to `--project-root .` unless noted. For the
meaning and restrictions of the configuration keys, see
[Configuration](configuration.md); for calculation details, see
[Specification](specification.md).

## Install and prepare inputs

```bash
python -m pip install .
python -c "import stanza; stanza.download('la', package='perseus')"
nlpo --help
```

For development, use `python -m pip install -e ".[dev]"` instead. Docker is
also supported; see below.

Put UTF-8 texts in `input/` and create `config/groups.config.yml`:

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

## Count: lemmas or surface forms

Check the configuration and inspect currently matching files *without running*
Cleaner preprocessing or the NLP backend:

```bash
nlpo count --dry-run --project-root . --config config/groups.config.yml
```

Count and write frequency tables:

```bash
nlpo count --project-root . --config config/groups.config.yml
```

Typical outputs include `output/frequency_text.csv`, `output/summary.txt`, and
`output/run_meta.json`. When dictionary checks are configured, the additional
files are `frequency_<label>.known.csv` and
`frequency_<label>.unknown.csv`.

To count each input file separately, add `--group-by-file`:

```bash
nlpo count --project-root . --config config/groups.config.yml --group-by-file
```

Equivalent YAML mode:

```yaml
grouping:
  mode: per_file
```

Output labels derive from file stems, with non-alphanumeric characters replaced
by underscores and numerical suffixes for collisions. `--group-by-file` and
`--auto-single-cleaned` cannot be combined.

When a Cleaner step is configured, use `--auto-single-cleaned` if you want to
select **exactly one** resulting `.txt` file without editing the group globs:

```bash
nlpo count --project-root . --config config/groups.config.yml \
  --auto-single-cleaned --run-name satyricon-cena
```

This fails if zero or multiple cleaned `.txt` files are found; `--run-name`
controls the archive name, **not** the selected input file. Use
`--error-on-empty-group` to fail when a group matches no files.

## Feature extraction

Create one feature row per group:

```bash
nlpo features --project-root . --config config/groups.config.yml \
  --out output/features.csv
```

Or create one row per input file:

```bash
nlpo features --project-root . --config config/groups.config.yml \
  --group-by-file --out output/features_by_file.csv
```

Most-frequent-lemma (MFW) feature columns:

```bash
nlpo features --project-root . --config config/groups.config.yml \
  --group-by-file --mfw 100 --field lemma --out output/features_mfw100.csv
```

For authorship or clustering experiments, generate fixed-size token windows:

```bash
nlpo features --project-root . --config config/groups.config.yml \
  --group-by-file --window-tokens 1000 --mfw 500 --field lemma \
  --out output/features_windows.csv
```

`--step-tokens 500` adds overlap to 1,000-token windows, and
`--include-partial-window` keeps a shorter trailing window. A `sample_id`
identifies each window. Windows do not cross source-file boundaries.

Other supported feature families include:

- `--lexical-diversity` (MATTR, MSTTR, MTLD, HD-D for tokens and lemmas).
- `--function-words config/latin_function_words.txt` (one reviewed term per
  line; `--function-word-field lemma` or `token`).
- `--char-ngram-size 3 --char-ngram-mode letters-spaces --char-ngram-top 500`.
- `--content-masked-char-ngram-size 3 --content-masked-char-ngram-top 300`.
- `--upos-ngram-size 2 --upos-ngram-top 100`.
- `--morphology --morph-attribute Case --morph-attribute Number`.

The n-gram options can be repeated where documented. Character n-gram features
require one source file per prepared corpus; use per-file grouping. See
[Specification](specification.md) for filtering, sampling, masking, and feature
selection behavior.

## Stylometry and authorship analysis

Calculate Burrows's Delta from a published Features table. The selected feature
family is **always explicit**; no numeric columns are selected implicitly:

```bash
nlpo stylometry delta --features output/features_windows.csv \
  --id-column sample_id --feature-prefix mfw_ \
  --out output/burrows_delta.csv
```

For an exploratory nearest-neighbor ranking:

```bash
nlpo stylometry neighbors --features output/features_windows.csv \
  --id-column sample_id --feature-prefix mfw_ \
  --metric burrows_delta --top 10 --out output/neighbors.csv
```

For table-input leave-one-work-out (LOWO) evaluation, prepare one metadata row
per Features observation, using `sample_id,author,work` columns:

```bash
nlpo stylometry evaluate-lowo \
  --features output/features_windows.csv \
  --metadata config/authorship_metadata.csv \
  --id-column sample_id --feature-prefix fw_ \
  --out output/lowo_folds.csv --summary-out output/lowo_summary.json
```

For candidate-author verification against reference and background works:

```bash
nlpo stylometry verify \
  --features output/features_windows.csv \
  --metadata config/authorship_metadata.csv \
  --candidate-author virgil --query-work ciris \
  --id-column sample_id --feature-prefix fw_ \
  --out output/verification_ciris.json \
  --calibration-out output/verification_ciris_calibration.csv
```

Prefer the corpus-input variants when automatic feature vocabulary must be fitted
**without** using the held-out or query work:

- `nlpo stylometry evaluate-lowo-corpus`
- `nlpo stylometry verify-corpus`
- `nlpo stylometry evaluate-verification-corpus`

`nlpo stylometry verify-stability` additionally tests sensitivity to resampling
of works, samples, and/or selected features. These results are statistical
comparisons, **not** proofs of authorship or authenticity. See
[Specification](specification.md) for critical evaluation constraints and
metadata examples.

## Compare frequency tables

`nlpo compare` uses previously generated frequency CSVs and **does not rerun
NLP**:

```bash
nlpo compare --inputs \
  output/frequency_virgil_aeneis.csv \
  output/frequency_virgil_georgica.csv \
  --labels aeneis georgica --min-total-count 3 --top 100 \
  --out output/compare_aeneis_georgica.csv
```

For comparisons within a single configured count run, the `comparisons` YAML
setting writes `group_comparison_<name>.csv` instead; see
[Configuration](configuration.md). These two comparison paths use distinct
smoothing behavior, described in [Specification](specification.md).

## Token artifacts, concordance, and n-grams

Enable the complete token artifact in `config/groups.config.yml`:

```yaml
artifacts:
  tokens:
    enabled: true
    path: output/tokens.tsv
```

Run `nlpo count` again. It writes `output/tokens.tsv` and the associated
`output/tokens.meta.json` metadata. These are the formal reusable token files;
a diagnostic trace TSV is **not** a substitute.

KWIC concordance (stdout by default):

```bash
nlpo concordance --tokens output/tokens.tsv --keys arma vir \
  --field lemma --window 5
```

Write concordance CSV:

```bash
nlpo concordance --tokens output/tokens.tsv --keys vir \
  --field lemma --format csv --out output/concordance_vir.csv
```

N-grams from token artifacts:

```bash
nlpo ngram --tokens output/tokens.tsv --n 2 --field lemma \
  --min-count 2 --top 100 --out output/ngram_2.tsv
```

Alternatively, count **surface-token** n-grams straight from configured text
files, without starting the NLP backend:

```bash
nlpo ngram --project-root . --config config/groups.config.yml \
  --field token --n 2 --out output/ngram_text.tsv
```

Config-input n-grams do not support lemma input; choose token-artifact input
for that. Artifact n-grams respect sequence boundaries. See the specification
for boundary and filtering behavior.

## Run archives and cache

Save a snapshot of a count run:

```bash
nlpo count --project-root . --config config/groups.config.yml \
  --run-name virgil_noun_test_01
```

Or automatically choose a timestamped directory:

```bash
nlpo count --project-root . --config config/groups.config.yml --archive-run
```

Archives are stored under `runs/<run-name>/` by default. By default, they do
not copy large source corpora; `--include-input` and `--include-cleaned` opt in
and `--runs-dir` selects another archive root. Existing run directories are
never overwritten.

Clear the configured cache safely through the CLI:

```bash
nlpo cache clear --project-root . --config config/groups.config.yml
```

Cache clearing is constrained to the project root; the command refuses paths
outside that root or the project root itself.

## Docker and development

```bash
docker compose build
docker compose run --rm nlpo count --project-root /workspace \
  --config config/groups.config.yml
```

`/workspace` is the mounted project root. Stanza data reside in the named
Docker volume at `/opt/stanza_resources`.

The original README documents these developer commands:

```bash
python scripts/generate_config_schema.py
PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider --ignore=tests/test_latin_vocab_pipeline.py
STANZA_RESOURCES_DIR=.stanza_resources PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider
```

The schema generator writes `config/groups.config.schema.json` from the
Pydantic model; do not edit that generated file by hand. Version numbers and
successful execution of these commands were not independently verified for
this documentation split.
