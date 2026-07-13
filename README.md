# nlpo_toolkit

`nlpo_toolkit` is the canonical Python package for Latin corpus cleaning,
NLP-backed vocabulary counting, and reproducible frequency-table workflows.

Corpus analysis functionality lives in `nlpo_toolkit.corpus_analysis` and is
exposed through the `nlpo` command.

CLI subcommands are implemented under `nlpo_toolkit.corpus_analysis.cli`.
Each command module defines its parser registration and execution handler.

## What Is Included

- Latin corpus cleaning utilities under `nlpo_toolkit.latin.cleaners`
- Stanza-backed NLP helpers under `nlpo_toolkit.nlp`
- Vocabulary counting CLI:
  - `nlpo count`
- Config-driven grouping, preprocessing, normalization, dictionary checks,
  reference-tag handling, trace output, and run metadata
- Docker setup with cached Stanza resources

## Install

Normal installation:

```bash
python -m pip install .
```

Editable development installation:

```bash
python -m pip install -e ".[dev]"
nlpo --help
```

Runtime and development dependencies are declared in `pyproject.toml`.

Stanza model download is required for real NLP runs unless you use Docker:

```bash
python -c "import stanza; stanza.download('la', package='perseus')"
```

## Count CLI

Use `nlpo count` when you want the full vocabulary-counting workflow:
optional cleaning, input grouping, NLP processing, frequency CSV output,
dictionary checks, tracing, and run metadata.

Run with an explicit project root and config:

```bash
nlpo count --project-root . --config config/groups.config.yml
```

To write one frequency table per input file, add `--group-by-file`:

```bash
nlpo count \
  --project-root . \
  --config config/groups.config.yml \
  --group-by-file
```

`--group-by-file` and `--auto-single-cleaned` are mutually exclusive.

Internally, Count, Features, config-input N-gram, and dry-run share the frozen
`CorpusPreparationRequest` DTO. Command requests compose this DTO, and its
single `grouping_override` value represents CLI grouping overrides. Config-input
N-grams always use tokens; dry-run is an independent use case rather than a
Count request mode.

To validate the config without running preprocessing or NLP, use `--dry-run`:

```bash
nlpo count \
  --dry-run \
  --project-root . \
  --config config/groups.config.yml
```

Example output:

```text
[OK] config loaded
[OK] preprocess cleaner config found: config/cleaner.yml
[OK] input files: 3
[OK] cleaned output dir: cleaned
[OK] group text matched files: 3
[WARN] duplicate YAML key: trace
[OK] dictcheck wordlist found
[OK] ref_tags patterns found
[OK] output dir: output
```

If `--config` is omitted, the CLI uses:

```text
<project-root>/config/groups.config.yml
```

Relative paths in the YAML config are resolved from `--project-root`, not from
the location of Python source files.

The groups configuration is validated by Pydantic v2. Unknown keys are errors
at every level, and YAML scalar values are not implicitly coerced (for example,
`enabled: "true"` is not accepted as a boolean). File paths remain relative to
the project root, and referenced-file existence is checked during planning.

### Frequency Output Naming

Frequency output naming changed from `noun_frequency_<label>.csv` to
`frequency_<label>.csv`, with dictcheck outputs named
`frequency_<label>.known.csv` and `frequency_<label>.unknown.csv`. The generic
name reflects that analyses may target surface forms or UPOS categories other
than nouns. Old output files are not deleted automatically. `nlpo compare` can
still read old CSV files when their columns are valid.

## Cache Management

Use `analysis_cache` to cache NLP token analysis records. The cache stores
ordered, filter-before NLP records, not final frequency Counters. This lets runs
reuse NLP output when changing `analysis_unit`, UPOS targets, minimum length,
Roman numeral filtering, diagnostic trace, token artifact output, dictcheck, or
archive settings. The cache is invalidated when the prepared text, backend,
language, model/package, chunking strategy, schema version, or analysis behavior
version changes.

```yaml
analysis_cache:
  enabled: true
  dir: .analysis_cache
  use_manifest: true
  manifest_key_mode: relative
  lock_timeout_sec: 300.0
```

The cache is an internal optimization and is not copied into run archives or
listed in `generated_outputs`. Use token artifacts for stable research outputs.

Use `nlpo cache clear` to remove the configured analysis cache for a project
instead of running `rm -rf .analysis_cache` manually.

```bash
nlpo cache clear
```

By default, the current directory is treated as the project root. If
`config/groups.config.yml` exists, the command reads `analysis_cache.dir` from
that config and clears that directory. If no cache directory is configured, it
clears `.analysis_cache` under the project root.

Use an explicit project root:

```bash
nlpo cache clear --project-root /path/to/project
```

Use an explicit config file:

```bash
nlpo cache clear --project-root . --config config/groups.config.yml
```

The resolved cache directory must stay inside the project root. The command
refuses to delete paths such as `../cache`, absolute paths outside the project,
or the project root itself.

### Minimal Workflow

1. Put UTF-8 text files under `input/`.

```text
input/
  text1.txt
  text2.txt
```

2. Create `config/groups.config.yml`.

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

3. Run the counter.

```bash
nlpo count --project-root . --config config/groups.config.yml
```

4. Read the generated files.

```text
output/
  frequency_text.csv
  summary.txt
  run_meta.json
```

### One Output Per Input File

Use this when you do not want multiple input files combined into one frequency
table.

CLI option:

```bash
nlpo count \
  --project-root . \
  --config config/groups.config.yml \
  --group-by-file
```

YAML option:

```yaml
grouping:
  mode: per_file

groups:
  all:
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

Each expanded input file is counted independently:

```text
input/
  virgil-aeneis.txt
  file2.txt
  file3.txt

output/
  frequency_virgil_aeneis.csv
  frequency_file2.csv
  frequency_file3.csv
  summary.txt
  run_meta.json
```

The output label is derived from the input file stem. Non-alphanumeric
characters are converted to underscores. If two files have the same stem, a
numeric suffix is added to keep output names unique.

### Group Partition Validation

Use `validations.partitions` to verify additive consistency between one
configured whole group and two or more configured part groups. The check compares
the final frequency `Counter` used for the base `frequency_<group>.csv`
files, after analysis unit selection, POS/token filters, Roman numeral filtering,
text normalization, reference tag removal, and `dictcheck.lemma_normalize`.

```yaml
groups:
  full:
    files:
      - input/full.txt
  part_a:
    files:
      - input/part_a.txt
  part_b:
    files:
      - input/part_b.txt

validations:
  partitions:
    - name: full_split
      whole: full
      parts:
        - part_a
        - part_b
      on_mismatch: error   # "warn" or "error"; default: "warn"
      report: mismatches   # "mismatches" or "all"; default: "mismatches"
```

Each partition writes `output/partition_validation_<name>.csv`, and all
partition summaries are written to `output/partition_validation.json`,
`summary.txt`, and `run_meta.json`. `on_mismatch: warn` reports a mismatch but
keeps exit code `0`; `on_mismatch: error` writes all validation outputs and then
returns exit code `1`.

`target_tokens` means the number of tokens accepted into the final frequency
table. It is not the raw Stanza token count.

The `whole` and `parts` groups are currently sent to Stanza independently, so a
perfect textual split can still produce different lemma or POS decisions near
section boundaries or chunk boundaries. This feature verifies additive
consistency of the final frequency Counters; it does not prove that the source
text was perfectly split as strings. For strict checks, use the same
preprocessing settings for all groups and split as much as possible at sentence
or clause boundaries.

Partition validation cannot be combined with `grouping.mode: per_file` or
`--group-by-file`, because configured group names no longer match output labels.

### Group Comparison / Keyness Analysis

Use top-level `comparisons` to compare the vocabulary distribution of any two
configured groups. This is separate from partition validation: the two groups do
not need to be whole/part relations, and the comparison uses the same final
frequency `Counter` that is written to `frequency_<group>.csv`.

```yaml
groups:
  corpus_a:
    files:
      - input/corpus_a.txt

  corpus_b:
    files:
      - input/corpus_b.txt

comparisons:
  - name: corpus_a_vs_corpus_b
    group_a: corpus_a
    group_b: corpus_b
    scale: 10000
    zero_correction: 0.5
    min_total_count: 2
```

Each comparison writes `output/group_comparison_<name>.csv`, and the overview is
written to `output/group_comparisons.json`, `summary.txt`, and `run_meta.json`.
The feature works with both `analysis_unit: lemma` and `analysis_unit: surface`.
CSV rows use generic columns such as `item`, `group_a_count`, and
`group_b_count`; actual group names are stored in the `group_a` and `group_b`
columns.

Configured group comparison and `nlpo compare` use the same comparison math
engine, but keep separate inputs and outputs. Group comparison reads in-memory
Counters from the current run and writes fixed keyness columns including
`log_likelihood`; `nlpo compare` reads existing CSV files and keeps its dynamic
CSV-comparison columns.

Rates are normalized frequencies:

```text
group_a_rate = group_a_count / sum(group_a_counter.values()) * scale
group_b_rate = group_b_count / sum(group_b_counter.values()) * scale
```

The denominator is the token total in the final Counter, after analysis-unit
selection, filters, reference tag handling, normalization, exclusions, and
dictionary normalization. It is not the raw whitespace token count or the raw
NLP token count.

`log_ratio` is a base-2 effect size. Positive values indicate that the item is
relatively more frequent in `group_a`; negative values indicate `group_b`.
Reversing `group_a` and `group_b` reverses `log_ratio` and `direction`.
Zero-frequency correction is used only for `log_ratio`, not for rates or log
likelihood.

`log_likelihood` is the G-squared statistic for the 2x2 table of target item vs.
other items. It measures the statistical strength of a frequency difference and
does not carry direction; use the `direction` column for direction. Reversing the
groups leaves `log_likelihood` unchanged. High-frequency items can have large
log likelihood even when the ratio difference is small, while rare items can
have large log ratio but small log likelihood.

Initial support does not implement p-values, q-values, or multiple-comparison
correction.

Group comparison cannot be combined with `grouping.mode: per_file` or
`--group-by-file`.

## Concordance CLI

Use `nlpo concordance` to build KWIC/concordance output from a complete token
artifact generated by `nlpo count`.

```bash
nlpo concordance \
  --tokens output/tokens.tsv \
  --keys arma vir \
  --field lemma \
  --window 5
```

Search by surface token instead:

```bash
nlpo concordance \
  --tokens output/tokens.tsv \
  --keys arma virumque \
  --field token \
  --window 3
```

The default output format is TSV on standard output. Use `--format csv` and
`--out` to write a CSV file:

```bash
nlpo concordance \
  --tokens output/tokens.tsv \
  --keys vir \
  --field lemma \
  --format csv \
  --out output/concordance_vir.csv
```

For reusable analysis data, enable the formal token artifact:

```yaml
artifacts:
  tokens:
    enabled: true
    path: output/tokens.tsv
```

Run `nlpo count --project-root . --config config/groups.config.yml`
to create the artifact. It records all NLP tokens with schema metadata in
`output/tokens.meta.json`. Concordance matches included tokens; excluded tokens
remain available in the surrounding context. File, group, and sentence metadata
are included in the concordance output when present.

## Compare CLI

Both configured group comparisons and `nlpo compare` use the shared
`nlpo_toolkit.comparison` package and mathematical engine. Configured
comparisons retain zero-only correction and scale 10000, while the CSV command
retains additive smoothing, scale 1 relative frequencies, and dynamic columns.

Use `nlpo compare` to compare existing frequency CSV files generated by
`nlpo count`. This command reads CSV files only; it does not rerun NLP.

Two-input comparison:

```bash
nlpo compare \
  --inputs \
    output/frequency_virgil_aeneis.csv \
    output/frequency_virgil_georgica.csv \
  --labels aeneis georgica \
  --metric log-ratio \
  --min-total-count 3 \
  --top 100 \
  --out output/compare_aeneis_georgica.csv
```

Run archive comparison:

```bash
nlpo compare \
  --inputs \
    runs/aeneis/outputs/frequency_text.csv \
    runs/cena/outputs/frequency_text.csv \
  --labels aeneis cena \
  --metric log-ratio \
  --min-total-count 3 \
  --out output/compare_aeneis_cena.csv
```

`log-ratio` is `log2(relative_a / relative_b)` with additive smoothing, so
positive values are more characteristic of the first input and negative values
are more characteristic of the second input. This smoothing is the existing
Compare CLI behavior and is distinct from configured group comparison's
zero-only `zero_correction`. For three or more inputs, the output reports each
label's count and relative frequency plus the max/min label and
relative-frequency range.

## Features CLI

Use `nlpo features` to build a feature matrix with one row per configured
group, or one row per file when `--group-by-file` is used. This is for
stylistic comparison, clustering, classification, and authorship experiments;
it is different from `count`, which writes frequency tables for
vocabulary inspection.

Basic feature matrix:

```bash
nlpo features \
  --project-root . \
  --config config/groups.config.yml \
  --out output/features.csv
```

Add most-frequent-lemma features:

```bash
nlpo features \
  --project-root . \
  --config config/groups.config.yml \
  --mfw 100 \
  --field lemma \
  --out output/features_mfw100.csv
```

One row per input file:

```bash
nlpo features \
  --project-root . \
  --config config/groups.config.yml \
  --group-by-file \
  --out output/features_by_file.csv
```

When using `grouping.mode: auto_single_cleaned` or `--auto-single-cleaned`,
features uses the same single-cleaned-file safety check as `count`:
exactly one cleaned `.txt` file must be present, otherwise the command fails.

`nlpo features` uses the same prepared text as `count`: cleaner output
is resolved first, group globs and `{cleaned_dir}` are expanded, files are
concatenated, configured text normalization is applied, and reference tags are
removed before NLP. If `ref_tags.enabled=true`, the pattern file must exist; a
missing pattern file is a configuration error.

Features and counting use the same chunked, pre-filter NLP analysis-record
extraction layer. Feature statistics are computed directly from those raw
records: they do not apply count-specific UPOS selection, require token
artifacts, or imply use of the analysis cache.

## N-Gram CLI

Use `nlpo ngram` to build n-gram frequency tables from a complete token artifact
or text files listed in `config/groups.config.yml`. Artifact input supports both
the `token` and `lemma` fields.

```bash
nlpo ngram \
  --tokens output/tokens.tsv \
  --n 2 \
  --field lemma \
  --min-count 2 \
  --top 100 \
  --out output/ngram_2.tsv
```

Group-specific output:

```bash
nlpo ngram \
  --tokens output/tokens.tsv \
  --n 3 \
  --field token \
  --by-group \
  --min-count 2 \
  --format csv \
  --out output/ngram_3_by_group.csv
```

Artifact n-grams use included tokens only and do not cross group, source-file,
section, chunk, or sentence boundaries. Empty values and punctuation-only
tokens are skipped.

Raw text input through config groups is also available for token n-grams:

```bash
nlpo ngram \
  --project-root . \
  --config config/groups.config.yml \
  --field token \
  --n 2 \
  --out output/ngram_text.tsv
```

Config input uses the same corpus planning and preparation path as `count` and
`features`, including cleaner execution, grouping mode, per-file and
auto-single-cleaned resolution, text normalization, and reference-tag removal.
All configured corpus files must be readable as UTF-8; if any file cannot be
read, preparation fails and no partial corpus is analyzed.
Token artifact input bypasses config loading, cleaner execution, and corpus
planning; it reads the existing validated artifact directly. Config input
supports `--field token`; use `--tokens` with an artifact for lemma n-grams.

Output columns are `ngram`, `count`, `n`, and `field`. When `--by-group` is
used, `group` is added.

## Example Config

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

filters:
  min_token_length: 2
  drop_roman_numerals: true
  roman_exceptions_file: config/roman_numeral_exceptions.txt
```

With cleaner preprocessing:

```yaml
preprocess:
  kind: cleaner
  config: config/cleaner.yml

groups:
  cleaned:
    files:
      - "{cleaned_dir}/*.txt"

out_dir: output
```

When `preprocess.kind` is `cleaner`, failure to load or run the cleaner stops
the command. Configured cleaner preprocessing is never silently skipped.

`groups.files: "{cleaned_dir}/*.txt"` counts every `.txt` file currently in the
cleaned directory. If you want to safely run one cleaned file at a time without
editing `groups`, use `--auto-single-cleaned`:

```bash
nlpo count \
  --project-root . \
  --config config/groups.config.yml \
  --auto-single-cleaned \
  --run-name satyricon-cena
```

The same behavior can be enabled in YAML:

```yaml
grouping:
  mode: auto_single_cleaned
  auto_group_name: text
```

Auto-single-cleaned selects the only `.txt` file in the resolved cleaned
directory. If there are zero or multiple cleaned `.txt` files, the command
fails instead of silently counting stale files. `--run-name` only names the run
archive; it does not change which input or cleaned file is counted.

For `count`, the input preparation order is: run cleaner if configured,
resolve the cleaned directory, resolve groups or per-file work items, concatenate
files, apply text normalization, remove and count reference tags, then run the
command-specific NLP/counting step. With `--group-by-file`, the output label is
derived from each file stem, duplicate files matched by multiple groups are
processed once, and label collisions receive numeric suffixes.

Useful config options:

```yaml
grouping:
  mode: groups          # "groups", "per_file", or "auto_single_cleaned"

analysis_unit: lemma      # "lemma" or "surface"

filters:
  min_token_length: 2
  drop_roman_numerals: true

dictcheck:
  enabled: true
  wordlist: data/wordlist/latin_words.txt
  lemma_normalize: config/lemma_normalize.tsv

ref_tags:
  enabled: true
  patterns: config/ref_tags.txt

trace:
  enabled: true
  path: output/trace.tsv
  max_rows: 500000

artifacts:
  tokens:
    enabled: false
    path: output/tokens.tsv
```

Diagnostic trace output is intended only for human inspection and debugging. It
may be filtered by `only_keys` or truncated by `max_rows`, and is not accepted
by concordance or n-gram commands. `artifacts.tokens` is the formal complete
token artifact for downstream reuse; if both are enabled, use different paths.

`filters.roman_exceptions_file` is a UTF-8, one-item-per-line file for tokens
that should not be removed by Roman numeral filtering. Blank lines and lines
starting with `#` are ignored, matching is case-insensitive, and missing files
are reported as configuration errors. The built-in surface-mode exceptions
`vi` and `di` remain available.

## Run Archive

`nlpo count` can save a reproducible snapshot of one successful run
under `runs/<run-name>/`. The archive includes frequency CSVs, `summary.txt`,
`run_meta.json`, token artifacts when enabled, trace files when enabled, a config snapshot, `manifest.json`,
and a short archive README.
Only files generated by the current run are copied into `outputs/`; stale CSVs
left in `out_dir` from older runs are not archived.
The archive uses the exact input, cleaned, trace, and output paths recorded
during the current run; it does not rescan corpus or output directories after
execution.

```bash
nlpo count \
  --project-root . \
  --config config/groups.config.yml \
  --run-name virgil_noun_test_01
```

If `--archive-run` is used without `--run-name`, the directory name is a
timestamp such as `runs/20260706-123456`.

```bash
nlpo count \
  --project-root . \
  --config config/groups.config.yml \
  --archive-run
```

Use `--runs-dir` to choose another archive root. Existing run directories are
never overwritten.

By default, large source corpora are not copied. Add these flags when the
archive should contain the files themselves:

```bash
nlpo count \
  --project-root . \
  --config config/groups.config.yml \
  --run-name virgil_noun_test_01 \
  --include-cleaned \
  --include-input
```

The same defaults can be set in `config/groups.config.yml`:

```yaml
archive:
  enabled: true
  runs_dir: runs
  include_input: true
  include_cleaned: true
```

When `archive.enabled` is true, `nlpo count` creates an archive even
without `--archive-run` or `--run-name`. CLI flags such as `--include-input`,
`--include-cleaned`, and `--runs-dir` override the config defaults.

After a successful archive, the CLI prints the saved path and copied file
counts:

```text
[ARCHIVE] saved run archive: runs/virgil_noun_test_01
[ARCHIVE] included input files: 3
[ARCHIVE] included cleaned files: 3
```

Use `--error-on-empty-group` to fail when any configured group matches zero
files. This is useful when a glob or `{cleaned_dir}` pattern no longer matches
the expected corpus. `--dry-run` prints the matched files under each group so
the config can be checked before running NLP.

`dictcheck.wordlist` is recorded in `manifest.json` with its path, size, and
SHA-256 hash, but is not copied into `config_snapshot/` by default.

All explicitly configured file references are resolved and checked while the
run plan is built, before cleaner execution, output creation, or NLP startup.
This check is strict even when the related feature is disabled. For example, a
configured but missing `dictcheck.lemma_normalize` file is now an error instead
of silently disabling lemma normalization. Dry runs apply the same checks.

## Configuration Roles

`nlpo_toolkit` uses several configuration files at different stages of the
workflow. These files should not be merged into a single file, because they have
different responsibilities.

| Setting | Role | When It Applies | Example |
|---|---|---|---|
| `lexicon_map.tsv` | Corrects or normalizes the input text itself | Before Stanza / NLP | `uod -> quod` |
| `lemma_normalize.tsv` | Corrects lemmas after NLP analysis | After Stanza / NLP | `omninus -> omnino` |
| `filters` | Mechanically removes tokens from counting | Before frequency counting | Drop one-character tokens; drop Roman numerals |
| `dictcheck.wordlist` | Classifies lemmas as known or unknown | After frequency tables are created | Check whether a lemma exists in the dictionary wordlist |

## Docker

Docker is the recommended way to avoid repeated local Stanza setup.

Build the image:

```bash
docker compose build
```

Run the CLI:

```bash
docker compose run --rm nlpo count --project-root /workspace --config config/groups.config.yml
```

Inside the container, the repository is mounted at `/workspace`, so use
`--project-root /workspace` for project-relative paths.

Stanza resources are stored at `/opt/stanza_resources` in a named Docker
volume. Normal container runs reuse that volume, so models are not redownloaded
each time.

Avoid this unless you intentionally want to remove the model cache:

```bash
docker compose down -v
```

To skip Stanza model download during image build:

```bash
docker compose build --build-arg DOWNLOAD_STANZA_MODELS=false
```

## Development Checks

Generate the JSON Schema directly from the canonical Pydantic model:

```bash
python scripts/generate_config_schema.py
```

This writes `config/groups.config.schema.json`; the generated file is not an
independent source of truth and should not be edited by hand.

Fast unit tests:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider --ignore=tests/test_latin_vocab_pipeline.py
```

Full tests that need real Stanza resources:

```bash
STANZA_RESOURCES_DIR=.stanza_resources \
PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider
```

## License

Licensed under the MIT License.
