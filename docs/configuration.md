# Configuration

Most corpus workflows use `config/groups.config.yml`. This document records
the configuration conventions and examples documented in the original README.
The source of truth for accepted keys remains the project's Pydantic models and
generated `config/groups.config.schema.json`; this document does not claim to
be an exhaustive JSON Schema.

## Location and validation

```bash
nlpo count --project-root . --config config/groups.config.yml
nlpo count --dry-run --project-root . --config config/groups.config.yml
```

When `--config` is omitted, the default is
`<project-root>/config/groups.config.yml`. Paths in the config are relative to
`--project-root`, **not** to the Python package directory or YAML file location.

The original README describes strict production YAML loading and Pydantic v2
validation:

- YAML is UTF-8, with a top-level mapping containing string keys.
- Duplicate mapping keys are invalid, including nested mappings in sequences.
- Unknown configuration keys are errors at every level.
- YAML scalars are not silently coerced: `enabled: "true"` is not a boolean.
- Referenced file paths are checked during planning, even when their feature is
  disabled; a configured-but-missing file can stop the run.
- Dry-run and normal execution use the same configuration reading and checks.
  Dry-run inspects currently matched inputs but does not run Cleaner or NLP.

## Minimal Latin lemmatization configuration

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

The text files are grouped together under `text`, and the default output is
`output/frequency_text.csv`. Use `analysis_unit: surface` to count surface
forms instead of lemmas.

## Grouping

```yaml
grouping:
  mode: groups  # groups | per_file | auto_single_cleaned
```

- `groups` counts all files matching each declared group.
- `per_file` produces one frequency table for each expanded input file. This
  is also accessible through `--group-by-file`.
- `auto_single_cleaned` requires exactly one cleaned `.txt` file and is also
  available as `--auto-single-cleaned`.

For single-cleaned-file runs, the README also documents:

```yaml
grouping:
  mode: auto_single_cleaned
  auto_group_name: text
```

The `--group-by-file` and `--auto-single-cleaned` flags are mutually exclusive.
Group comparison and partition validation are incompatible with per-file mode,
because the output labels no longer correspond to configured group names.

## Cleaner preprocessing

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

When Cleaner is configured, failure to load or execute it stops the command.
Configured Cleaner is not silently skipped. Normal runs execute Cleaner
**before** resolving input globs; dry-run inspects files without executing it.
`"{cleaned_dir}/*.txt"` matches all current text files in the resolved cleaned
directory. For a directory Cleaner input, a configured
`output_filename_template` is used exactly as written.

## Token filtering and postprocessing

```yaml
analysis_unit: lemma  # lemma | surface

filters:
  min_token_length: 2
  drop_roman_numerals: true
  roman_exceptions_file: config/roman_numeral_exceptions.txt

dictcheck:
  enabled: true
  wordlist: data/wordlist/latin_words.txt
  lemma_normalize: config/lemma_normalize.tsv

ref_tags:
  enabled: true
  patterns: config/ref_tags.txt
```

`roman_exceptions_file` contains one UTF-8 item per line. Empty lines and
`#` comments are ignored, and matching is case-insensitive. The original
README additionally records built-in exceptions `vi` and `di` for surface
mode.

`lemma_normalize.tsv` is an optional, project-specific **post-NLP lemma**
correction map. Each non-comment line has two tab-separated values, with no
header: the emitted lemma and the canonical lemma used for counting. The
repository-provided map is intentionally empty by default. Only add verified
corrections; it does not change the source text.

## Output traces, token artifacts, and caching

```yaml
trace:
  enabled: true
  path: output/trace.tsv
  max_rows: 500000

artifacts:
  tokens:
    enabled: true
    path: output/tokens.tsv

analysis_cache:
  enabled: true
  dir: .analysis_cache
  lock_timeout_sec: 300.0
```

Diagnostic traces are for debugging and may be filtered or truncated; they
cannot be used as concordance or n-gram token input. The complete formal
artifact is `tokens.tsv` together with its associated metadata. If trace and
token output are both enabled, give them different paths.

The analysis cache stores NLP records rather than final frequency counts;
changes to lemma filtering or reporting settings may reuse records, while
changes to input text, backend, models, or analysis behavior invalidate it.
The cache is excluded from archived research outputs. Clear it with
`nlpo cache clear`, not a manual recursive delete.

## Group partition validation

When the corpus has one whole group and two or more parts, validate whether the
**final frequency tables** sum to the whole:

```yaml
groups:
  full:
    files: [input/full.txt]
  part_a:
    files: [input/part_a.txt]
  part_b:
    files: [input/part_b.txt]

validations:
  partitions:
    - name: full_split
      whole: full
      parts: [part_a, part_b]
      on_mismatch: error  # warn | error; default warn
      report: mismatches  # mismatches | all; default mismatches
```

Outputs include `partition_validation_<name>.csv` and
`partition_validation.json`. With `on_mismatch: warn`, mismatches are reported
without changing exit code 0; with `error`, validation outputs are written and
the exit code becomes 1. NLP decisions at split boundaries can differ even
when the input text is perfectly partitioned. See
[Specification](specification.md).

## Configured group comparisons

```yaml
groups:
  corpus_a:
    files: [input/corpus_a.txt]
  corpus_b:
    files: [input/corpus_b.txt]

comparisons:
  - name: corpus_a_vs_corpus_b
    group_a: corpus_a
    group_b: corpus_b
    scale: 10000
    zero_correction: 0.5
    min_total_count: 2
```

Writes `group_comparison_<name>.csv` and `group_comparisons.json`, in addition
to the usual summary/run metadata. These comparisons operate on final group
frequency counters in the current run. Their normalization and smoothing are
**not identical** to the `nlpo compare` command's behavior.

## Run archive

```yaml
archive:
  enabled: true
  runs_dir: runs
  include_input: true
  include_cleaned: true
```

With `archive.enabled: true`, Count creates an archive even without
`--run-name` or `--archive-run`. Flags such as `--runs-dir`, `--include-input`,
and `--include-cleaned` override configuration defaults. Without opt-in,
large source texts are not copied into archives. Existing archive directories
are not overwritten. The archive manifest records configured dictionary
wordlist path, size and SHA-256 but does not copy the wordlist by default.

## Other configuration files and their roles

| File or setting | Stage | Role |
| --- | --- | --- |
| `lexicon_map.tsv` | Before NLP | Corrects/normalizes source text |
| `lemma_normalize.tsv` | After NLP | Maps erroneous emitted lemmas to canonical lemmas |
| `filters` | Counting | Excludes tokens by mechanical rules |
| `dictcheck.wordlist` | After counting | Classifies lemmas as known/unknown |
| `config/latin_function_words.txt` | Feature extraction | Reviewed function-word list for explicitly selected features |
| `config/authorship_metadata.csv` | Table-input stylometry | Maps each feature observation to author/work |
| `config/authorship_works.csv` | Corpus-input stylometry | Maps each prepared group to author/work |

Do not merge these files into one correction map: they act at different stages
and have different semantics. See [Usage](usage.md) for corresponding commands.

## Schema and development checks

```bash
python scripts/generate_config_schema.py
```

This generates `config/groups.config.schema.json` from the project's canonical
Pydantic model. The generated schema is not a separate authoritative source;
it should not be edited manually. The original README does not provide a
validated operating-system/Python-version support matrix.
