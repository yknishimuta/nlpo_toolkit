# nlpo_toolkit

`nlpo_toolkit` is the canonical Python package for Latin corpus cleaning,
NLP-backed vocabulary counting, and reproducible frequency-table workflows.

Corpus analysis functionality lives in `nlpo_toolkit.corpus_analysis` and is
exposed through the `nlpo` command.

CLI subcommands are implemented under `nlpo_toolkit.corpus_analysis.cli`.
Each command module defines its parser registration and execution handler.

## What Is Included

- Latin corpus cleaning utilities under `nlpo_toolkit.latin.cleaners`
- Low-level chunking, normalization, Roman-numeral policy, and vocabulary
  loading utilities under `nlpo_toolkit.nlp`; backend construction lives under
  `nlpo_toolkit.backends`
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

Dependency boundaries for corpus analysis are explicit: `ports.py` defines the
interfaces and frozen containers required by application services, while
`composition.py` selects production implementations. Application services import
only ports, CLI/bootstrap code creates production dependencies, and unit tests
normally construct port containers directly.

Backend-independent NLP models, protocols, backend metadata, and backend build
specifications are owned by `nlpo_toolkit.nlp.contracts`. Backend modules
implement those contracts without importing `corpus_analysis`; application code
depends on the contracts and its ports. `corpus_analysis/composition.py` is the
production boundary that converts the Pydantic `NLPConfig` into an
`NLPBackendSpec` and assembles the selected backend.

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

Corpus preparation is split across a one-way `planning` package. Immutable
models live in `planning.models`; `planning.build` validates configuration,
paths, Cleaner configuration, and referenced files and returns a static plan
without running the Cleaner or resolving corpus globs. Pure Count structure and
resolved-group checks live in `planning.validate`.
`planning.resolve.inspect_analysis_plan()` resolves currently existing inputs
for dry-run and never runs the Cleaner. Count, Features, and config-input N-gram
use `planning.resolve.prepare_analysis_plan()`, which runs the Cleaner first and
then resolves inputs from the resulting filesystem state. Finally,
`prepare_corpora()` reads those resolved inputs and applies normalization and
reference-tag removal. `ResolvedAnalysisPlan` contains its static
`AnalysisPlan` as `definition`; it does not duplicate static values through
proxy fields or properties. The dependency direction is models to validation
to build or resolve, with resolve alone calling preprocessing.

Cleaner execution is a typed application-service call to
`latin.cleaners.service.execute_cleaner()`, not an invocation of the standalone
Cleaner CLI. The `CleanerConfigInspection` produced during corpus planning is
passed directly to execution, so Count, Features, and config-input N-gram do not
reload the Cleaner config. The Cleaner program is built once per run. The CLI
adapter only converts its config argument to a request, presents the typed
result, and maps domain errors to a process exit code. For directory inputs, a
configured `output_filename_template` is always used exactly as configured.

Execution commands compose these stages through two typed sessions.
`CorpusExecutionSession` owns the resolved plan and prepared corpora shared by
Count, Features, and config-input N-gram. `NLPExecutionSession` adds the built
backend, extraction policy, and Roman-numeral exceptions shared by Count and
Features. Both pass each prepared corpus string unchanged to the common chunked
analysis-record extraction path. Sentence splitting is the responsibility of the
selected main NLP backend and that shared extraction path; Count has no dedicated
pre-tokenization backend. Config-input N-gram uses only the corpus session and
starts no NLP backend.

Count application services depend only on typed publication ports and frozen
publication requests. Production filesystem adapters implement the group,
partition, comparison, run-report, token-artifact, and diagnostic-trace
publication boundaries by delegating to the concrete serializers and writers.
The production composition root injects all of these publication dependencies;
application orchestration neither selects concrete writers nor serializes CSV,
JSON, TSV, or text output.

Count also obtains NLP analysis records only through the typed
`AnalysisRecordProvider` port. The production composition root injects the
cached provider, which owns both cache-enabled and cache-disabled extraction.
Cache fingerprints and keys, repositories, locking, codecs, hit/miss checks,
and atomic cache writes remain inside the Analysis Cache infrastructure; Count
application orchestration does not construct or import them.

Features keeps application orchestration separate from calculation:
`features.service` connects the execution sessions to `features.engine`, while
the engine extracts analysis records once per prepared corpus and applies the
eligibility filter once. Lexical, UPOS, and MFW modules then calculate their
columns from that shared filtered record population.

The canonical token-analysis path is
`iter_nlp_analysis_records_from_text()` followed by
`evaluate_analysis_record()`. Reference tags are removed and counted during
corpus preparation before prepared text reaches NLP; token evaluation does not
repeat reference-tag detection.

Group analysis data has one canonical store: the immutable
`AnalysisResults.groups` mapping. Generated-file inventory is separate and has
one canonical store of its own: Count builds an immutable `ArtifactPlan` after
corpus preparation has established the final group labels and before starting
the NLP backend or creating the output directory. The plan contains the typed,
absolute paths for frequency, dictcheck, reference-tag, diagnostic trace, token
artifact and metadata, partition validation, group comparison, summary, and run
metadata outputs. Every planned path is checked together for collisions before
any output is written.

Count postprocessing applies lemma normalization and dictionary classification
to immutable count mappings and returns typed results. The lemma-normalization
TSV loader is an infrastructure boundary owned by the postprocessing package.
Artifact writers
receive a `PlannedArtifact`, validate its `ArtifactKind`, and publish UTF-8
output through a same-directory temporary file; writers never reconstruct
filenames. Reporting is split into typed report models, a pure summary renderer,
a pure run-metadata builder/serializer, and a publication service. The service
publishes `summary.txt` before `run_meta.json`, and metadata
`generated_outputs` comes directly from `ArtifactPlan`, including
`run_meta.json` itself. Analysis, partition, and comparison results do not carry
generated-path inventories; `CountRunResult` exposes only read-only views derived
from its single `artifact_plan` field.

Archive request/result contracts are owned by `corpus_analysis.archive`; Count
and archive adapters depend on that package-level contract rather than a generic
top-level types module.

Count execution state is owned by `count_context`, and the final Count result is
owned by `count_result`. Partition-validation and configured-comparison run-level
results live in their own application-result modules, independently of the
services that construct them. Archive consumes `CountRunResult` but does not own
or redefine it.

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

All production YAML configuration files use the same strict loader. Files must
be UTF-8 and their top level must be a mapping whose keys are strings. Duplicate
mapping keys are errors at every depth, including mappings nested in sequences;
a later value never silently overwrites an earlier value. Dry-run and normal
execution use the same single configuration read and therefore accept or reject
the same YAML document.

### Frequency Output Naming

Frequency outputs are named `frequency_<label>.csv`. Dictionary-check outputs
are named `frequency_<label>.known.csv` and
`frequency_<label>.unknown.csv`.

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
  lock_timeout_sec: 300.0
```

The cache is an internal optimization and is not copied into run archives or
listed in `generated_outputs`. Use token artifacts for stable research outputs.
Internally, the production `AnalysisRecordProvider` coordinates cache keys and
paths, fingerprinting, record codecs, writers, repositories, and hit/miss
handling. It returns a context-managed record stream, so locks are released and
interrupted consumers cannot publish incomplete cache objects. Cache statistics
snapshots and their application-side collector are independent of the
filesystem cache package; reporting consumes only those typed result models.

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
remain available in the surrounding context. Context is built only from formal
token records and never by re-tokenizing the sentence metadata string. File,
group, and sentence metadata are included in the concordance output when present.

Internally, the version 1 token-artifact protocol is split into schema, row
codec, writer, reader, integrity, and full-validation modules. N-gram and
Concordance pass those typed records through the same token sequence collection.
Its canonical `TokenSequenceId` boundary consists of group, source file, section,
chunk, and sentence; tokens within each sequence are ordered by token index with
global token index as the deterministic tie-breaker. Metadata JSON is
strictly validated: strings are never coerced to booleans or integers, unknown
keys are rejected, and row, included/excluded, byte-size, and SHA-256 values
must agree with the TSV. The writer uses unique temporary files and publishes
the TSV before publishing metadata as the final commit marker. Concordance and
N-gram share the same reader. An artifact copied into a run archive remains
readable because validation does not require its current path to equal the
informational `artifact_path` stored when it was created.

## Compare CLI

Both configured group comparisons and `nlpo compare` use the same pure
`nlpo_toolkit.comparison.engine`. `comparison.models` owns the concrete
`FrequencyTable` engine input, and the engine compares that model without
depending on filesystems or Pydantic configuration. `comparison.config` owns
configured-comparison settings, while `comparison.results` owns concrete,
non-generic engine and application results. Configured results hold a concrete
`ComparisonSpec`; configured and CSV comparison services share the same table
and result models without generic table protocols or type ignores. In-memory
configured Counters and frequency CSV files enter through separate adapters in
`comparison.services.configured` and `comparison.services.csv`.

Configured comparisons retain zero-only correction and scale 10000. CSV
comparisons retain additive smoothing and scale 1. CSV/JSON serialization and
dynamic CLI columns are rendered outside the comparison package in Corpus
analysis artifact and CLI boundaries. The comparison package does not know
ArtifactPlan, output paths, or output filename conventions.

Across the analysis pipeline, domain and application layers retain concrete
dataclasses, Pydantic models, enums, and typed rows. Recursive JSON/YAML values
and scalar CSV rows are introduced only at parsing or serialization boundaries
through `nlpo_toolkit.serialization.types`. Token artifact metadata, analysis
cache metadata and fingerprints, archive manifests, and run metadata remain
strict typed models until their final serializer runs. Paths, datetimes, and
enums are not converted to strings prematurely.

Raw JSON, YAML, and third-party adapter values enter as `object` and are
validated at the same boundary before typed values are constructed. Malformed
Transformers token-classification output is rejected instead of silently
coerced. Application results do not use `Any` or untyped
`Mapping[str, object]` payloads.

Use `nlpo compare` to compare existing frequency CSV files generated by
`nlpo count`. This command reads CSV files only; it does not rerun NLP.

Two-input comparison:

```bash
nlpo compare \
  --inputs \
    output/frequency_virgil_aeneis.csv \
    output/frequency_virgil_georgica.csv \
  --labels aeneis georgica \
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

Add length-adjusted lexical-diversity features for both normalized surface
tokens and lemmas:

```bash
nlpo features \
  --project-root . \
  --config config/groups.config.yml \
  --lexical-diversity \
  --out output/features_lexdiv.csv
```

The project defaults are a MATTR/MSTTR window of 100, an MTLD threshold of
0.72, and an HD-D sample size of 42. They can be set explicitly with
`--lexdiv-window`, `--mtld-threshold`, and `--hdd-sample-size`; specifying any
one of these also enables lexical diversity. The output columns are, in order,
`mattr_token`, `mattr_lemma`, `msttr_token`, `msttr_lemma`, `mtld_token`,
`mtld_lemma`, `hdd_token`, and `hdd_lemma`. The existing `--field` option only
selects the MFW field and never suppresses either lexical-diversity variant.

All four metrics use the ordered records that pass the shared Features
eligibility filter. MATTR averages overlapping moving-window TTRs; MSTTR
averages non-overlapping full segments and ignores a trailing remainder when a
full segment exists. MTLD averages forward and reverse factor calculations.
HD-D uses an effective sample size of the smaller of the configured size and
the available token count. For sequences shorter than the configured window,
MATTR and MSTTR fall back to whole-sequence TTR; empty sequences produce `0.0`
for every metric. With fixed-token sampling, every metric is calculated inside
each sample and never crosses its boundary. Use identical window, threshold,
and sample-size settings when comparing texts.

Explicit parameter example:

```bash
nlpo features \
  --project-root . \
  --config config/groups.config.yml \
  --lexdiv-window 100 \
  --mtld-threshold 0.72 \
  --hdd-sample-size 42 \
  --out output/features_lexdiv.csv
```

Explicit Latin function-word features can be supplied from a research-specific
UTF-8 list:

```bash
nlpo features \
  --project-root . \
  --config config/groups.config.yml \
  --function-words config/latin_function_words.txt \
  --out output/features_function_words.csv
```

The list contains one token per line. Blank lines and lines whose stripped
content starts with `#` are ignored; duplicate normalized terms, tab-separated
content, multiword expressions, and lists with no terms are rejected. Terms are
matched case-insensitively after stripping. `--function-word-field lemma` is
the default and uses the normal lemma-with-surface-fallback rule;
`--function-word-field token` selects normalized surface tokens. This setting
is independent of the MFW-only `--field` option.

Each listed term produces one `fw_<term>` column in list order. Its value is
the term count divided by all lexical tokens in that feature row, including
tokens that are not in the list. These explicit columns coexist with the UPOS
aggregate `function_word_count` and `function_word_ratio`, and with independently
selected MFW columns. Use the same reviewed list for every text in a comparison;
the toolkit does not provide or infer an authoritative Latin function-word set.

Only records passing the shared Features eligibility filter participate. Thus
a one-character term may remain at zero when the configured minimum token
length excludes it. Matching is exact: attached enclitics such as `-que`,
`-ve`, and `-ne` are not split or suffix-matched, and multiword expressions are
not supported. With fixed-token windows, frequencies and denominators are
calculated independently inside each full, partial, or overlapping sample.

One row per input file:

```bash
nlpo features \
  --project-root . \
  --config config/groups.config.yml \
  --group-by-file \
  --out output/features_by_file.csv
```

For authorship and clustering experiments, split each file into fixed windows
of eligible word tokens:

```bash
nlpo features \
  --project-root . \
  --config config/groups.config.yml \
  --group-by-file \
  --window-tokens 1000 \
  --mfw 500 \
  --field lemma \
  --out output/features_windows.csv
```

Use `--step-tokens 500` with `--window-tokens 1000` for overlapping windows,
or add `--include-partial-window` to retain at most one shorter trailing
window. Window size is measured after the shared Features eligibility filter,
so punctuation, empty or too-short tokens, and excluded Roman numerals do not
consume window positions. Windows never cross source-file boundaries; use
`--group-by-file` or `grouping.mode: per_file` when a configured group resolves
to multiple files.

Sample rows include `source_file`, a deterministic `sample_id`, a one-based
`sample_index`, `sample_kind`, and filtered-token offsets.
`sample_start_token` is inclusive and `sample_end_token` is exclusive. Raw
`token_count`, sentence count, and character count describe the span between
the sample's first and last eligible records, including filtered records inside
that span. Global MFW terms are selected once from each unsampled filtered
corpus before windows are created, so overlap does not bias the MFW vocabulary.
Without `--window-tokens`, the existing one-row-per-corpus schema and values are
unchanged.

Basic features describe both sentence-length and surface-token-length
distributions. For each distribution the output includes the mean, population
variance (dividing by `N`), median, and linearly interpolated 25th and 75th
percentiles. A sentence is identified by both chunk index and sentence index;
its length is the number of tokens that pass the Features eligibility filter,
and a raw sentence with no eligible lexical tokens contributes a length of
zero. Token length is the Python string length of the stripped, lowercased
surface token, never the lemma. Empty distributions produce `0.0` for every
statistic. With fixed windows, these values are calculated independently from
each sample's raw and lexical records, including sentences cut by a window
boundary.

When using `grouping.mode: auto_single_cleaned` or `--auto-single-cleaned`,
features uses the same single-cleaned-file safety check as `count`:
exactly one cleaned `.txt` file must be present, otherwise the command fails.

`nlpo features` uses the same prepared text as `count`: cleaner output
is resolved first, group globs and `{cleaned_dir}` are expanded, files are
concatenated, configured text normalization is applied, and reference tags are
removed before NLP. If `ref_tags.enabled=true`, the pattern file must exist; a
missing pattern file is a configuration error.

Features and counting use the same chunked NLP analysis-record extraction
layer. Features retains the complete post-NLP, pre-filter raw record tuple, then
applies one eligibility filter per corpus to create an immutable lexical record
tuple. Basic lexical statistics, UPOS statistics, global MFW selection, and
per-corpus MFW frequencies all consume that same lexical population. The raw
population remains available for raw token counts, sentence/chunk boundaries,
and punctuation or sampling spans. Eligibility uses
the surface token and includes word-token detection, `min_token_length`, and
the Roman-numeral policy including configured and built-in surface exceptions.
The MFW `lemma`/`token` field changes only the value selected from eligible
records, never the eligible record population.

`filters.upos_targets` is intentionally Count-only and is not applied to
Features: UPOS distribution and content/function ratios require the full
eligible lexical population. In feature output, `token_count` is the raw NLP
record count, while `word_token_count` is the count that passed the feature
eligibility filter. Features do not require token artifacts or imply use of the
analysis cache.

## Stylometry / Burrows's Delta

`nlpo stylometry delta` reads an already-published Features CSV or TSV. It does
not rerun NLP, corpus preparation, Cleaner, or feature calculation. Every input
row is treated as one stylometric vector, and the feature family is always
selected explicitly:

```bash
nlpo stylometry delta \
  --features output/features.csv \
  --id-column sample_id \
  --feature-prefix mfw_ \
  --out output/burrows_delta.csv
```

For fixed-token Features output, use the unique `sample_id` rather than a
repeated group label. Prefixes and explicit columns may be combined, for
example `--feature-prefix fw_ --feature-prefix mfw_`, or individual columns can
be supplied repeatedly with `--feature-column`. No numeric columns are selected
implicitly, so MFW count, function-word list, fixed-window size, and all other
feature-generation settings should be held constant across an experiment.

The initial CLI fits one z-score model on all input rows and transforms those
same rows. Each retained feature uses its sample standard deviation with
denominator `N - 1`; this differs from the population variance used by Features
descriptive statistics. Burrows's Delta is the mean absolute difference between
two standardized vectors. A smaller distance indicates greater stylometric
proximity, but distance alone does not establish authorship or authenticity.

Features with exactly zero sample variance are excluded and reported on stderr;
the command fails if none remain. Output is a long-form table containing each
unordered sample pair once as `sample_a`, `sample_b`, and `burrows_delta`, sorted
by distance. Rigorous unknown-work evaluation will require a future workflow
that fits the standardization model on reference works only; the fit and
transform calculations are already separated internally for that extension.

### Leave-one-work-out evaluation

`evaluate-lowo` holds out one entire known work at a time. All fixed-window
samples from that work stay together, so no window from the test work can enter
training. Metadata contains one row per Features observation:

```csv
sample_id,author,work
aeneid__sample_0001,virgil,aeneid
aeneid__sample_0002,virgil,aeneid
georgics__sample_0001,virgil,georgics
thebaid__sample_0001,statius,thebaid
silvae__sample_0001,statius,silvae
```

```bash
nlpo stylometry evaluate-lowo \
  --features output/features_windows.csv \
  --metadata config/authorship_metadata.csv \
  --id-column sample_id \
  --feature-prefix fw_ \
  --out output/lowo_folds.csv \
  --summary-out output/lowo_summary.json
```

Each author must have at least two distinct works. Samples are first averaged
within each work, giving every work one equal-weight profile regardless of its
window count. In every fold, z-score fit and zero-variance detection use only
training work profiles. Standardized training works are averaged by author,
again by work rather than sample count, and the held-out work is assigned to
the nearest author centroid using Burrows's Delta.

The fold table has one row per work. The JSON summary contains overall work
accuracy, per-author work accuracy, and macro author accuracy—the unweighted
mean of author accuracies. This is closed-set evaluation: it cannot reject an
author outside the candidates, and high accuracy alone cannot establish
authorship or authenticity. Use identical Features settings for every work.
For fixed windows, prefer a common `window_tokens`, consistent overlap step,
and consistent partial-window policy; partial and full windows receive the same
sample weight in the initial work mean.

LOWO prevents a held-out work from entering z-score fit, zero-variance checks,
author centroids, or training through another window. It cannot audit feature
selection performed before the table was created. In particular, an MFW
vocabulary selected from all works may already reflect the held-out work.
Strict experiments should use a pre-registered function-word list, an MFW
vocabulary fixed from an external reference corpus, or a future fold-specific
MFW-selection workflow. The command evaluates the supplied feature matrix and
does not reject `mfw_` columns automatically.

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
section, chunk, or sentence boundaries. They use the same `TokenSequenceId` and
sequence collection as Concordance. Empty values and punctuation-only tokens
end the current lexical run. Excluded tokens are omitted without ending a run.
`--by-group` changes only the destination counter and output column; it never
changes a sequence boundary.

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
Each prepared corpus becomes one typed token sequence with no invented file,
section, chunk, or sentence detail, and this path does not start an NLP backend.
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
Only files in the actual run's `ArtifactPlan` are copied; stale CSVs left in
`out_dir` from older runs are not archived. The archive classifies diagnostic
trace artifacts by their typed kind and treats every other planned artifact as
an output. It does not infer kinds from filenames or rescan the output directory.

Internally, archive inventory validates and classifies the `ArtifactPlan` held
by `CountRunResult`, copying executes only the resulting copy plan, and manifest generation
converts typed file and Git metadata into the existing schema. A thin archive
service orchestrates those stages and removes only the directory it created if
any stage fails.

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

The repository-provided `config/lemma_normalize.tsv` is intentionally empty by
default. It is a project-specific correction map, so entries should be added
only after an NLP lemma error has been verified against the configured backend
and source corpus. Each mapping is one tab-separated pair: the left column is
the lemma emitted by NLP and the right column is the canonical lemma used in
frequency tables. Empty lines and `#` comments are ignored; do not add a header
row. This map does not correct input text, and while it contains no mappings,
lemma normalization is a no-op. The `omninus -> omnino` row above is an
illustrative example, not a default mapping.

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
