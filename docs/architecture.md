# Architecture

This document describes the current **module responsibilities and dependency
boundaries as stated in the supplied former README**. It is an implementation
design reference, not a user-facing command guide or a claim that the current
repository was independently audited. For behavior rather than implementation,
see [Specification](specification.md).

## 1. Package layout and boundaries

| Area | Responsibility |
| --- | --- |
| `nlpo_toolkit.latin.cleaners` | Latin corpus cleaning and its typed execution service |
| `nlpo_toolkit.nlp` | Backend-neutral NLP contracts, chunking, filtering and analysis-record utilities |
| `nlpo_toolkit.backends` | Concrete NLP backend implementations and construction |
| `nlpo_toolkit.corpus_analysis` | Corpus planning, execution, Count, Features, output publication, cache composition, archives |
| `nlpo_toolkit.corpus_analysis.cli` | CLI subcommand parser registration, request translation, presentation and exit codes |
| `nlpo_toolkit.comparison` | Pure comparison engine and configured/CSV-entry adapters |
| `nlpo_toolkit.serialization.types` | Typed parsing/serialization boundary for JSON/YAML/CSV scalar structures |

The original README gives the following dependency principle:

- `corpus_analysis/ports.py` owns interfaces and frozen containers needed by
  application services.
- Application services import ports rather than concrete production adapters.
- `corpus_analysis/composition.py` selects and constructs production dependencies.
- Tests can instantiate port containers without constructing production I/O.
- `nlpo_toolkit.nlp.contracts` owns backend-independent models, protocols,
  metadata, and backend build specifications. Backend implementations do not
  import `corpus_analysis`.
- The composition boundary turns Pydantic `NLPConfig` into `NLPBackendSpec` and
  assembles the selected backend.

The architecture therefore separates **contracts, application orchestration,
backend adapters, and publication infrastructure**.

## 2. Corpus planning and preparation

The preparation pipeline is described as a one-way `planning` package:

```text
configuration / CLI options
           |
           v
planning.models  (immutable plan / resolved-plan data)
           |
           v
planning.build   (validate configuration, paths, Cleaner config and references)
           |
           +--> planning.resolve.inspect_analysis_plan()
           |      (dry-run: inspect currently existing inputs, no Cleaner)
           |
           +--> planning.resolve.prepare_analysis_plan()
                  (normal run: execute Cleaner, then resolve files)
                                  |
                                  v
                           prepare_corpora()
                    (read text, normalize, remove reference tags)
```

`planning.validate` holds pure Count-specific structure and resolved-group
checks. `planning.build` creates a static plan without running the Cleaner or
resolving globs. `planning.resolve` is the component that can invoke
preprocessing. `ResolvedAnalysisPlan` has one `definition` field holding its
static `AnalysisPlan` rather than duplicating fields with forwarding properties.
The documented dependency direction is from models to validation to build or
resolve, with preprocessing side effects confined to resolution.

The shared frozen `CorpusPreparationRequest` captures preparation settings for
Count, Features, config-input N-gram, and dry-run. Each command request composes
this request and uses a single `grouping_override` value for CLI grouping
settings. Dry-run is an independent use case rather than a mode of a Count
request. Config-input N-gram uses tokens and does not run the NLP backend.

## 3. Cleaner integration

Cleaner is invoked through the typed
`latin.cleaners.service.execute_cleaner()` application service, **not** by
shelling out to its standalone CLI. The `CleanerConfigInspection` produced by
planning is passed into execution, avoiding another config read. The Cleaner
program is built once per run. The CLI adapter merely builds the typed request,
presents the result, and maps domain errors to process exit codes. For directory
input, `output_filename_template` is obeyed literally.

## 4. Execution sessions and NLP pipeline

Two typed sessions provide shared execution state:

- `CorpusExecutionSession` owns the resolved plan and prepared corpora used by
  Count, Features and config-input N-gram.
- `NLPExecutionSession` additionally owns the selected backend, extraction
  policy and Roman numeral exceptions for Count and Features.

The prepared text is passed unchanged from those sessions into the common
chunked analysis-record extraction path. Sentence splitting is performed by the
selected main backend and the shared extraction layer. There is no Count-only
pre-tokenization backend.

The canonical sequence identified by the former README is:

```text
prepared corpus text
      |
      v
iter_nlp_analysis_records_from_text()
      |
      v
evaluate_analysis_record()
      |
      +--> Count: selection and frequency accumulation
      +--> Features: shared lexical-record filtering, feature calculations
```

Reference tags have already been removed/counted during preparation, so
token-level evaluation does not detect them again. The pipeline distinguishes
**raw NLP analysis records** from the **eligible lexical record** population.

## 5. Count services, cache and publication ports

Count obtains NLP records through a typed `AnalysisRecordProvider` port.
Production composition injects its cached implementation; both cache-on and
cache-off extraction are hidden behind the same provider. This prevents Count
orchestration from depending on fingerprint generation, cache key formats,
repository access, locks, codecs, hits/misses or atomic cache writes.

A provider returns a context-managed record stream. Cache locks are released
when the consumer exits, and interrupted consumers do not publish incomplete
cache objects. Cache-statistic snapshots and the application-side collector are
separate from the filesystem cache package; reporting reads typed results.

Publication is also port-driven: frozen publication requests and typed ports
cover group frequency tables, partition validation, comparisons, run reports,
formal token artifacts and diagnostic traces. Production filesystem adapters
delegate to concrete serializers and writers, injected by the composition
root. Application orchestration neither selects concrete CSV/TSV/JSON writers
nor serializes their payloads.

## 6. Results, output inventory, and atomic writes

The original README describes two separate canonical stores:

- `AnalysisResults.groups`: immutable map of the actual group analysis results.
- `ArtifactPlan`: immutable plan of every generated file path.

The `ArtifactPlan` is built **after** final group labels are known and
**before** startup of the NLP backend or creation of the output directory. It
holds typed absolute paths for frequency/dictionary results, reference tags,
traces, token artifacts and metadata, partition validation, group comparisons,
summary, and run metadata. The entire set is checked for path collisions.

Postprocessing accepts immutable count mappings and returns typed normalized
and dictionary-classified results. The TSV loader for lemma normalization is an
infrastructure boundary within postprocessing.

Each output writer accepts a `PlannedArtifact`, checks its `ArtifactKind`, and
publishes UTF-8 output using a temporary file in the destination directory;
it does not reconstruct file names. Reporting is divided into typed report
models, a pure summary renderer, a pure metadata builder/serializer, and a
publication service. The service writes `summary.txt` before `run_meta.json`.
The `generated_outputs` field comes from the artifact plan and includes
`run_meta.json` itself. Other result types do not maintain duplicate output
path inventories.

Count execution state lives in `count_context`; the final result lives in
`count_result`. Partition and configured comparison run-level result types
live in their own modules instead of being defined inside service code.
`CountRunResult` exposes read-only output-plan views and retains a single
`artifact_plan` field.

## 7. Features engine and column calculations

`features.service` connects the common execution sessions to
`features.engine`. The engine performs analysis-record extraction once per
prepared corpus and applies the eligibility filter once. Basic lexical, UPOS,
and MFW calculators then consume that shared lexical population; raw records
remain available for sentence/chunk metadata, sample bounds, and unfiltered
record counts. Choosing lemma versus surface values for MFW does not change
which lexical records are eligible.

`filters.upos_targets` is deliberately Count-only; Features' UPOS distribution
and lexical class ratios require all otherwise eligible lexical tokens. The
formal token-artifact stream is not a prerequisite for Features calculations.

## 8. Comparison engine

Configured group comparisons and the `nlpo compare` CSV command both use the
pure `nlpo_toolkit.comparison.engine`. The comparison engine accepts a concrete
`FrequencyTable` model from `comparison.models`, with no filesystem or Pydantic
configuration dependency.

- `comparison.config` defines configured-comparison settings.
- `comparison.results` defines typed engine/application results and the
  concrete `ComparisonSpec` in configured output.
- `comparison.services.configured` adapts live Count Counters.
- `comparison.services.csv` adapts published frequency files.
- CSV/JSON serialization and dynamic command output columns stay outside the
  pure comparison package, within corpus-analysis artifacts and CLI adapters.

Configured comparisons and CSV comparison preserve their distinct smoothing
and scale conventions. The comparison engine knows neither `ArtifactPlan` nor
output filenames.

## 9. Token artifacts and archive design

The version-1 formal token-artifact protocol is split into schema, row codec,
writer, reader, integrity, and full-validation components. Concordance and
N-gram share its typed token sequence reader/collector. The sequence key is
`TokenSequenceId(group, source file, section, chunk, sentence)`. The writer
uses unique temporary files and publishes TSV **before** its validating
metadata JSON, which acts as the final commit marker. Reader validation checks
strict metadata types, row/count/byte/hash agreement and schema compatibility.
The original artifact location is informational, so archive copies still read.

Run-archive request/result contracts belong to `corpus_analysis.archive`.
Archive consumes the Count run result and its `ArtifactPlan` but does not own
or redefine those types. The archive inventory classifies the already planned
files rather than scanning the output directory. Copying uses that inventory;
manifest generation converts typed file and Git metadata. On failure, the
archive service removes only the new directory it created for that attempt.

## 10. Typed boundary policy

Domain and application code retain concrete dataclasses, Pydantic models,
enums, typed rows, and typed result objects as long as possible. Conversion
to recursive JSON/YAML or scalar CSV structures occurs at parsing and
serialization edges, using `nlpo_toolkit.serialization.types`.

Raw JSON/YAML and third-party adapter data enter as `object` and are validated
before typed structures are constructed. The documented implementation rejects
malformed Transformers token-classification output instead of silently coercing
it. Application results avoid generic `Any` or untyped
`Mapping[str, object]` payloads. Paths, times, and enums are not prematurely
converted to strings.

## 11. Development and verification

The original project README lists these development checks:

```bash
python scripts/generate_config_schema.py
PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider --ignore=tests/test_latin_vocab_pipeline.py
STANZA_RESOURCES_DIR=.stanza_resources \
  PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider
```

The first command regenerates `config/groups.config.schema.json` from the
canonical Pydantic model. The other commands run quick and Stanza-dependent
tests, respectively. They were **not rerun** as part of this documentation-only
reorganization, since the source repository and its dependency lockfiles were
not supplied. Consult [the archived full README](legacy-readme.md) for the
former wording and fine-grained rationale behind these boundaries.
