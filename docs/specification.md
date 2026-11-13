# Specification

This document records the **observable behavior and research-methodology
constraints** described by the former `README.md`. It is separated from
[Usage](usage.md) (commands), [Configuration](configuration.md) (YAML), and
[Architecture](architecture.md) (implementation boundaries).

> Documentation status: these descriptions were reorganized from the supplied
> former README. They have **not** been independently checked against the
> current source tree or executed tests. The full source document is preserved
> as [Original README](legacy-readme.md) for auditing any omitted detail.

## 1. Corpus preparation and analysis units

The documented normal preparation order is: inspect the configuration and
referenced files, run Cleaner preprocessing when configured, resolve cleaned
inputs and group globs, read/concatenate UTF-8 texts, apply text normalization,
remove and count reference tags, then perform NLP and the command-specific
analysis. `--dry-run` checks the static plan and currently present inputs but
does **not** execute Cleaner or NLP.

`analysis_unit` selects `lemma` or `surface` for Count. The `filters` controls,
Roman-numeral policy, and optional post-NLP lemma-normalization map affect the
final frequency Counter. Count, Features, and NLP-backed workflows share the
prepared text and chunked analysis-record extraction, but their eligibility
filters are **not identical**: the documented `filters.upos_targets` restriction
applies to Count, **not** Features.

The original README identifies the canonical token-analysis path as
`iter_nlp_analysis_records_from_text()` followed by
`evaluate_analysis_record()`. Reference-tag detection occurs during corpus
preparation and is not repeated by token evaluation.

## 2. Count outputs and reproducibility

- Base frequency files: `frequency_<label>.csv`.
- Optional dictionary-check files: `frequency_<label>.known.csv` and
  `frequency_<label>.unknown.csv`.
- Default run reports: `summary.txt` and `run_meta.json`.
- Optional partition, comparison, trace, and token-artifact outputs depend on
  configuration.

A complete, typed artifact inventory is assembled after prepared groups and
labels are resolved and before the NLP backend is started or output directories
are created. All output paths are checked for collisions together. Output
writers publish UTF-8 data through same-directory temporary files. The summary
is published before the run metadata; the latter lists generated outputs,
including itself.

A Count run archive contains generated outputs, a configuration snapshot,
`manifest.json`, and a short archive README. Diagnostic traces and formal token
artifacts are distinguished. Only files planned for that run are copied;
existing stale output CSVs are not included. By default source corpora are not
copied, and existing run directories are never overwritten. When configured,
wordlist path/size/SHA-256 are recorded in the manifest without copying the
wordlist.

## 3. Group partition validation

`validations.partitions` compares the **final** Counter of a configured whole
group with the sum of two or more part-group Counters. It runs after selection
of analysis units, token/POS filters, Roman-numeral handling, source
normalization, reference-tag removal, and dictionary lemma normalization.

`target_tokens` is the total of accepted tokens in the **final** frequency
table, not a raw Stanza token count. The check is an additive-frequency
consistency test, **not** proof that source strings were partitioned exactly.
When whole and part texts are processed separately, boundary context may alter
NLP lemma or POS decisions; splitting at sentence/clause boundaries and using
identical preprocessing reduces spurious mismatches.

For `on_mismatch: warn`, the mismatch is reported but the exit code remains 0.
For `on_mismatch: error`, output reports are still written, then exit code 1 is
returned. Per-file grouping is incompatible with this validation.

## 4. Group comparisons and keyness

Configured comparisons operate on final group Counters from a single Count
run. For a group-specific token count \(c\), total accepted tokens \(N\), and
configured scale \(s\), the documented normalized rate is:

\[
\mathrm{rate} = \frac{c}{N}\,s.
\]

Denominators count final accepted token totals after all relevant filters and
normalization—not whitespace tokens or raw NLP tokens. `log_ratio` is a base-2
relative-frequency effect size: positive indicates relative overrepresentation
in `group_a`, negative in `group_b`. Reversing the groups reverses `log_ratio`
and `direction`. Configured comparisons use `zero_correction` for log-ratio
zero-frequency handling, **not** for the rates or log likelihood.

`log_likelihood` is the G-squared statistic for a 2×2 table (target item versus
other items); it measures the strength of evidence for distributional
difference and has no direction. Reversing groups leaves it unchanged.
High-frequency words can have high likelihood with small ratio differences,
and rare words can have a large log ratio but small likelihood. The described
implementation has no p-values, q-values, or multiple-testing correction.

`nlpo compare` is a **different entry path**: it compares existing CSVs,
without NLP, and uses additive smoothing and scale 1. Its log ratio is
`log2(relative_a / relative_b)` after that smoothing. For three or more inputs,
it reports per-label counts/relative frequencies, largest/smallest labels, and
the relative-frequency range. Do not interpret the two comparison outputs as
using identical smoothing or normalization defaults.

## 5. Formal token artifacts and concordance

When `artifacts.tokens.enabled` is true, Count writes a complete token TSV and
strict metadata JSON. Metadata records the schema, row/byte counts,
included/excluded counts, and SHA-256; these values are validated against the
TSV when read. The TSV is published before metadata, which acts as the final
commit marker. A copied artifact can still be read from an archive: its actual
path need not match the informational original path in the metadata.

The canonical token-sequence boundary consists of **group, source file,
section, chunk, and sentence**. Tokens in a sequence are ordered by token
index, with global token index as deterministic tie-breaker. Concordance and
artifact-input N-gram use the same reader and sequence collection.

Concordance matches **included** tokens but can display excluded tokens in
surrounding context. It constructs context from formal token records; it does
not re-tokenize a sentence-metadata string. Group, source-file, and sentence
metadata are propagated when available. Diagnostic trace files, which may be
truncated or filtered, cannot substitute for formal token artifacts.

## 6. N-gram calculation

With formal token-artifact input, `nlpo ngram` supports both token and lemma
values and never crosses the group/file/section/chunk/sentence sequence
boundary. It uses included tokens only: punctuation-only or empty values end
the lexical run; excluded tokens are omitted **without** ending a run.
`--by-group` alters the destination Counter and output column, not the boundary.

With configuration/input-text input, only surface-token n-grams are supported.
The workflow reuses Cleaner, grouping, text normalization, and reference-tag
preparation but **does not** start an NLP backend. Each prepared corpus becomes
one token sequence, without fabricated sentence/chunk details. Failure to read
an input as UTF-8 stops preparation. Use token-artifact input for lemma n-grams.
Output columns are `ngram`, `count`, `n`, `field`, plus `group` when requested.

## 7. Basic Features table and windows

`nlpo features` produces one row per configured corpus, per source file, or per
fixed-length sample window. The pipeline retains complete raw post-NLP records
and creates one filtered lexical-record population used by basic lexical
statistics, UPOS counts, MFW features, and most derived calculations.
The `--field` selector changes MFW's field (`lemma` or `token`) **not** the
population of eligible lexical records.

`token_count` describes raw NLP records, while `word_token_count` counts
eligible lexical records. Sentence length is the number of eligible tokens in
a sentence identified by `(chunk_index, sentence_index)`; a sentence with no
eligible lexical tokens contributes zero. Surface-token length is the length
of the stripped lowercased surface string, **not** the lemma. Both sentence
and token length distributions include mean, population variance (divisor N),
median, and linearly interpolated quartiles. Empty distributions yield `0.0`.

Fixed-token windows count **eligible lexical tokens**, excluding punctuation,
short/empty tokens, and excluded Roman numerals. A window never crosses a
source-file boundary. `--step-tokens` can overlap samples; the option to include
a final partial window retains at most one trailing remainder. Sample metadata
includes `source_file`, stable `sample_id`, one-based `sample_index`,
`sample_kind`, and inclusive start/exclusive end eligible-token offsets.
Raw token/character/sentence statistics describe the span covered by the sample,
including filtered records within the span. Per-corpus MFW vocabulary is chosen
from unsampled filtered input, so overlapping windows do not reweight its
selection. With no windows, the ordinary one-row-per-corpus schema is retained.

## 8. Lexical diversity and function words

`--lexical-diversity` calculates **MATTR, MSTTR, MTLD, and HD-D**, for both
normalized surface tokens and lemmas. Documented defaults:

| Metric/parameter | Documented behavior |
| --- | --- |
| MATTR | Mean of overlapping moving-window type/token ratios |
| MSTTR | Mean of non-overlapping full-window ratios; a trailing partial segment is ignored if a full window exists |
| MTLD | Average of forward and reverse factor-based calculations |
| HD-D | Effective sample size is min(configured sample size, available token count) |
| Window size | `100` by default for MATTR/MSTTR |
| MTLD threshold | `0.72` by default |
| HD-D sample size | `42` by default |

For text shorter than a configured window, MATTR and MSTTR fall back to
whole-sequence TTR; empty sequences produce `0.0`. For fixed-token windows,
all metrics are calculated **within** each sample. The `--field` flag only
selects MFW field, not which lexical-diversity variants are emitted. Use the
same parameter values when comparing works.

Explicit function-word features load a reviewed UTF-8 file with one token per
line. Blank and `#` comment lines are ignored; duplicates after normalization,
tab-separated or multiword entries, and empty lists are invalid. Terms match
case-insensitively; `lemma` is the default field with surface fallback, while
`token` uses the normalized surface. Each `fw_<term>` value divides its count
by **all eligible lexical tokens in that row**, not just listed function words.
The list is distinct from UPOS-derived aggregate `function_word_count` and
`function_word_ratio`. No authoritative Latin function-word list is supplied or
inferred. One-character terms excluded by the configured minimum length can
remain at zero, and attached enclitics such as `-que`, `-ve`, `-ne` are not
split or suffix-matched.

## 9. Character n-grams and masking

Character n-grams operate on **prepared text**, not reconstructed NLP token
strings. They lowercase with Python `str.lower()` and do not normalize u/v,
i/j, ligatures, accents, or macrons. Modes:

| Mode | Characters retained |
| --- | --- |
| `full` | All characters; collapse Unicode whitespace runs to one ASCII space |
| `no-punctuation` | Replace Unicode punctuation (category `P*`) with spaces |
| `letters-spaces` | Letters/combining marks plus collapsed word boundaries |
| `letters-only` | Letters/combining marks concatenated, without former word boundaries |

The `--char-ngram-top` selection limit applies **per mode and per size**.
Candidate vocabulary is fitted by pooled counts from unsampled corpora,
frequency first then lexical tie-break. Longer works therefore influence the
selection more, while overlapping sampled windows do not affect which columns
are selected. Frequency values divide occurrence counts by all available
positions of the specified size, including positions for unselected terms.
N-grams may span word and sentence boundaries, but not a Features unit or
source-file boundary. The ordinary character family requires **one input file
per prepared corpus**. For fixed windows it slices the exact prepared text from
the first to last eligible token, retaining internal punctuation, excluded
tokens, and whitespace; missing exact offsets are errors, not invitations to
reconstruct token-joined text.

Content-masked character n-grams form a distinct UPOS-dependent family. They
mask `NOUN`, `PROPN`, `VERB`, `ADJ`, `ADV`, `NUM`, and `X`; retain the surface of
`AUX`, `PRON`, `ADP`, `CCONJ`, `SCONJ`, `PART`, and `DET`; and retain punctuation.
A word-like token without UPOS is masked conservatively. Letters, combining
marks, and numbers within a masked token span become `*`; punctuation,
whitespace, token boundaries, and code-point length remain observable. Masking
uses raw NLP records independent of Features lexical filtering or selected
MFW field. A backend treating an attached enclitic as one content token may
mask the whole span.

Column prefixes include `char3_` for ordinary full-mode trigrams,
`char_nopunct3_`, `char_letters_spaces3_`, `char_letters_only3_`, and
`char_masked3_`. Character feature vocabulary fitted to **all** corpora can
leak information from test works into table-input LOWO or verification. The
corpus-input variants refit vocabulary from references/training works only.
Punctuation patterns and orthography may also reflect the editor/edition, not
necessarily an author.

## 10. UPOS n-grams and morphological features

UPOS n-gram features count tag sequences of sizes 2 or 3. Tag values are
stripped and uppercased. Included lexical records appear in order and n-grams
never cross a `(chunk_index, sentence_index)` boundary. Missing/empty UPOS
values break the sequence; they do not create `UNK` tags. No BOS/EOS markers
are added. The selected vocabulary is fitted from all unsampled corpora by
pooled count, with lexical tie-breaking. Per-row values divide observed counts
by **all valid positions** in that row, including unselected patterns.

Morphological Features use UD `feats` supplied by the NLP backend, rather than
treebank XPOS or dependency relations. For each selected attribute:

- `morph_coverage_*` is the share of eligible lexical tokens with an annotation.
- `morph_value_*` is an annotated-value count over **all** eligible tokens.
- `morph_conditional_*` divides by tokens annotated for that attribute.
- `morph_other_*` and corresponding conditional columns capture values absent
  from the fitted vocabulary; missing annotations instead reduce coverage.
- `--morph-bundle-top` selects common combined bundles of selected attributes,
  with deterministic frequency and lexical ordering.

Models and package choices affect UPOS and morphology annotations. For
meaningful comparison, keep NLP model, edition, cleaning, filters, and window
settings consistent. Corpus-input LOWO fits morphological vocabularies using
training works only; ordinary Features fits from all inputs.

## 11. Distances and nearest neighbors

`stylometry delta` takes a **published feature table** (no NLP) and explicitly
selected columns. It computes sample z-scores with the **sample** standard
deviation (denominator `N - 1`), unlike Features' descriptive population
variance. Zero-variance features are dropped and reported; all-zero variance
fails the run. Burrows's Delta is the **mean absolute difference** between two
standardized feature vectors. Smaller distances indicate greater proximity,
not proof of common authorship.

`stylometry neighbors` fits one z-score model on all supplied rows and ranks
other observations for each query. Supported metrics include:

- `burrows_delta`: mean absolute standardized difference.
- `manhattan`: sum of absolute standardized differences.
- `cosine_delta`: 1 minus cosine similarity.
- `cosine_similarity`: cosine similarity itself (larger ranks first).

Cosine metrics fail for zero-norm vectors; they do not invent scores. Output is
directed, excludes self-neighbors, and sorts before `--top` is applied.
Ordinary Delta and neighbors are exploratory and fit on **all** supplied rows.

## 12. LOWO attribution and leakage controls

Table-input `evaluate-lowo` holds out **one whole work**, including all its
sample windows. It averages samples within each work so that all works have
equal weight. Each fold fits z-scores and zero-variance selection on training
work profiles only, builds one standardized centroid per author (work-weighted),
and assigns the held-out work to the nearest centroid by Burrows's Delta.
Each author needs at least two distinct works. The summary provides overall
work accuracy, per-author work accuracy, and unweighted macro author accuracy.
This is a **closed-set** test; it cannot reject a previously unseen author.

Table-input LOWO cannot undo selection leakage if the previously created CSV
already selected MFW, character n-grams, or UPOS n-grams using the held-out
work. A registered function-word list, external vocabulary, or corpus-input
fold-specific fitting is preferable for stringent tests.

`evaluate-lowo-corpus` analyzes configured texts once but refits automatic
MFW/character/UPOS and supported morphology vocabularies within each fold,
using training works only. All samples of each work remain in the same fold;
a vocabulary audit supplies selected terms and deterministic SHA-256. Its
metadata columns are `group,author,work`, one per prepared group. These are
still closed-set attribution results, not authenticity proofs.

## 13. Candidate-author verification

Table-input `stylometry verify` tests one query work against one candidate
author's known works plus background works. It produces **accept**, **reject**,
or **inconclusive**. It requires at least three candidate reference works and
two background works. Input metadata is per observation:
`sample_id,author,work`; the query author's field is a placeholder and does
not enter the decision.

Work profiles are equally weighted. The query is excluded while computing the
reference-only z-score fit, retained features, candidate centroid, calibration,
and thresholds. Genuine calibration uses leave-one-reference-work-out candidate
distances; impostor calibration compares background works with the full
candidate centroid. The configured genuine/impostor quantiles define two
thresholds: the **smaller** is the accept boundary and the **larger** the reject
boundary. Scores **between or equal to** these boundaries are inconclusive.
The nearest background work is diagnostic, not the decision rule.

`verify-corpus` also prevents the query from influencing automatic feature
vocabulary selection and writes a vocabulary audit. `evaluate-verification-
corpus` performs pseudo-unknown work evaluation, holding out each labeled work
in turn. It requires at least four candidate and three background works so the
ordinary verification minimum remains after each holdout. The summary reports
false accept/reject rates, inconclusive outcomes, coverage, decisive accuracy,
overall correct rate, and balanced correct rate. Quantiles must not be tuned on
the evaluation outcomes.

`verify-stability` recomputes table-input verification under explicit
resampling axes (`works`, `samples`, `features`) in that order. Work resampling
is stratified without replacement, within-work samples are bootstrapped, and
features are sampled from **already selected columns**. Each replication
refits z-scores, calibration and thresholds without the query. Its stability
status summarizes agreement under the chosen perturbation **only**, not a
formal confidence guarantee. It does not refit or repair upstream vocabulary
selection. All verification decisions are sensitive to period, genre,
background works, sample size, edition, and NLP quality.

## 14. Cache validity and result integrity

The analysis cache stores ordered **pre-filter NLP analysis records**, not
frequency Counters. Runs may reuse records while changing selection filters,
lemma normalization, dictionary checks, token artifacts, tracing, or archive
settings. It is invalidated by changes to prepared text, backend,
language/model/package, chunking strategy, schema version, or analysis behavior
version. Locks and atomic writes prevent incomplete objects from becoming
cache results; the cache is an optimization, not a stable research artifact.

The published token artifacts and run archives are the stable reusable research
outputs. Treat SHA-256 and recorded metadata as integrity/provenance checks,
not proof of correctness of the linguistic analysis.

## 15. Limits of the supplied specification

This document mirrors the supplied README's behavior descriptions; it does not
supply an exhaustive command/parser schema, a verified runtime environment,
or a formal specification for every input CSV/YAML column. Consult the
project's source, tests, generated config schema, and the preserved
[original README](legacy-readme.md) before changing backward compatibility.
