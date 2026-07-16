# Latin Text Cleaners

This module provides configurable cleaning utilities for Latin texts, with a focus on:

- **Corpus Corporum** exports
- **Scholastic texts** (e.g. Summa Theologiae)

------

## Installation

Install this module as part of the project with `python -m pip install .`.
Dependencies are declared in the repository `pyproject.toml`.

## Usage

This script cleans a Latin text using a YAML configuration file that specifies:

- the **kind** of cleaner (e.g., `"corpus_corporum"` or `"scholastic_text"`)
- the **input** file path
- the **output** file path

The CLI validates the config once and delegates the run to the typed Cleaner
application service.

### Basic Usage

From the project root (or from the module path):

```
python -m nlpo_toolkit.latin.cleaners.run_clean_corpus
```

By default, it uses:

```
nlpo_toolkit/latin/cleaners/config/sample.yml
```

------

# Using a Custom Config File

You can specify a different YAML config file by passing the path as an argument:

```
python -m nlpo_toolkit.latin.cleaners.run_clean_corpus \
    /path/to/your/custom_config.yml
```

By default, the script uses a sample config under the cleaners directory.

------

# YAML Config Format

A configuration file should look like:

```
kind: corpus_corporum          # cleaner type ("corpus_corporum", "scholastic_text", etc.)
input: ../input/input_file.txt 
output: ../output/cleaned.txt
```

## Internal design

- `rule_loader` validates YAML and constructs an immutable, typed `RuleSet`.
- `rule_engine` applies corpus-independent line removal and substitution rules.
- `corpora` profiles define body extraction, default rules, and corpus-specific
  line finalization.
- `pipeline` combines a profile, rules, common normalization, and the lexicon
  map without performing filesystem writes.
- `service.execute_cleaner()` accepts a `CleanerExecutionRequest`, plans and
  validates every output, builds the Cleaner program once, performs file I/O,
  writes the run's event TSV, and returns a typed `CleanerExecutionResult`.
- `run_clean_corpus` is only the standalone CLI adapter: it parses the optional
  config path, inspects the config once, calls the service, presents the result,
  and converts domain errors to exit codes.
- Directory-mode `output_filename_template` values are always respected; they
  are never replaced based on whether input stems happen to collide.
- Corpus analysis reuses the inspection created during planning and calls the
  service directly. It never calls this CLI module.
