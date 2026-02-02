# Latin Text Cleaners

This module provides configurable cleaning utilities for Latin texts, with a focus on:

- **Corpus Corporum** exports
- **Scholastic texts** (e.g. Summa Theologiae)

------

## Installation

This module only requires standard library plus **PyYAML**:

```
pip install pyyaml
```

## Usage

This script cleans a Latin text using a YAML configuration file that specifies:

- the **kind** of cleaner (e.g., `"corpus_corporum"` or `"scholastic_text"`)
- the **input** file path
- the **output** file path

The script loads the config, reads the text, applies the appropriate cleaning rules, and writes the cleaned file.

### Basic Usage

From the project root (or from the module path):

```
python -m nlpo_toolkit.latin.cleaners.run_clean_corpus
```

By default, it uses:

```
clean_configs/sample.yml
```

as defined in:

```
DEFAULT_CONFIG: Path = BASE_DIR.parent / "cleaners" / "config" /"sample.yml"
```

------

# Using a Custom Config File

You can specify a different YAML config file by passing the path as an argument:

```
python -m nlpo_toolkit.latin.cleaners.run_clean_corpus \
    /path/to/your/custom_config.yml
```

or, if calling the script directly:

```
python nlpo_toolkit/latin/cleaners/run_clean_config.py \
    /path/to/config.yml
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