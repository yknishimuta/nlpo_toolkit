# nlpo_toolkit

A lightweight Python toolkit for processing and analyzing Latin (and other classical / scholarly) corpora. Designed to support digital-humanities research, NLP preprocessing, and reproducible vocabulary / frequency analysis workflows.

### Prerequisites

- Python **3.8** (or a reasonably recent 3.x)
- Recommended: use a virtual environment

### Installation

```
# Option 1: install via pip from PyPI or GitHub (if packaged)
pip install git+https://github.com/yknishimuta/nlpo_toolkit.git

# Option 2: install from source
git clone https://github.com/yknishimuta/nlpo_toolkit.git
cd nlpo_toolkit
pip install -r requirements.txt
```

### Count Vocabula CLI

`count_corpus_vocabula` is integrated as an `nlpo_toolkit` subcommand. The
package and repository name remain `nlpo_toolkit`.

```
nlpo count-vocabula --project-root . --config config/groups.config.yml
nlpo count --project-root . --config config/groups.config.yml
```

`--project-root` is used to resolve relative paths in the YAML config. If
`--config` is omitted, the CLI uses `<project-root>/config/groups.config.yml`.
The legacy script name is kept as a wrapper:

```
python count_corpus_vocabula_local.py --project-root . --config config/groups.config.yml
```

## License

Licensed under the **MIT License** 
