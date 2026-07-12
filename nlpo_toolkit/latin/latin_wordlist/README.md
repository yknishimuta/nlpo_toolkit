# Latin Wordlist Builder

This development tool builds a Latin wordlist from CoNLL-U treebanks, plain
text corpora, and optional existing wordlists.

All inputs, filtering values, and the output path are declared in a YAML file.
The packaged `config/latin_wordlist.yml` is a template: copy it to a working
directory and adjust its project-relative paths before running the builder.

```bash
python -m nlpo_toolkit.latin.latin_wordlist.build_latin_wordlist \
  --config path/to/latin_wordlist.yml
```

The YAML sections are:

- `inputs`: `conllu_dir`, `latin_text_dir`, and `extra_wordlists`
- `output`: `latin_wordlist_out`
- `filters`: `min_length`, `min_form_freq`, and `min_text_freq`
- `tokenize`: optional `extra_punct`

Missing optional input directories or wordlists are reported and skipped. The
output is a sorted UTF-8 text file containing one entry per line.
