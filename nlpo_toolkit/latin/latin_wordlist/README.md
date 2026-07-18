# Latin Wordlist Builder

This independent application builds a general-purpose Latin wordlist from
CoNLL-U treebanks, plain-text corpora, and optional existing wordlists.

```bash
python -m nlpo_toolkit.latin.latin_wordlist \
  --config path/to/latin_wordlist.yml
```

The strict YAML configuration has four sections: `inputs` (`conllu_dir`,
`latin_text_dir`, `extra_wordlists`), `output` (`latin_wordlist_out`), `filters`
(`min_length`, `min_form_freq`, `min_text_freq`), and `tokenize`
(`extra_punct`). Unknown keys are rejected, thresholds must be positive, and
relative paths are resolved from the configuration file's directory. An output
`.txt` file inside `latin_text_dir` is rejected to prevent self-ingestion.

Missing source directories and extra wordlists are reported as warnings and
skipped. Existing sources must be readable strict UTF-8; decoding or read
failures stop the run. CoNLL-U files are traversed and opened once while both
lemma candidates and form frequencies are collected. The vocabulary keeps the
existing lowercase, alphabetic, punctuation, threshold, union, and lexical-sort
semantics.

The application service consumes typed collector and publication ports and
returns a typed result containing notices and source statistics. It performs no
display or filesystem work. The CLI alone renders messages and maps errors to
exit codes. Publication writes a temporary file beside the destination and
atomically replaces the destination only after a complete UTF-8 write.
