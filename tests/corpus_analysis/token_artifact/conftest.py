from nlpo_toolkit.corpus_analysis.analysis_records import TokenRecord


def make_record(**overrides) -> TokenRecord:
    values = {
        "group": "text", "source_file": "input/text.txt", "section": None,
        "chunk_index": 0, "sentence_index": 0, "token_index": 0,
        "global_token_index": 0, "char_start_in_chunk": 0,
        "char_end_in_chunk": 6, "char_start_in_text": 0,
        "char_end_in_text": 6, "sentence": "Puella amat.", "token": "Puella",
        "lemma": "puella", "upos": "NOUN", "analysis_key": "puella",
        "included": True, "exclusion_reason": None, "ref_tag": None,
    }
    values.update(overrides)
    return TokenRecord(**values)
