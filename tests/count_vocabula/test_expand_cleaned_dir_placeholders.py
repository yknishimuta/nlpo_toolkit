from __future__ import annotations

from pathlib import Path

from nlpo_toolkit.count_vocabula.preprocess import expand_cleaned_dir_placeholders


def test_expand_cleaned_dir_noop_when_none():
    patterns = ["data/*.txt", "{cleaned_dir}/*.txt"]
    out = expand_cleaned_dir_placeholders(patterns, None)
    assert out == patterns


def test_expand_cleaned_dir_replaces_placeholder():
    patterns = ["{cleaned_dir}/*.txt", "other/*.txt"]
    cleaned_dir = Path("/tmp/cleaned")
    out = expand_cleaned_dir_placeholders(patterns, cleaned_dir)
    assert out[0] == "/tmp/cleaned/*.txt"
    assert out[1] == "other/*.txt"


def test_expand_cleaned_dir_does_not_break_other_braces():
    patterns = ["{cleaned_dir}/*.txt", "{not_a_placeholder}/*.txt"]
    cleaned_dir = Path("/tmp/cleaned")
    out = expand_cleaned_dir_placeholders(patterns, cleaned_dir)

    assert out[0] == "/tmp/cleaned/*.txt"
    assert out[1] == "{not_a_placeholder}/*.txt"
