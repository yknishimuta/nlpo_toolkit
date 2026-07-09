from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from nlpo_toolkit.corpus_analysis.config import ensure_app_config
from nlpo_toolkit.corpus_analysis.corpus import (
    CorpusPreparationError,
    CorpusWorkItem,
    build_corpus_work_items,
    cleaned_txt_files,
    label_from_file,
    prepare_corpora,
    prepare_corpus_text,
    resolve_auto_single_cleaned_group,
    resolve_corpus_work_items,
    resolve_group_files,
)
from nlpo_toolkit.corpus_analysis.config import load_config


def _config(data: dict):
    return ensure_app_config(data)


def test_resolve_group_files_project_relative_absolute_cleaned_and_empty(tmp_path: Path) -> None:
    project_root = tmp_path
    input_dir = project_root / "input"
    cleaned_dir = project_root / "cleaned"
    input_dir.mkdir()
    cleaned_dir.mkdir()
    a = input_dir / "a.txt"
    b = input_dir / "b.txt"
    c = cleaned_dir / "c.txt"
    for path in (b, a, c):
        path.write_text(path.name, encoding="utf-8")

    cfg = _config(
        {
            "groups": {
                "relative": {"files": ["input/*.txt"]},
                "absolute": {"files": [str(a)]},
                "cleaned": {"files": ["{cleaned_dir}/*.txt"]},
                "empty": {"files": ["missing/*.txt"]},
            }
        }
    )

    group_files = resolve_group_files(
        groups=cfg.groups,
        project_root=project_root,
        cleaned_dir=cleaned_dir,
    )

    assert group_files["relative"] == (a.resolve(), b.resolve())
    assert group_files["absolute"] == (a.resolve(),)
    assert group_files["cleaned"] == (c.resolve(),)
    assert group_files["empty"] == ()


def test_resolve_corpus_work_items_errors_on_empty_group(tmp_path: Path) -> None:
    cfg = _config({"groups": {"empty": {"files": ["missing/*.txt"]}}})

    with pytest.raises(CorpusPreparationError, match="No files matched"):
        resolve_corpus_work_items(
            config=cfg,
            project_root=tmp_path,
            cleaned_dir=None,
            error_on_empty_group=True,
        )


def test_group_by_file_deduplicates_and_labels_deterministically(tmp_path: Path) -> None:
    one = tmp_path / "group one.txt"
    two = tmp_path / "sub" / "group one.txt"
    two.parent.mkdir()
    one.write_text("one", encoding="utf-8")
    two.write_text("two", encoding="utf-8")
    group_files = {
        "group_a": (one, two),
        "group_b": (one,),
    }

    items = build_corpus_work_items(group_files=group_files, group_by_file=True)

    assert label_from_file(one) == "group_one"
    assert [item.label for item in items] == ["group_one", "group_one_2"]
    assert [item.files for item in items] == [(one.resolve(),), (two.resolve(),)]


def test_auto_single_cleaned_success_and_errors(tmp_path: Path) -> None:
    cleaned_dir = tmp_path / "cleaned"
    cleaned_dir.mkdir()
    (cleaned_dir / ".DS_Store").write_text("ignored", encoding="utf-8")
    (cleaned_dir / ".gitkeep").write_text("", encoding="utf-8")

    assert cleaned_txt_files(cleaned_dir) == ()
    with pytest.raises(CorpusPreparationError, match="no \\.txt files"):
        resolve_auto_single_cleaned_group(cleaned_dir=cleaned_dir, group_name="corpus_a")

    selected = cleaned_dir / "sample_text_a.txt"
    selected.write_text("text", encoding="utf-8")
    assert cleaned_txt_files(cleaned_dir) == (selected.resolve(),)
    assert resolve_auto_single_cleaned_group(
        cleaned_dir=cleaned_dir,
        group_name="corpus_a",
    )["corpus_a"] == (selected.resolve(),)

    extra = cleaned_dir / "sample_text_b.txt"
    extra.write_text("text", encoding="utf-8")
    with pytest.raises(CorpusPreparationError, match="found 2"):
        resolve_auto_single_cleaned_group(cleaned_dir=cleaned_dir, group_name="corpus_a")


def test_prepare_corpus_text_normalizes_and_counts_ref_tags(tmp_path: Path) -> None:
    text_path = tmp_path / "sample_text_a.txt"
    text_path.write_text("VITA ref. item_a", encoding="utf-8")
    ref_path = tmp_path / "ref_tags.txt"
    ref_path.write_text("ref\tref\\.\n", encoding="utf-8")
    cfg = _config(
        {
            "groups": {"corpus_a": {"files": [str(text_path)]}},
            "normalization": {"casefold": True},
            "ref_tags": {"enabled": True, "patterns": str(ref_path)},
        }
    )

    prepared = prepare_corpora(
        work_items=(CorpusWorkItem("corpus_a", (text_path,)),),
        config=cfg,
        project_root=tmp_path,
    )[0]

    assert prepared.files == (text_path,)
    assert prepared.raw_text == "VITA ref. item_a"
    assert "ref." not in prepared.prepared_text
    assert "vita" in prepared.prepared_text
    assert prepared.ref_tag_counts == Counter({"ref": 1})


def test_prepare_corpus_text_uses_supplied_patterns_without_reload(tmp_path: Path, monkeypatch) -> None:
    text_path = tmp_path / "sample_text_a.txt"
    text_path.write_text("ref. item_a", encoding="utf-8")
    ref_path = tmp_path / "ref_tags.txt"
    ref_path.write_text("ref\tref\\.\n", encoding="utf-8")
    cfg = _config(
        {
            "groups": {"corpus_a": {"files": [str(text_path)]}},
            "ref_tags": {"enabled": True, "patterns": str(ref_path)},
        }
    )
    patterns = tuple(__import__("nlpo_toolkit.corpus_analysis.ref_tags", fromlist=["load_ref_tag_patterns"]).load_ref_tag_patterns(ref_path))

    import nlpo_toolkit.corpus_analysis.corpus as corpus_mod

    monkeypatch.setattr(corpus_mod, "load_ref_tag_patterns", lambda path: pytest.fail("patterns reloaded"))
    prepared = prepare_corpus_text(
        work_item=CorpusWorkItem("corpus_a", (text_path,)),
        config=cfg,
        ref_tag_patterns=patterns,
    )

    assert prepared.ref_tag_counts == Counter({"ref": 1})


def test_missing_ref_tag_file_is_error(tmp_path: Path) -> None:
    text_path = tmp_path / "sample_text_a.txt"
    text_path.write_text("item_a", encoding="utf-8")
    cfg = _config(
        {
            "groups": {"corpus_a": {"files": [str(text_path)]}},
            "ref_tags": {"enabled": True, "patterns": "missing_ref_tags.txt"},
        }
    )

    with pytest.raises(CorpusPreparationError, match="pattern file was not found"):
        prepare_corpora(
            work_items=(CorpusWorkItem("corpus_a", (text_path,)),),
            config=cfg,
            project_root=tmp_path,
        )


def test_count_features_and_ngram_config_receive_same_prepared_text(tmp_path: Path, monkeypatch) -> None:
    import nlpo_toolkit.corpus_analysis.features as features_mod
    import nlpo_toolkit.corpus_analysis.ngram as ngram_mod
    import nlpo_toolkit.corpus_analysis.runner as runner_mod

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    text_path = input_dir / "corpus_a.txt"
    text_path.write_text("VITA ref. item_a", encoding="utf-8")
    ref_path = tmp_path / "ref_tags.txt"
    ref_path.write_text("ref\tref\\.\n", encoding="utf-8")
    config_path = tmp_path / "groups.config.yml"
    config_path.write_text(
        "\n".join(
            [
                "groups:",
                "  corpus_a:",
                "    files:",
                "      - input/corpus_a.txt",
                "normalization:",
                "  casefold: true",
                "ref_tags:",
                "  enabled: true",
                "  patterns: ref_tags.txt",
                "out_dir: output",
                "",
            ]
        ),
        encoding="utf-8",
    )
    received: dict[str, str] = {}

    def count_group_fn(text, nlp, **kwargs):
        received["count"] = text
        return Counter({"item_a": 1})

    runner_mod.run(
        project_root=tmp_path,
        config_path=config_path,
        load_config_fn=load_config,
        clean_mod=object(),
        build_pipeline_fn=lambda *a, **k: (object(), "package_a"),
        build_sentence_splitter_fn=None,
        count_group_fn=count_group_fn,
        render_stanza_package_table_fn=lambda *a, **k: [],
    )

    class CaptureNLP:
        def __call__(self, text):
            received["features"] = text
            return type("Doc", (), {"sentences": []})()

    features_mod.run_features(
        project_root=tmp_path,
        config_path=config_path,
        out=tmp_path / "features.csv",
        build_pipeline_fn=lambda *a, **k: (CaptureNLP(), "package_a"),
        clean_mod=object(),
        load_config_fn=load_config,
    )

    def capture_rows_from_text(text: str, group: str):
        received["ngram"] = text
        return [{"group": group, "token": "item_a"}]

    monkeypatch.setattr(ngram_mod, "_rows_from_text", capture_rows_from_text)
    ngram_mod.write_ngrams_from_config(
        project_root=tmp_path,
        config_path=config_path,
        n=1,
        field="token",
        by_group=True,
        min_count=1,
        top=None,
        output_format="tsv",
        out_path=tmp_path / "ngrams.tsv",
        clean_mod=object(),
    )

    assert received["count"] == received["features"] == received["ngram"]
    assert " ".join(received["count"].split()) == "vita item_a"
