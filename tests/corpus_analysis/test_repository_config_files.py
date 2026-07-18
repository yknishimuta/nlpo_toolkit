from pathlib import Path

from nlpo_toolkit.corpus_analysis.config import load_config
from nlpo_toolkit.corpus_analysis.postprocessing.lemma_normalization import (
    apply_lemma_normalization,
)
from nlpo_toolkit.corpus_analysis.postprocessing.lemma_normalization_io import (
    load_lemma_normalization_map,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LEMMA_NORMALIZATION_PATH = REPOSITORY_ROOT / "config" / "lemma_normalize.tsv"


def test_repository_lemma_normalization_file_exists_and_matches_config() -> None:
    config = load_config(REPOSITORY_ROOT / "config" / "groups.config.yml")

    assert config.dictcheck.lemma_normalize == "config/lemma_normalize.tsv"
    assert LEMMA_NORMALIZATION_PATH.is_file()
    assert (REPOSITORY_ROOT / config.dictcheck.lemma_normalize).resolve() == (
        LEMMA_NORMALIZATION_PATH.resolve()
    )


def test_repository_lemma_normalization_file_is_valid_noop() -> None:
    normalization = load_lemma_normalization_map(LEMMA_NORMALIZATION_PATH)

    assert normalization == {}
    assert dict(apply_lemma_normalization({"arma": 2, "vir": 1}, normalization)) == {
        "arma": 2,
        "vir": 1,
    }


def test_repository_lemma_normalization_file_encoding_and_layout() -> None:
    raw = LEMMA_NORMALIZATION_PATH.read_bytes()

    assert not raw.startswith(b"\xef\xbb\xbf")
    assert b"\r\n" not in raw
    assert raw.endswith(b"\n")
    assert all(
        not line.strip() or line.lstrip().startswith(b"#")
        for line in raw.splitlines()
    )
