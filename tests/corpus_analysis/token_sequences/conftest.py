from dataclasses import dataclass

import pytest


@dataclass(frozen=True)
class Item:
    group: str = "g"
    source_file: str | None = "a.txt"
    section: str | None = "s"
    chunk_index: int = 0
    sentence_index: int = 0
    token_index: int = 0
    global_token_index: int = 0
    sentence: str = "metadata"
    token: str = "arma"
    lemma: str | None = "arma"
    included: bool = True


@pytest.fixture
def item_type():
    return Item
