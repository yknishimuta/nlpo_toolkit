from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class NLPToken:
    text: str
    lemma: Optional[str]
    upos: str
    start_char: int = 0

@dataclass
class NLPSentence:
    tokens: List[NLPToken] = field(default_factory=list)
    text: Optional[str] = None

@dataclass
class NLPDocument:
    sentences: List[NLPSentence] = field(default_factory=list)
    text: Optional[str] = None