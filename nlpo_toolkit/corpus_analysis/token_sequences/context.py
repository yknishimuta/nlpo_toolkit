from __future__ import annotations

from dataclasses import dataclass

from .models import SequenceItem, TokenLocation


@dataclass(frozen=True)
class TokenContext:
    left: tuple[str, ...]
    node: str
    right: tuple[str, ...]

    def left_text(self) -> str:
        return " ".join(self.left)

    def right_text(self) -> str:
        return " ".join(self.right)


def build_token_context(
    location: TokenLocation[SequenceItem], *, window: int,
) -> TokenContext:
    if window < 0:
        raise ValueError("window must be zero or greater.")
    items = location.sequence.items
    offset = location.offset
    return TokenContext(
        left=tuple(item.token for item in items[max(0, offset - window):offset]),
        node=items[offset].token,
        right=tuple(item.token for item in items[offset + 1:offset + 1 + window]),
    )
