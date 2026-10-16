"""A deterministic pseudo-tokenizer used only to keep the demo self-contained."""

from __future__ import annotations

import re


class PseudoTokenizer:
    """Split words and punctuation, assigning IDs in first-seen order."""

    def __init__(self) -> None:
        self.token_to_id = {"<mask>": 0}
        self.id_to_token = {0: "<mask>"}

    @property
    def mask_token_id(self) -> int:
        return 0

    def encode(self, text: str) -> list[int]:
        pieces = re.findall(r"\w+|[^\w\s]", text)
        ids = []
        for piece in pieces:
            key = piece.lower()
            if key not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[key] = idx
                self.id_to_token[idx] = piece
            ids.append(self.token_to_id[key])
        return ids

    def tokens(self, ids: list[int]) -> list[str]:
        return [self.id_to_token[idx] for idx in ids]
