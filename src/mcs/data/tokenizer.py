"""Thin wrapper around a tiktoken encoding (GPT-2 BPE by default).

Two entry points on purpose:
- ``encode`` is for user text. Special tokens are refused, so a prompt that
  contains the literal string ``<|endoftext|>`` cannot end a document early
  (tiktoken raises ``ValueError`` by default; we never pass ``allowed_special``).
- ``encode_document`` is for pretraining data and appends the end-of-text id
  so documents stay separated after they are concatenated.
"""

import tiktoken


class Tokenizer:
    def __init__(self, name: str = "gpt2") -> None:
        self._enc = tiktoken.get_encoding(name)

    @property
    def vocab_size(self) -> int:
        return self._enc.n_vocab

    @property
    def eot_id(self) -> int:
        return self._enc.eot_token

    def encode(self, text: str) -> list[int]:
        return self._enc.encode(text)

    def encode_document(self, text: str) -> list[int]:
        return [*self.encode(text), self.eot_id]

    def decode(self, ids: list[int]) -> str:
        return self._enc.decode(ids)
