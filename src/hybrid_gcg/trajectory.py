"""Serializable baseline trajectory captured from llama.cpp."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class Trajectory:
    schema_version: int
    prompt_text: str
    prompt_token_ids: tuple[int, ...]
    hop1_context_text: str
    hop1_context_ids: tuple[int, ...]
    hop1_prompt_span: tuple[int, int]
    hop1_output_ids: tuple[int, ...]
    hop1_text: str
    hop1_visible_text: str
    hop2_context_text: str
    hop2_context_ids: tuple[int, ...]
    hop2_prompt_span: tuple[int, int]
    target_eog_text: str
    target_eog_id: int
    llama_cpp_version: str
    gguf_name: str
    gguf_size_bytes: int
    gguf_vocab_size: int

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"Unsupported trajectory schema {self.schema_version}")
        if not self.prompt_token_ids or not self.hop1_output_ids:
            raise ValueError("Trajectory is missing prompt or hop-1 tokens")
        if self.gguf_size_bytes <= 0 or self.gguf_vocab_size <= 0:
            raise ValueError("Trajectory has invalid GGUF provenance")
        for context, span, label in (
            (self.hop1_context_ids, self.hop1_prompt_span, "hop1"),
            (self.hop2_context_ids, self.hop2_prompt_span, "hop2"),
        ):
            start, end = span
            if start < 0 or end < start or end > len(context):
                raise ValueError(f"Invalid {label} prompt span {span}")
            if tuple(context[start:end]) != self.prompt_token_ids:
                raise ValueError(f"{label} prompt span does not match prompt tokens")

    def context_with_prompt(
        self, hop: int, prompt_ids: Sequence[int]
    ) -> tuple[int, ...]:
        prompt = tuple(int(token) for token in prompt_ids)
        if len(prompt) != len(self.prompt_token_ids):
            raise ValueError("GCG candidates must preserve prompt token length")
        if hop == 1:
            context, span = self.hop1_context_ids, self.hop1_prompt_span
        elif hop == 2:
            context, span = self.hop2_context_ids, self.hop2_prompt_span
        else:
            raise ValueError("hop must be 1 or 2")
        start, end = span
        return tuple(context[:start]) + prompt + tuple(context[end:])

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "prompt_token_ids",
            "hop1_context_ids",
            "hop1_prompt_span",
            "hop1_output_ids",
            "hop2_context_ids",
            "hop2_prompt_span",
        ):
            payload[key] = list(payload[key])
        return payload

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Trajectory":
        tuple_fields = (
            "prompt_token_ids",
            "hop1_context_ids",
            "hop1_prompt_span",
            "hop1_output_ids",
            "hop2_context_ids",
            "hop2_prompt_span",
        )
        values = dict(payload)
        for key in tuple_fields:
            values[key] = tuple(int(value) for value in values[key])
        trajectory = cls(**values)
        trajectory.validate()
        return trajectory

    @classmethod
    def load(cls, path: str | Path) -> "Trajectory":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
