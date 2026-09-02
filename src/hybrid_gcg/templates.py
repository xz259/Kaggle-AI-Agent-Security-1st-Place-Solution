"""Text templates with explicit, tokenizer-checked prompt boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


PROMPT = "{{PROMPT}}"
HOP1 = "{{HOP1}}"
HOP1_VISIBLE = "{{HOP1_VISIBLE}}"


class TokenBoundaryError(ValueError):
    """The prompt does not form an independent token span in a template."""


@dataclass(frozen=True)
class RenderedTemplate:
    text: str
    token_ids: tuple[int, ...]
    prompt_span: tuple[int, int]


@dataclass(frozen=True)
class TextTemplate:
    text: str

    @classmethod
    def load(cls, path: str | Path) -> "TextTemplate":
        return cls(Path(path).read_text(encoding="utf-8"))

    def _parts(
        self,
        *,
        hop1: str = "",
        hop1_visible: str = "",
    ) -> tuple[str, str]:
        if self.text.count(PROMPT) != 1:
            raise ValueError(f"Template must contain {PROMPT} exactly once")
        expanded = self.text.replace(HOP1_VISIBLE, hop1_visible).replace(HOP1, hop1)
        return tuple(expanded.split(PROMPT, 1))  # type: ignore[return-value]

    def render(
        self,
        prompt: str,
        *,
        hop1: str = "",
        hop1_visible: str = "",
    ) -> str:
        prefix, suffix = self._parts(hop1=hop1, hop1_visible=hop1_visible)
        return prefix + prompt + suffix

    def render_token_aligned(
        self,
        prompt: str,
        tokenize: Callable[[str, bool], Sequence[int]],
        *,
        add_bos: bool,
        hop1: str = "",
        hop1_visible: str = "",
    ) -> RenderedTemplate:
        prefix, suffix = self._parts(hop1=hop1, hop1_visible=hop1_visible)
        text = prefix + prompt + suffix
        full_ids = tuple(int(x) for x in tokenize(text, add_bos))
        prefix_ids = tuple(int(x) for x in tokenize(prefix, add_bos))
        prompt_ids = tuple(int(x) for x in tokenize(prompt, False))
        suffix_ids = tuple(int(x) for x in tokenize(suffix, False))
        reconstructed = prefix_ids + prompt_ids + suffix_ids
        if reconstructed != full_ids:
            mismatch = next(
                (
                    index
                    for index, pair in enumerate(zip(reconstructed, full_ids))
                    if pair[0] != pair[1]
                ),
                min(len(reconstructed), len(full_ids)),
            )
            raise TokenBoundaryError(
                "The template/prompt boundary changes tokenization at token "
                f"{mismatch}. Add an explicit model-appropriate delimiter around "
                f"{PROMPT}."
            )
        start = len(prefix_ids)
        return RenderedTemplate(
            text=text,
            token_ids=full_ids,
            prompt_span=(start, start + len(prompt_ids)),
        )
