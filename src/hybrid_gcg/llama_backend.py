"""Small authoritative llama.cpp backend used by bootstrap and validation."""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import ModelConfig
from .objectives import SequenceScore, TokenScore, score_logits


@dataclass(frozen=True)
class GreedyResult:
    token_ids: tuple[int, ...]
    stopped_on_eog: bool


class LlamaBackend:
    """One-sequence llama.cpp wrapper with full and prefix-reused scoring."""

    def __init__(self, config: ModelConfig) -> None:
        if not config.gguf_path.is_file():
            raise FileNotFoundError(config.gguf_path)
        try:
            import llama_cpp
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "llama-cpp-python is required; see the README installation notes"
            ) from exc

        self._module = llama_cpp
        self.version = importlib.metadata.version("llama-cpp-python")
        self.llm = llama_cpp.Llama(
            model_path=str(config.gguf_path),
            n_ctx=config.n_ctx,
            n_batch=config.n_batch,
            n_ubatch=config.n_ubatch,
            n_gpu_layers=config.n_gpu_layers,
            logits_all=False,
            verbose=config.verbose,
        )
        self.n_vocab = int(self.llm.n_vocab())

    def tokenize(self, text: str, add_bos: bool = False) -> tuple[int, ...]:
        return tuple(
            int(token)
            for token in self.llm.tokenize(
                text.encode("utf-8"),
                add_bos=bool(add_bos),
                special=True,
            )
        )

    def detokenize(self, token_ids: Sequence[int], *, special: bool = True) -> str:
        raw = self.llm.detokenize(list(map(int, token_ids)), special=special)
        return raw.decode("utf-8", errors="strict")

    def token_piece(self, token_id: int) -> str:
        return self.detokenize([int(token_id)], special=True)

    def is_eog(self, token_id: int) -> bool:
        low = self._module.llama_cpp
        return bool(
            low.llama_vocab_is_eog(self.llm._model.vocab, int(token_id))
        )

    def reset_and_eval(self, token_ids: Sequence[int]) -> None:
        tokens = list(map(int, token_ids))
        if not tokens:
            raise ValueError("Cannot evaluate an empty context")
        self.llm.reset()
        self.llm.eval(tokens)

    def _truncate_to(self, n_tokens: int, fallback_prefix: Sequence[int]) -> bool:
        if n_tokens <= 0:
            return False
        try:
            ok = bool(self.llm._ctx.kv_cache_seq_rm(-1, int(n_tokens), -1))
        except Exception:
            ok = False
        if ok:
            self.llm.n_tokens = int(n_tokens)
            if hasattr(self.llm, "_requires_eval"):
                self.llm._requires_eval = True
            return True
        self.reset_and_eval(fallback_prefix)
        return False

    def current_logits(self, *, copy: bool = True) -> np.ndarray:
        if int(self.llm.n_tokens) <= 0:
            raise RuntimeError("No tokens have been evaluated")
        pointer = self.llm._ctx.get_logits()
        if not pointer:
            raise RuntimeError("llama.cpp returned a null logits pointer")
        view = np.ctypeslib.as_array(pointer, shape=(self.n_vocab,))
        return np.array(view, copy=True) if copy else view

    def score_next(self, context_ids: Sequence[int], target_id: int) -> TokenScore:
        self.reset_and_eval(context_ids)
        return score_logits(self.current_logits(), target_id)

    def score_next_panel(
        self,
        contexts: Sequence[Sequence[int]],
        *,
        common_prefix_length: int,
        target_id: int,
    ) -> list[TokenScore]:
        """Score a panel while reusing only its byte-identical token prefix.

        This is a shortlist scorer. Search promotion always repeats scoring with
        ``score_next`` from a clean full context.
        """
        if not contexts:
            return []
        prefix_length = int(common_prefix_length)
        first = tuple(map(int, contexts[0]))
        if prefix_length <= 0:
            return [self.score_next(context, target_id) for context in contexts]
        prefix = first[:prefix_length]
        if not prefix:
            raise ValueError("Common prefix unexpectedly empty")
        for context in contexts:
            row = tuple(map(int, context))
            if row[:prefix_length] != prefix:
                raise ValueError("Panel contexts do not share the requested prefix")

        self.reset_and_eval(prefix)
        scores: list[TokenScore] = []
        for index, context in enumerate(contexts):
            if index and not self._truncate_to(prefix_length, prefix):
                pass
            tail = list(map(int, context[prefix_length:]))
            if not tail:
                raise ValueError("Candidate tail unexpectedly empty")
            self.llm.eval(tail)
            scores.append(score_logits(self.current_logits(), target_id))
        return scores

    def score_sequence(
        self,
        context_ids: Sequence[int],
        target_ids: Sequence[int],
    ) -> SequenceScore:
        targets = tuple(map(int, target_ids))
        if not targets:
            raise ValueError("Target sequence cannot be empty")
        self.reset_and_eval(context_ids)
        rows: list[TokenScore] = []
        for index, target in enumerate(targets):
            rows.append(score_logits(self.current_logits(), target))
            if index + 1 < len(targets):
                self.llm.eval([target])
        return SequenceScore(tuple(rows))

    def greedy_decode(
        self,
        context_ids: Sequence[int],
        *,
        max_tokens: int,
    ) -> GreedyResult:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self.reset_and_eval(context_ids)
        output: list[int] = []
        for _ in range(max_tokens):
            token = int(np.argmax(self.current_logits(copy=False)))
            output.append(token)
            if self.is_eog(token):
                return GreedyResult(tuple(output), True)
            self.llm.eval([token])
        return GreedyResult(tuple(output), False)

    def close(self) -> None:
        close = getattr(self.llm, "close", None)
        if callable(close):
            close()


def gguf_display_name(path: Path) -> str:
    return path.name
