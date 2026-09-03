"""BF16 HotFlip rankings and Faster-GCG-style radius-1 sampling."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .config import ModelConfig
from .trajectory import Trajectory


@dataclass(frozen=True)
class Rankings:
    positions: tuple[int, ...]
    token_ids: np.ndarray
    predicted_deltas: np.ndarray
    loss: float
    competitor_id: int


@dataclass(frozen=True)
class Proposal:
    position: int
    replacement_id: int
    gradient_rank: int
    predicted_delta: float


def transformers_dtype_keyword(version: str) -> str:
    """Return the non-deprecated model-loading dtype keyword.

    Transformers 5 renamed ``torch_dtype`` to ``dtype``.  Retaining the older
    keyword for Transformers 4 keeps the presentation package compatible with
    its declared minimum version.
    """
    try:
        major = int(version.split(".", 1)[0])
    except (AttributeError, ValueError):
        major = 4
    return "dtype" if major >= 5 else "torch_dtype"


def rank_probabilities(top_k: int, temperature: float) -> np.ndarray:
    """Power-law rank distribution used by Faster-GCG.

    Rank 0 is best. As temperature approaches zero the distribution becomes
    greedier; large temperatures approach uniform sampling.
    """
    if top_k <= 0 or temperature <= 0.0:
        raise ValueError("top_k and temperature must be positive")
    descending = np.arange(top_k, 0, -1, dtype=np.float64)
    log_weights = np.log(descending) / float(temperature)
    log_weights -= float(np.max(log_weights))
    weights = np.exp(log_weights)
    return weights / weights.sum()


def prompt_hash(prompt_ids: Sequence[int]) -> str:
    import hashlib

    payload = ",".join(str(int(token)) for token in prompt_ids).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def build_radius1_panel(
    current_ids: Sequence[int],
    rankings: Rankings,
    *,
    budget: int,
    temperature: float,
    rng: np.random.Generator,
    visited: set[str],
) -> list[Proposal]:
    if budget <= 0:
        raise ValueError("budget must be positive")
    top_k = int(rankings.token_ids.shape[1])
    probabilities = rank_probabilities(top_k, temperature)
    positions = list(rankings.positions)
    if not positions:
        raise ValueError("No mutable positions")

    proposals: list[Proposal] = []
    panel_keys: set[str] = set()

    def consider(position_index: int, rank: int) -> bool:
        position = positions[position_index]
        replacement = int(rankings.token_ids[position_index, rank])
        if replacement == int(current_ids[position]):
            return False
        candidate = list(map(int, current_ids))
        candidate[position] = replacement
        key = prompt_hash(candidate)
        if key in visited or key in panel_keys:
            return False
        panel_keys.add(key)
        proposals.append(
            Proposal(
                position=position,
                replacement_id=replacement,
                gradient_rank=int(rank),
                predicted_delta=float(
                    rankings.predicted_deltas[position_index, rank]
                ),
            )
        )
        return True

    # Give every coordinate one opportunity before allocating the rest randomly.
    for position_index in rng.permutation(len(positions)):
        for rank in rng.choice(top_k, size=top_k, replace=False, p=probabilities):
            if consider(int(position_index), int(rank)):
                break
        if len(proposals) >= budget:
            break

    attempts = 0
    attempt_limit = max(10_000, budget * 50)
    while len(proposals) < budget and attempts < attempt_limit:
        attempts += 1
        position_index = int(rng.integers(0, len(positions)))
        rank = int(rng.choice(top_k, p=probabilities))
        consider(position_index, rank)

    if len(proposals) < budget:
        fallback = sorted(
            (
                (
                    float(rankings.predicted_deltas[position_index, rank]),
                    position_index,
                    rank,
                )
                for position_index in range(len(positions))
                for rank in range(top_k)
            ),
            key=lambda row: row[0],
        )
        for _delta, position_index, rank in fallback:
            consider(position_index, rank)
            if len(proposals) >= budget:
                break
    return proposals


class HotFlipProposer:
    """Load the upstream checkpoint only to rank discrete replacements."""

    def __init__(
        self,
        config: ModelConfig,
        trajectory: Trajectory,
        *,
        top_k: int,
        vocab_chunk_size: int,
    ) -> None:
        try:
            import torch
            import transformers
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "PyTorch and Transformers are required for gradient proposals"
            ) from exc

        self.torch = torch
        self.top_k = int(top_k)
        self.vocab_chunk_size = int(vocab_chunk_size)
        self.gguf_vocab_size = int(trajectory.gguf_vocab_size)
        tokenizer_class = getattr(transformers, "AutoTokenizer")
        self.tokenizer = tokenizer_class.from_pretrained(
            config.hf_model,
            trust_remote_code=config.trust_remote_code,
        )
        encoded = tuple(
            int(token)
            for token in self.tokenizer.encode(
                trajectory.hop2_context_text,
                add_special_tokens=False,
            )
        )
        if encoded != trajectory.hop2_context_ids:
            raise RuntimeError(
                "HF and GGUF tokenizers disagree on the captured hop-2 context. "
                "Hybrid token-ID proposals would be unsafe for this model pair."
            )
        target_ids = tuple(
            int(token)
            for token in self.tokenizer.encode(
                trajectory.target_eog_text,
                add_special_tokens=False,
            )
        )
        if target_ids != (trajectory.target_eog_id,):
            raise RuntimeError(
                "HF and GGUF tokenizers disagree on the target EOG token ID"
            )
        if len(self.tokenizer) != trajectory.gguf_vocab_size:
            raise RuntimeError(
                "HF and GGUF vocabulary sizes differ: "
                f"{len(self.tokenizer)} != {trajectory.gguf_vocab_size}"
            )

        model_class = getattr(transformers, config.model_class, None)
        if model_class is None:
            raise ValueError(f"Unknown Transformers model class {config.model_class!r}")
        dtype = getattr(torch, config.dtype, None)
        if dtype is None:
            raise ValueError(f"Unknown torch dtype {config.dtype!r}")
        kwargs: dict[str, object] = {
            "device_map": config.device_map,
            "trust_remote_code": config.trust_remote_code,
        }
        kwargs[transformers_dtype_keyword(transformers.__version__)] = dtype
        if config.attn_implementation:
            kwargs["attn_implementation"] = config.attn_implementation
        self.model = model_class.from_pretrained(config.hf_model, **kwargs)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def _forward_last_logits(self, embeddings):
        parameters = inspect.signature(self.model.forward).parameters
        kwargs = {"inputs_embeds": embeddings, "use_cache": False}
        if "logits_to_keep" in parameters:
            kwargs["logits_to_keep"] = 1
        elif "num_logits_to_keep" in parameters:
            kwargs["num_logits_to_keep"] = 1
        output = self.model(**kwargs)
        return output.logits[:, -1, :]

    def rank(
        self,
        prompt_ids: Sequence[int],
        trajectory: Trajectory,
        mutable_positions: Sequence[int],
    ) -> Rankings:
        torch = self.torch
        context = trajectory.context_with_prompt(2, prompt_ids)
        embedding_layer = self.model.get_input_embeddings()
        device = embedding_layer.weight.device
        input_ids = torch.tensor([context], dtype=torch.long, device=device)
        embeddings = embedding_layer(input_ids).detach().requires_grad_(True)
        logits = self._forward_last_logits(embeddings)[0]
        if int(logits.shape[-1]) < self.gguf_vocab_size:
            raise RuntimeError(
                "HF output vocabulary is smaller than the GGUF vocabulary"
            )
        logits = logits[: self.gguf_vocab_size]
        target = int(trajectory.target_eog_id)
        masked = logits.detach().clone()
        masked[target] = -torch.inf
        competitor = int(torch.argmax(masked).item())
        loss = torch.nn.functional.softplus(logits[competitor] - logits[target])
        gradient = torch.autograd.grad(loss, embeddings, retain_graph=False)[0][0]

        start, _end = trajectory.hop2_prompt_span
        positions = tuple(int(position) for position in mutable_positions)
        context_positions = torch.tensor(
            [start + position for position in positions],
            dtype=torch.long,
            device=device,
        )
        selected_gradient = gradient.index_select(0, context_positions).float()
        weight = embedding_layer.weight.detach()
        if int(weight.shape[0]) < self.gguf_vocab_size:
            raise RuntimeError(
                "HF embedding vocabulary is smaller than GGUF vocabulary"
            )
        vocab_size = self.gguf_vocab_size
        if self.top_k >= vocab_size:
            raise ValueError("top_k must be smaller than the vocabulary")

        best_values = torch.empty(
            (len(positions), 0), dtype=torch.float32, device=device
        )
        best_ids = torch.empty(
            (len(positions), 0), dtype=torch.long, device=device
        )
        current = torch.tensor(
            [int(prompt_ids[position]) for position in positions],
            dtype=torch.long,
            device=device,
        )
        for chunk_start in range(0, vocab_size, self.vocab_chunk_size):
            chunk_end = min(vocab_size, chunk_start + self.vocab_chunk_size)
            candidate_weight = weight[chunk_start:chunk_end].float()
            values = selected_gradient @ candidate_weight.T
            in_chunk = (current >= chunk_start) & (current < chunk_end)
            if bool(in_chunk.any()):
                rows = torch.nonzero(in_chunk, as_tuple=False).flatten()
                columns = current[rows] - chunk_start
                values[rows, columns] = torch.inf
            ids = torch.arange(chunk_start, chunk_end, device=device).expand(
                len(positions), -1
            )
            merged_values = torch.cat((best_values, values), dim=1)
            merged_ids = torch.cat((best_ids, ids), dim=1)
            keep = min(self.top_k, merged_values.shape[1])
            best_values, indices = torch.topk(
                merged_values, keep, dim=1, largest=False, sorted=True
            )
            best_ids = torch.gather(merged_ids, 1, indices)

        current_weight = weight.index_select(0, current).float()
        current_values = (selected_gradient * current_weight).sum(dim=1, keepdim=True)
        deltas = best_values - current_values
        return Rankings(
            positions=positions,
            token_ids=best_ids.cpu().numpy(),
            predicted_deltas=deltas.cpu().numpy(),
            loss=float(loss.detach().float().cpu()),
            competitor_id=competitor,
        )
