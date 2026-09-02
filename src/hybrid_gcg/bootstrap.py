"""Capture the trajectory that GCG must preserve."""

from __future__ import annotations

from .config import Config
from .llama_backend import LlamaBackend, gguf_display_name
from .templates import TextTemplate
from .trajectory import Trajectory


def capture_trajectory(config: Config, backend: LlamaBackend) -> Trajectory:
    prompt = config.task.prompt_file.read_text(encoding="utf-8")
    hop1_template = TextTemplate.load(config.task.hop1_template_file)
    hop2_template = TextTemplate.load(config.task.hop2_template_file)

    hop1 = hop1_template.render_token_aligned(
        prompt,
        backend.tokenize,
        add_bos=config.task.add_bos,
    )
    prompt_ids = tuple(hop1.token_ids[slice(*hop1.prompt_span)])
    if not prompt_ids:
        raise RuntimeError("The mutable prompt has zero tokens")

    generated = backend.greedy_decode(
        hop1.token_ids,
        max_tokens=config.task.max_hop1_tokens,
    )
    if not generated.stopped_on_eog:
        raise RuntimeError(
            "Hop 1 did not reach an EOG token within max_hop1_tokens; no stable "
            "two-hop trajectory was captured"
        )
    hop1_text = backend.detokenize(generated.token_ids, special=True)
    visible_ids = generated.token_ids[:-1]
    hop1_visible = backend.detokenize(visible_ids, special=True)

    hop2 = hop2_template.render_token_aligned(
        prompt,
        backend.tokenize,
        add_bos=config.task.add_bos,
        hop1=hop1_text,
        hop1_visible=hop1_visible,
    )
    if tuple(hop2.token_ids[slice(*hop2.prompt_span)]) != prompt_ids:
        raise RuntimeError("Hop-1 and hop-2 prompt tokenizations do not match")

    target_ids = backend.tokenize(config.task.target_eog, False)
    if len(target_ids) != 1:
        raise RuntimeError(
            f"target_eog must be one GGUF token, got {len(target_ids)}"
        )
    target_id = int(target_ids[0])
    decoded_target = backend.detokenize(target_ids, special=True)
    if decoded_target != config.task.target_eog:
        raise RuntimeError(
            "target_eog does not round-trip through the GGUF tokenizer: "
            f"{config.task.target_eog!r} -> {decoded_target!r}"
        )
    if not backend.is_eog(target_id):
        raise RuntimeError(
            f"target_eog token {target_id} is not classified as EOG by llama.cpp"
        )

    trajectory = Trajectory(
        schema_version=1,
        prompt_text=prompt,
        prompt_token_ids=prompt_ids,
        hop1_context_text=hop1.text,
        hop1_context_ids=hop1.token_ids,
        hop1_prompt_span=hop1.prompt_span,
        hop1_output_ids=generated.token_ids,
        hop1_text=hop1_text,
        hop1_visible_text=hop1_visible,
        hop2_context_text=hop2.text,
        hop2_context_ids=hop2.token_ids,
        hop2_prompt_span=hop2.prompt_span,
        target_eog_text=config.task.target_eog,
        target_eog_id=target_id,
        llama_cpp_version=backend.version,
        gguf_name=gguf_display_name(config.model.gguf_path),
        gguf_size_bytes=config.model.gguf_path.stat().st_size,
        gguf_vocab_size=backend.n_vocab,
    )
    trajectory.validate()
    return trajectory
