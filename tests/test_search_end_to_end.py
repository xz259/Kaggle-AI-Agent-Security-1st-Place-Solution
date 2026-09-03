import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from hybrid_gcg.config import (
    Config,
    ModelConfig,
    RidgeConfig,
    SearchConfig,
    TaskConfig,
)
from hybrid_gcg.hotflip import Rankings
from hybrid_gcg.llama_backend import GreedyResult
from hybrid_gcg.objectives import SequenceScore, TokenScore, score_logits
from hybrid_gcg.search import SearchRunner
from hybrid_gcg.templates import TextTemplate
from hybrid_gcg.trajectory import Trajectory


def byte_tokenizer(text: str, add_bos: bool = False) -> tuple[int, ...]:
    prefix = (255,) if add_bos else ()
    return prefix + tuple(text.encode("utf-8"))


class FakeBackend:
    version = "test"
    n_vocab = 256

    def tokenize(self, text: str, add_bos: bool = False) -> tuple[int, ...]:
        return byte_tokenizer(text, add_bos)

    def detokenize(self, token_ids, *, special: bool = True) -> str:
        del special
        return bytes(map(int, token_ids)).decode("utf-8")

    def score_next(self, context_ids, target_id: int) -> TokenScore:
        values = np.zeros(256)
        values[1] = 2.0
        values[target_id] = 3.0 if ord("c") in context_ids else 1.0
        return score_logits(values, target_id)

    def score_next_panel(
        self,
        contexts,
        *,
        common_prefix_length: int,
        target_id: int,
    ) -> list[TokenScore]:
        del common_prefix_length
        return [self.score_next(context, target_id) for context in contexts]

    def score_sequence(self, context_ids, target_ids) -> SequenceScore:
        del context_ids
        rows = []
        for target_id in target_ids:
            values = np.zeros(256)
            values[target_id] = 2.0
            rows.append(score_logits(values, target_id))
        return SequenceScore(tuple(rows))

    def greedy_decode(self, context_ids, *, max_tokens: int) -> GreedyResult:
        if max_tokens == 2:
            return GreedyResult((ord("O"), 0), True)
        if ord("c") in context_ids:
            return GreedyResult((ord("!"),), True)
        return GreedyResult((1,), False)

    def token_piece(self, token_id: int) -> str:
        return bytes([token_id]).decode("utf-8")


class FakeProposer:
    def rank(self, prompt_ids, trajectory, mutable_positions) -> Rankings:
        del prompt_ids, trajectory
        self.positions = tuple(mutable_positions)
        return Rankings(
            positions=self.positions,
            token_ids=np.array([[ord("c")]], dtype=np.int64),
            predicted_deltas=np.array([[-1.0]], dtype=np.float32),
            loss=1.0,
            competitor_id=1,
        )


class RidgeFakeProposer:
    def rank(self, prompt_ids, trajectory, mutable_positions) -> Rankings:
        del prompt_ids, trajectory
        return Rankings(
            positions=tuple(mutable_positions),
            token_ids=np.array([[ord("c"), ord("d")]], dtype=np.int64),
            predicted_deltas=np.array([[-2.0, -1.0]], dtype=np.float32),
            loss=1.0,
            competitor_id=1,
        )


class RejectingHop1Backend(FakeBackend):
    def score_sequence(self, context_ids, target_ids) -> SequenceScore:
        if ord("c") not in context_ids:
            return super().score_sequence(context_ids, target_ids)
        rows = []
        for target_id in target_ids:
            values = np.zeros(256)
            values[target_id] = -1.0
            values[1 if target_id != 1 else 2] = 1.0
            rows.append(score_logits(values, target_id))
        return SequenceScore(tuple(rows))


class SearchEndToEndTests(unittest.TestCase):
    def test_accepts_only_after_full_score_and_hop1_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prompt_path = root / "prompt.txt"
            hop1_path = root / "hop1.txt"
            hop2_path = root / "hop2.txt"
            gguf_path = root / "fake.gguf"
            prompt_path.write_text("ab", encoding="utf-8")
            hop1_path.write_text("H|{{PROMPT}}|A", encoding="utf-8")
            hop2_path.write_text(
                "H|{{PROMPT}}|A{{HOP1}}TOOL|", encoding="utf-8"
            )
            gguf_path.write_bytes(b"fake")

            hop1 = TextTemplate.load(hop1_path).render_token_aligned(
                "ab", byte_tokenizer, add_bos=False
            )
            hop2 = TextTemplate.load(hop2_path).render_token_aligned(
                "ab",
                byte_tokenizer,
                add_bos=False,
                hop1="O\x00",
                hop1_visible="O",
            )
            trajectory = Trajectory(
                schema_version=1,
                prompt_text="ab",
                prompt_token_ids=tuple(b"ab"),
                hop1_context_text=hop1.text,
                hop1_context_ids=hop1.token_ids,
                hop1_prompt_span=hop1.prompt_span,
                hop1_output_ids=(ord("O"), 0),
                hop1_text="O\x00",
                hop1_visible_text="O",
                hop2_context_text=hop2.text,
                hop2_context_ids=hop2.token_ids,
                hop2_prompt_span=hop2.prompt_span,
                target_eog_text="!",
                target_eog_id=ord("!"),
                llama_cpp_version="test",
                gguf_name="fake.gguf",
                gguf_size_bytes=4,
                gguf_vocab_size=256,
            )
            config = Config(
                source=root / "config.toml",
                model=ModelConfig(gguf_path=gguf_path, hf_model="unused"),
                task=TaskConfig(
                    prompt_file=prompt_path,
                    hop1_template_file=hop1_path,
                    hop2_template_file=hop2_path,
                    target_eog="!",
                ),
                search=SearchConfig(
                    output_dir=root / "run",
                    steps=1,
                    candidate_budget=1,
                    top_k=1,
                    full_recheck_count=1,
                    mutable_positions=(0,),
                ),
            )

            with contextlib.redirect_stdout(io.StringIO()):
                result = SearchRunner(
                    config,
                    trajectory,
                    FakeBackend(),
                    proposer=FakeProposer(),
                ).run()

            self.assertTrue(result["solved"])
            self.assertTrue(result["hop1_preserved"])
            self.assertEqual(result["prompt"], "cb")
            self.assertEqual(result["accepted_steps"], 1)
            self.assertTrue((root / "run" / "checkpoint.json").is_file())
            self.assertTrue((root / "run" / "events.jsonl").is_file())

    def test_rejects_hop2_improvement_that_changes_hop1(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prompt_path = root / "prompt.txt"
            hop1_path = root / "hop1.txt"
            hop2_path = root / "hop2.txt"
            gguf_path = root / "fake.gguf"
            prompt_path.write_text("ab", encoding="utf-8")
            hop1_path.write_text("H|{{PROMPT}}|A", encoding="utf-8")
            hop2_path.write_text(
                "H|{{PROMPT}}|A{{HOP1}}TOOL|", encoding="utf-8"
            )
            gguf_path.write_bytes(b"fake")
            hop1 = TextTemplate.load(hop1_path).render_token_aligned(
                "ab", byte_tokenizer, add_bos=False
            )
            hop2 = TextTemplate.load(hop2_path).render_token_aligned(
                "ab",
                byte_tokenizer,
                add_bos=False,
                hop1="O\x00",
                hop1_visible="O",
            )
            trajectory = Trajectory(
                schema_version=1,
                prompt_text="ab",
                prompt_token_ids=tuple(b"ab"),
                hop1_context_text=hop1.text,
                hop1_context_ids=hop1.token_ids,
                hop1_prompt_span=hop1.prompt_span,
                hop1_output_ids=(ord("O"), 0),
                hop1_text="O\x00",
                hop1_visible_text="O",
                hop2_context_text=hop2.text,
                hop2_context_ids=hop2.token_ids,
                hop2_prompt_span=hop2.prompt_span,
                target_eog_text="!",
                target_eog_id=ord("!"),
                llama_cpp_version="test",
                gguf_name="fake.gguf",
                gguf_size_bytes=4,
                gguf_vocab_size=256,
            )
            config = Config(
                source=root / "config.toml",
                model=ModelConfig(gguf_path=gguf_path, hf_model="unused"),
                task=TaskConfig(
                    prompt_file=prompt_path,
                    hop1_template_file=hop1_path,
                    hop2_template_file=hop2_path,
                    target_eog="!",
                ),
                search=SearchConfig(
                    output_dir=root / "run",
                    steps=1,
                    candidate_budget=1,
                    top_k=1,
                    full_recheck_count=1,
                    mutable_positions=(0,),
                ),
            )

            with contextlib.redirect_stdout(io.StringIO()):
                result = SearchRunner(
                    config,
                    trajectory,
                    RejectingHop1Backend(),
                    proposer=FakeProposer(),
                ).run()

            self.assertFalse(result["solved"])
            self.assertEqual(result["prompt"], "ab")
            self.assertEqual(result["accepted_steps"], 0)

    def test_ridge_calibration_resets_after_an_accepted_edit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prompt_path = root / "prompt.txt"
            hop1_path = root / "hop1.txt"
            hop2_path = root / "hop2.txt"
            gguf_path = root / "fake.gguf"
            prompt_path.write_text("ab", encoding="utf-8")
            hop1_path.write_text("H|{{PROMPT}}|A", encoding="utf-8")
            hop2_path.write_text(
                "H|{{PROMPT}}|A{{HOP1}}TOOL|", encoding="utf-8"
            )
            gguf_path.write_bytes(b"fake")
            hop1 = TextTemplate.load(hop1_path).render_token_aligned(
                "ab", byte_tokenizer, add_bos=False
            )
            hop2 = TextTemplate.load(hop2_path).render_token_aligned(
                "ab",
                byte_tokenizer,
                add_bos=False,
                hop1="O\x00",
                hop1_visible="O",
            )
            trajectory = Trajectory(
                schema_version=1,
                prompt_text="ab",
                prompt_token_ids=tuple(b"ab"),
                hop1_context_text=hop1.text,
                hop1_context_ids=hop1.token_ids,
                hop1_prompt_span=hop1.prompt_span,
                hop1_output_ids=(ord("O"), 0),
                hop1_text="O\x00",
                hop1_visible_text="O",
                hop2_context_text=hop2.text,
                hop2_context_ids=hop2.token_ids,
                hop2_prompt_span=hop2.prompt_span,
                target_eog_text="!",
                target_eog_id=ord("!"),
                llama_cpp_version="test",
                gguf_name="fake.gguf",
                gguf_size_bytes=4,
                gguf_vocab_size=256,
            )
            config = Config(
                source=root / "config.toml",
                model=ModelConfig(gguf_path=gguf_path, hf_model="unused"),
                task=TaskConfig(
                    prompt_file=prompt_path,
                    hop1_template_file=hop1_path,
                    hop2_template_file=hop2_path,
                    target_eog="!",
                ),
                search=SearchConfig(
                    output_dir=root / "run",
                    steps=1,
                    candidate_budget=2,
                    top_k=2,
                    full_recheck_count=2,
                    mutable_positions=(0,),
                    ridge=RidgeConfig(
                        enabled=True,
                        minimum_observations=1,
                        exploration_fraction=0.0,
                    ),
                ),
            )

            with contextlib.redirect_stdout(io.StringIO()):
                result = SearchRunner(
                    config,
                    trajectory,
                    FakeBackend(),
                    proposer=RidgeFakeProposer(),
                ).run()

            self.assertTrue(result["solved"])
            self.assertEqual(result["proposal_strategy"], "gguf_ridge")
            self.assertEqual(result["ridge_observation_count"], 0)
            checkpoint = json.loads(
                (root / "run" / "checkpoint.json").read_text(encoding="utf-8")
            )
            self.assertEqual(checkpoint["ridge_observations"], [])
            event = checkpoint["history"][0]
            self.assertEqual(len(event["proposal_phases"]), 2)
            self.assertEqual(event["proposal_phases"][0]["name"], "calibration")
            self.assertEqual(event["proposal_phases"][1]["name"], "regression")
            self.assertGreater(event["ridge_observation_count_after_scoring"], 0)


if __name__ == "__main__":
    unittest.main()
