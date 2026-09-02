import unittest

from hybrid_gcg.trajectory import Trajectory


def make_trajectory() -> Trajectory:
    return Trajectory(
        schema_version=1,
        prompt_text="ab",
        prompt_token_ids=(10, 11),
        hop1_context_text="xabz",
        hop1_context_ids=(1, 10, 11, 2),
        hop1_prompt_span=(1, 3),
        hop1_output_ids=(20, 21),
        hop1_text="OK<EOG>",
        hop1_visible_text="OK",
        hop2_context_text="xabOK",
        hop2_context_ids=(3, 10, 11, 4),
        hop2_prompt_span=(1, 3),
        target_eog_text="<EOG>",
        target_eog_id=21,
        llama_cpp_version="test",
        gguf_name="test.gguf",
        gguf_size_bytes=4,
        gguf_vocab_size=256,
    )


class TrajectoryTests(unittest.TestCase):
    def test_context_replacement_is_length_preserving(self) -> None:
        trajectory = make_trajectory()
        trajectory.validate()
        self.assertEqual(
            trajectory.context_with_prompt(1, [30, 31]), (1, 30, 31, 2)
        )
        self.assertEqual(
            trajectory.context_with_prompt(2, [30, 31]), (3, 30, 31, 4)
        )


if __name__ == "__main__":
    unittest.main()
