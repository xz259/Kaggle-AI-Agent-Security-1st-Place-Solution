import tempfile
import unittest
from pathlib import Path

from hybrid_gcg.config import load_config


class ConfigTests(unittest.TestCase):
    def test_nested_ridge_configuration_is_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.toml"
            path.write_text(
                """
[model]
gguf_path = "model.gguf"
hf_model = "model"

[task]
prompt_file = "prompt.txt"
hop1_template_file = "hop1.txt"
hop2_template_file = "hop2.txt"
target_eog = "<eog>"

[search]
output_dir = "run"

[search.ridge]
enabled = true
minimum_observations = 32
regularization = 0.05
exploration_fraction = 0.2
""".strip()
                + "\n",
                encoding="utf-8",
            )
            ridge = load_config(path).search.ridge
            self.assertTrue(ridge.enabled)
            self.assertEqual(ridge.minimum_observations, 32)
            self.assertEqual(ridge.regularization, 0.05)
            self.assertEqual(ridge.exploration_fraction, 0.2)


if __name__ == "__main__":
    unittest.main()
