import unittest

from hybrid_gcg.objectives import score_logits
from hybrid_gcg.search import better_hop2


class SearchPolicyTests(unittest.TestCase):
    def test_only_strict_full_margin_improvement_promotes(self) -> None:
        incumbent = score_logits([3.0, 2.0], 1)
        better = score_logits([2.0, 2.5], 1)
        equal = score_logits([3.0, 2.0], 1)
        self.assertTrue(better_hop2(better, incumbent))
        self.assertFalse(better_hop2(equal, incumbent))


if __name__ == "__main__":
    unittest.main()
