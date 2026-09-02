import unittest

from hybrid_gcg.templates import TextTemplate, TokenBoundaryError


def byte_tokenizer(text: str, add_bos: bool):
    prefix = [999] if add_bos else []
    return prefix + list(text.encode("utf-8"))


class TemplateTests(unittest.TestCase):
    def test_template_finds_exact_prompt_span(self) -> None:
        template = TextTemplate("system|{{PROMPT}}|assistant")
        rendered = template.render_token_aligned(
            "hello", byte_tokenizer, add_bos=True
        )
        start, end = rendered.prompt_span
        self.assertEqual(rendered.token_ids[start:end], tuple(b"hello"))

    def test_hop1_placeholders_are_expanded(self) -> None:
        template = TextTemplate("{{PROMPT}}/{{HOP1_VISIBLE}}/{{HOP1}}")
        self.assertEqual(
            template.render("P", hop1="FULL", hop1_visible="VISIBLE"),
            "P/VISIBLE/FULL",
        )

    def test_boundary_merge_is_rejected(self) -> None:
        def merging_tokenizer(text: str, add_bos: bool):
            if text == "ab":
                return [7]
            return list(text.encode())

        with self.assertRaises(TokenBoundaryError):
            TextTemplate("a{{PROMPT}}").render_token_aligned(
                "b", merging_tokenizer, add_bos=False
            )


if __name__ == "__main__":
    unittest.main()
