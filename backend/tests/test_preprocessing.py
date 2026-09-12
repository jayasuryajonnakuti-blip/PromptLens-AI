import unittest

from app.engines.preprocessing import preprocess_prompt


class PreprocessingEngineTests(unittest.TestCase):
    def test_empty_prompt(self) -> None:
        result = preprocess_prompt("")

        self.assertEqual(result.text.word_count, 0)
        self.assertEqual(result.text.sentence_count, 0)
        self.assertTrue(result.flags.is_empty)
        self.assertFalse(result.flags.is_near_empty)

    def test_very_short_prompt(self) -> None:
        result = preprocess_prompt("Hi.")

        self.assertEqual(result.text.character_count, 3)
        self.assertEqual(result.text.word_count, 1)
        self.assertEqual(result.text.average_word_length, 2.0)
        self.assertTrue(result.flags.is_near_empty)

    def test_normal_educational_prompt(self) -> None:
        result = preprocess_prompt(
            "Explain photosynthesis to a high school student. Include three key steps."
        )

        self.assertEqual(result.text.sentence_count, 2)
        self.assertEqual(result.structure.question_count, 0)
        self.assertEqual(result.structure.instruction_count_estimated, 2)
        self.assertIn("audience", result.sections)
        self.assertIn("task_goal", result.sections)

    def test_coding_prompt_with_code_block(self) -> None:
        result = preprocess_prompt(
            "Write a Python function that adds two numbers.\n```python\nreturn a + b\n```"
        )

        self.assertTrue(result.structure.has_code_block)
        self.assertIn("code", result.output_formats)
        self.assertIn("task_goal", result.sections)

    def test_prompt_with_urls_and_email(self) -> None:
        result = preprocess_prompt(
            "Review https://example.com and email analyst@example.com with a summary."
        )

        self.assertTrue(result.structure.has_url)
        self.assertTrue(result.structure.has_email)

    def test_prompt_with_placeholders(self) -> None:
        result = preprocess_prompt(
            "Write a welcome note for [NAME] about {TOPIC} using <tone>."
        )

        self.assertEqual(
            [placeholder.value for placeholder in result.placeholders],
            ["[NAME]", "{TOPIC}", "<tone>"],
        )

    def test_prompt_with_bullets(self) -> None:
        result = preprocess_prompt("- Explain the concept\n- Include an example")

        self.assertEqual(result.structure.bullet_count, 2)
        self.assertEqual(result.structure.instruction_count_estimated, 2)
        self.assertIn("bullets", result.output_formats)

    def test_long_prompt(self) -> None:
        result = preprocess_prompt("word " * 2_000)

        self.assertEqual(result.text.word_count, 2_000)
        self.assertTrue(result.flags.is_very_long)

    def test_prompt_with_repeated_phrases(self) -> None:
        result = preprocess_prompt("make it clear and make it clear")

        phrases = {item.phrase: item.count for item in result.repeated_phrases}
        self.assertEqual(phrases["make it clear"], 2)

    def test_prompt_with_multiple_output_formats(self) -> None:
        result = preprocess_prompt(
            "Return the answer as JSON and Markdown, then show it in a table and as plain text."
        )

        self.assertEqual(
            result.output_formats,
            ["JSON", "Markdown", "table", "plain_text"],
        )

    def test_prompt_with_at_least_five_thousand_words(self) -> None:
        result = preprocess_prompt("analysis " * 5_001)

        self.assertEqual(result.text.word_count, 5_001)
        self.assertTrue(result.flags.is_very_long)
        self.assertFalse(result.flags.is_empty)


if __name__ == "__main__":
    unittest.main()