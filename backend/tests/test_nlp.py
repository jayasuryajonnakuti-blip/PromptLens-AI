import unittest

from app.engines.nlp import analyze_prompt


class NlpEngineTests(unittest.TestCase):
    def test_basic_educational_prompt(self) -> None:
        result = analyze_prompt("Explain photosynthesis to a high school student.")

        self.assertGreater(result.token_count, 0)
        self.assertGreater(result.pos_counts.verbs, 0)
        self.assertGreater(result.sentence_statistics.sentence_count, 0)

    def test_coding_prompt(self) -> None:
        result = analyze_prompt("Write a Python function that sorts a list.")

        self.assertGreater(result.pos_counts.nouns, 0)
        self.assertGreater(result.verb_statistics.verb_count, 0)
        self.assertIn("function", result.technical_terms.technical_terms)

    def test_named_entities(self) -> None:
        result = analyze_prompt("Ada Lovelace studied mathematics in London.")

        entities = {(entity.text, entity.label) for entity in result.named_entities}
        self.assertTrue(entities)
        self.assertIn(("London", "GPE"), entities)

    def test_questions(self) -> None:
        result = analyze_prompt("What is gravity? Why does it matter?")

        self.assertEqual(result.complexity.question_count, 2)

    def test_instructions(self) -> None:
        result = analyze_prompt("Explain the topic. Summarize the key points.")

        self.assertGreaterEqual(result.complexity.instruction_count_estimated, 2)

    def test_multiple_sentences(self) -> None:
        result = analyze_prompt("One idea is here. Another idea follows. A final idea ends.")

        self.assertEqual(result.sentence_statistics.sentence_count, 3)
        self.assertEqual(result.sentence_statistics.longest_sentence_length, 4)

    def test_punctuation(self) -> None:
        result = analyze_prompt("Really, explain this: now! Is it clear?")

        self.assertGreaterEqual(result.punctuation.comma_count, 1)
        self.assertEqual(result.punctuation.question_mark_count, 1)
        self.assertEqual(result.punctuation.exclamation_mark_count, 1)
        self.assertGreaterEqual(result.punctuation.colon_count, 1)

    def test_numbers(self) -> None:
        result = analyze_prompt("Compare 2 models and 3 datasets in 2026.")

        self.assertEqual(result.number_token_count, 3)

    def test_empty_and_near_empty_input(self) -> None:
        empty = analyze_prompt("")
        short = analyze_prompt("Hi.")

        self.assertEqual(empty.token_count, 0)
        self.assertEqual(empty.sentence_statistics.sentence_count, 0)
        self.assertEqual(short.token_count, 2)

    def test_large_input(self) -> None:
        result = analyze_prompt("analysis " * 5_001)

        self.assertGreaterEqual(result.token_count, 5_001)
        self.assertGreater(result.sentence_statistics.sentence_count, 0)


if __name__ == "__main__":
    unittest.main()