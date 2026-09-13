import unittest

from app.ml.quality.categories import (
    QualityLabel,
    SUPPORTED_LABELS,
    label_score_is_consistent,
    score_to_label,
)


class QualityCategoryTests(unittest.TestCase):
    def test_boundary_zero(self) -> None:
        self.assertEqual(score_to_label(0), QualityLabel.POOR)

    def test_boundary_thirty_nine(self) -> None:
        self.assertEqual(score_to_label(39), QualityLabel.POOR)
        self.assertEqual(score_to_label(39.99), QualityLabel.POOR)

    def test_boundary_forty(self) -> None:
        self.assertEqual(score_to_label(40), QualityLabel.FAIR)

    def test_boundary_fifty_nine(self) -> None:
        self.assertEqual(score_to_label(59), QualityLabel.FAIR)
        self.assertEqual(score_to_label(59.99), QualityLabel.FAIR)

    def test_boundary_sixty(self) -> None:
        self.assertEqual(score_to_label(60), QualityLabel.GOOD)

    def test_boundary_seventy_four(self) -> None:
        self.assertEqual(score_to_label(74), QualityLabel.GOOD)
        self.assertEqual(score_to_label(74.99), QualityLabel.GOOD)

    def test_boundary_seventy_five(self) -> None:
        self.assertEqual(score_to_label(75), QualityLabel.STRONG)

    def test_boundary_eighty_nine(self) -> None:
        self.assertEqual(score_to_label(89), QualityLabel.STRONG)
        self.assertEqual(score_to_label(89.99), QualityLabel.STRONG)

    def test_boundary_ninety(self) -> None:
        self.assertEqual(score_to_label(90), QualityLabel.EXCELLENT)

    def test_boundary_one_hundred(self) -> None:
        self.assertEqual(score_to_label(100), QualityLabel.EXCELLENT)

    def test_out_of_bounds_clamping(self) -> None:
        self.assertEqual(score_to_label(-10), QualityLabel.POOR)
        self.assertEqual(score_to_label(150), QualityLabel.EXCELLENT)

    def test_supported_labels_set(self) -> None:
        expected = {"POOR", "FAIR", "GOOD", "STRONG", "EXCELLENT"}
        self.assertEqual(SUPPORTED_LABELS, expected)

    def test_label_score_consistency_checks(self) -> None:
        self.assertTrue(label_score_is_consistent(25.0, "POOR"))
        self.assertTrue(label_score_is_consistent(45.0, "FAIR"))
        self.assertTrue(label_score_is_consistent(65.0, "GOOD"))
        self.assertTrue(label_score_is_consistent(80.0, "STRONG"))
        self.assertTrue(label_score_is_consistent(95.0, "EXCELLENT"))
        self.assertFalse(label_score_is_consistent(25.0, "STRONG"))
        self.assertFalse(label_score_is_consistent(95.0, "POOR"))


if __name__ == "__main__":
    unittest.main()
