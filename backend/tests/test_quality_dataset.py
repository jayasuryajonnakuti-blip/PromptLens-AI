import tempfile
import unittest
from pathlib import Path

from app.ml.quality.dataset import (
    DatasetValidationError,
    load_and_validate_dataset,
)


class QualityDatasetValidationTests(unittest.TestCase):
    def test_load_valid_project_dataset(self) -> None:
        dataset_path = Path(__file__).resolve().parents[1] / "data" / "prompt_quality_dataset_v1.csv"
        dataset = load_and_validate_dataset(dataset_path)

        self.assertGreaterEqual(len(dataset.prompts), 500)
        self.assertEqual(len(dataset.prompts), len(dataset.scores))
        self.assertEqual(len(dataset.prompts), len(dataset.labels))
        for label in ["POOR", "FAIR", "GOOD", "STRONG", "EXCELLENT"]:
            self.assertIn(label, dataset.distribution)
            self.assertGreaterEqual(dataset.distribution[label], 50)

    def test_missing_file_raises_error(self) -> None:
        with self.assertRaises(DatasetValidationError):
            load_and_validate_dataset(Path("data/non_existent_dataset.csv"))

    def test_missing_required_column_raises_error(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
            f.write("prompt,quality_score\nhello,50\n")
            temp_path = Path(f.name)
        try:
            with self.assertRaises(DatasetValidationError) as ctx:
                load_and_validate_dataset(temp_path)
            self.assertIn("missing required columns", str(ctx.exception).lower())
        finally:
            temp_path.unlink(missing_ok=True)

    def test_empty_prompt_raises_error(self) -> None:
        csv_content = (
            "prompt,quality_score,quality_label,clarity,specificity,context,goal_definition,"
            "constraints,output_format,role_persona,audience,ambiguity,completeness,actionability,consistency\n"
            ",50,FAIR,50,50,0,50,0,0,0,0,50,50,50,50\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
            f.write(csv_content)
            temp_path = Path(f.name)
        try:
            with self.assertRaises(DatasetValidationError) as ctx:
                load_and_validate_dataset(temp_path)
            self.assertIn("empty or missing prompt", str(ctx.exception).lower())
        finally:
            temp_path.unlink(missing_ok=True)

    def test_duplicate_prompt_raises_error(self) -> None:
        csv_content = (
            "prompt,quality_score,quality_label,clarity,specificity,context,goal_definition,"
            "constraints,output_format,role_persona,audience,ambiguity,completeness,actionability,consistency\n"
            "Hello world,50,FAIR,50,50,0,50,0,0,0,0,50,50,50,50\n"
            "hello WORLD,50,FAIR,50,50,0,50,0,0,0,0,50,50,50,50\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
            f.write(csv_content)
            temp_path = Path(f.name)
        try:
            with self.assertRaises(DatasetValidationError) as ctx:
                load_and_validate_dataset(temp_path)
            self.assertIn("duplicate prompt", str(ctx.exception).lower())
        finally:
            temp_path.unlink(missing_ok=True)

    def test_invalid_label_raises_error(self) -> None:
        csv_content = (
            "prompt,quality_score,quality_label,clarity,specificity,context,goal_definition,"
            "constraints,output_format,role_persona,audience,ambiguity,completeness,actionability,consistency\n"
            "Sample prompt,50,MEDIOCRE,50,50,0,50,0,0,0,0,50,50,50,50\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
            f.write(csv_content)
            temp_path = Path(f.name)
        try:
            with self.assertRaises(DatasetValidationError) as ctx:
                load_and_validate_dataset(temp_path)
            self.assertIn("unsupported quality_label", str(ctx.exception).lower())
        finally:
            temp_path.unlink(missing_ok=True)

    def test_score_out_of_bounds_raises_error(self) -> None:
        csv_content = (
            "prompt,quality_score,quality_label,clarity,specificity,context,goal_definition,"
            "constraints,output_format,role_persona,audience,ambiguity,completeness,actionability,consistency\n"
            "Sample prompt,150,EXCELLENT,50,50,0,50,0,0,0,0,50,50,50,50\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
            f.write(csv_content)
            temp_path = Path(f.name)
        try:
            with self.assertRaises(DatasetValidationError) as ctx:
                load_and_validate_dataset(temp_path)
            self.assertIn("out of [0, 100]", str(ctx.exception).lower())
        finally:
            temp_path.unlink(missing_ok=True)

    def test_dimension_out_of_bounds_raises_error(self) -> None:
        csv_content = (
            "prompt,quality_score,quality_label,clarity,specificity,context,goal_definition,"
            "constraints,output_format,role_persona,audience,ambiguity,completeness,actionability,consistency\n"
            "Sample prompt,50,FAIR,-5,50,0,50,0,0,0,0,50,50,50,50\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
            f.write(csv_content)
            temp_path = Path(f.name)
        try:
            with self.assertRaises(DatasetValidationError) as ctx:
                load_and_validate_dataset(temp_path)
            self.assertIn("out of [0, 100]", str(ctx.exception).lower())
        finally:
            temp_path.unlink(missing_ok=True)

    def test_label_score_inconsistency_raises_error(self) -> None:
        csv_content = (
            "prompt,quality_score,quality_label,clarity,specificity,context,goal_definition,"
            "constraints,output_format,role_persona,audience,ambiguity,completeness,actionability,consistency\n"
            "Sample prompt,20,EXCELLENT,50,50,0,50,0,0,0,0,50,50,50,50\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
            f.write(csv_content)
            temp_path = Path(f.name)
        try:
            with self.assertRaises(DatasetValidationError) as ctx:
                load_and_validate_dataset(temp_path)
            self.assertIn("inconsistent", str(ctx.exception).lower())
        finally:
            temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
