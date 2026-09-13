import asyncio
import json
import unittest
from typing import Any

from app.main import app


class ScoringApiTests(unittest.TestCase):
    def test_valid_score_request(self) -> None:
        status, body = _request(
            {"prompt": "Explain machine learning to a beginner using a simple real-world example."}
        )

        self.assertEqual(status, 200)
        self.assertIn("overall_score", body)
        self.assertIn("dimensions", body)
        self.assertIn("signals", body)
        self.assertIn("findings", body)
        self.assertIn("recommendations", body)
        self.assertIn("metadata", body)

        # Check overall score
        overall = body["overall_score"]
        self.assertGreaterEqual(overall["score"], 0.0)
        self.assertLessEqual(overall["score"], 100.0)
        self.assertIn(overall["category"], ["POOR", "FAIR", "GOOD", "STRONG", "EXCELLENT"])
        self.assertEqual(overall["status"], overall["category"])

        # Check all 12 dimensions present
        expected_dims = {
            "clarity",
            "specificity",
            "context",
            "goal_definition",
            "constraints",
            "output_format",
            "role_persona",
            "audience",
            "ambiguity",
            "completeness",
            "actionability",
            "consistency",
        }
        self.assertEqual(set(body["dimensions"].keys()), expected_dims)
        for dim_key, dim_val in body["dimensions"].items():
            self.assertGreaterEqual(dim_val["score"], 0.0)
            self.assertLessEqual(dim_val["score"], 100.0)
            self.assertIn("status", dim_val)
            self.assertIn("reason", dim_val)
            self.assertIn("recommendation", dim_val)

        # Check signals and LLM unavailable state
        signals = body["signals"]
        self.assertIn("rules", signals)
        self.assertIn("nlp", signals)
        self.assertIn("quality_ml", signals)
        self.assertIn("llm", signals)

        self.assertTrue(signals["rules"]["available"])
        self.assertTrue(signals["nlp"]["available"])
        self.assertTrue(signals["quality_ml"]["available"])
        self.assertFalse(signals["llm"]["available"])
        self.assertEqual(signals["llm"]["effective_weight"], 0.0)
        self.assertIsNone(signals["llm"]["raw_score"])

        # Effective weights of available signals must sum to 1.0
        eff_sum = (
            signals["rules"]["effective_weight"]
            + signals["nlp"]["effective_weight"]
            + signals["quality_ml"]["effective_weight"]
        )
        self.assertAlmostEqual(eff_sum, 1.0, places=4)

        # Check metadata
        self.assertFalse(body["metadata"]["llm_available"])
        self.assertEqual(body["metadata"]["signals_evaluated"], 3)

    def test_structured_prompt_scoring(self) -> None:
        status, body = _request({
            "prompt": (
                "You are a senior DevOps engineer. Write a GitHub Actions workflow for a Python FastAPI service. "
                "Include unit tests with coverage above 80%, linting with flake8, and container build. "
                "Format output as YAML."
            )
        })

        self.assertEqual(status, 200)
        self.assertGreaterEqual(body["overall_score"]["score"], 65.0)

    def test_long_prompt_scoring(self) -> None:
        long_prompt = "You are an enterprise system architect. Explain cloud resilience. " * 200
        status, body = _request({"prompt": long_prompt})

        self.assertEqual(status, 200)
        self.assertGreaterEqual(body["overall_score"]["score"], 0.0)
        self.assertLessEqual(body["overall_score"]["score"], 100.0)

    def test_empty_prompt_is_rejected(self) -> None:
        status, body = _request({"prompt": ""})

        self.assertEqual(status, 422)
        self.assertIn("detail", body)

    def test_whitespace_prompt_is_rejected(self) -> None:
        status, body = _request({"prompt": "   \n\t  "})

        self.assertEqual(status, 422)
        self.assertIn("detail", body)

    def test_malformed_request_body_is_rejected(self) -> None:
        status, body = _request({"prompt": 98765})

        self.assertEqual(status, 422)
        self.assertIn("detail", body)

    def test_missing_prompt_key_is_rejected(self) -> None:
        status, body = _request({"text": "Hello"})

        self.assertEqual(status, 422)
        self.assertIn("detail", body)


def _request(payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    body = json.dumps(payload).encode()
    messages: list[dict[str, Any]] = []
    request_sent = False

    async def receive() -> dict[str, Any]:
        nonlocal request_sent
        if request_sent:
            return {"type": "http.disconnect"}
        request_sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        messages.append(message)

    path = "/api/v1/score"
    scope: dict[str, Any] = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [(b"content-type", b"application/json")],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
    }

    asyncio.run(app(scope, receive, send))
    status = next(
        message["status"]
        for message in messages
        if message["type"] == "http.response.start"
    )
    response_body = b"".join(
        message.get("body", b"")
        for message in messages
        if message["type"] == "http.response.body"
    )
    return status, json.loads(response_body)


if __name__ == "__main__":
    unittest.main()
