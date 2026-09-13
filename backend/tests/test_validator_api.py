"""Tests for POST /api/v1/validator/validate endpoint (Step 18)."""
import asyncio
import json
import unittest

from app.llm.config import LLMConfig
from app.llm.model import MockLLMProvider
from app.llm.schemas import LLMHealthStatus
from app.main import app
from app.services.llm_service import LLMService, set_llm_service
from app.services.validator_service import set_validator_service
from app.validator.schemas import ValidationMode


class ValidatorApiMockProvider(MockLLMProvider):
    def _valid_payload(self) -> str:
        return json.dumps({
            "decision": "PASS",
            "safety_score": 92.0,
            "justification": "Optimization is completely valid and preserves all explicit instructions.",
            "issues": [],
            "passed_checks": ["intent_preservation", "negative_constraints", "output_format"],
            "failed_checks": [],
        })


def _call_endpoint(method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
    scope = {
        "type": "http",
        "method": method.upper(),
        "path": path,
        "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
    }
    response_started: list[dict] = []
    response_body: list[bytes] = []

    async def receive():
        if body is not None:
            content = json.dumps(body).encode()
            return {"type": "http.request", "body": content, "more_body": False}
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        if message["type"] == "http.response.start":
            response_started.append(message)
        elif message["type"] == "http.response.body":
            response_body.append(message.get("body", b""))

    asyncio.run(app(scope, receive, send))
    status_code = response_started[0]["status"] if response_started else 500
    raw_body = b"".join(response_body)
    try:
        parsed = json.loads(raw_body.decode())
    except Exception:
        parsed = {"raw": raw_body.decode()}
    return status_code, parsed


class TestValidatorEndpointUnavailable(unittest.TestCase):
    def setUp(self):
        config = LLMConfig(enabled=False, provider="transformers")
        provider = MockLLMProvider(health_status=LLMHealthStatus.LLM_DISABLED)
        svc = LLMService(config=config, provider=provider)
        set_llm_service(svc)

    def tearDown(self):
        set_llm_service(None)
        set_validator_service(None)

    def test_200_ok_unavailable_mode(self):
        orig = "Write a Python script to sort items."
        opt = "Write a clean Python script to sort items in ascending order."
        status, body = _call_endpoint(
            "POST",
            "/api/v1/validator/validate",
            {"original_prompt": orig, "optimized_prompt": opt},
        )
        self.assertEqual(status, 200)
        self.assertIn("result", body)
        self.assertEqual(body["original_prompt_length"], len(orig))
        self.assertEqual(body["optimized_prompt_length"], len(opt))
        self.assertEqual(
            body["result"]["metadata"]["validation_mode"],
            ValidationMode.UNAVAILABLE.value,
        )
        self.assertIn(body["result"]["decision"], {"PASS", "FAIL", "NEEDS_REVIEW"})

    def test_422_on_blank_original(self):
        status, _ = _call_endpoint(
            "POST",
            "/api/v1/validator/validate",
            {"original_prompt": "   ", "optimized_prompt": "Valid"},
        )
        self.assertEqual(status, 422)

    def test_422_on_blank_optimized(self):
        status, _ = _call_endpoint(
            "POST",
            "/api/v1/validator/validate",
            {"original_prompt": "Valid", "optimized_prompt": "   "},
        )
        self.assertEqual(status, 422)

    def test_422_on_missing_fields(self):
        status, _ = _call_endpoint(
            "POST",
            "/api/v1/validator/validate",
            {"original_prompt": "Valid"},
        )
        self.assertEqual(status, 422)


class TestValidatorEndpointMock(unittest.TestCase):
    def setUp(self):
        config = LLMConfig(enabled=True, provider="mock", max_retries=2)
        provider = ValidatorApiMockProvider(mode="valid")
        svc = LLMService(config=config, provider=provider)
        set_llm_service(svc)

    def tearDown(self):
        set_llm_service(None)
        set_validator_service(None)

    def test_200_ok_mock_mode(self):
        orig = "Write a function in Python."
        opt = "Write a typed Python function to parse JSON with error handling."
        status, body = _call_endpoint(
            "POST",
            "/api/v1/validator/validate",
            {"original_prompt": orig, "optimized_prompt": opt},
        )
        self.assertEqual(status, 200)
        res_data = body["result"]
        self.assertEqual(res_data["metadata"]["validation_mode"], ValidationMode.MOCK.value)
        self.assertEqual(res_data["decision"], "PASS")
        self.assertTrue(res_data["is_valid"])
        self.assertIn("original_metadata", res_data)
        self.assertIn("optimized_metadata", res_data)
        self.assertIn("evidence", res_data)


if __name__ == "__main__":
    unittest.main()
