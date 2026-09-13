"""Tests for POST /api/v1/optimizer/optimize endpoint (Step 16).

Uses ASGI test harness with MockLLMProvider injected via set_llm_service().
Always resets state in tearDown().
"""
import asyncio
import json
import unittest

from app.main import app
from app.llm.model import MockLLMProvider
from app.llm.schemas import LLMHealthStatus
from app.optimizer.schemas import OptimizationMode
from app.services.llm_service import LLMService, set_llm_service
from app.services.optimizer_service import set_optimizer_service


# ---------------------------------------------------------------------------
# Optimizer-compatible mock provider for API tests
# ---------------------------------------------------------------------------
class OptimizerApiMockProvider(MockLLMProvider):
    def _valid_payload(self) -> str:
        return json.dumps({
            "optimized_prompt": (
                "You are an expert Python developer. Write a REST API endpoint for user authentication. "
                "Requirements:\n"
                "- Framework: [WEB_FRAMEWORK]\n"
                "- Token mechanism: JWT\n"
                "- Return format: JSON with status and access_token\n"
                "- Handle errors with appropriate HTTP status codes (e.g., 401 Unauthorized)"
            ),
            "summary": "Clarified requirements, added placeholders for missing framework, and structured outputs.",
            "changes": [
                {"category": "structure", "description": "Formatted requirements into structured list."},
                {"category": "placeholder_inserted", "description": "Added [WEB_FRAMEWORK] placeholder."}
            ],
            "preserved_requirements": [
                {"requirement": "Write a REST API endpoint for user authentication.", "reason": "Original core objective preserved."}
            ],
            "placeholders_inserted": ["[WEB_FRAMEWORK]"],
            "improvement_score_delta": 15.0
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


class TestOptimizerEndpointUnavailable(unittest.TestCase):
    def setUp(self):
        from app.llm.config import LLMConfig
        config = LLMConfig(enabled=False, provider="transformers")
        provider = MockLLMProvider(health_status=LLMHealthStatus.LLM_DISABLED)
        svc = LLMService(config=config, provider=provider)
        set_llm_service(svc)

    def tearDown(self):
        set_llm_service(None)
        set_optimizer_service(None)

    def test_200_ok_unavailable_mode(self):
        prompt = "Write a function to compute Fibonacci sequence."
        status, body = _call_endpoint(
            "POST",
            "/api/v1/optimizer/optimize",
            {"prompt": prompt, "mode": "balanced"}
        )
        self.assertEqual(status, 200)
        self.assertIn("result", body)
        self.assertEqual(body["result"]["metadata"]["optimizer_mode"], "UNAVAILABLE")
        self.assertEqual(body["result"]["optimized_prompt"], prompt)

    def test_422_on_blank_prompt(self):
        status, _ = _call_endpoint("POST", "/api/v1/optimizer/optimize", {"prompt": "   "})
        self.assertEqual(status, 422)

    def test_422_on_invalid_mode(self):
        status, _ = _call_endpoint("POST", "/api/v1/optimizer/optimize", {"prompt": "Valid", "mode": "unsupported_mode"})
        self.assertEqual(status, 422)


class TestOptimizerEndpointMock(unittest.TestCase):
    def setUp(self):
        from app.llm.config import LLMConfig
        config = LLMConfig(enabled=True, provider="mock", max_retries=2)
        provider = OptimizerApiMockProvider(mode="valid")
        svc = LLMService(config=config, provider=provider)
        set_llm_service(svc)

    def tearDown(self):
        set_llm_service(None)
        set_optimizer_service(None)

    def test_200_ok_mock_mode(self):
        status, body = _call_endpoint(
            "POST",
            "/api/v1/optimizer/optimize",
            {
                "prompt": "Write a REST API endpoint for authentication in Python.",
                "mode": "analytical"
            }
        )
        self.assertEqual(status, 200)
        self.assertIn("result", body)
        res = body["result"]
        self.assertEqual(res["metadata"]["optimizer_mode"], "MOCK")
        self.assertEqual(res["metadata"]["optimization_mode"], "analytical")
        self.assertIn("[WEB_FRAMEWORK]", res["placeholders_inserted"])
        self.assertGreater(len(res["changes"]), 0)
        self.assertGreater(len(res["preserved_requirements"]), 0)

    def test_with_analyzer_result(self):
        status, body = _call_endpoint(
            "POST",
            "/api/v1/optimizer/optimize",
            {
                "prompt": "Write a REST API endpoint.",
                "mode": "expert",
                "analyzer_result": {
                    "interpreted_goal": "Create API",
                    "intent": {"label": "CODING", "confidence": 0.95}
                }
            }
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["result"]["metadata"]["optimization_mode"], "expert")


if __name__ == "__main__":
    unittest.main()
