"""Tests for POST /api/v1/critic/evaluate endpoint (Step 17)."""
import asyncio
import json
import unittest

from app.critic.critic import PromptCritic
from app.critic.schemas import CriticAnalysisMode
from app.llm.config import LLMConfig
from app.llm.model import MockLLMProvider
from app.llm.schemas import LLMHealthStatus
from app.main import app
from app.services.critic_service import set_critic_service
from app.services.llm_service import LLMService, set_llm_service


class CriticApiMockProvider(MockLLMProvider):
    def _valid_payload(self) -> str:
        return json.dumps({
            "decision": "PASS",
            "overall_critique_score": 85.0,
            "intent_preservation": {
                "score": 90.0,
                "status": "EXCELLENT",
                "reason": "Task goal preserved.",
            },
            "requirement_preservation": {
                "score": 88.0,
                "status": "STRONG",
                "reason": "All constraints retained.",
            },
            "clarity_assessment": "Clearer structure.",
            "specificity_assessment": "More specific parameters.",
            "ambiguity_assessment": "Reduced ambiguity.",
            "completeness_assessment": "Added error handling.",
            "issues": [],
            "preserved_requirements": ["Python script", "CSV parsing"],
            "lost_requirements": [],
            "introduced_requirements": [],
            "unsupported_assumptions": [],
            "strengths": ["Clear instructions"],
            "weaknesses": [],
            "recommendations": [],
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


class TestCriticEndpointUnavailable(unittest.TestCase):
    def setUp(self):
        config = LLMConfig(enabled=False, provider="transformers")
        provider = MockLLMProvider(health_status=LLMHealthStatus.LLM_DISABLED)
        svc = LLMService(config=config, provider=provider)
        set_llm_service(svc)

    def tearDown(self):
        set_llm_service(None)
        set_critic_service(None)

    def test_200_ok_unavailable_mode(self):
        orig = "Write a Python script to sort items."
        opt = "Write a clean Python script to sort items in ascending order."
        status, body = _call_endpoint(
            "POST",
            "/api/v1/critic/evaluate",
            {"original_prompt": orig, "optimized_prompt": opt},
        )
        self.assertEqual(status, 200)
        self.assertIn("evaluation", body)
        self.assertEqual(body["original_prompt_length"], len(orig))
        self.assertEqual(body["optimized_prompt_length"], len(opt))
        self.assertEqual(
            body["evaluation"]["metadata"]["analysis_mode"],
            CriticAnalysisMode.UNAVAILABLE.value,
        )

    def test_422_on_blank_original(self):
        status, _ = _call_endpoint(
            "POST",
            "/api/v1/critic/evaluate",
            {"original_prompt": "   ", "optimized_prompt": "Valid"},
        )
        self.assertEqual(status, 422)

    def test_422_on_blank_optimized(self):
        status, _ = _call_endpoint(
            "POST",
            "/api/v1/critic/evaluate",
            {"original_prompt": "Valid", "optimized_prompt": "   "},
        )
        self.assertEqual(status, 422)

    def test_422_on_missing_fields(self):
        status, _ = _call_endpoint(
            "POST",
            "/api/v1/critic/evaluate",
            {"original_prompt": "Valid"},
        )
        self.assertEqual(status, 422)


class TestCriticEndpointMock(unittest.TestCase):
    def setUp(self):
        config = LLMConfig(enabled=True, provider="mock", max_retries=2)
        provider = CriticApiMockProvider(mode="valid")
        svc = LLMService(config=config, provider=provider)
        set_llm_service(svc)

    def tearDown(self):
        set_llm_service(None)
        set_critic_service(None)

    def test_200_ok_mock_mode(self):
        orig = "Write a function in Python."
        opt = "Write a typed Python function to parse JSON."
        status, body = _call_endpoint(
            "POST",
            "/api/v1/critic/evaluate",
            {"original_prompt": orig, "optimized_prompt": opt},
        )
        self.assertEqual(status, 200)
        eval_data = body["evaluation"]
        self.assertEqual(eval_data["metadata"]["analysis_mode"], CriticAnalysisMode.MOCK.value)
        self.assertIn(eval_data["decision"], {"PASS", "FAIL", "NEEDS_REVIEW"})
        self.assertIn("clarity_change", eval_data)
        self.assertIn("specificity_change", eval_data)
        self.assertIn("ambiguity_change", eval_data)
        self.assertIn("completeness_change", eval_data)


if __name__ == "__main__":
    unittest.main()
