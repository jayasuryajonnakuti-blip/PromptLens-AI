import asyncio
import json
import unittest
from typing import Any

from app.llm.config import LLMConfig
from app.llm.model import MockLLMProvider
from app.llm.schemas import LLMHealthStatus
from app.main import app
from app.services.llm_service import LLMService, set_llm_service


class LLMApiTests(unittest.TestCase):
    def tearDown(self) -> None:
        set_llm_service(None)

    def test_disabled_llm_returns_503(self) -> None:
        # Default configuration is disabled
        status, body = _request("POST", "/api/v1/llm/analyze", {"prompt": "Analyze this prompt."})
        self.assertEqual(status, 503)
        self.assertIn("detail", body)
        self.assertIn("disabled", body["detail"].lower())

    def test_valid_request_with_mock_provider(self) -> None:
        mock_provider = MockLLMProvider(model_name="mock-test-llm")
        test_service = LLMService(
            config=LLMConfig(enabled=True, provider="mock"),
            provider=mock_provider,
        )
        set_llm_service(test_service)

        status, body = _request("POST", "/api/v1/llm/analyze", {
            "prompt": "Write a Python function to parse JSON with error handling."
        })

        self.assertEqual(status, 200)
        self.assertIn("analysis", body)
        self.assertIn("model", body)
        self.assertIn("provider", body)
        self.assertIn("latency_ms", body)
        self.assertTrue(body["structured_output_valid"])

        analysis = body["analysis"]
        self.assertIn("interpreted_goal", analysis)
        self.assertIn("instruction_quality", analysis)
        self.assertGreaterEqual(analysis["instruction_quality"]["score"], 0.0)
        self.assertLessEqual(analysis["instruction_quality"]["score"], 100.0)
        self.assertIn("strengths", analysis)
        self.assertIn("recommendations", analysis)

    def test_empty_prompt_is_rejected(self) -> None:
        status, body = _request("POST", "/api/v1/llm/analyze", {"prompt": ""})
        self.assertEqual(status, 422)

    def test_whitespace_prompt_is_rejected(self) -> None:
        status, body = _request("POST", "/api/v1/llm/analyze", {"prompt": "   \n\t  "})
        self.assertEqual(status, 422)

    def test_malformed_request_is_rejected(self) -> None:
        status, body = _request("POST", "/api/v1/llm/analyze", {"prompt": 12345})
        self.assertEqual(status, 422)

    def test_health_endpoint(self) -> None:
        status, body = _request("GET", "/api/v1/llm/health", None)
        self.assertEqual(status, 200)
        self.assertIn("status", body)
        self.assertIn("model", body)
        self.assertIn("provider", body)

    def test_long_prompt_with_mock_provider(self) -> None:
        mock_provider = MockLLMProvider(model_name="mock-test-llm")
        test_service = LLMService(
            config=LLMConfig(enabled=True, provider="mock", context_length=1024),
            provider=mock_provider,
        )
        set_llm_service(test_service)

        long_prompt = "You are a software architect. Design a system. " * 200
        status, body = _request("POST", "/api/v1/llm/analyze", {"prompt": long_prompt})
        self.assertEqual(status, 200)
        self.assertTrue(body["structured_output_valid"])


def _request(method: str, path: str, payload: dict[str, Any] | None) -> tuple[int, dict[str, Any]]:
    body = json.dumps(payload).encode() if payload is not None else b""
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

    headers = [(b"content-type", b"application/json")] if payload is not None else []
    scope: dict[str, Any] = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": headers,
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
