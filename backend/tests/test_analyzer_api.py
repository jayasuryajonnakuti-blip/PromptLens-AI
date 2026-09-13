"""Tests for POST /api/v1/analyzer/analyze endpoint (Step 15).

Uses ASGI test harness (no network) with MockLLMProvider injected via
set_llm_service() for clean test isolation.  set_analyzer_service(None)
is always called in tearDown() to reset the singleton.
"""
import asyncio
import json
import unittest

from app.main import app
from app.llm.model import MockLLMProvider
from app.llm.schemas import LLMHealthStatus
from app.services.analyzer_service import set_analyzer_service
from app.services.llm_service import LLMService, set_llm_service


# ---------------------------------------------------------------------------
# Analyzer-compatible mock provider (same as test_analyzer.py)
# ---------------------------------------------------------------------------
class AnalyzerMockProvider(MockLLMProvider):
    """Mock provider returning analyzer-compatible JSON for endpoint tests."""

    def _valid_payload(self) -> str:
        return json.dumps({
            "interpreted_goal": "Implement a login endpoint with token-based authentication.",
            "context_summary": "Web backend context, developer role.",
            "intent_label": "code_generation",
            "intent_confidence": 0.85,
            "ambiguities": [
                {"issue": "Token type not specified.", "evidence": "No mention of JWT or session.", "confidence": 0.70}
            ],
            "missing_information": [
                {"item": "Programming language/framework", "why_it_matters": "Affects implementation.", "confidence": 0.80}
            ],
            "contradictions": [],
            "instruction_quality": {"score": 68.0, "reason": "Clear goal, missing constraints."},
            "strengths": [{"strength": "Explicit endpoint purpose.", "evidence": "Task stated clearly."}],
            "weaknesses": [{"weakness": "Missing error handling.", "evidence": "No constraint language."}],
            "recommendations": [
                {"recommendation": "Specify auth token type.", "reason": "Reduces ambiguity.", "priority": "HIGH"}
            ],
        })


# ---------------------------------------------------------------------------
# Minimal ASGI test harness (reuses pattern from Steps 13 & 14 tests)
# ---------------------------------------------------------------------------
def _call_endpoint(method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
    """Send a request to the ASGI app and return (status_code, json_body)."""
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


class TestAnalyzerEndpointUnavailableMode(unittest.TestCase):
    """Tests for UNAVAILABLE mode (LLM disabled) via the HTTP endpoint."""

    def setUp(self):
        from app.llm.config import LLMConfig

        config = LLMConfig(enabled=False, provider="transformers")
        provider = MockLLMProvider(health_status=LLMHealthStatus.LLM_DISABLED)
        svc = LLMService(config=config, provider=provider)
        set_llm_service(svc)

    def tearDown(self):
        set_llm_service(None)
        set_analyzer_service(None)

    def test_200_response_unavailable(self):
        status, body = _call_endpoint(
            "POST",
            "/api/v1/analyzer/analyze",
            {"prompt": "Write a Python function to sort a list."},
        )
        self.assertEqual(status, 200)

    def test_response_has_analysis_key(self):
        _, body = _call_endpoint(
            "POST",
            "/api/v1/analyzer/analyze",
            {"prompt": "Explain the difference between REST and GraphQL."},
        )
        self.assertIn("analysis", body)

    def test_analysis_mode_is_unavailable(self):
        _, body = _call_endpoint(
            "POST",
            "/api/v1/analyzer/analyze",
            {"prompt": "Explain the difference between REST and GraphQL."},
        )
        mode = body["analysis"]["metadata"]["analysis_mode"]
        self.assertEqual(mode, "UNAVAILABLE")

    def test_prompt_length_in_response(self):
        prompt = "Summarize the following article in three bullet points."
        _, body = _call_endpoint("POST", "/api/v1/analyzer/analyze", {"prompt": prompt})
        self.assertEqual(body["prompt_length"], len(prompt))

    def test_step13_score_in_response(self):
        _, body = _call_endpoint(
            "POST",
            "/api/v1/analyzer/analyze",
            {"prompt": "Write a unit test for the add() function."},
        )
        self.assertIn("step13_overall_score", body)
        self.assertIn("step13_quality_category", body)
        self.assertIsInstance(body["step13_overall_score"], (int, float))

    def test_evidence_is_list(self):
        _, body = _call_endpoint(
            "POST",
            "/api/v1/analyzer/analyze",
            {"prompt": "You are a senior developer. Build a REST API."},
        )
        self.assertIsInstance(body["analysis"]["evidence"], list)

    def test_422_on_blank_prompt(self):
        status, _ = _call_endpoint("POST", "/api/v1/analyzer/analyze", {"prompt": "   "})
        self.assertEqual(status, 422)

    def test_422_on_empty_prompt(self):
        status, _ = _call_endpoint("POST", "/api/v1/analyzer/analyze", {"prompt": ""})
        self.assertEqual(status, 422)

    def test_422_on_missing_prompt_field(self):
        status, _ = _call_endpoint("POST", "/api/v1/analyzer/analyze", {"not_prompt": "x"})
        self.assertEqual(status, 422)


class TestAnalyzerEndpointMockMode(unittest.TestCase):
    """Tests for MOCK mode (AnalyzerMockProvider valid output)."""

    def setUp(self):
        from app.llm.config import LLMConfig

        config = LLMConfig(enabled=True, provider="mock", max_retries=2)
        provider = AnalyzerMockProvider(mode="valid")
        svc = LLMService(config=config, provider=provider)
        set_llm_service(svc)

    def tearDown(self):
        set_llm_service(None)
        set_analyzer_service(None)

    def test_200_response_mock_mode(self):
        status, _ = _call_endpoint(
            "POST",
            "/api/v1/analyzer/analyze",
            {"prompt": "Build a login endpoint with token authentication."},
        )
        self.assertEqual(status, 200)

    def test_mode_is_mock(self):
        _, body = _call_endpoint(
            "POST",
            "/api/v1/analyzer/analyze",
            {"prompt": "Build a login endpoint with token authentication."},
        )
        mode = body["analysis"]["metadata"]["analysis_mode"]
        self.assertEqual(mode, "MOCK")

    def test_interpreted_goal_populated(self):
        _, body = _call_endpoint(
            "POST",
            "/api/v1/analyzer/analyze",
            {"prompt": "Build a login endpoint with token authentication."},
        )
        self.assertGreater(len(body["analysis"]["interpreted_goal"]), 0)

    def test_intent_present(self):
        _, body = _call_endpoint(
            "POST",
            "/api/v1/analyzer/analyze",
            {"prompt": "Build a login endpoint with token authentication."},
        )
        intent = body["analysis"]["intent"]
        self.assertIn("label", intent)
        self.assertIn("confidence", intent)

    def test_instruction_quality_score_in_range(self):
        _, body = _call_endpoint(
            "POST",
            "/api/v1/analyzer/analyze",
            {"prompt": "Build a login endpoint with token authentication."},
        )
        score = body["analysis"]["instruction_quality"]["score"]
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 100.0)

    def test_recommendations_have_priority(self):
        _, body = _call_endpoint(
            "POST",
            "/api/v1/analyzer/analyze",
            {"prompt": "Build a login endpoint with token authentication."},
        )
        for rec in body["analysis"].get("recommendations", []):
            self.assertIn(rec["priority"], ("HIGH", "MEDIUM", "LOW"))

    def test_step13_score_present(self):
        _, body = _call_endpoint(
            "POST",
            "/api/v1/analyzer/analyze",
            {"prompt": "Build a login endpoint with token authentication."},
        )
        self.assertIn("step13_overall_score", body)

    def test_analyzer_version_present(self):
        _, body = _call_endpoint(
            "POST",
            "/api/v1/analyzer/analyze",
            {"prompt": "Build a login endpoint with token authentication."},
        )
        version = body["analysis"]["metadata"]["analyzer_version"]
        self.assertGreater(len(version), 0)

    def test_latency_ms_positive(self):
        _, body = _call_endpoint(
            "POST",
            "/api/v1/analyzer/analyze",
            {"prompt": "Build a login endpoint with token authentication."},
        )
        latency = body["analysis"]["metadata"]["latency_ms"]
        self.assertGreaterEqual(latency, 0.0)


class TestAnalyzerEndpointResponseStructure(unittest.TestCase):
    """Verify the complete structure of the response JSON."""

    def setUp(self):
        from app.llm.config import LLMConfig

        config = LLMConfig(enabled=True, provider="mock", max_retries=1)
        provider = AnalyzerMockProvider(mode="valid")
        svc = LLMService(config=config, provider=provider)
        set_llm_service(svc)

    def tearDown(self):
        set_llm_service(None)
        set_analyzer_service(None)

    def test_top_level_keys(self):
        _, body = _call_endpoint(
            "POST",
            "/api/v1/analyzer/analyze",
            {"prompt": "You are a data scientist. Analyze the sales dataset for anomalies."},
        )
        for key in ("analysis", "prompt_length", "step13_overall_score", "step13_quality_category"):
            self.assertIn(key, body, f"Missing top-level key: {key}")

    def test_analysis_keys(self):
        _, body = _call_endpoint(
            "POST",
            "/api/v1/analyzer/analyze",
            {"prompt": "You are a data scientist. Analyze the sales dataset for anomalies."},
        )
        analysis = body["analysis"]
        for key in (
            "interpreted_goal",
            "intent",
            "context_summary",
            "ambiguities",
            "missing_information",
            "contradictions",
            "instruction_quality",
            "strengths",
            "weaknesses",
            "recommendations",
            "evidence",
            "metadata",
        ):
            self.assertIn(key, analysis, f"Missing analysis key: {key}")

    def test_metadata_keys(self):
        _, body = _call_endpoint(
            "POST",
            "/api/v1/analyzer/analyze",
            {"prompt": "Classify these customer reviews as positive or negative."},
        )
        meta = body["analysis"]["metadata"]
        for key in ("analyzer_version", "llm_model", "analysis_mode", "latency_ms"):
            self.assertIn(key, meta, f"Missing metadata key: {key}")


if __name__ == "__main__":
    unittest.main()
