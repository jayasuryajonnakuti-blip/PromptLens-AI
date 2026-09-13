import asyncio
import json
import unittest
from typing import Any

from app.main import app


class QualityApiTests(unittest.TestCase):
    def test_valid_quality_prediction_request(self) -> None:
        status, body = _request(
            {"prompt": "Explain machine learning to a beginner using a simple example."}
        )

        self.assertEqual(status, 200)
        self.assertIn("score", body)
        self.assertIn("label", body)
        self.assertIn("model", body)
        self.assertGreaterEqual(body["score"], 0.0)
        self.assertLessEqual(body["score"], 100.0)
        self.assertIn(body["label"], ["POOR", "FAIR", "GOOD", "STRONG", "EXCELLENT"])
        self.assertEqual(body["model"]["name"], "promptlens-quality-model")
        self.assertEqual(body["model"]["version"], "0.1.0")

    def test_empty_prompt_is_rejected(self) -> None:
        status, body = _request({"prompt": ""})

        self.assertEqual(status, 422)
        self.assertIn("detail", body)

    def test_whitespace_prompt_is_rejected(self) -> None:
        status, body = _request({"prompt": "   \n\t  "})

        self.assertEqual(status, 422)
        self.assertIn("detail", body)

    def test_malformed_request_body_is_rejected(self) -> None:
        status, body = _request({"prompt": 12345})

        self.assertEqual(status, 422)
        self.assertIn("detail", body)

    def test_missing_prompt_field_is_rejected(self) -> None:
        status, body = _request({"text": "wrong key"})

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

    path = "/api/v1/quality/predict"
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
