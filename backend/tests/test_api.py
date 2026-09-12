import asyncio
import json
import unittest
from typing import Any

from app.main import app


class ApiTests(unittest.TestCase):
    def test_health_endpoint(self) -> None:
        status, body = _request("GET", "/health")

        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "ok")

    def test_preprocess_endpoint(self) -> None:
        status, body = _request(
            "POST",
            "/api/v1/preprocess",
            {"prompt": "Explain gravity as JSON."},
        )

        self.assertEqual(status, 200)
        self.assertEqual(body["text"]["word_count"], 4)
        self.assertIn("JSON", body["output_formats"])

    def test_preprocess_rejects_invalid_request_body(self) -> None:
        status, body = _request("POST", "/api/v1/preprocess", {"prompt": 42})

        self.assertEqual(status, 422)
        self.assertIn("detail", body)


def _request(
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
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