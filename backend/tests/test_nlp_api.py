import asyncio
import json
import unittest
from typing import Any

from app.main import app


class NlpApiTests(unittest.TestCase):
    def test_nlp_endpoint(self) -> None:
        status, body = _request(
            {"prompt": "Explain gravity. Return the answer as JSON."}
        )

        self.assertEqual(status, 200)
        self.assertGreater(body["token_count"], 0)
        self.assertIn("sentence_statistics", body)


def _request(payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    request_sent = False
    body = json.dumps(payload).encode()

    async def receive() -> dict[str, Any]:
        nonlocal request_sent
        if request_sent:
            return {"type": "http.disconnect"}
        request_sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        messages.append(message)

    path = "/api/v1/nlp/analyze"
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