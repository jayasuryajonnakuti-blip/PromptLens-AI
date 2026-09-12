import asyncio
import json
import unittest
from typing import Any

from app.main import app


class EmbeddingApiTests(unittest.TestCase):
    def test_generate_embedding_endpoint(self) -> None:
        status, body = _request("/api/v1/embeddings/generate", {"prompt": "Explain machine learning."})

        self.assertEqual(status, 200)
        self.assertEqual(body["dimensions"], 384)
        self.assertEqual(len(body["embedding"]), 384)
        self.assertTrue(body["normalized"])
        self.assertIn("name", body["model"])

    def test_similarity_endpoint(self) -> None:
        status, body = _request(
            "/api/v1/embeddings/similarity",
            {"prompt_a": "Explain machine learning.", "prompt_b": "Describe machine learning."},
        )

        self.assertEqual(status, 200)
        self.assertEqual(body["metric"], "cosine")
        self.assertGreaterEqual(body["similarity"], -1.0)
        self.assertLessEqual(body["similarity"], 1.0)

    def test_empty_embedding_request_is_rejected(self) -> None:
        status, body = _request("/api/v1/embeddings/generate", {"prompt": " "})

        self.assertEqual(status, 422)
        self.assertIn("detail", body)


def _request(path: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
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
    status = next(message["status"] for message in messages if message["type"] == "http.response.start")
    response_body = b"".join(
        message.get("body", b"")
        for message in messages
        if message["type"] == "http.response.body"
    )
    return status, json.loads(response_body)


if __name__ == "__main__":
    unittest.main()