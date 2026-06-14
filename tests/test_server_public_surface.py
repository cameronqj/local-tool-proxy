import json
from unittest.mock import AsyncMock, patch

import httpx
from fastapi.testclient import TestClient

import proxy.server as srv


def test_upstream_url_avoids_duplicate_v1_prefix():
    previous = srv.OLLAMA_BASE
    try:
        srv.OLLAMA_BASE = "http://localhost:11434/v1"
        assert srv._upstream_url("models") == "http://localhost:11434/v1/models"
        assert srv._upstream_url("v1/models") == "http://localhost:11434/v1/models"

        srv.OLLAMA_BASE = "http://localhost:11434"
        assert srv._upstream_url("models") == "http://localhost:11434/v1/models"
        assert srv._upstream_url("v1/models") == "http://localhost:11434/v1/models"
    finally:
        srv.OLLAMA_BASE = previous


def test_models_endpoint_proxies_upstream_and_marks_compat():
    client = TestClient(srv.app)
    srv.COMPAT_MODELS = {"gemma4:e4b-mlx"}

    upstream_response = httpx.Response(
        200,
        json={
            "object": "list",
            "data": [
                {"id": "gemma4:e4b-mlx", "object": "model"},
                {"id": "llama3.2", "object": "model"},
            ],
        },
        headers={"content-type": "application/json"},
    )

    with patch.object(srv.CLIENT, "get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = upstream_response
        resp = client.get("/v1/models")

    assert resp.status_code == 200
    data = resp.json()["data"]
    assert [m["id"] for m in data] == ["gemma4:e4b-mlx", "llama3.2"]
    assert data[0]["local_tool_proxy_compat"] is True
    assert data[1]["local_tool_proxy_compat"] is False


def test_models_endpoint_falls_back_to_configured_compat_models():
    client = TestClient(srv.app)
    srv.COMPAT_MODELS = {"gemma4:e4b-mlx"}

    with patch.object(srv.CLIENT, "get", new_callable=AsyncMock) as mock_get:
        mock_get.side_effect = httpx.ConnectError("no upstream")
        resp = client.get("/v1/models")

    assert resp.status_code == 200
    assert resp.json()["data"][0]["id"] == "gemma4:e4b-mlx"
    assert resp.json()["data"][0]["local_tool_proxy_compat"] is True


def test_stream_content_extraction_collects_content_and_metadata():
    chunks = [
        b'data: {"id":"abc","object":"chat.completion.chunk","created":1,'
        b'"model":"gemma4:e4b-mlx","choices":[{"delta":{"content":"run_command"}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"{\\"command\\": \\"ls\\"}"}}]}\n\n',
        b"data: [DONE]\n\n",
    ]

    content, metadata = srv._extract_stream_content(chunks)

    assert content == 'run_command{"command": "ls"}'
    assert metadata["id"] == "abc"
    assert metadata["model"] == "gemma4:e4b-mlx"


def test_trace_event_writes_sanitized_jsonl(tmp_path):
    trace_file = tmp_path / "trace.jsonl"
    previous = srv.TRACE_FILE
    srv.TRACE_FILE = trace_file
    try:
        srv._write_trace_event(
            "ltp-test",
            "rewrite",
            model="gemma4:e4b-mlx",
            tool_names=["write_file"],
        )
    finally:
        srv.TRACE_FILE = previous

    record = json.loads(trace_file.read_text().strip())
    assert record["trace_id"] == "ltp-test"
    assert record["event"] == "rewrite"
    assert record["tool_names"] == ["write_file"]
    assert "prompt" not in record
    assert "content" not in record
