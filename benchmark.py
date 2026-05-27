#!/usr/bin/env python3
"""
Reproducible benchmark harness for local Ollama models on Apple Silicon.
Matches the spirit of the original S/M/L testing in the README.
Uses Ollama HTTP API directly for precise eval_count / duration stats.
"""

import json
import time
import requests
import subprocess
import sys
from pathlib import Path

OLLAMA_HOST = "http://localhost:11434"
API_GENERATE = f"{OLLAMA_HOST}/api/generate"
API_CHAT = f"{OLLAMA_HOST}/api/chat"

# Representative prompts approximating the original lengths (S~59 tok, M~1800, L~8500+)
# Self-contained and realistic for coding/reasoning workloads.

SHORT_PROMPT = "Explain the key differences between synchronous and asynchronous I/O in Python, including when to use each and one common pitfall. Answer in one concise paragraph."

MEDIUM_PROMPT = '''You are an expert Python engineer. Here is a small but realistic FastAPI dependency:

```python
from fastapi import Depends, FastAPI
from sqlalchemy.orm import Session
from . import models, schemas

app = FastAPI()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/items")
def read_items(db: Session = Depends(get_db), skip: int = 0, limit: int = 100):
    return db.query(models.Item).offset(skip).limit(limit).all()
```

Problems visible: N+1 risk on relationships, no pagination metadata, sync session, missing response model, no error handling.

Refactor this endpoint + dependency for 2026 best practices (async-capable, proper pagination response, lifespan DB management, Pydantic v2 if relevant, clear error responses). Provide the full improved code snippet and list the 3 most important changes with one-sentence justification each.'''

LONG_PROMPT = '''You are reviewing a legacy 3-file Python microservice for a URL shortener (already checked out in the test dir).

Files involved (realistic sizes):
- app/main.py (~420 LOC, messy)
- app/models.py
- app/schemas.py
- tests/ with some existing tests
- requirements.txt pinned to old versions
- README with outdated instructions

Specific requirements for this task:
- Security: fix any obvious injection, missing auth on admin routes, weak token generation.
- Performance: identify and fix N+1, missing indexes suggestions, synchronous I/O in hot paths.
- Maintainability: add type hints everywhere, improve error handling to structured JSON, update to modern dependency injection.
- Add or update tests for the fixed critical paths.
- Produce a short migration guide in MIGRATION.md.

The prompt context is long on purpose to simulate real 8k-16k token agent sessions with full specs + existing code + constraints. Do the analysis, make the minimal set of high-impact edits using tools, run the test suite, and confirm everything still works. Prioritize correctness over cleverness.'''

def count_approx_tokens(text: str) -> int:
    return max(1, len(text.split()) * 4 // 3)  # rough

def run_generate(model: str, prompt: str, num_predict: int = 300, temperature: float = 0.0, num_ctx: int = 32768):
    """Use /api/generate for clean prefill + decode stats."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
            "num_ctx": num_ctx,
            "top_p": 0.95,
        },
    }
    start = time.time()
    resp = requests.post(API_GENERATE, json=payload, timeout=300)
    elapsed = time.time() - start
    resp.raise_for_status()
    data = resp.json()
    return {
        "total_duration_s": elapsed,
        "prompt_eval_count": data.get("prompt_eval_count", 0),
        "prompt_eval_duration_ns": data.get("prompt_eval_duration", 0),
        "eval_count": data.get("eval_count", 0),
        "eval_duration_ns": data.get("eval_duration", 0),
        "done": data.get("done", False),
        "model": model,
    }

def compute_stats(result: dict) -> dict:
    prefill_tok_s = 0.0
    if result["prompt_eval_duration_ns"] > 0 and result["prompt_eval_count"] > 0:
        prefill_tok_s = result["prompt_eval_count"] / (result["prompt_eval_duration_ns"] / 1e9)
    decode_tok_s = 0.0
    if result["eval_duration_ns"] > 0 and result["eval_count"] > 0:
        decode_tok_s = result["eval_count"] / (result["eval_duration_ns"] / 1e9)
    return {
        **result,
        "prefill_tok_s": round(prefill_tok_s, 1),
        "decode_tok_s": round(decode_tok_s, 1),
        "approx_prompt_tokens": count_approx_tokens(result.get("prompt", "")),
    }

def benchmark_model(model: str, sizes=("S", "M", "L")):
    print(f"\n=== Benchmarking {model} ===")
    prompts = {
        "S": SHORT_PROMPT,
        "M": MEDIUM_PROMPT,
        "L": LONG_PROMPT,
    }
    for size in sizes:
        prompt = prompts[size]
        print(f"  Running {size} (approx {count_approx_tokens(prompt)} prompt tok, requesting 300 gen tok)...")
        try:
            raw = run_generate(model, prompt)
            stats = compute_stats(raw)
            print(f"    prefill: {stats['prefill_tok_s']} tok/s ({stats['prompt_eval_count']} tok)")
            print(f"    decode : {stats['decode_tok_s']} tok/s ({stats['eval_count']} tok)")
            print(f"    total wall: {stats['total_duration_s']:.1f}s")
        except Exception as e:
            print(f"    ERROR on {size}: {e}")
        time.sleep(2)  # brief cool-down between runs on fanless

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python benchmark.py <model1> [model2 ...]")
        print("Example: python benchmark.py gpt-oss:20b qwen3.5:9b-nvfp4 gemma4:e4b-mlx")
        sys.exit(1)
    models = sys.argv[1:]
    for m in models:
        benchmark_model(m)
    print("\nDone. Record resident memory separately via Activity Monitor / `memory_pressure` / `ollama ps` during the L runs.")
