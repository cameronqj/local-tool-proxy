#!/usr/bin/env python3
"""
Reproducible benchmark harness for local Ollama models on Apple Silicon.
Matches the spirit of the original S/M/L testing in the README.
Uses Ollama HTTP API directly for precise eval_count / duration stats.

Also supports *direct* llama.cpp (local GGUF + custom builds) for MTP/draft-spec
testing using the same S/M/L prompts (for cases where Ollama does not expose
the drafter or MTP path, or to measure specific -ngl / build variants).
"""

import json
import re
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

# Direct llama.cpp (custom build / local GGUF / MTP drafter) support.
# These bypass Ollama so we can exercise --spec-type draft-mtp etc and exact -ngl.
# Tweak ngl / ctx / ngld to fit your hardware's memory and offload headroom.
DIRECT_LLAMA = str(Path.home() / "llama.cpp-mtp" / "build" / "bin" / "llama-cli")
# Note: this build's llama-cli prints a warning that --no-conversation is not supported
# and suggests llama-completion. However, the MTP --spec-type draft-mtp + --model-draft
# support we need for the 12B QAT tests lives in the llama-cli binary in this fork.
# llama-completion supports the no-conv flags but may not carry the full spec-draft-mtp changes.
# We therefore stay on llama-cli for direct MTP runs and tolerate the warning + residual UI.

DIRECT_MODELS = {
    # 12B QAT MTP via the downloaded local files + MTP-enabled build. Full offload works with
    # --no-conversation --single-turn --reasoning off (chat/thinking mode + drafter was the OOM trigger).
    "gemma4:12b-qat-mtp": {
        "label": "gemma4-12b-qat-mtp (direct)",
        "model": "~/models/gemma4-12b-qat/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf",
        "model_draft": "~/models/gemma4-12b-qat/mtp-gemma-4-12B-it.gguf",
        "spec_type": "draft-mtp",
        "spec_draft_n_max": 2,
        "ngl": 48,
        "ngld": 999,
        "ctx": 4096,
        "extra": ["--no-conversation", "--single-turn", "--reasoning", "off"],
    },
    # Non-MTP 12B QAT local baseline (same weights, no drafter).
    "gemma4:12b-qat": {
        "label": "gemma4-12b-qat (direct)",
        "model": "~/models/gemma4-12b-qat/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf",
        "model_draft": None,
        "ngl": 48,
        "ngld": 0,
        "ctx": 4096,
        "extra": ["--no-conversation", "--single-turn", "--reasoning", "off"],
    },
}

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


# --- Direct llama.cpp support (for MTP / local GGUF runs) ---

def _parse_llama_speeds(text: str) -> dict:
    """Parse either classic llama_print_timings or the live [ Prompt: X t/s | Generation: Y t/s ] lines."""
    prefill = 0.0
    decode = 0.0
    # live banner used by this build
    m = re.search(r'\[ Prompt:\s*([\d.]+)\s*t/s\s*\|\s*Generation:\s*([\d.]+)\s*t/s', text)
    if m:
        prefill = float(m.group(1))
        decode = float(m.group(2))
    else:
        # classic
        m = re.search(r'prompt eval time =.*?([\d.]+)\s+tokens per second', text)
        if m: prefill = float(m.group(1))
        m = re.search(r'eval time =.*?([\d.]+)\s+tokens per second', text)
        if m: decode = float(m.group(1))
    # also try to grab token counts if present
    pc = 0
    m = re.search(r'prompt eval time =.*?/\s*(\d+)\s*tokens', text)
    if m: pc = int(m.group(1))
    ec = 0
    m = re.search(r'eval time =.*?/\s*(\d+)\s*(runs|tokens)', text)
    if m: ec = int(m.group(1))
    # fallback counts from live if present (rare)
    return {"prefill_tok_s": prefill, "decode_tok_s": decode, "prompt_eval_count": pc, "eval_count": ec}


def run_direct(model_key: str, prompt: str, num_predict: int = 300) -> dict:
    if model_key not in DIRECT_MODELS:
        raise ValueError(f"Unknown direct model key: {model_key}. Known: {list(DIRECT_MODELS)}")
    cfg = DIRECT_MODELS[model_key]
    llama = DIRECT_LLAMA
    model_path = str(Path(cfg["model"]).expanduser())
    cmd = [
        llama,
        "--model", model_path,
        "-c", str(cfg.get("ctx", 4096)),
        "-ngl", str(cfg.get("ngl", 48)),
        "--temp", "0", "--top-p", "0.95", "--top-k", "64",
        "-p", prompt,
        "-n", str(num_predict),
        "--no-display-prompt",
    ]
    if cfg.get("model_draft"):
        draft_path = str(Path(cfg["model_draft"]).expanduser())
        cmd += ["--model-draft", draft_path]
        if cfg.get("spec_type"):
            cmd += ["--spec-type", cfg["spec_type"]]
        if cfg.get("spec_draft_n_max") is not None:
            cmd += ["--spec-draft-n-max", str(cfg["spec_draft_n_max"])]
        if cfg.get("ngld"):
            cmd += ["-ngld", str(cfg["ngld"])]
    cmd += cfg.get("extra", [])
    label = cfg.get("label", model_key)
    print(f"    [direct llama.cpp] {label}  ngl={cfg.get('ngl')} ngld={cfg.get('ngld')} ctx={cfg.get('ctx')} n={num_predict}")
    start = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        wall = time.time() - start
        out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        speeds = _parse_llama_speeds(out)

        # Best-effort extraction of the actual generated text (between last prompt echo / thinking and the speed banner)
        generated = ""
        m = re.search(r'(?:\n> .*?\n|\[ Start thinking\].*?\n)(.*?)(?:\n\[ Prompt:|\nExiting|\nllama_print)', out, re.DOTALL)
        if m:
            generated = " ".join(m.group(1).split())[:280]

        # If the live banner gave us speeds but 0 counts, synthesize rough counts from requested
        # (we mainly care about the reported tok/s for these direct runs)
        result = {
            "model": label,
            "total_duration_s": wall,
            "prompt_eval_count": speeds.get("prompt_eval_count") or 0,
            "prompt_eval_duration_ns": 0,
            "eval_count": speeds.get("eval_count") or num_predict,
            "eval_duration_ns": 0,
            "done": proc.returncode == 0,
            "prefill_tok_s": round(speeds.get("prefill_tok_s", 0), 1),
            "decode_tok_s": round(speeds.get("decode_tok_s", 0), 1),
            "approx_prompt_tokens": count_approx_tokens(prompt),
            "direct_cmd_tail": out[-400:].strip(),
            "generated_preview": generated,
        }
        if proc.returncode != 0:
            print("    WARN: non-zero exit. tail:", out[-300:])
        return result
    except subprocess.TimeoutExpired:
        return {
            "model": label, "total_duration_s": 300, "prompt_eval_count": 0,
            "eval_count": 0, "done": False, "prefill_tok_s": 0, "decode_tok_s": 0,
            "error": "timeout",
        }

def benchmark_model(model: str, sizes=("S", "M", "L")):
    print(f"\n=== Benchmarking {model} ===")
    prompts = {
        "S": SHORT_PROMPT,
        "M": MEDIUM_PROMPT,
        "L": LONG_PROMPT,
    }
    is_direct = model in DIRECT_MODELS
    for size in sizes:
        prompt = prompts[size]
        n_predict = 300 if size != "L" else 200  # be gentler on L for mem
        print(f"  Running {size} (approx {count_approx_tokens(prompt)} prompt tok, requesting {n_predict} gen tok)...")
        try:
            if is_direct:
                raw = run_direct(model, prompt, num_predict=n_predict)
                pre = raw.get("prefill_tok_s", 0)
                dec = raw.get("decode_tok_s", 0)
                pc = raw.get("prompt_eval_count", 0)
                ec = raw.get("eval_count", 0)
                print(f"    prefill: {pre} tok/s ({pc} tok)")
                print(f"    decode : {dec} tok/s ({ec} tok)")
                print(f"    total wall: {raw.get('total_duration_s', 0):.1f}s")
                if raw.get("generated_preview"):
                    print(f"    sample: {raw['generated_preview'][:220]}{'...' if len(raw.get('generated_preview',''))>220 else ''}")
                if not raw.get("done"):
                    print(f"    (note: direct run had issues; tail: {raw.get('direct_cmd_tail', '')[:160]})")
            else:
                raw = run_generate(model, prompt, num_predict=n_predict)
                stats = compute_stats(raw)
                print(f"    prefill: {stats['prefill_tok_s']} tok/s ({stats['prompt_eval_count']} tok)")
                print(f"    decode : {stats['decode_tok_s']} tok/s ({stats['eval_count']} tok)")
                print(f"    total wall: {stats['total_duration_s']:.1f}s")
        except Exception as e:
            print(f"    ERROR on {size}: {e}")
        time.sleep(2)  # brief cool-down between runs

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python benchmark.py <model1> [model2 ...]")
        print("Ollama example: python benchmark.py gpt-oss:20b qwen3.5:9b-nvfp4 gemma4:e4b-mlx")
        print("Direct (MTP/local GGUF) example: python benchmark.py gemma4:12b-qat-mtp gemma4:12b-qat")
        print("Direct keys:", list(DIRECT_MODELS.keys()))
        sys.exit(1)
    models = sys.argv[1:]
    for m in models:
        benchmark_model(m)
    print("\nDone. Record resident memory separately via Activity Monitor / `memory_pressure` / `ollama ps` during the L runs.")
