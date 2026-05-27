# Local LLMs on a 24 GB MacBook Air (M4) — Field Notes

What actually works, what doesn't, and the specific gotchas that don't appear in any vendor marketing. Built from empirical testing on **one specific machine**:

- **MacBook Air 15" (M4, 2025)** — Mac16,13, base M4 (10-core CPU, 10-core GPU)
- **24 GB unified memory**
- **Fanless** — sustained throughput will throttle thermally
- macOS 26.5

If you have an M-series Pro / Max with active cooling and 32 GB+ unified, **almost everything below gets easier and faster**. The numbers and constraints here are the *worst-case-but-most-common* MacBook profile.

---

## TL;DR — what to actually use

| Want | Use |
|---|---|
| Best daily-driver chat / coding / long context | **`ollama run gpt-oss:20b`** — ~24 tok/s, 14 GB resident, MoE so bandwidth-sparse |
| Lighter / safer footprint | `ollama run qwen3.5:9b-nvfp4` — ~13 tok/s, 8 GB resident |
| Tool calling / structured output / agentic workflows | **Unsloth Studio** + `Qwen3.5-9B-MTP-GGUF UD-Q4_K_XL` (NOT gpt-oss) |
| Max model quality, willing to wait | MTPLX + `Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed` — ~5-7 tok/s, 20 GB resident |
| Anything 27B+ at long context (>16k) | **Don't.** You will hit the Metal cap. See below. |

---

## The single most important fact about Apple Silicon LLMs

**On a 24 GB MacBook Air, you have ~19 GB of usable GPU memory, not 24 GB.**

macOS imposes a per-process **Metal "working set" cap** (`recommendedMaxWorkingSetSize`) which on a 24 GB Air is approximately **19 GB**. This is the real ceiling for model weights + KV cache + compute scratch buffers. Exceed it and you get:

```
error: Insufficient Memory (kIOGPUCommandBufferCallbackErrorOutOfMemory)
ggml_metal_synchronize: error: command buffer 0 failed with status 5
llama_decode: failed to decode, ret = -3
```

This appears in the wild as **"Compute error"** from Unsloth Studio / LM Studio / llama-server.

**Practical rules:**
- A model whose download size is ≤14 GB is comfortable.
- 15-17 GB models work only with short context (8k-16k) and no other GPU-heavy apps.
- 18+ GB models are unreliable; will load and immediately OOM during decode.
- The vendor "Needs 20 GB RAM" line you see on Twitter/HF cards is **model footprint**, not system requirement. Subtract ~5 GB from your unified memory total to get the real ceiling.

Quick check:
```bash
grep -aE 'recommendedMaxWorkingSet' /tmp/lsv*.log 2>/dev/null
# Or after a model loads:
grep -aE 'recommendedMaxWorkingSetSize|peak_memory_bytes' <server-log>
```

---

## MoE memory math — the other common confusion

For Mixture-of-Experts models (gpt-oss-20b, Qwen3.6-35B-A3B, Qwen3-Coder-30B-A3B, etc.):

- **Resident memory (does it fit?)** scales with **total parameters**. All experts must be loaded. A 21B MoE needs ~13 GB at MXFP4 just like a dense 21B would.
- **Generation tok/s (how fast?)** scales with **active parameters per token**. gpt-oss-20b has 3.6B active = only ~1.9 GB of weights read per token = bandwidth ÷ 1.9 GB ≈ 60 tok/s theoretical on M4 Air.

This is why gpt-oss-20b runs at ~24 tok/s while a dense 9B runs at ~13 — the MoE is actually faster despite being bigger.

**Common error:** assuming a "3B-active" MoE has a 3B memory footprint. It does not. The 35B-A3B is a 32 GB+ machine model regardless of its sparse compute.

---

## Quantization formats — what actually differs

| Format | What it is | Speed on Apple Silicon | Notes |
|---|---|---|---|
| **MXFP4** | Microscaling 4-bit float (OpenAI) | Native, fast | gpt-oss ships in this format. Excellent quality:size ratio. |
| **nvfp4** | NVIDIA 4-bit float | **MLX path engages** on Ollama 0.19+ | Use for Qwen if you want MLX speed via Ollama. |
| **mxfp4** / **mxfp8** | Microscaling float (various bits) | MLX path engages | Newer Ollama tag prefixes. |
| **q4_K_M / q5_K_M / q8_0** | llama.cpp GGUF k-quants | **Falls through to llama.cpp/Metal, not MLX** | ~20-30% slower than nvfp4 of similar size. |
| **UD-Q4_K_XL** etc | Unsloth Dynamic 2.0 GGUF | llama.cpp/Metal | Higher quality than standard Q4 at similar size. |
| **bf16 / fp16** | Half precision | MLX or llama.cpp | 2× the memory of any 4-bit variant. Rarely justified on 24 GB. |

**The Ollama-MLX engagement rule:** when you `ollama pull qwen3.5:9b-q8_0` you get a GGUF that runs via llama.cpp/Metal (slower). When you pull `qwen3.5:9b-nvfp4` (or any `-mlx-*` / `-nvfp4` / `-mxfp4` tag) you get the MLX path. **The "Ollama uses MLX since 0.19" headline is true but only for MLX-format tags, not default `:latest` GGUF tags.**

Verify which path is in use:
```bash
# Right after a model loads, check ollama serve logs:
grep -iE 'mlx engine|ggml_metal_init|starting mlx runner' <ollama-log>
# MLX path: "starting mlx runner subprocess" / "MLX engine initialized"
# llama.cpp path: "ggml_metal_init: picking default device: Apple M4"
```

---

## Speculative decoding / MTP — bimodal, not a free lunch

MTP (Multi-Token Prediction) and ngram-mod chained spec decoding give 2-3× speedups in marketing, but in practice on M4 Air:

| Workload | Draft acceptance | Speedup |
|---|---|---|
| Predictable / templated output (boilerplate code, JSON, structured response) | 60-93% | **2-3× faster** |
| Novel prose / creative writing / varied content | 35-45% | **2-3× SLOWER** than no MTP |

Why: each draft cycle costs N drafter passes + 1 verifier pass. Math says spec is net-positive only above ~50% acceptance. Below that, you're paying for wasted draft compute.

**Rule of thumb:** turn MTP on for tool-use/structured output, off for chat/creative.

| Stack | Has MTP | When to use |
|---|---|---|
| Ollama nvfp4 (qwen3.5:9b) | No | Daily chat |
| llama.cpp + `*-MTP-GGUF` (via Unsloth Studio) | Yes | Tool calling, agents (high draft acceptance regime) |
| MTPLX + Qwen3.6-MTPLX-Optimized-Speed | Yes (MLX-native) | 27B-quality with reasonable speed; ~2× MTP gain that holds up |

---

## Tools installed & where

### Ollama (recommended default runtime)
- **Install:** `brew install ollama` → `/opt/homebrew/opt/ollama/`
- **Models on disk:** `~/.ollama/models/` (blob store + manifests)
- **Start:** `OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve`
- **Auto-start at login:** `brew services start ollama`
- **OpenAI-compatible endpoint:** `http://localhost:11434/v1` (key can be any string)
- **Native endpoint:** `http://localhost:11434/api/chat` (richer — exposes `thinking` field, native `think:false`)

### Unsloth Studio (UI + tool-calling)
- Pre-existing on this machine. Config at `~/.unsloth/`.
- Bundled llama.cpp at `~/.unsloth/llama.cpp/llama-server`
- HF model cache at `~/.cache/huggingface/hub/` (shared with HF CLI)
- Best for: GUI experimentation, tool-calling with MTP-GGUF models, watching speculative-decoding stats live
- **Critical bug avoidance:** for MTP support need Studio ≥v0.1.405-beta (May 18, 2026) — earlier versions ship stale `llama-server` binary that pre-dates the May 13 `--spec-type mtp` → `--spec-type draft-mtp` flag rename.

### MTPLX (MLX-native MTP serving)
- **Install:** `python3 -m venv ~/mtplx-env && ~/mtplx-env/bin/pip install -U mtplx`
- **Models on disk:** `~/.mtplx/models/`
- **Config:** `~/.mtplx/config.toml`
- **Setup + download model:** `mtplx setup --model Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed --profile sustained --download --force`
- **Start server:** `mtplx quickstart --model Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed --port 58902 --profile sustained --mtp --reasoning off --no-stats-footer --yes`
- **Exposes:** OpenAI-compatible `/v1/chat/completions` + Anthropic-compatible `/v1/messages`
- Best for: 27B-class quality with real MTP speedup on Apple Silicon. Only one currently shipping that *actually delivers* MLX-native MTP for Qwen3.6.

### OpenCode (agentic harness)
- Pre-existing. Binary at `/opt/homebrew/bin/opencode`.
- Per-project config: `<project>/opencode.json`
- Auth: `~/.local/share/opencode/auth.json`
- For local Ollama models, add this to `opencode.json`:
  ```json
  {
    "$schema": "https://opencode.ai/config.json",
    "provider": {
      "ollama": {
        "npm": "@ai-sdk/openai-compatible",
        "name": "Ollama (local)",
        "options": { "baseURL": "http://localhost:11434/v1" },
        "models": { "qwen3.5:9b-nvfp4": { "name": "Qwen3.5 9B nvfp4" } }
      }
    }
  }
  ```
  Then `opencode providers list` should show `ollama` after seeding `auth.json` with any string for the `ollama` provider key.
- Run non-interactive: `opencode run --dir <dir> --model ollama/<tag> --print-logs --dangerously-skip-permissions "<prompt>"`
- **Background gotcha:** `opencode run` with stdout redirected and no TTY can silently hang in pre-Ollama bootstrap. Run foreground or use `--format json` for fully detached mode.

### Codex CLI (gpt-oss's native harness)
- **Install:** `npm i -g @openai/codex` → `/opt/homebrew/bin/codex`
- Config: `~/.codex/config.toml` and `~/.codex/auth.json`
- **Native Ollama support:** `--oss --local-provider ollama -m <model>`
- **Isolate from existing config:** `--ignore-user-config --ephemeral` runs without reading or writing your config
- Run: `codex exec --oss --local-provider ollama -m gpt-oss:20b --cd <dir> --skip-git-repo-check --dangerously-bypass-approvals-and-sandbox --ephemeral --ignore-user-config "<prompt>"`
- **Important:** default `reasoning effort: none` for `--oss` runs. gpt-oss tool-use degrades severely without reasoning. Override with `-c model_reasoning_effort='"medium"'`.

---

## Performance numbers (measured on this M4 Air)

Identical prompt set across all stacks: S=59 tokens, M=1821 tokens, L=8923 tokens. `temperature=0`, `max_tokens=300`, thinking off. **Generation tok/s steady-state:**

| Stack | Model | S | M | L | Resident |
|---|---|---|---|---|---|
| Ollama MLX | qwen3.5:9b-nvfp4 | 13.5 | 13.6 | 12.4 | 8 GB |
| Ollama llama.cpp | qwen3.5:9b-q8_0 | 8.2 | — | — | 12 GB |
| llama.cpp + MTP | Qwen3.5-9B-MTP-GGUF UD-Q4_K_XL | 32.3 ⚠ | 4.4 | 4.4 | 9 GB |
| MTPLX MLX-MTP | Qwen3.6-27B-MTPLX-Optimized-Speed | 6.9 | 6.4 | 4.8 | 20 GB |
| **Ollama MLX (MXFP4)** | **gpt-oss:20b** | **24.4** | **23.9** | **19.0** | **14 GB** |

⚠ MTP at S was 93% draft acceptance — textbook content. Drops to 35-45% on novel content where it becomes *net-negative*.

**Prefill (matters for long prompts):**
- Ollama MLX: 150-360 tok/s
- llama.cpp + MTP: 60-80 tok/s (slower)
- MTPLX 27B: 33-43 tok/s (slow, painful at long context: ~4.4 min for 9k-token prefill)
- gpt-oss MLX: 200-360 tok/s (best)

For headline comparisons: M5 Max benchmarks circulating on X (e.g. "63 tok/s on Qwen3.6-27B") will roughly **3-10× faster** on that hardware than on this M4 Air, both because of bandwidth (~120 vs ~400 GB/s) and active cooling.

---

## Setup recipes (working, copy-paste)

### Fastest path to a working local model
```bash
brew install ollama
brew services start ollama   # auto-start at login (optional)
ollama pull gpt-oss:20b
ollama run gpt-oss:20b "/no_think hello"
```

### Use as OpenAI-compatible endpoint
```bash
# Any tool that takes an OpenAI base URL works:
export OPENAI_BASE_URL=http://localhost:11434/v1
export OPENAI_API_KEY=ollama   # any non-empty string
```

### Force 8-bit KV cache to save memory at long context
```bash
# Drop ~50% off KV memory for context-heavy work
OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve
```

### Run gpt-oss with thinking effort control via API
```bash
curl -s http://localhost:11434/api/chat -d '{
  "model":"gpt-oss:20b",
  "messages":[{"role":"user","content":"explain X"}],
  "stream":false,
  "think":false,
  "options":{"temperature":0,"num_predict":300,"num_ctx":32768}
}'
# /v1/chat/completions does NOT honor the "think" flag — only /api/chat does.
```

### MTPLX 27B for a real "quality over speed" run
```bash
~/mtplx-env/bin/mtplx setup --model Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed \
  --profile sustained --download --force
~/mtplx-env/bin/mtplx quickstart \
  --model Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed \
  --port 58902 --profile sustained --mtp --reasoning off --no-stats-footer --yes
# Then POST to http://localhost:58902/v1/chat/completions
```

### Stop everything cleanly
```bash
# Ollama
ollama stop <model-name>   # unloads from VRAM but keeps server up
brew services stop ollama  # or kill the ollama serve process
# llama-server (Unsloth Studio or manual)
pkill -f llama-server
# MTPLX — IMPORTANT: must kill the worker, not the wrapper
pkill -9 -f mtplx.server.openai
```

---

## Troubleshooting playbook

### "Compute error" / "Insufficient Memory" mid-generation
**Cause:** Metal working-set cap (~19 GB) exceeded by weights + KV cache + scratch.
**Fix:** Reduce context length (`-c 16384` or `-c 8192`), use a smaller quant (UD-Q3_K_XL instead of Q4_K_XL), or use a smaller model.

### Model loads but generation is 1-3 tok/s
**Cause:** macOS is swapping pages to SSD because the working set won't fit.
**Verify:** `sysctl vm.swapusage` should show low used; `memory_pressure` should show >60% free.
**Fix:** Pick a smaller model or shorter context.

### Tool calls fail with "Model tried to call unavailable tool 'container.exec'"
**Cause:** You're running **gpt-oss in OpenCode (or similar harness with renamed tools)**. gpt-oss was trained against OpenAI Codex CLI's tool ontology (`container.exec`, `python`, `browser.*`, `assistant`). When a harness exposes a `bash` tool instead, gpt-oss confidently reaches for the trained-in name and fails.
**Fix:** Use **Codex CLI** for gpt-oss (`codex exec --oss --local-provider ollama -m gpt-oss:20b`). Or use a Qwen-family model in OpenCode (they follow declared tool names).

### Model generates a "what I would do" narrative instead of executing
**Cause:** Prompt contains action-fiction triggers:
- "**simulate** starting the server"
- "**show every command** you execute"
- "Every action or response must be **numbered**"
- "Never add extra commentary"

These reliably push gpt-oss into describe-mode regardless of harness or reasoning effort.
**Fix:** Rewrite the prompt in imperative, tool-direct form: *"Use the bash tool to git init. Use the write tool to create requirements.txt. Do not produce a narrative."*

### Ollama claims it's using MLX but performance feels like llama.cpp
**Cause:** You pulled a GGUF tag (`:q4_K_M`, `:q8_0`, `:latest` etc) which routes to llama.cpp/Metal, not MLX.
**Fix:** Pull an MLX-format tag (`:nvfp4`, `:mxfp4`, `:mlx-bf16`). Verify in `ollama serve` logs:
```
grep -iE 'starting mlx runner|MLX engine initialized' <log>
```

### Unsloth Studio "Compute error 500" on tool calls (but plain chat works)
**Cause:** Stale `llama-server` binary in Studio pre-dating May 2026 MTP flag rename, **OR** model is 17+ GB hitting Metal cap during tool-call grammar buffer allocation.
**Fix:**
1. Update Studio to ≥v0.1.405-beta (May 18, 2026).
2. Drop model size — e.g. UD-Q3_K_XL instead of UD-Q4_K_XL for 27B.
3. Drop context to ≤32k.

### `pkill -f "mtplx quickstart"` doesn't free memory
**Cause:** `mtplx quickstart` is a wrapper that spawns `python -m mtplx.server.openai` as the actual worker. `pkill -f mtplx quickstart` only matches the wrapper.
**Fix:**
```bash
pkill -9 -f mtplx.server.openai
pkill -9 -f 'tee.*mtplx_server.log'   # if tee was used
```
Verify with `pgrep -afl mtplx`.

### OpenCode run hangs silently when backgrounded
**Cause:** OpenCode wants a TTY for some setup paths. When stdout is fully detached (no terminal at all), it stalls during bootstrap.
**Fix:** Run foreground, or use `--format json` which is designed for detached mode.

### macOS reports 20 GB swap used even though Activity Monitor shows free memory
**Cause:** A previous large model load (16+ GB) got swapped out when something else needed memory. macOS doesn't proactively zero swap.
**Verify:** `sysctl vm.swapusage` shows the actual used swap. Check `pgrep -afl 'mtplx|llama-server|ollama'` for stuck processes holding ghost residency.
**Fix:** Properly kill leaked workers (see MTPLX gotcha above). After kill, swap shrinks within a few minutes.

---

## Disk layout & cleanup

| Path | Purpose | Typical size |
|---|---|---|
| `~/.ollama/models/` | Ollama blob store | 10-50 GB |
| `~/.mtplx/models/` | MTPLX-verified models | 15 GB per 27B model |
| `~/.cache/huggingface/hub/` | HF downloads (Unsloth Studio, MTPLX setup, direct hf-cli pulls) | grows large fast |
| `~/.unsloth/` | Unsloth Studio config + bundled llama.cpp | a few GB |
| `~/.codex/` | Codex CLI config + auth | small |
| `~/.local/share/opencode/` | OpenCode session DB + auth | small |
| `~/mtplx-env/` (or wherever you venv'd) | MTPLX Python deps | ~400 MB |

**Inspect what's eating space:**
```bash
du -sh ~/.ollama ~/.mtplx ~/.cache/huggingface ~/.unsloth ~/.codex 2>/dev/null
du -sh ~/.cache/huggingface/hub/models--* 2>/dev/null | sort -hr | head
```

**Safe deletes (won't break anything else):**
```bash
# Remove a specific Ollama model
ollama rm qwen3.5:9b-q8_0

# Remove an HF-cached model (Unsloth GGUFs, MTPLX downloads in HF cache, etc.)
rm -rf ~/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-MTP-GGUF

# Remove a MTPLX model
rm -rf ~/.mtplx/models/Youssofal--Qwen3.6-27B-MTPLX-Optimized-Speed
```

---

## Honest reality-check on common marketing claims

| Claim | Reality on M4 Air 24 GB |
|---|---|
| "Runs on 20 GB RAM" (about a 21 GB model) | Refers to model footprint, not system requirement. Won't fit comfortably alongside macOS+apps within Metal's 19 GB cap. |
| "63 tok/s on Qwen3.6-27B" | Measured on M5 Max. Expect ~7 tok/s here. The ratio scales; the absolute number doesn't. |
| "Ollama auto-uses MLX since 0.19" | True *only for MLX-format tags* (`-nvfp4`, `-mxfp4`, `-mlx-bf16`). Default GGUF tags still go through llama.cpp/Metal. |
| "3× MTP speedup" | Best-case on M5 Max with predictable content. On M4 Air with novel prose, MTP can be net-negative (~3× *slower*). |
| "10/10 PASS on agentic coding tasks" (gpt-oss-20b) | True only with a harness that uses gpt-oss's trained tool names (`container.exec`, etc.) AND prompts that don't contain "simulate" / "show commands" / "numbered narrative" triggers. In a random Ollama+OpenCode setup with a typical multi-step spec, expect 0 tool calls. |
| "Apple Neural Engine accelerates local LLMs" | Not for MLX or llama.cpp on Mac. Both run on GPU via Metal. ANE is only used by Core ML conversions (Core ML-LLM, etc.) — useful for iPhone-style low-power inference, not desktop throughput. |

---

## Recommended starting setup (if you wiped this machine and started fresh)

1. **Install:** `brew install ollama && brew services start ollama`
2. **Pull:** `ollama pull gpt-oss:20b` and `ollama pull qwen3.5:9b-nvfp4`
3. **Daily driver:** gpt-oss:20b for everything except tool-calling
4. **For tool-calling/agents:** install Unsloth Studio, use `Qwen3.5-9B-MTP-GGUF UD-Q4_K_XL`, set Speculative Decoding=Auto + KV q8_0
5. **For OpenAI-API-compatible app dev:** point `OPENAI_BASE_URL=http://localhost:11434/v1` at any code that supports custom OpenAI endpoints
6. **For agentic gpt-oss specifically:** install Codex CLI (`npm i -g @openai/codex`), use `codex exec --oss --local-provider ollama -m gpt-oss:20b -c model_reasoning_effort='"medium"'`

Skip until you need more capability:
- MTPLX (only if you need 27B-class quality and can wait for ~5 tok/s)
- 27B/35B-A3B models (Metal cap pain isn't worth it)
- Custom Modelfiles unless you have a specific need

---

## References

- [Ollama MLX backend announcement](https://ollama.com/blog/mlx) (March 2026)
- [Gemma 4 MTP drafters](https://ai.google.dev/gemma/docs/mtp/mtp) (May 2026)
- [Apple Neural Engine for LLM Inference: What Actually Works](https://insiderllm.com/guides/apple-neural-engine-llm-inference/)
- [MTPLX GitHub](https://github.com/youssofal/MTPLX)
- [Unsloth Dynamic 2.0 GGUF docs](https://unsloth.ai/docs/basics/unsloth-dynamic-v2.0-gguf)
- [llama.cpp MTP PR #22673](https://github.com/ggerganov/llama.cpp/pull/22673) (merged May 16, 2026)

---

*Document built from empirical testing on May 24-25, 2026. Numbers above are from the specific M4 Air described at top — they will be different on your hardware, but the relative ordering and the gotchas tend to hold.*
