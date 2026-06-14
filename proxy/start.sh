#!/usr/bin/env bash
#
# Convenience launcher for the local-tool-proxy
# tuned for stock OpenCode + Gemma 4 on Apple Silicon.
#
# Usage:
#   ./proxy/start.sh
#   ./proxy/start.sh --port 9000 --models gemma4:e4b-mlx,gemma4:e2b-mlx
#

set -euo pipefail

PORT=${PORT:-9000}
MODELS=${MODELS:-gemma4:e4b-mlx,gemma4:e2b-mlx}
OLLAMA_BASE=${OLLAMA_BASE:-http://localhost:11434/v1}

echo "Starting local-tool-proxy for stock OpenCode + small models"
echo "   Port:        $PORT"
echo "   Compat:      $MODELS"
echo "   Upstream:    $OLLAMA_BASE"
echo ""
echo "   Experimental modes (opt-in):"
echo "     --mode stabilize --planner soft --stabilize-max-retries 1"
echo ""
echo "   After it starts, test with:"
echo "     curl http://localhost:$PORT/health"
echo "     curl http://localhost:$PORT/v1/models"
echo ""
echo "   Then point stock OpenCode at http://localhost:$PORT/v1"
echo "   (see proxy/examples/opencode-for-proxy.json and proxy/TESTING.md)"
echo ""

exec python3 -m proxy.server \
  --port "$PORT" \
  --ollama-base "$OLLAMA_BASE" \
  --compat-models "$MODELS" \
  "$@"
