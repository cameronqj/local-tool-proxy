# Minimal image for local-tool-proxy.
#
# Build:  docker build -t local-tool-proxy .
# Demo:   docker run --rm --entrypoint python local-tool-proxy demo.py  # no Ollama
# Run:    docker run --rm -p 9000:9000 local-tool-proxy \
#             --ollama-base http://host.docker.internal:11434/v1
#
# The proxy binds 0.0.0.0 inside the container so the published port is
# reachable from the host. See SECURITY.md before exposing it beyond localhost.
FROM python:3.12-slim

WORKDIR /app

# Install the package first (only the files the build needs) so this layer
# caches independently of the demo script.
COPY pyproject.toml README.md ./
COPY proxy ./proxy
RUN pip install --no-cache-dir .

# The self-contained demo (mock upstream + proxy, no Ollama) ships in the image.
COPY demo.py ./

EXPOSE 9000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:9000/health', timeout=2).status==200 else 1)"

# Default to serving the proxy. host.docker.internal points at the host's Ollama
# on Docker Desktop (macOS/Windows); on Linux add --add-host or override here.
ENTRYPOINT ["local-tool-proxy", "--host", "0.0.0.0", "--port", "9000"]
CMD ["--ollama-base", "http://host.docker.internal:11434/v1", \
     "--compat-models", "gemma4:e4b-mlx,gemma4:e2b-mlx,gpt-oss:20b"]
