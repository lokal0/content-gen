#!/usr/bin/env bash
set -euo pipefail

echo "=== Lokal — Local Business SEO Intelligence Platform ==="
echo ""

# Check .env exists
if [ ! -f .env ]; then
  if [ -f .env.example ]; then
    cp .env.example .env
    echo "[!] Created .env from .env.example — fill in your API keys before running."
    echo ""
    echo "    Required keys:"
    echo "      DATAFORSEO_API_KEY   — base64 of login:password (https://dataforseo.com)"
    echo "      DATABASE_URL         — Neon Postgres connection string"
    echo "      TAVILY_API_KEY       — https://tavily.com"
    echo "      GEMINI_API_KEY       — https://ai.google.dev"
    echo "      ANTHROPIC_API_KEY    — https://console.anthropic.com"
    echo "      PIONEER_API_KEY      — https://pioneer.ai"
    echo ""
    echo "    Edit .env then run this script again."
    exit 1
  else
    echo "[!] No .env or .env.example found."
    exit 1
  fi
fi

# Check Docker
if ! command -v docker &> /dev/null; then
  echo "[!] Docker is required. Install it from https://docker.com"
  exit 1
fi

if ! docker info &> /dev/null; then
  echo "[!] Docker daemon is not running. Start Docker first."
  exit 1
fi

echo "[1/3] Building services..."
docker compose build

echo ""
echo "[2/3] Starting all services..."
docker compose up -d

echo ""
echo "[3/3] Waiting for health checks..."
sleep 5

check_health() {
  local name=$1 url=$2
  for i in $(seq 1 20); do
    if curl -sf "$url" > /dev/null 2>&1; then
      echo "  ✓ $name is ready"
      return 0
    fi
    sleep 2
  done
  echo "  ✗ $name failed to start"
  return 1
}

check_health "seo-api"     "http://localhost:3000/health"
check_health "content-gen"  "http://localhost:8000/health"
check_health "lokal-next"   "http://localhost:3001"

echo ""
echo "=== All services running ==="
echo ""
echo "  Frontend:      http://localhost:3001"
echo "  Content API:   http://localhost:8000/docs"
echo "  SEO API:       http://localhost:3000"
echo ""
echo "  Logs:          docker compose logs -f"
echo "  Stop:          docker compose down"
echo ""
