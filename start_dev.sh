#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════
# SPLAT·FORGE — Local Dev Startup
# Usage: ./start_dev.sh
# ══════════════════════════════════════════════════════════
set -euo pipefail

CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${CYAN}"
echo "  ███████╗██████╗ ██╗      █████╗ ████████╗"
echo "  ██╔════╝██╔══██╗██║     ██╔══██╗╚══██╔══╝"
echo "  ███████╗██████╔╝██║     ███████║   ██║   "
echo "  ╚════██║██╔═══╝ ██║     ██╔══██║   ██║   "
echo "  ███████║██║     ███████╗██║  ██║   ██║   "
echo "  ╚══════╝╚═╝     ╚══════╝╚═╝  ╚═╝   ╚═╝   "
echo -e "  ${YELLOW}FORGE${NC} ${CYAN}— 3D Gaussian Splatting Platform${NC}"
echo ""

# ── Check Python ──────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
  echo "❌  python3 not found"; exit 1
fi
PYTHON=$(command -v python3.11 || command -v python3 || echo "python3")
echo -e "${GREEN}✓${NC}  Python: $($PYTHON --version)"

# ── Check CUDA ────────────────────────────────────────────
if $PYTHON -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  GPU=$($PYTHON -c "import torch; print(torch.cuda.get_device_name(0))")
  echo -e "${GREEN}✓${NC}  GPU: $GPU"
else
  echo -e "${YELLOW}⚠${NC}  No GPU found — 3DGS will run on CPU (slow)"
fi

# ── Create data dirs ──────────────────────────────────────
mkdir -p data/uploads data/outputs
echo -e "${GREEN}✓${NC}  Data dirs ready"

# ── Check deps ────────────────────────────────────────────
echo ""
echo "Checking dependencies..."
MISSING=()
for pkg in fastapi uvicorn cv2 PIL numpy torch; do
  if ! $PYTHON -c "import $pkg" 2>/dev/null; then
    MISSING+=("$pkg")
  fi
done

if [ ${#MISSING[@]} -ne 0 ]; then
  echo -e "${YELLOW}Installing missing: ${MISSING[*]}${NC}"
  pip install -r requirements.txt
fi

for pkg in gsplat pycolmap; do
  if ! $PYTHON -c "import $pkg" 2>/dev/null; then
    echo -e "${YELLOW}⚠  $pkg not installed — some features will be limited${NC}"
  else
    echo -e "${GREEN}✓${NC}  $pkg available"
  fi
done

# ── Launch ────────────────────────────────────────────────
echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "  🚀  Launching on ${GREEN}http://localhost:8000${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

cd "$(dirname "$0")"
exec $PYTHON -m uvicorn app.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --reload \
  --reload-dir app \
  --log-level info
