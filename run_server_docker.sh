#!/usr/bin/env bash
# ==============================================================================
# VeRLab Server Deployment & Execution Script for Overcooked MARL
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=================================================="
echo "1. Pulling latest code updates from GitHub..."
echo "=================================================="
git pull origin main

echo "=================================================="
echo "2. Building Docker image (overcooked_marl:latest)..."
echo "=================================================="
docker build -t overcooked_marl:latest .

echo "=================================================="
echo "3. Running Docker Container with GPU access..."
echo "=================================================="
docker run --rm -it \
  --gpus all \
  --name overcooked_container \
  -v "$PWD":/app \
  overcooked_marl:latest
