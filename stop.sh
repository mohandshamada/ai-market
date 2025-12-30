#!/bin/bash

# AI Market - Stop Script
# This script stops all services

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "  AI Market - Stopping All Services"
echo "=========================================="

# Stop Angular frontend
echo ""
echo "[1/2] Stopping Angular frontend..."
pkill -f "ng serve" 2>/dev/null && echo "Angular stopped." || echo "Angular was not running."

# Stop Docker services
echo ""
echo "[2/2] Stopping Docker services..."
docker compose down

echo ""
echo "=========================================="
echo "  All services stopped."
echo "=========================================="
echo ""
echo "  To start again: ./start.sh"
echo "  To reset database: ./reset-db.sh"
echo "=========================================="
