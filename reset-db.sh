#!/bin/bash

# AI Market - Database Reset Script
# This script resets the database to a fresh state

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "  AI Market - Database Reset"
echo "=========================================="
echo ""
echo "WARNING: This will delete all data!"
read -p "Are you sure? (y/N): " confirm

if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "[1/3] Stopping services..."
./stop.sh

echo ""
echo "[2/3] Removing database volume..."
docker volume rm ai-market_postgres_data 2>/dev/null || true
docker volume rm ai-market_pgadmin_data 2>/dev/null || true

echo ""
echo "[3/3] Restarting services with fresh database..."
./start.sh

echo ""
echo "=========================================="
echo "  Database reset complete!"
echo "=========================================="
