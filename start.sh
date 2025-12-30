#!/bin/bash

# AI Market - Startup Script
# This script starts all services (Docker + Angular frontend)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "  AI Market - Starting All Services"
echo "=========================================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker first."
    exit 1
fi

# Stop any existing Angular process
echo ""
echo "[1/4] Cleaning up existing processes..."
pkill -f "ng serve" 2>/dev/null || true

# Start Docker services
echo ""
echo "[2/4] Starting Docker services..."
docker compose up -d

# Wait for services to be healthy
echo ""
echo "[3/4] Waiting for services to be ready..."
echo "Waiting for PostgreSQL..."
sleep 5

# Check if API is healthy
for i in {1..30}; do
    if curl -s http://localhost:8003/health > /dev/null 2>&1; then
        echo "API is ready!"
        break
    fi
    echo "Waiting for API... ($i/30)"
    sleep 2
done

# Show Docker services status
echo ""
echo "Docker services status:"
docker compose ps

# Start Angular frontend
echo ""
echo "[4/4] Starting Angular frontend..."
cd frontend

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "Installing npm dependencies..."
    npm install
fi

# Start Angular in background
echo "Starting Angular dev server..."
nohup npm start > /tmp/angular-frontend.log 2>&1 &

# Wait for Angular to compile
echo "Waiting for Angular to compile..."
for i in {1..30}; do
    if lsof -i :4200 > /dev/null 2>&1; then
        break
    fi
    sleep 1
done

sleep 5

# Final status check
echo ""
if lsof -i :4200 > /dev/null 2>&1; then
    echo "=========================================="
    echo "  All services are running!"
    echo "=========================================="
    echo ""
    echo "  Frontend:  http://localhost:4200"
    echo "  API:       http://localhost:8003"
    echo "  pgAdmin:   http://localhost:8082"
    echo "             (admin@ai-market.com / admin)"
    echo "  Postgres:  localhost:5434"
    echo ""
    echo "  Logs:"
    echo "    Angular: tail -f /tmp/angular-frontend.log"
    echo "    API:     docker compose logs -f api-service"
    echo ""
    echo "  To stop all services: ./stop.sh"
    echo "=========================================="
else
    echo "Warning: Angular may still be starting."
    echo "Check logs: tail -f /tmp/angular-frontend.log"
fi
