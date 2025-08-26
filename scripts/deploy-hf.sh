#!/bin/bash

echo "🚀 Deploying Quantum SERP with Hugging Face models..."

# Build and start services
docker-compose -f docker-compose.hf.yml build --no-cache
docker-compose -f docker-compose.hf.yml up -d

# Wait for services with timeout
echo "⏳ Waiting for services to start..."
timeout 300 bash -c 'until curl -f http://localhost:8000/health; do sleep 5; done' || echo "Service startup timeout"

# Health check
echo "🔍 Health check..."
curl -f http://localhost:8000/health || exit 1

echo "✅ Deployment successful!"
echo "📡 API: http://localhost:8000"
echo "📚 Docs: http://localhost:8000/docs"