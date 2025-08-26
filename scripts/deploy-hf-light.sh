#!/bin/bash

echo "🚀 Fast deploy without pre-downloading models..."

# Use light version
docker-compose -f docker-compose.hf.light.yml up -d --build

echo "⏳ Models will download at first request..."
echo "📡 API: http://localhost:8000"