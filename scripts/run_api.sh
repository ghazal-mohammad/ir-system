#!/usr/bin/env bash
# Run the API Gateway locally (Linux / Mac)
cd "$(dirname "$0")/.."
uvicorn api.main:app --port 8000
