#!/bin/sh
# API container start-up: apply migrations, create demo users (keys -> artifacts/demo_api_keys.json),
# seed vendor history if the bootstrap job produced it, then serve.
set -e

python -m scripts.init_db
exec uvicorn api.main:app --host 0.0.0.0 --port 8000 --proxy-headers --forwarded-allow-ips="*"
