#!/usr/bin/env sh
# Start the Chainlit UI on the configured port. Using a script lets us expand
# the PORT environment variable (which a JSON-array CMD cannot do) and keeps the
# run command identical to the one in the README.
set -e

PORT="${PORT:-8000}"
echo "Starting iPhone Guide Assistant on port ${PORT}..."

exec chainlit run app/ui/chainlit_app.py \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --headless
