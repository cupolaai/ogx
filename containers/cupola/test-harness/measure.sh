#!/bin/sh
# What OGX answers for a provider error on each path a Cupola client uses.
B=${1:-http://localhost:18321}
for m in ${MODELS:-openai-r429/gemma openai-r401/bad401 openai-r403/bad403}; do
  echo "=================== $m"
  echo "--- POST /v1/chat/completions (no stream)"
  curl -s -m 120 -o /tmp/perr.body -w 'HTTP %{http_code}\n' "$B/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$m\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}"; cat /tmp/perr.body; echo
  echo "--- POST /v1/chat/completions (stream)"
  curl -s -N -m 120 -o /tmp/perr.body -w 'HTTP %{http_code}\n' "$B/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$m\",\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}"; cat /tmp/perr.body; echo
  echo "--- POST /v1/responses (no stream)"
  curl -s -m 120 -o /tmp/perr.body -w 'HTTP %{http_code}\n' "$B/v1/responses" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$m\",\"input\":\"hi\",\"store\":false}"; cat /tmp/perr.body; echo
  echo "--- POST /v1/responses (stream)"
  curl -s -N -m 120 -o /tmp/perr.body -w 'HTTP %{http_code}\n' "$B/v1/responses" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$m\",\"input\":\"hi\",\"store\":false,\"stream\":true}"; grep -E 'failed|"error"' /tmp/perr.body; echo
done
