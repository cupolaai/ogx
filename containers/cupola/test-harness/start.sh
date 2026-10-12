#!/bin/sh
# Runs inside the OGX image: the fake provider, then OGX on the harness run.yaml.
# /tmp/ogx survives a docker restart (rag_restart.sh relies on it).
mkdir -p /tmp/ogx
python3 /h/fake_provider.py &
exec ogx stack run /h/run.yaml
