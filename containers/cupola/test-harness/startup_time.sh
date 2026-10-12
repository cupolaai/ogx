#!/bin/sh
# Seconds from container start to OGX answering /v1/health, N runs per image.
# Usage: startup_time.sh <image> <port> <runs>
IMG=$1; PORT=$2; N=${3:-3}; NAME=perr-start-$PORT
H=$(cd "$(dirname "$0")" && pwd)
for r in $(seq 1 $N); do
  docker rm -f $NAME >/dev/null 2>&1
  t0=$(python3 -c 'import time;print(time.time())')
  docker run -d --name $NAME --platform linux/amd64 -p $PORT:8321 -v "$H":/h:ro --entrypoint sh "$IMG" /h/start.sh >/dev/null
  until curl -s -m 2 http://localhost:$PORT/v1/health >/dev/null; do sleep 0.5; done
  t1=$(python3 -c 'import time;print(time.time())')
  python3 -c "print(f'run $r: {$t1-$t0:.1f}s')"
done
docker rm -f $NAME >/dev/null 2>&1
