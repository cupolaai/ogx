#!/bin/sh
# Run Cupola's tests inside a built OGX image, with the image's own Python packages.
#
#   containers/cupola/test.sh <image> [--postgres]
#   e.g. containers/cupola/test.sh cupola-cr.io/cupola/cupola-distribution-starter:1.1.3-cupola.14@sha256:...
#
# uv.lock's development environment has older anyio / starlette / openai / uvicorn
# than the image (image-lock.txt), and the client-disconnect bugs of Cupola
# faz8.0.576 reproduced only with the image's versions. So the Cupola tests run
# here, in a throwaway container of the image itself: ogx and every third-party
# package come from the image, the test files from this checkout (read-only).
#   1. the test-only packages: exactly the files of test-lock.txt (by sha256),
#      downloaded natively;
#   2. the tests (tests/unit/**/test_cupola_*.py and the Cupola-changed
#      test_openai_base_url_config.py), linux/amd64, without network
#      (--network=none: the network tests use 127.0.0.1 only); test-lock.txt is
#      installed into the container (--no-index --no-deps --require-hashes),
#      nothing is written to the image.
# --postgres also runs the PostgreSQL tests against a throwaway postgres on an
# internal docker network; without it they are skipped.
set -eu

IMAGE="${1:?usage: test.sh <image> [--postgres]}"
POSTGRES="${2:-}"
PY="docker.io/library/python@sha256:6c4dd321d176d61ea848dc8c73a4f7dbae8f70e0ee48bb411ea2f045b599fa8e"  # python:3.12.13-slim-trixie, as build.sh
PG="docker.io/library/postgres@sha256:2d2b8998d31037bf721cfdf764d76ba74171b4fab3431b7f72c27c56ddbdf9e3"   # postgres:17

ROOT="$(git rev-parse --show-toplevel)"
HERE="$ROOT/containers/cupola"
step() { echo "[$(date -u +%H:%M:%S)] $*"; }

STAGE="$(mktemp -d)"
RUN_ID="cupola-test-$$"
cleanup() {
  if [ "$POSTGRES" = "--postgres" ]; then
    docker rm -f "$RUN_ID-pg" >/dev/null 2>&1 || true
    docker network rm "$RUN_ID" >/dev/null 2>&1 || true
  fi
  rm -rf "$STAGE"
}
trap cleanup EXIT

step "1/2 test packages = test-lock.txt"
mkdir -p "$STAGE/wheels"
docker run --rm -e PIP_DISABLE_PIP_VERSION_CHECK=1 -v "$STAGE:/s" -v "$HERE:/c:ro" "$PY" \
  pip download -q --no-cache-dir --no-deps --only-binary=:all: --require-hashes \
    --platform any --implementation cp --python-version 3.12 --abi none \
    -r /c/test-lock.txt -d /s/wheels

NETWORK="--network=none"
PGENV=""
if [ "$POSTGRES" = "--postgres" ]; then
  step "   postgres on the internal network $RUN_ID"
  docker network create --internal "$RUN_ID" >/dev/null
  docker run -d --name "$RUN_ID-pg" --network "$RUN_ID" \
    -e POSTGRES_USER=ogx -e POSTGRES_PASSWORD=ogx -e POSTGRES_DB=ogx "$PG" >/dev/null
  i=0
  until docker exec "$RUN_ID-pg" pg_isready -q -U ogx -d ogx; do
    i=$((i + 1)); [ "$i" -le 60 ] || { echo "test.sh: postgres did not start" >&2; exit 1; }
    sleep 1
  done
  sleep 2  # the image's first start restarts the server once after initdb
  NETWORK="--network=$RUN_ID"
  PGENV="-e CUPOLA_TEST_POSTGRES_HOST=$RUN_ID-pg"
fi

step "2/2 tests in $IMAGE"
# shellcheck disable=SC2086
docker run --rm --platform linux/amd64 $NETWORK $PGENV \
  -v "$ROOT:/w:ro" -v "$STAGE/wheels:/wheels:ro" -e PYTHONPATH=/w \
  --entrypoint sh "$IMAGE" -c '
    set -e
    pip install -q --no-cache-dir --root-user-action=ignore --no-index --no-deps --require-hashes \
      --find-links /wheels -r /w/containers/cupola/test-lock.txt
    cd /tmp
    python -m pytest -q -p no:cacheprovider --rootdir /w \
      $(cd /w && find tests/unit -name "test_cupola_*.py" | sort | sed "s#^#/w/#") \
      /w/tests/unit/providers/inference/test_openai_base_url_config.py'
step "tests passed: $IMAGE"
