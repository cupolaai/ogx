#!/bin/sh
# Build Cupola's OGX distribution image from this fork's source.
#
#   containers/cupola/build.sh <tag> [--push]
#   e.g. containers/cupola/build.sh 1.1.3-cupola.9          (local image, --load)
#        containers/cupola/build.sh 1.1.3-cupola.9 --push   (to cupola-cr.io)
#
# Needs a clean working tree (the image is the committed source), docker with
# buildx, and network for the download steps only. The image build itself
# installs offline (--network=none) from what these steps fetched:
#   1. ogx + ogx-api wheels from this commit: sdist -> wheel, as PyPI's 1.1.3
#      was built (setuptools 80.10.2; build-constraints.txt), version
#      1.1.3+cupola.N;
#   2. the wheelhouse: exactly the files of image-lock.txt (by sha256),
#      downloaded natively (no emulation), PyTorch CPU index for +cpu;
#   3. tiktoken's cl100k_base, checked by sha256;
#   4. the image (linux/amd64), 45-minute limit, every step timestamped.
set -eu

TAG="${1:?usage: build.sh <tag, e.g. 1.1.3-cupola.9> [--push]}"
PUSH="${2:-}"
IMAGE="cupola-cr.io/cupola/cupola-distribution-starter:${TAG}"
VERSION="1.1.3+${TAG#1.1.3-}"            # 1.1.3-cupola.9 -> 1.1.3+cupola.9
LIMIT_SECONDS=2700
TIKTOKEN_URL="https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
TIKTOKEN_SHA256="223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"
PY="docker.io/library/python@sha256:6c4dd321d176d61ea848dc8c73a4f7dbae8f70e0ee48bb411ea2f045b599fa8e"  # python:3.12.13-slim-trixie of 2026-06-24, all platforms (native here)

ROOT="$(git rev-parse --show-toplevel)"
HERE="$ROOT/containers/cupola"
step() { echo "[$(date -u +%H:%M:%S)] $*"; }

if [ -n "$(git -C "$ROOT" status --porcelain)" ]; then
  echo "build.sh: the working tree is not clean; the image is built from committed source" >&2
  exit 1
fi
COMMIT="$(git -C "$ROOT" rev-parse HEAD)"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
step "commit $COMMIT, version $VERSION, staging $STAGE"

step "1/4 ogx + ogx-api wheels from git (sdist -> wheel)"
git clone -q "$ROOT" "$STAGE/src"
git -C "$STAGE/src" checkout -q "$COMMIT"
mkdir -p "$STAGE/ctx/ogx-wheels"
docker run --rm -v "$STAGE:/s" -w /tmp \
  -e PIP_CONSTRAINT=/s/src/containers/cupola/build-constraints.txt \
  -e SETUPTOOLS_SCM_PRETEND_VERSION_FOR_OGX="$VERSION" \
  -e SETUPTOOLS_SCM_PRETEND_VERSION_FOR_OGX_API="$VERSION" \
  "$PY" sh -c '
    set -e
    apt-get update -qq >/dev/null && apt-get install -y -qq git >/dev/null
    git config --global --add safe.directory "*"
    pip install -q --no-cache-dir -c /s/src/containers/cupola/build-constraints.txt build
    mkdir -p /tmp/sd
    python -m build -q --sdist --outdir /tmp/sd /s/src
    python -m build -q --sdist --outdir /tmp/sd /s/src/src/ogx_api
    for s in /tmp/sd/*.tar.gz; do d=$(mktemp -d); tar -xzf "$s" -C "$d"; python -m build -q --wheel --outdir /s/ctx/ogx-wheels "$d"/*; done'
( cd "$STAGE/ctx/ogx-wheels" && for w in *.whl; do
    n="${w%%-*}"; echo "$n==$VERSION --hash=sha256:$(shasum -a 256 "$w" | cut -d' ' -f1)"
  done > ogx-wheels.txt )
cat "$STAGE/ctx/ogx-wheels/ogx-wheels.txt"

step "2/4 wheelhouse = image-lock.txt"
mkdir -p "$STAGE/ctx/wheelhouse"
# pip does not widen a manylinux_2_X platform to the older ones: name them all
# (glibc 2.41 = Debian trixie, the image's base).
PLATFORMS="--platform linux_x86_64 --platform manylinux1_x86_64 --platform manylinux2010_x86_64 --platform manylinux2014_x86_64"
i=5; while [ "$i" -le 41 ]; do PLATFORMS="$PLATFORMS --platform manylinux_2_${i}_x86_64"; i=$((i + 1)); done
docker run --rm -v "$STAGE:/s" -v "$HERE:/c:ro" "$PY" \
  pip download -q --no-cache-dir --no-deps --only-binary=:all: --require-hashes \
    $PLATFORMS \
    --implementation cp --python-version 3.12 --abi cp312 --abi abi3 --abi none \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r /c/image-lock.txt -d /s/ctx/wheelhouse
want="$(grep -c '^[a-z0-9]' "$HERE/image-lock.txt")"
got="$(ls "$STAGE/ctx/wheelhouse" | wc -l | tr -d ' ')"
[ "$want" = "$got" ] || { echo "build.sh: wheelhouse has $got files, image-lock.txt pins $want" >&2; exit 1; }
step "   $got wheels"

step "3/4 tiktoken cl100k_base"
mkdir -p "$STAGE/ctx/tiktoken"
name="$(printf '%s' "$TIKTOKEN_URL" | shasum -a 1 | cut -d' ' -f1)"
curl -fsSL "$TIKTOKEN_URL" -o "$STAGE/ctx/tiktoken/$name"
[ "$(shasum -a 256 "$STAGE/ctx/tiktoken/$name" | cut -d' ' -f1)" = "$TIKTOKEN_SHA256" ] || {
  echo "build.sh: cl100k_base sha256 differs" >&2; exit 1; }

step "4/4 image $IMAGE"
cp "$HERE/Containerfile" "$HERE/image-lock.txt" "$ROOT/uv.lock" "$STAGE/ctx/"
cp -R "$HERE/checks" "$STAGE/ctx/checks"
if [ "$PUSH" = "--push" ]; then OUT="--push"; else OUT="--load"; fi
docker buildx build --platform linux/amd64 --progress plain \
  --build-arg OGX_VERSION="$VERSION" --build-arg OGX_COMMIT="$COMMIT" \
  -f "$STAGE/ctx/Containerfile" -t "$IMAGE" $OUT "$STAGE/ctx" > "$STAGE/build.log" 2>&1 &
pid=$!
( sleep "$LIMIT_SECONDS"; kill "$pid" 2>/dev/null && echo "build.sh: stopped after ${LIMIT_SECONDS}s" >&2 ) &
guard=$!
if ! wait "$pid"; then
  kill "$guard" 2>/dev/null || true
  tail -40 "$STAGE/build.log" >&2
  exit 1
fi
kill "$guard" 2>/dev/null || true
grep -E "cupola image checks|^#[0-9]+ DONE" "$STAGE/build.log" | tail -25
step "done: $IMAGE (commit $COMMIT)"
if [ "$PUSH" = "--push" ]; then docker buildx imagetools inspect "$IMAGE" | head -3; fi
