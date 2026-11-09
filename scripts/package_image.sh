#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

TEAM_ID="${TEAM_ID:-JTI}"
VERSION="${VERSION:-0.1.0}"
SAFE_TEAM="${TEAM_ID//[^A-Za-z0-9._-]/_}"
IMAGE="research-pilot:${VERSION}"
TAR="dist/research-pilot-${SAFE_TEAM}-${VERSION}.tar"

docker build --platform linux/amd64 \
  --build-arg TEAM_ID="$TEAM_ID" \
  --build-arg AGENT_VERSION="$VERSION" \
  -t "$IMAGE" .

mkdir -p dist
docker save "$IMAGE" -o "$TAR"

echo "image: $IMAGE"
echo "tar: $TAR"
if command -v shasum >/dev/null 2>&1; then
  shasum -a 256 "$TAR"
else
  sha256sum "$TAR"
fi
ls -lh "$TAR"
