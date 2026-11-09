#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
docker build --platform linux/amd64 -t research-pilot:0.1.0 .
