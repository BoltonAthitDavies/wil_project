#!/usr/bin/env bash
# Build the wil_project image with provenance.
#
# Records the superproject HEAD and submodule pins into docker/build_info.txt
# (copied into the image at /home/ambushee/wil_project/docker/build_info.txt)
# and passes the describe string as an OCI label, because the image excludes
# .git to save ~600 MB. Run from anywhere; extra args go to `docker compose build`.
#
#   ./docker/build.sh                 # full image (default target)
#   ./docker/build.sh --no-cache      # from scratch
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DESCRIBE="$(git describe --always --dirty --long 2>/dev/null || echo unknown)"
{
    echo "built_at:   $(date --iso-8601=seconds)"
    echo "host:       $(hostname)"
    echo "head:       $(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "describe:   $DESCRIBE"
    echo "branch:     $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
    echo "submodules:"
    git submodule status 2>/dev/null | sed 's/^/  /' || true
    echo "dirty_files:"
    git status --porcelain 2>/dev/null | sed 's/^/  /' || true
} > docker/build_info.txt
cat docker/build_info.txt

export USER_UID="$(id -u)" USER_GID="$(id -g)" WIL_GIT_DESCRIBE="$DESCRIBE"
exec docker compose -f docker/compose.yml build "$@" wil
