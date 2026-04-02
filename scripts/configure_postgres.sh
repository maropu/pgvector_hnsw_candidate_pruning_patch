#!/usr/bin/env bash
# configure_postgres.sh
#
# Runs inside docker-entrypoint-initdb.d (before other initdb scripts):
# appends tuning parameters to postgresql.conf so they take effect when
# PostgreSQL starts after initialization completes.
#
# Environment variables (all optional, have defaults set in the Dockerfile):
#   PG_SHARED_BUFFERS   Value for shared_buffers  (default: 4GB)

set -euo pipefail

: "${PG_SHARED_BUFFERS:=4GB}"

cat >> "${PGDATA}/postgresql.conf" <<EOF

# --- Tuning parameters set by configure_postgres.sh ---
shared_buffers = ${PG_SHARED_BUFFERS}
EOF

echo "[configure] shared_buffers = ${PG_SHARED_BUFFERS}"
