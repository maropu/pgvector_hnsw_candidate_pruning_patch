#!/usr/bin/env bash
# create_hnsw_index.sh
#
# Runs inside docker-entrypoint-initdb.d (after 20-load-sift1m.sh): creates an
# HNSW index with neighbor_metadata on the base SIFT1M table.
#
# Only the "base" subset is indexed — query/learn subsets are not search targets
# and do not need a vector index.
#
# Environment variables (all optional, have defaults set in the Dockerfile):
#   SIFT_SUBSETS    Space-separated list of splits that were loaded.
#                   Index is skipped if "base" is not in this list.
#   SIFT_TABLE      Base table name (default: sift1m)
#   SIFT_COLUMN     Vector column name (default: embedding)
#
# Connection: Unix socket via PGHOST (set by docker-entrypoint.sh), no password.

set -uo pipefail

: "${POSTGRES_DB:=${POSTGRES_USER:-postgres}}"
: "${SIFT_SUBSETS:=base}"
: "${SIFT_TABLE:=sift1m}"
: "${SIFT_COLUMN:=embedding}"

# Only create the index when the base subset was loaded.
if [[ " ${SIFT_SUBSETS} " != *" base "* ]]; then
    echo "[hnsw-index] 'base' not in SIFT_SUBSETS — skipping index creation."
    exit 0
fi

echo "[hnsw-index] Creating HNSW index on ${SIFT_TABLE}(${SIFT_COLUMN}) ..."

psql -v ON_ERROR_STOP=1 \
     --username "${POSTGRES_USER:-postgres}" \
     --dbname   "${POSTGRES_DB}" \
     <<SQL
CREATE INDEX ON ${SIFT_TABLE}
    USING hnsw (${SIFT_COLUMN} vector_l2_ops)
    WITH (m = 24, ef_construction = 200, neighbor_metadata = on);
SQL

echo "[hnsw-index] Index created successfully."
