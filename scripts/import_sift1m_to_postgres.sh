#!/usr/bin/env bash
# import_sift1m_to_postgres.sh  (Docker initdb name: 20-load-sift1m.sh)
#
# Orchestrates loading of a TexMex SIFT dataset into PostgreSQL by invoking
# import_sift1m_to_postgres.py once per requested split.  Placed in
# /docker-entrypoint-initdb.d/ so it runs automatically on the very first
# container start, after the cluster is initialized but before PostgreSQL
# accepts external connections.
#
# -----------------------------------------------------------------------------
# WHAT THIS SCRIPT DOES
# -----------------------------------------------------------------------------
# 1. Reads environment variables (see below) with safe defaults.
# 2. Exits immediately (no error) if SIFT_SUBSETS is empty — useful for
#    skipping data loading during quick dev/test runs.
# 3. Iterates over each split in SIFT_SUBSETS, deriving the destination table
#    name:
#      subset "base"           -> SIFT_TABLE          (e.g. sift1m)
#      subset "query"/"learn"  -> SIFT_TABLE_<subset>  (e.g. sift1m_query)
# 4. Calls import_sift1m_to_postgres.py for each split.  A non-zero exit from
#    the Python script is logged as a WARNING but does NOT abort the loop, so
#    the remaining subsets are still attempted.
#
# -----------------------------------------------------------------------------
# ENVIRONMENT VARIABLES  (all optional; defaults match Dockerfile ENV values)
# -----------------------------------------------------------------------------
#   SIFT_DATASET    Dataset variant to download and load.
#                     sift1m     : 1,000,000 vectors, 128 dims  (default)
#                     siftsmall  : 10,000 vectors, 128 dims (fast dev/test)
#
#   SIFT_SUBSETS    Space-separated list of splits to load.
#                   Valid values: base  query  learn  (any combination)
#                   Set to "" to skip all loading.
#                   Default: "base query"
#                   Example: SIFT_SUBSETS="base query learn"
#
#   SIFT_TABLE      Base table name used for the "base" split and as the prefix
#                   for other splits (<table>_query, <table>_learn).
#                   Default: sift1m
#
#   SIFT_COLUMN     Name of the pgvector vector column in each table.
#                   Default: embedding
#
#   SIFT_CACHE_DIR  Host-side directory (bind-mounted into the container) used
#                   to cache the downloaded .tar.gz and extracted .fvecs files
#                   across container restarts.
#                   Default: /var/lib/postgresql/sift_cache
#                   Override: -v /local/cache:/sift_cache -e SIFT_CACHE_DIR=/sift_cache
#
#   POSTGRES_DB     Target database.  Inherited from docker-entrypoint.sh.
#                   Falls back to POSTGRES_USER, then "postgres".
#   POSTGRES_USER   PostgreSQL superuser name.  Inherited from Docker env.
#
# -----------------------------------------------------------------------------
# TABLE NAMING CONVENTION
# -----------------------------------------------------------------------------
#   SIFT_TABLE=sift1m, SIFT_SUBSETS="base query learn"  =>
#     sift1m        (base vectors — the primary search target)
#     sift1m_query  (query vectors — used as probe vectors in ANN benchmarks)
#     sift1m_learn  (learn vectors — used for quantization training, optional)
#
# -----------------------------------------------------------------------------
# USAGE EXAMPLES (docker run)
# -----------------------------------------------------------------------------
#   # Default: load sift1m base split only
#   docker run -e POSTGRES_PASSWORD=postgres -e SIFT_SUBSETS="base" ...
#
#   # Load base + query for ANN benchmark
#   docker run -e POSTGRES_PASSWORD=postgres -e SIFT_SUBSETS="base query" ...
#
#   # Use siftsmall for fast patch iteration (downloads ~3 MB instead of ~161 MB)
#   docker run -e POSTGRES_PASSWORD=postgres \
#              -e SIFT_DATASET=siftsmall -e SIFT_SUBSETS="base" ...
#
#   # Skip data loading entirely (PostgreSQL only, no SIFT data)
#   docker run -e POSTGRES_PASSWORD=postgres -e SIFT_SUBSETS="" ...
#
#   # Reuse cached archive across container runs
#   docker run -e POSTGRES_PASSWORD=postgres \
#              -v /local/cache:/sift_cache \
#              -e SIFT_CACHE_DIR=/sift_cache ...
#
# -----------------------------------------------------------------------------
# DEPENDENCIES
# -----------------------------------------------------------------------------
#   - /opt/sift-venv/bin/python3       Python venv with psycopg[binary]
#   - /usr/local/lib/import_sift1m_to_postgres.py  (the actual loader)
#   - PostgreSQL reachable via Unix socket (PGHOST set by docker-entrypoint.sh)
#
# Connection: Unix socket via PGHOST (set by docker-entrypoint.sh), no password.

set -uo pipefail

PYTHON=/opt/sift-venv/bin/python3
SCRIPT=/usr/local/lib/import_sift1m_to_postgres.py

# Respect docker-entrypoint.sh environment
: "${POSTGRES_DB:=${POSTGRES_USER:-postgres}}"
: "${SIFT_DATASET:=sift1m}"
: "${SIFT_SUBSETS:=base query}"
: "${SIFT_TABLE:=sift1m}"
: "${SIFT_COLUMN:=embedding}"
: "${SIFT_CACHE_DIR:=/sift_cache}"

# Skip if no subsets requested
if [ -z "${SIFT_SUBSETS}" ]; then
    echo "[sift] SIFT_SUBSETS is empty — skipping dataset load."
    exit 0
fi

echo "[sift] Starting load (dataset: ${SIFT_DATASET}, subsets: ${SIFT_SUBSETS})"
echo "[sift] Cache dir : ${SIFT_CACHE_DIR}"
echo "[sift] Target DB : ${POSTGRES_DB}"
echo "[sift] Table     : ${SIFT_TABLE}"

mkdir -p "${SIFT_CACHE_DIR}"

for subset in ${SIFT_SUBSETS}; do
    # "base" goes into the plain table; other subsets get a _<subset> suffix
    if [ "${subset}" = "base" ]; then
        table="${SIFT_TABLE}"
    else
        table="${SIFT_TABLE}_${subset}"
    fi

    echo "[sift] Loading subset='${subset}' into table='${table}' ..."

    if "${PYTHON}" "${SCRIPT}" \
        --dbname   "${POSTGRES_DB}" \
        --user     "${POSTGRES_USER:-postgres}" \
        --dataset  "${SIFT_DATASET}" \
        --subset   "${subset}" \
        --table    "${table}" \
        --column   "${SIFT_COLUMN}" \
        --cache-dir "${SIFT_CACHE_DIR}" \
        --batch-rows 10000 \
        --verbose; then
        echo "[sift] Finished subset='${subset}'."
    else
        echo "[sift] WARNING: failed to load subset='${subset}' (exit $?). Continuing." >&2
    fi
done

echo "[sift] All subsets loaded successfully."
