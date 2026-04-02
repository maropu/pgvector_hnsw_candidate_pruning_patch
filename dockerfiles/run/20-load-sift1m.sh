#!/usr/bin/env bash
# 20-load-sift1m.sh
#
# Runs inside docker-entrypoint-initdb.d: loads SIFT1M subset(s) into the
# PostgreSQL cluster that was just initialized by docker-entrypoint.sh.
#
# Environment variables (all optional, have defaults set in the Dockerfile):
#   SIFT_SUBSETS    Space-separated list of splits to load: "base" "query" "learn"
#                   Set to "" to skip loading entirely. (default: "base query")
#   SIFT_TABLE      Base table name; each subset is loaded into <table>_<subset>
#                   except "base" which uses the plain <table>. (default: sift1m)
#   SIFT_COLUMN     Vector column name. (default: embedding)
#   SIFT_CACHE_DIR  Directory for caching the downloaded .tar.gz. (default: /sift_cache)
#
# Connection: Unix socket via PGHOST (set by docker-entrypoint.sh), no password.

set -uo pipefail

PYTHON=/opt/sift-venv/bin/python3
SCRIPT=/usr/local/lib/import_sift1m_to_postgres.py

# Respect docker-entrypoint.sh environment
: "${POSTGRES_DB:=${POSTGRES_USER:-postgres}}"
: "${SIFT_SUBSETS:=base query}"
: "${SIFT_TABLE:=sift1m}"
: "${SIFT_COLUMN:=embedding}"
: "${SIFT_CACHE_DIR:=/sift_cache}"

# Skip if no subsets requested
if [ -z "${SIFT_SUBSETS}" ]; then
    echo "[sift] SIFT_SUBSETS is empty — skipping dataset load."
    exit 0
fi

echo "[sift] Starting SIFT1M load (subsets: ${SIFT_SUBSETS})"
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
