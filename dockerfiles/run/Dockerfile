# syntax=docker/dockerfile:1
#
# pgvector run image w/ candidate pruning patch + SIFT1M dataset
#
# Builds pgvector from source with the specified patch applied, installs it
# into a standard postgres image, and loads the SIFT1M dataset on first start.
#
# Requirements:
#   - Docker BuildKit (for `ADD <git-url>#<ref>`)
#
# Build:
#   docker build \
#     --build-arg PG_MAJOR=17 \
#     --build-arg PGVECTOR_REF=v0.8.0 \
#     --build-arg PATCH_FILE=pgvector_v0.8.0_hnsw_candidate_pruning_turboquant.patch \
#     -t pgvector-run:pg17-turboquant \
#     -f dockerfiles/run/Dockerfile \
#     .
#
# Run:
#   docker run --rm \
#     -e POSTGRES_PASSWORD=postgres \
#     -e SIFT_SUBSETS="base query" \
#     -p 5432:5432 \
#     pgvector-run:pg17-turboquant
#
# Environment variables:
#   POSTGRES_PASSWORD  (required) PostgreSQL superuser password
#   SIFT_SUBSETS       Space-separated list of SIFT splits to load on first
#                      start: "base", "query", and/or "learn".
#                      Default: "base query"
#                      Set to "" (empty) to skip dataset loading.
#   SIFT_TABLE         Destination table name (default: sift1m)
#   SIFT_COLUMN        Vector column name (default: embedding)
#   SIFT_CACHE_DIR     Host path to cache the downloaded archive
#                      (mount as -v /local/path:/sift_cache for reuse)
#
# Connect:
#   psql -h localhost -U postgres
#   \dt          -- shows sift1m table after loading
#
# Notes:
#   - SIFT1M download (~165 MB) happens at container first-start time.
#     Mount -v /local/cache:/sift_cache to reuse across runs.
#   - ABI compatibility: the build uses the same postgres:<major>-bookworm
#     base image as the final stage, ensuring vector.so is ABI-compatible.
#   - Data persistence: use -v pgdata:/var/lib/postgresql/data

ARG PG_MAJOR=17
ARG PGVECTOR_REF=v0.8.0
ARG PATCH_FILE=pgvector_v0.8.0_hnsw_candidate_pruning_turboquant.patch

# ---- Build stage ----
FROM postgres:${PG_MAJOR}-bookworm AS build
ARG PG_MAJOR
ARG PGVECTOR_REF
ARG PATCH_FILE

# Fetch pgvector source at the specified git ref using BuildKit remote git context
ADD https://github.com/pgvector/pgvector.git#${PGVECTOR_REF} /tmp/pgvector

# Copy the patch file from the repository root (build context)
COPY ${PATCH_FILE} /tmp/pgvector/

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install build dependencies
RUN set -eux; \
    export DEBIAN_FRONTEND=noninteractive; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      build-essential \
      libpq-dev \
      postgresql-server-dev-"${PG_MAJOR}" \
      ca-certificates \
      patch; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/*

WORKDIR /tmp/pgvector

# Verify patch applies cleanly (dry-run first for diagnostics)
RUN set -eux; \
    patch -p1 --forward --dry-run < "${PATCH_FILE}"

# Apply patch
RUN set -eux; \
    patch -p1 --forward < "${PATCH_FILE}"

# Build and install into DESTDIR
RUN set -eux; \
    make clean; \
    make USE_PGXS=1 OPTFLAGS=""

RUN set -eux; \
    make USE_PGXS=1 install DESTDIR=/pkgroot

# ---- Runtime stage ----
# Use the same base image so vector.so is ABI-compatible with the server binary.
FROM postgres:${PG_MAJOR}-bookworm AS run
ARG PG_MAJOR
ARG PATCH_FILE

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install Python3 + psycopg (binary) for the SIFT import script,
# plus curl/wget for FTP download fallback.
RUN set -eux; \
    export DEBIAN_FRONTEND=noninteractive; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      python3 \
      python3-pip \
      python3-venv \
      ca-certificates; \
    python3 -m venv /opt/sift-venv; \
    /opt/sift-venv/bin/pip install --no-cache-dir "psycopg[binary]"; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/*

# Copy built extension files from the build stage
RUN set -eux; \
    PKGLIBDIR="$(pg_config --pkglibdir)"; \
    EXT_DIR="$(pg_config --sharedir)/extension"; \
    mkdir -p "${PKGLIBDIR}" "${EXT_DIR}"

COPY --from=build /pkgroot /

# Copy SIFT import script
COPY scripts/import_sift1m_to_postgres.py /usr/local/lib/import_sift1m_to_postgres.py

# Verify the extension is visible to PostgreSQL
RUN set -eux; \
    EXT_DIR="$(pg_config --sharedir)/extension"; \
    test -f "${EXT_DIR}/vector.control"; \
    ls "${EXT_DIR}"/vector--*.sql | grep -q .; \
    test -f "$(pg_config --pkglibdir)/vector.so"; \
    echo "pgvector extension installed successfully."

# Default environment variables for SIFT loading.
# SIFT_CACHE_DIR defaults to a path inside the postgres data volume so the
# postgres user can write there without needing host-side chmod.
# Override at runtime with -e SIFT_CACHE_DIR=/sift_cache when you mount a
# host directory that is writable by UID 999 (the postgres user).
ENV SIFT_SUBSETS="base query" \
    SIFT_TABLE="sift1m" \
    SIFT_COLUMN="embedding" \
    SIFT_CACHE_DIR="/var/lib/postgresql/sift_cache"

# initdb script 1: create the vector extension
RUN echo "CREATE EXTENSION IF NOT EXISTS vector;" \
      > /docker-entrypoint-initdb.d/10-vector.sql

# initdb script 2: load SIFT1M dataset
# Runs as the postgres user; connects via Unix socket (no password needed).
# Skipped when SIFT_SUBSETS is empty.
COPY --chmod=755 scripts/import_sift1m_to_postgres.sh \
     /docker-entrypoint-initdb.d/import_sift1m_to_postgres.sh

# Create the default cache directory and give it to the postgres user *before*
# the VOLUME declaration.  Docker copies the layer content (including ownership)
# when initialising the anonymous volume, so the postgres user can write there.
RUN install -d -o postgres -g postgres /var/lib/postgresql/sift_cache

# Convenience mount point for caching the downloaded SIFT archive.
# To reuse a host cache across runs, bind-mount a directory writable by
# UID 999: -v /local/cache:/var/lib/postgresql/sift_cache
VOLUME ["/var/lib/postgresql/sift_cache"]

# Embed the patch name as a label for traceability
LABEL org.opencontainers.image.title="pgvector (patched) + SIFT1M" \
      org.opencontainers.image.description="PostgreSQL with pgvector candidate-pruning patch and SIFT1M dataset" \
      pgvector.patch="${PATCH_FILE}"

# Expose default PostgreSQL port
EXPOSE 5432

# Inherit ENTRYPOINT and CMD from the base postgres image:
#   ENTRYPOINT ["docker-entrypoint.sh"]
#   CMD ["postgres"]
