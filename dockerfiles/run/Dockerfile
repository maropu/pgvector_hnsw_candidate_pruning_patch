# syntax=docker/dockerfile:1
#
# pgvector run image w/ candidate pruning patch + SIFT1M dataset
#
# Builds pgvector from source with the specified patch applied, installs it
# into a standard postgres image, and loads the SIFT1M (or siftsmall) dataset
# on first container start.
#
# =============================================================================
# WHAT THIS DOCKERFILE DOES
# =============================================================================
# Build stage  (AS build):
#   1. Installs build tools (gcc, make, postgresql-server-dev) via apt.
#   2. Fetches pgvector source at the specified git ref from GitHub (BuildKit).
#   3. Copies the patch file from the repository root into the source tree.
#   4. Dry-runs the patch for early diagnostics, then applies it.
#   5. Compiles vector.so and installs extension files into /pkgroot (DESTDIR).
#
# Runtime stage  (AS run):
#   6. Installs Python 3 + psycopg[binary] in a venv (/opt/sift-venv) for the
#      SIFT data-loading script, and tini as PID 1.
#   7. Copies vector.so and .sql/.control extension files from the build stage.
#   8. Places four initdb scripts in /docker-entrypoint-initdb.d/ that run in
#      order on the very first container start:
#        00-configure-postgres.sh  — applies shared_buffers / maintenance_work_mem
#        10-vector.sql             — CREATE EXTENSION vector
#        20-load-sift1m.sh         — downloads and loads SIFT dataset
#        30-create-hnsw-index.sh   — builds HNSW index on the base table
#
# =============================================================================
# REQUIREMENTS
# =============================================================================
#   - Docker with BuildKit enabled (required for ADD <git-url>#<ref>).
#     BuildKit is the default in Docker 23+; for older versions set:
#       DOCKER_BUILDKIT=1
#   - Build context must be the repository root (contains the .patch file).
#   - Internet access at build time to fetch pgvector source from GitHub.
#   - Internet access at first container start to download the SIFT archive
#     from ftp://ftp.irisa.fr (unless SIFT_CACHE_DIR is pre-populated).
#
# =============================================================================
# BUILD ARGUMENTS
# =============================================================================
#   PG_MAJOR      PostgreSQL major version.  Default: 17
#   PGVECTOR_REF  pgvector git ref (tag or commit SHA).  Default: v0.8.0
#   PATCH_FILE    Patch filename (must exist in the repository root).
#                 Default: pgvector_v0.8.0_hnsw_candidate_pruning_turboquant.patch
#                 Available patches (see repo root):
#                   pgvector_v0.8.0_hnsw_candidate_pruning_turboquant.patch
#                   pgvector_v0.8.0_hnsw_candidate_pruning_simhash.patch
#                   pgvector_v0.8.0_hnsw_candidate_pruning_pq.patch
#
# =============================================================================
# BUILD EXAMPLES
# =============================================================================
#   # simhash patch (recommended naming: pg17-simhash)
#   docker build \
#     --build-arg PATCH_FILE=pgvector_v0.8.0_hnsw_candidate_pruning_simhash.patch \
#     -t pgvector-run:pg17-simhash \
#     -f dockerfiles/run/Dockerfile .
#
#   # turboquant patch
#   docker build \
#     --build-arg PATCH_FILE=pgvector_v0.8.0_hnsw_candidate_pruning_turboquant.patch \
#     -t pgvector-run:pg17-turboquant \
#     -f dockerfiles/run/Dockerfile .
#
#   # Different PostgreSQL version
#   docker build \
#     --build-arg PG_MAJOR=16 \
#     --build-arg PATCH_FILE=pgvector_v0.8.0_hnsw_candidate_pruning_simhash.patch \
#     -t pgvector-run:pg16-simhash \
#     -f dockerfiles/run/Dockerfile .
#
# =============================================================================
# LAYER CACHING NOTE (patch iteration)
# =============================================================================
#   apt-get and the GitHub source fetch are ordered BEFORE COPY ${PATCH_FILE},
#   so those layers remain cached even when the patch file changes.  Only the
#   patch-apply and make steps re-run on each patch iteration.
#
# =============================================================================
# RUNTIME ENVIRONMENT VARIABLES
# =============================================================================
#   POSTGRES_PASSWORD       (required) PostgreSQL superuser password.
#
#   PG_SHARED_BUFFERS       PostgreSQL shared_buffers.  Default: "4GB"
#   PG_MAINTENANCE_WORK_MEM PostgreSQL maintenance_work_mem.  Default: "1GB"
#
#   SIFT_DATASET    Dataset to load on first start.
#                     sift1m    : 1,000,000 vectors, 128 dims, ~161 MB  (default)
#                     siftsmall : 10,000 vectors, 128 dims,  ~3 MB  (dev/test)
#
#   SIFT_SUBSETS    Space-separated list of splits to load.
#                   Values: base  query  learn  (any combination)
#                   Set to "" to skip all data loading.
#                   Default: "base"
#
#   SIFT_TABLE      Base table name (default: sift1m).
#                   "base" split  -> sift1m
#                   Other splits  -> sift1m_query, sift1m_learn
#
#   SIFT_COLUMN     Vector column name (default: embedding).
#
#   SIFT_CACHE_DIR  Path inside the container for caching the downloaded
#                   archive and extracted .fvecs files.
#                   Default: /var/lib/postgresql/sift_cache
#                   To persist across runs, bind-mount a host directory
#                   writable by UID 999 (the postgres user):
#                     -v /local/cache:/sift_cache -e SIFT_CACHE_DIR=/sift_cache
#
# =============================================================================
# RUN EXAMPLES
# =============================================================================
#   # Production: load sift1m base split and build HNSW index
#   docker run --rm \
#     -e POSTGRES_PASSWORD=postgres \
#     -e SIFT_SUBSETS="base" \
#     -p 5432:5432 \
#     pgvector-run:pg17-simhash
#
#   # Dev/test: use siftsmall for fast startup (~3 MB download, seconds to load)
#   docker run --rm \
#     -e POSTGRES_PASSWORD=postgres \
#     -e SIFT_DATASET=siftsmall \
#     -e SIFT_SUBSETS="base" \
#     -p 5432:5432 \
#     pgvector-run:pg17-simhash
#
#   # Skip data loading entirely (PostgreSQL + pgvector only)
#   docker run --rm \
#     -e POSTGRES_PASSWORD=postgres \
#     -e SIFT_SUBSETS="" \
#     -p 5432:5432 \
#     pgvector-run:pg17-simhash
#
#   # Persist data and reuse SIFT cache across runs
#   docker run \
#     -e POSTGRES_PASSWORD=postgres \
#     -v pgdata:/var/lib/postgresql/data \
#     -v /local/cache:/sift_cache \
#     -e SIFT_CACHE_DIR=/sift_cache \
#     -p 5432:5432 \
#     pgvector-run:pg17-simhash
#
# =============================================================================
# CONNECT
# =============================================================================
#   psql -h 127.0.0.1 -U postgres --password
#   \dt          -- lists tables (sift1m appears after loading completes)
#
#   -- Check index build progress while loading:
#   SELECT phase, blocks_done, blocks_total,
#          round(100.0 * blocks_done / NULLIF(blocks_total,0), 1) AS pct
#   FROM pg_stat_progress_create_index;
#
# =============================================================================
# RESTART PostgreSQL WITHOUT STOPPING THE CONTAINER
# =============================================================================
#   tini is PID 1; postgres runs as its child, so pg_ctl restart is safe.
#
#   docker exec -u postgres <container> \
#     /usr/lib/postgresql/17/bin/pg_ctl restart -D /var/lib/postgresql/data
#
# =============================================================================
# NOTES
# =============================================================================
#   - SIFT download happens at first container start, not at build time.
#     Use SIFT_CACHE_DIR bind-mount to avoid re-downloading across runs.
#   - ABI compatibility: build stage and runtime stage share the same
#     postgres:<major>-bookworm base image, ensuring vector.so is compatible.
#   - Data persistence: use -v pgdata:/var/lib/postgresql/data to survive
#     container removal.
#   - Shared memory: always pass --shm-size=8g (or larger) to docker run.
#     PostgreSQL shared_buffers (default 4 GB) is backed by POSIX shared memory;
#     Docker's default /dev/shm is only 64 MB and will cause
#     "No space left on device" when creating an HNSW index without it.

ARG PG_MAJOR=17
ARG PGVECTOR_REF=v0.8.0
ARG PATCH_FILE=pgvector_v0.8.0_hnsw_candidate_pruning_turboquant.patch

# ---- Build stage ----
FROM postgres:${PG_MAJOR}-bookworm AS build
ARG PG_MAJOR
ARG PGVECTOR_REF
ARG PATCH_FILE

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install build dependencies before copying the patch so this layer is cached
# even when the patch file changes.
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

# Fetch pgvector source at the specified git ref using BuildKit remote git context
ADD https://github.com/pgvector/pgvector.git#${PGVECTOR_REF} /tmp/pgvector

# Copy the patch file from the repository root (build context)
COPY ${PATCH_FILE} /tmp/pgvector/

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
# plus tini as a minimal init process so postgres is not PID 1.
# With tini as PID 1, pg_ctl restart works inside the container without
# stopping it (restarting PID 1 would terminate the container).
RUN set -eux; \
    export DEBIAN_FRONTEND=noninteractive; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      python3 \
      python3-pip \
      python3-venv \
      ca-certificates \
      tini; \
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

# Default environment variables.
# PG_SHARED_BUFFERS: passed to configure_postgres.sh; override at runtime with
#   -e PG_SHARED_BUFFERS=8GB
# SIFT_CACHE_DIR defaults to a path inside the postgres data volume so the
# postgres user can write there without needing host-side chmod.
# Override at runtime with -e SIFT_CACHE_DIR=/sift_cache when you mount a
# host directory that is writable by UID 999 (the postgres user).
ENV PG_SHARED_BUFFERS="4GB" \
    PG_MAINTENANCE_WORK_MEM="1GB" \
    SIFT_DATASET="sift1m" \
    SIFT_SUBSETS="base" \
    SIFT_TABLE="sift1m" \
    SIFT_COLUMN="embedding" \
    SIFT_CACHE_DIR="/var/lib/postgresql/sift_cache"

# initdb script 0: apply PostgreSQL tuning parameters (runs first)
COPY --chmod=755 scripts/configure_postgres.sh \
     /docker-entrypoint-initdb.d/00-configure-postgres.sh

# initdb script 1: create the vector extension
RUN echo "CREATE EXTENSION IF NOT EXISTS vector;" \
      > /docker-entrypoint-initdb.d/10-vector.sql

# initdb script 2: load SIFT1M dataset
# Runs as the postgres user; connects via Unix socket (no password needed).
# Skipped when SIFT_SUBSETS is empty.
COPY --chmod=755 scripts/import_sift1m_to_postgres.sh \
     /docker-entrypoint-initdb.d/20-load-sift1m.sh

# initdb script 3: create HNSW index on the base table (runs after data load)
COPY --chmod=755 scripts/create_hnsw_index.sh \
     /docker-entrypoint-initdb.d/30-create-hnsw-index.sh

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

# Use a wrapper script so that tini's direct child is a long-lived shell, not
# postgres itself.  When docker-entrypoint.sh runs it eventually does
# `exec postgres`, which would make postgres PID 2 (tini's direct child).
# With that layout, `pg_ctl restart` kills postgres → tini sees its child exit
# → container terminates.
#
# The wrapper keeps tini's child alive across pg_ctl restarts:
#   1. docker-entrypoint.sh postgres  starts (and initialises) postgres
#   2. Once init is done postgres is running in the foreground inside the wrapper
#   3. pg_ctl restart can now stop/start postgres without the wrapper exiting
#
# Signal handling: tini forwards SIGTERM to the wrapper, which propagates it to
# the postgres process group, so `docker stop` still shuts everything down
# cleanly.
COPY --chmod=755 <<'EOF' /usr/local/bin/pg-entrypoint.sh
#!/bin/bash
# Wrapper that keeps this shell (tini's direct child) alive across pg_ctl
# restarts.  Without this wrapper, docker-entrypoint.sh eventually does
# `exec postgres`, making postgres tini's direct child.  pg_ctl restart then
# kills postgres → tini sees its child exit → container terminates.
#
# Solution: run docker-entrypoint.sh in the background so THIS shell is
# always tini's direct child.  After postgres exits during a restart, we fall
# through to an interruptible sleep loop that keeps the container alive until
# `docker stop` sends SIGTERM.

_stop() {
    /usr/lib/postgresql/17/bin/pg_ctl stop \
        -D /var/lib/postgresql/data -m fast 2>/dev/null || true
    exit 0
}
trap '_stop' TERM INT

# Start postgres (including initdb on first run) in the background.
docker-entrypoint.sh "$@" &
wait $! || true

# postgres exited (e.g. during pg_ctl restart).  Stay alive so the restarted
# postgres can continue running.  The sleep loop is signal-interruptible so
# `docker stop` → SIGTERM → _stop() still shuts everything down cleanly.
while true; do
    sleep 60 &
    wait $!
done
EOF

ENTRYPOINT ["tini", "--", "pg-entrypoint.sh"]
CMD ["postgres"]
