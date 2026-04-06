# syntax=docker/dockerfile:1
#
# pgvector build image w/ candidate pruning patch
#
# Compiles pgvector from source with the specified patch applied and collects
# the resulting extension files into /artifacts inside the image.  No PostgreSQL
# server is started; the image exists solely to produce build outputs.
#
# =============================================================================
# WHAT THIS DOCKERFILE DOES
# =============================================================================
# Build stage  (AS build):
#   1. Fetches pgvector source at the specified git ref from GitHub (BuildKit).
#   2. Copies the patch file from the repository root into the source tree.
#   3. Installs build tools (gcc, make, postgresql-server-dev) via apt.
#   4. Dry-runs the patch for early diagnostics, then applies it.
#   5. Compiles vector.so and installs extension files into /pkgroot (DESTDIR).
#   6. Collects artifacts into /artifacts:
#        lib/vector.so
#        extension/vector.control
#        extension/vector--*.sql
#        doc/LICENSE  doc/README.md
#        pgvector-<ref>-pg<major>-bookworm-<arch>.tar.gz  (convenience tarball)
#   7. Strips build residues (source tree, DESTDIR, build packages) to keep
#      the image lean.
#
# out stage  (AS out — Buildx only):
#   Scratch image that contains only /artifacts; used with
#   --output type=local,dest=./dist to write artifacts directly to the host.
#
# artifacts stage  (AS artifacts — default build target):
#   debian:bookworm-slim image with /artifacts in place; use docker cp to
#   extract files without ever starting the container.
#
# =============================================================================
# REQUIREMENTS
# =============================================================================
#   - Docker with BuildKit enabled (required for ADD <git-url>#<ref>).
#     BuildKit is the default in Docker 23+; for older versions set:
#       DOCKER_BUILDKIT=1
#   - Build context must be the repository root (contains the .patch file).
#   - Internet access at build time to fetch pgvector source from GitHub.
#
# =============================================================================
# BUILD ARGUMENTS
# =============================================================================
#   PG_MAJOR      PostgreSQL major version.  Default: 17
#   PGVECTOR_REF  pgvector git ref (tag or commit SHA).  Default: v0.8.0
#   PATCH_FILE    Patch filename (must exist in the repository root).
#                 Accepts both plain .patch and gzip-compressed .patch.gz files.
#                 Default: pgvector_v0.8.0_hnsw_candidate_pruning_simhash.patch
#                 Available patches (see repo root):
#                   pgvector_v0.8.0_hnsw_candidate_pruning_turboquant.patch[.gz]
#                   pgvector_v0.8.0_hnsw_candidate_pruning_simhash.patch[.gz]
#                   pgvector_v0.8.0_hnsw_candidate_pruning_pq.patch[.gz]
#
# =============================================================================
# BUILD EXAMPLES
# =============================================================================
#   # Default (simhash patch, PostgreSQL 17)
#   docker build \
#     -t pgvector-artifacts:pg17-simhash \
#     -f dockerfiles/build/Dockerfile .
#
#   # turboquant patch
#   docker build \
#     --build-arg PATCH_FILE=pgvector_v0.8.0_hnsw_candidate_pruning_turboquant.patch \
#     -t pgvector-artifacts:pg17-turboquant \
#     -f dockerfiles/build/Dockerfile .
#
#   # PQ patch, PostgreSQL 16
#   docker build \
#     --build-arg PG_MAJOR=16 \
#     --build-arg PATCH_FILE=pgvector_v0.8.0_hnsw_candidate_pruning_pq.patch \
#     -t pgvector-artifacts:pg16-pq \
#     -f dockerfiles/build/Dockerfile .
#
# =============================================================================
# LAYER CACHING NOTE (patch iteration)
# =============================================================================
#   The GitHub source fetch and apt-get install steps are ordered BEFORE
#   COPY ${PATCH_FILE}, so those layers remain cached even when the patch
#   file changes.  Only the patch-apply and make steps re-run on each
#   patch iteration.
#
# =============================================================================
# ARTIFACT LAYOUT
# =============================================================================
#   /artifacts/
#     lib/
#       vector.so                              shared library loaded by PostgreSQL
#     extension/
#       vector.control                         extension metadata
#       vector--<version>.sql                  extension SQL script(s)
#     doc/
#       LICENSE
#       README.md
#     pgvector-<ref>-pg<major>-bookworm-<arch>.tar.gz   convenience tarball
#
#   Install paths on the target host (per pg_config):
#     $libdir              <- vector.so
#     $sharedir/extension  <- vector.control, vector--*.sql
#
# =============================================================================
# EXTRACTING ARTIFACTS
# =============================================================================
#   Method 1 — docker cp (works with plain docker build):
#     cid=$(docker create pgvector-artifacts:pg17-simhash)
#     docker cp "$cid":/artifacts ./artifacts
#     docker rm "$cid"
#     ls -al ./artifacts
#
#   Method 2 — Buildx local output (writes directly to host, no image needed):
#     docker buildx build \
#       --build-arg PATCH_FILE=pgvector_v0.8.0_hnsw_candidate_pruning_simhash.patch \
#       --target out \
#       --output type=local,dest=./dist \
#       -f dockerfiles/build/Dockerfile .
#     ls -al ./dist
#
# =============================================================================
# NOTES
# =============================================================================
#   - ABI compatibility: the build uses postgres:<major>-bookworm as its base
#     image.  Install vector.so only on a PostgreSQL host built from the same
#     major version and OS (bookworm).
#   - The tarball inside /artifacts is named after the ref, PG major, OS, and
#     CPU architecture so it can be stored and distributed unambiguously.

ARG PG_MAJOR=17
ARG PGVECTOR_REF=v0.8.0
ARG PATCH_FILE=pgvector_v0.8.0_hnsw_candidate_pruning_simhash.patch

# ---- Build stage (keeps ABI with target Postgres by using postgres:<major>-bookworm) ----
FROM postgres:${PG_MAJOR}-bookworm AS build
ARG PG_MAJOR
ARG PGVECTOR_REF
ARG PATCH_FILE

# Fetch source at the specified git ref (tag/SHA) using BuildKit remote git context
ADD https://github.com/pgvector/pgvector.git#${PGVECTOR_REF} /tmp/pgvector

# Copy the specified patch file from build context
COPY ${PATCH_FILE} /tmp/pgvector/

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# --- Decompress patch if gzip-compressed ---
RUN set -eux; \
    if [[ "${PATCH_FILE}" == *.gz ]]; then \
      gunzip "${PATCH_FILE}"; \
    fi

# --- Install build dependencies ---
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

# --- Prepare source workspace ---
WORKDIR /tmp/pgvector

# --- Patch (dry-run for diagnostics) ---
RUN set -eux; \
    patch -p1 --forward --dry-run < "${PATCH_FILE%.gz}"

# --- Apply patch ---
RUN set -eux; \
    patch -p1 --forward < "${PATCH_FILE%.gz}"

# --- Build extension ---
RUN set -eux; \
    make clean; \
    make USE_PGXS=1 OPTFLAGS=""

# --- Staged install into DESTDIR ---
RUN set -eux; \
    DESTDIR="/pkgroot"; \
    make USE_PGXS=1 install DESTDIR="${DESTDIR}"

# --- Collect artifacts into /artifacts ---
RUN set -eux; \
    PKGLIBDIR="$(pg_config --pkglibdir)"; \
    EXT_DIR="$(pg_config --sharedir)/extension"; \
    DESTDIR="/pkgroot"; \
    mkdir -p /artifacts/lib /artifacts/extension /artifacts/doc; \
    cp "${DESTDIR}${PKGLIBDIR}/vector.so" /artifacts/lib/; \
    cp "${DESTDIR}${EXT_DIR}/vector.control" /artifacts/extension/; \
    cp "${DESTDIR}${EXT_DIR}/vector--"*.sql /artifacts/extension/; \
    cp LICENSE README.md /artifacts/doc/

# --- Package tarball (avoid self-inclusion by writing to /tmp first) ---
RUN set -eux; \
    ARCH="$(dpkg --print-architecture)"; \
    ARTIFACT_NAME="pgvector-${PGVECTOR_REF}-pg${PG_MAJOR}-bookworm-${ARCH}.tar.gz"; \
    TMP_TARBALL="/tmp/${ARTIFACT_NAME}"; \
    tar -C /artifacts -czf "${TMP_TARBALL}" .; \
    mv "${TMP_TARBALL}" "/artifacts/${ARTIFACT_NAME}"

# --- Cleanup build residues ---
RUN set -eux; \
    rm -rf /tmp/pgvector "/pkgroot"; \
    apt-get update; \
    apt-get purge -y --auto-remove build-essential postgresql-server-dev-"${PG_MAJOR}" patch libpq-dev; \
    rm -rf /var/lib/apt/lists/*

# ---- Optional out stage (for Buildx local output) ----
FROM scratch AS out
COPY --from=build /artifacts /

# ---- Final artifacts image (default build target) ----
FROM debian:bookworm-slim AS artifacts
COPY --from=build /artifacts /artifacts
LABEL org.opencontainers.image.title="pgvector artifacts" \
      org.opencontainers.image.description="pgvector build outputs under /artifacts" \
      org.opencontainers.image.source="https://github.com/pgvector/pgvector"
