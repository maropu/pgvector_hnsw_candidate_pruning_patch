# syntax=docker/dockerfile:1
#
# pgvector test image — compile + regression test suite
#
# Fetches pgvector at the specified git ref, applies a candidate-pruning
# patch, compiles the extension, and runs two test suites:
#
#   1. SQL regression tests  (make installcheck)
#      Classic pg_regress tests in test/sql/*.sql.
#
#   2. TAP tests  (make prove_installcheck)
#      Perl-driven tests in test/t/*.pl that each spin up their own
#      temporary PostgreSQL cluster.  The subset defined by PROVE_TESTS
#      includes the tests that are modified or added by the patches.
#
# =============================================================================
# BUILD ARGUMENTS
# =============================================================================
#   PG_MAJOR      PostgreSQL major version.  Default: 17
#   PGVECTOR_REF  pgvector git ref (tag or commit SHA).  Default: v0.8.0
#   PATCH_FILE    Patch filename (must exist in the repository root).
#                 Default: pgvector_v0.8.0_hnsw_candidate_pruning_turboquant.patch
#
# =============================================================================
# BUILD EXAMPLE
# =============================================================================
#   docker build \
#     --build-arg PATCH_FILE=pgvector_v0.8.0_hnsw_candidate_pruning_turboquant.patch \
#     -t pgvector-test:pg17-turboquant \
#     -f dockerfiles/test/Dockerfile .
#
# =============================================================================
# NOTES
# =============================================================================
#   - The image exits 0 only when all tests pass; a non-zero exit code means
#     at least one test failed — check the build log for details.
#   - ABI compatibility: the base image pins the PostgreSQL major version, so
#     the compiled vector.so matches the server that runs the tests.
#   - Internet access is required at build time to fetch pgvector from GitHub.

ARG PG_MAJOR=17
ARG PGVECTOR_REF=v0.8.0
ARG PATCH_FILE=pgvector_v0.8.0_hnsw_candidate_pruning_turboquant.patch

FROM postgres:${PG_MAJOR}-bookworm
ARG PG_MAJOR
ARG PGVECTOR_REF
ARG PATCH_FILE

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# ---- Install build + test dependencies ----
RUN set -eux; \
    export DEBIAN_FRONTEND=noninteractive; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      build-essential \
      libpq-dev \
      postgresql-server-dev-"${PG_MAJOR}" \
      ca-certificates \
      patch \
      perl \
      libipc-run-perl \
      libtest-simple-perl; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/*

# ---- Fetch pgvector source ----
ADD https://github.com/pgvector/pgvector.git#${PGVECTOR_REF} /tmp/pgvector

# ---- Copy and apply patch ----
COPY ${PATCH_FILE} /tmp/pgvector/
WORKDIR /tmp/pgvector

RUN set -eux; \
    patch -p1 --forward --dry-run < "${PATCH_FILE}"; \
    patch -p1 --forward < "${PATCH_FILE}"

# ---- Compile ----
RUN set -eux; \
    make clean; \
    make USE_PGXS=1 OPTFLAGS=""

# ---- Install into the system PostgreSQL ----
RUN set -eux; \
    make USE_PGXS=1 install

# ---- Fix ownership so postgres can write regression output ----
RUN chown -R postgres:postgres /tmp/pgvector

# ---- SQL regression tests (make installcheck) ----
#
# Requires a running server.  We start a temporary cluster as the postgres
# user, run installcheck, then stop it.  The cluster is discarded at the
# end of this layer.
RUN set -eux; \
    export PGDATA=/tmp/testdb; \
    su postgres -c "initdb -D ${PGDATA} --no-locale --encoding=UTF8"; \
    su postgres -c "pg_ctl start -D ${PGDATA} -l /tmp/pg.log -w"; \
    su postgres -c "make installcheck USE_PGXS=1 -C /tmp/pgvector" \
        || { echo '--- PostgreSQL log ---'; cat /tmp/pg.log; \
             echo '--- regression.diffs ---'; \
             cat /tmp/pgvector/regression.diffs 2>/dev/null || true; \
             su postgres -c "pg_ctl stop -D ${PGDATA}"; exit 1; }; \
    su postgres -c "pg_ctl stop -D ${PGDATA}"

# ---- TAP tests (prove_installcheck) ----
#
# PostgreSQL::Test::Cluster starts its own temporary server per test script,
# so no pre-running server is needed here.
#
# PROVE_TESTS covers:
#   012  HNSW build recall          — core correctness
#   013  HNSW insert recall         — incremental insert correctness
#   017  HNSW filtering             — modified by candidate-pruning patches
#   039  HNSW cost estimation       — modified by candidate-pruning patches
#   043  HNSW iterative scan        — modified by candidate-pruning patches
#   044  HNSW iterative scan recall — modified by candidate-pruning patches
RUN set -eux; \
    PROVE_TESTS="test/t/012_hnsw_vector_build_recall.pl \
                 test/t/013_hnsw_vector_insert_recall.pl \
                 test/t/017_hnsw_filtering.pl \
                 test/t/039_hnsw_cost.pl \
                 test/t/043_hnsw_iterative_scan.pl \
                 test/t/044_hnsw_iterative_scan_recall.pl"; \
    su postgres -c \
        "cd /tmp/pgvector && make prove_installcheck USE_PGXS=1 PROVE_TESTS='${PROVE_TESTS}'" \
        || { echo '--- TAP test log ---'; \
             find /tmp -name '*.log' -newer /tmp/pgvector/Makefile \
               -exec echo '=== {} ===' \; -exec cat {} \; 2>/dev/null || true; \
             exit 1; }

# ---- Done ----
#
# Reaching this point means all tests passed.  The image itself is not
# meant to be run as a container; it is used purely as a build-time test.
RUN echo "All tests passed."
