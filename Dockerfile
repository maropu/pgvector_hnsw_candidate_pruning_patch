# ==== global ARGs (must be before the first FROM if used in FROM) ====
ARG PG_MAJOR=17
ARG PGVECTOR_REF=v0.8.0

# ---- Build stage (split RUN for debuggability) ----
FROM postgres:${PG_MAJOR}-bookworm AS build
# re-declare to use inside this stage (RUN, ENV, etc.)
ARG PG_MAJOR
ARG PGVECTOR_REF

# BuildKit remote git context
ADD https://github.com/pgvector/pgvector.git#${PGVECTOR_REF} /tmp/pgvector
COPY pgvector_v0.8.0_hnsw_pruning.patch /tmp/pgvector/

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# 1) deps
RUN set -eux; \
    export DEBIAN_FRONTEND=noninteractive; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      build-essential \
      libpq-dev \
      postgresql-server-dev-${PG_MAJOR} \
      ca-certificates \
      patch \
      findutils \
      file; \
    apt-get clean

WORKDIR /tmp/pgvector

# 2) patch dry-run
RUN set -eux; \
    patch -p1 --forward --dry-run < pgvector_v0.8.0_hnsw_pruning.patch

# 3) patch apply
RUN set -eux; \
    patch -p1 --forward < pgvector_v0.8.0_hnsw_pruning.patch

# 4) build
RUN set -eux; \
    make clean; \
    make USE_PGXS=1 OPTFLAGS=""

# 5) staged install
RUN set -eux; \
    DESTDIR="/pkgroot"; \
    make USE_PGXS=1 install DESTDIR="${DESTDIR}"; \
    echo "pkglibdir=$(pg_config --pkglibdir)"; \
    echo "sharedir=$(pg_config --sharedir)"; \
    ls -al "${DESTDIR}$(pg_config --pkglibdir)" || true; \
    ls -al "${DESTDIR}$(pg_config --sharedir)/extension" || true

# 6) collect artifacts
RUN set -eux; \
    PKGLIBDIR="$(pg_config --pkglibdir)"; \
    EXT_DIR="$(pg_config --sharedir)/extension"; \
    DESTDIR="/pkgroot"; \
    install -d /artifacts/lib /artifacts/extension /artifacts/doc; \
    test -f "${DESTDIR}${PKGLIBDIR}/vector.so"; \
    install -m 0644 "${DESTDIR}${PKGLIBDIR}/vector.so" /artifacts/lib/; \
    test -f "${DESTDIR}${EXT_DIR}/vector.control"; \
    install -m 0644 "${DESTDIR}${EXT_DIR}/vector.control" /artifacts/extension/; \
    if compgen -G "${DESTDIR}${EXT_DIR}/vector--"*.sql > /dev/null; then \
      cp -v "${DESTDIR}${EXT_DIR}/vector--"*.sql /artifacts/extension/; \
    else \
      echo "ERROR: No SQL files matched ${DESTDIR}${EXT_DIR}/vector--*.sql"; exit 1; \
    fi; \
    install -m 0644 LICENSE README.md /artifacts/doc/

# 7) tarball
RUN set -eux; \
    echo "==> pack tarball"; \
    # 念のため中身があることを確認
    test -f /artifacts/lib/vector.so; \
    test -f /artifacts/extension/vector.control; \
    ARCH="$(dpkg --print-architecture)"; \
    ARTIFACT_NAME="pgvector-${PGVECTOR_REF}-pg${PG_MAJOR}-bookworm-${ARCH}.tar.gz"; \
    TMP_TARBALL="/tmp/${ARTIFACT_NAME}"; \
    # /artifacts を -C で基点にしつつ、出力は /tmp に逃がす
    tar -C /artifacts -czf "${TMP_TARBALL}" .; \
    mv "${TMP_TARBALL}" "/artifacts/${ARTIFACT_NAME}"; \
    ls -al /artifacts

# 8) cleanup (optional)
RUN set -eux; \
    rm -rf /tmp/pgvector "/pkgroot"

