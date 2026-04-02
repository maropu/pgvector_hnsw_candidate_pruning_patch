#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
import_sift1m_to_postgres.py

Downloads a TexMex SIFT dataset from FTP, caches it locally, and streams one
split (base / learn / query) into a PostgreSQL table using the pgvector
extension.  Designed to be called by import_sift1m_to_postgres.sh inside the
Docker initdb sequence, but also usable standalone from the host or any
container that can reach the PostgreSQL Unix socket on port 5432.

-----------------------------------------------------------------------------
WHAT THIS SCRIPT DOES
-----------------------------------------------------------------------------
1. Downloads the dataset archive (FTP) to --cache-dir if not already cached.
2. Extracts the tarball into --cache-dir if the .fvecs files are not already
   present.
3. Opens a psycopg connection via Unix socket (no host, fixed port 5432).
4. Creates the pgvector extension if missing.
5. Creates a new table with a bigserial PRIMARY KEY and a vector(N) column.
   Raises an error if the table already exists (no upsert / truncate logic).
6. Streams all rows from the chosen .fvecs split into the table via COPY.
   No index is created — index creation is handled separately by
   create_hnsw_index.sh.

-----------------------------------------------------------------------------
SUPPORTED DATASETS  (--dataset)
-----------------------------------------------------------------------------
  sift1m    (default)
    1,000,000 base vectors, 128 dimensions, float32
    Archive : ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz  (~161 MB)
    Splits  : base (1 M rows)  |  learn (100 K rows)  |  query (10 K rows)

  siftsmall
    10,000 base vectors, 128 dimensions, float32
    Archive : ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz  (~3 MB)
    Splits  : base (10 K rows)  |  learn (25 K rows)  |  query (100 rows)
    Recommended for quick smoke-tests and patch iteration.

-----------------------------------------------------------------------------
ARGUMENTS
-----------------------------------------------------------------------------
  --dbname      (required) Target PostgreSQL database name.
  --user        PostgreSQL user name. Default: postgres
  --dataset     Dataset variant: sift1m | siftsmall. Default: sift1m
  --subset      Which split to load: base | learn | query. Default: base
  --table       Destination table name (schema-qualified OK, e.g. public.sift1m).
                Default: sift1m
  --column      Name of the vector column. Default: embedding
  --cache-dir   Directory used to cache the downloaded archive and extracted
                .fvecs files across runs. Default: /tmp/sift_texmex
  --batch-rows  Progress-log interval (rows). Default: 10000
  --verbose     Print download / extract / copy progress to stdout.

-----------------------------------------------------------------------------
USAGE EXAMPLES
-----------------------------------------------------------------------------
  # Load sift1m base split (production default)
  python import_sift1m_to_postgres.py \
    --dbname postgres --user postgres \
    --dataset sift1m --subset base \
    --table sift1m --column embedding \
    --verbose

  # Load siftsmall base split (fast dev/test)
  python import_sift1m_to_postgres.py \
    --dbname postgres --user postgres \
    --dataset siftsmall --subset base \
    --table sift1m --column embedding \
    --verbose

  # Load query split into a separate table
  python import_sift1m_to_postgres.py \
    --dbname postgres --user postgres \
    --dataset sift1m --subset query \
    --table sift1m_query --column embedding \
    --verbose

-----------------------------------------------------------------------------
SCHEMA CREATED
-----------------------------------------------------------------------------
  CREATE TABLE <table> (
    id       bigserial PRIMARY KEY,
    <column> vector(<dim>)
  );
  -- dim = 128 for both sift1m and siftsmall

-----------------------------------------------------------------------------
ERROR CONDITIONS
-----------------------------------------------------------------------------
  - Table already exists          → psycopg raises DuplicateTable; re-run
                                    after DROP TABLE <table>.
  - FTP unreachable               → urllib raises URLError.
  - Corrupt / incomplete archive  → tarfile / struct raises an exception.
  - pgvector extension missing    → script auto-runs CREATE EXTENSION vector;
                                    fails only if the .so is not installed.

-----------------------------------------------------------------------------
CACHING BEHAVIOUR
-----------------------------------------------------------------------------
  Both the .tar.gz archive and the extracted .fvecs directory are reused on
  subsequent runs.  Delete --cache-dir (or the specific archive/directory
  inside it) to force a fresh download or re-extraction.
  Default cache layout:
    <cache-dir>/sift.tar.gz          (sift1m archive)
    <cache-dir>/sift/                (sift1m extracted files)
    <cache-dir>/siftsmall.tar.gz     (siftsmall archive)
    <cache-dir>/siftsmall/           (siftsmall extracted files)
"""

from __future__ import annotations
import argparse
import io
import struct
import tarfile
import urllib.request
from pathlib import Path
from typing import Generator, Iterable, Tuple

import psycopg
from psycopg import sql as psql


DEFAULT_CACHE_DIR = Path("/tmp/sift_texmex")

# Per-dataset configuration: URL, archive name, extracted dir, split filenames
DATASET_CONFIG = {
    "sift1m": {
        "url":       "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz",
        "archive":   "sift.tar.gz",
        "extracted": "sift",
        "splits": {
            "base":  "sift_base.fvecs",
            "learn": "sift_learn.fvecs",
            "query": "sift_query.fvecs",
        },
    },
    "siftsmall": {
        "url":       "ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz",
        "archive":   "siftsmall.tar.gz",
        "extracted": "siftsmall",
        "splits": {
            "base":  "siftsmall_base.fvecs",
            "learn": "siftsmall_learn.fvecs",
            "query": "siftsmall_query.fvecs",
        },
    },
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Download/cache TexMex SIFT (.tar.gz) and load a split into Postgres w/pgvector."
    )

    # Connection (Unix socket; fixed port 5432)
    ap.add_argument("--dbname", required=True, help="Target database name")
    ap.add_argument("--user", default="postgres", help="User name (default: postgres)")

    # Dataset variant
    ap.add_argument("--dataset", choices=list(DATASET_CONFIG.keys()), default="sift1m",
                    help="Which TexMex dataset to load: sift1m (1M vectors) or siftsmall (10K vectors) (default: sift1m)")

    # Dataset split & cache
    ap.add_argument("--subset", choices=["base", "learn", "query"], default="base",
                    help="Which SIFT split to load (default: base)")
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR,
                    help="Cache directory under /tmp (default: /tmp/sift_texmex)")

    # Destination
    ap.add_argument("--table", default="sift1m", help="Destination table name (schema-qualified allowed)")
    ap.add_argument("--column", default="embedding", help="Vector column name (default: embedding)")

    # Misc
    ap.add_argument("--batch-rows", type=int, default=10000, help="Progress interval (rows) (default: 10000)")
    ap.add_argument("--verbose", action="store_true", help="Verbose progress output")
    return ap.parse_args()


def ensure_cache_dirs(cache_dir: Path) -> None:
    """Create cache directory if it does not exist."""
    cache_dir.mkdir(parents=True, exist_ok=True)


def download_if_missing(url: str, dst: Path, verbose: bool) -> None:
    """
    Download the .tar.gz only if it's not present already.
    If present, reuse it as-is.
    """
    if dst.exists():
        if verbose:
            print(f"[cache] Using cached archive: {dst}")
        return
    if verbose:
        print(f"[download] Fetching: {url} -> {dst}")
    tmp = dst.with_suffix(dst.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, tmp)  # supports FTP
        tmp.replace(dst)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass
    if verbose:
        print(f"[download] Completed: {dst}")


def extract_tar_gz_if_missing(archive: Path, out_dir: Path, extracted_dirname: str,
                              split_filenames: dict, verbose: bool) -> Path:
    """
    Extract the .tar.gz if the extracted root doesn't already contain expected .fvecs.
    Returns the extracted root directory path (e.g., /tmp/sift_texmex/sift).
    """
    if archive.suffixes[-2:] != [".tar", ".gz"]:
        raise ValueError(f"Only .tar.gz is supported: {archive}")

    extracted_root = out_dir / extracted_dirname
    expected = set(split_filenames.values())
    if extracted_root.exists():
        have = {p.name for p in extracted_root.glob("*.fvecs")}
        if expected.issubset(have):
            if verbose:
                print(f"[cache] Using extracted data in: {extracted_root}")
            return extracted_root

    if verbose:
        print(f"[extract] Extracting: {archive} -> {out_dir}")
    with tarfile.open(archive, mode="r:gz") as tf:
        tf.extractall(out_dir)
    if verbose:
        print("[extract] Done")
    return extracted_root


def get_split_path(extracted_root: Path, subset: str, split_filenames: dict) -> Path:
    """Resolve the target .fvecs path for the requested split."""
    fname = split_filenames[subset]
    fpath = extracted_root / fname
    if not fpath.exists():
        raise FileNotFoundError(f"Split file not found: {fpath}")
    return fpath


def ident_qualified(name: str) -> psql.Composed:
    """Return a safe schema-qualified Identifier composition."""
    parts = [p.strip() for p in name.split(".")]
    return psql.SQL(".").join(psql.Identifier(p) for p in parts)


def ensure_extension_vector(cur: psycopg.Cursor) -> None:
    """CREATE EXTENSION vector if missing."""
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")


def create_table_fail(cur: psycopg.Cursor, table: str, column: str, dim: int) -> None:
    """
    Always CREATE TABLE. Fail if it already exists.
    This mirrors the requested 'always fail' semantics.
    """
    tbl_ident = ident_qualified(table)
    col_ident = psql.Identifier(column)
    cur.execute(
        psql.SQL(
            "CREATE TABLE {} ("
            "  id bigserial PRIMARY KEY,"
            "  {} vector({})"
            ")"
        ).format(tbl_ident, col_ident, psql.Literal(dim))
    )


def iter_fvecs(path: Path) -> Generator[Tuple[int, Tuple[float, ...]], None, None]:
    """
    Stream records from .fvecs:
      yields (dim, tuple(float32,..))
    """
    with path.open("rb") as f:
        rd = f.read
        unpack_i32 = struct.Struct("<i").unpack
        while True:
            hdr = rd(4)
            if not hdr:
                break
            if len(hdr) != 4:
                raise IOError("Corrupt .fvecs header (short read)")
            (d,) = unpack_i32(hdr)
            payload = rd(4 * d)
            if len(payload) != 4 * d:
                raise IOError(f"Unexpected EOF: expected {4*d} bytes, got {len(payload)}")
            vec = struct.unpack("<" + "f" * d, payload)
            yield d, vec


def copy_vectors(conn: psycopg.Connection,
                 table: str,
                 column: str,
                 dim: int,
                 rows: Iterable[Tuple[int, Tuple[float, ...]]],
                 batch_rows: int,
                 verbose: bool) -> int:
    """
    Stream rows into COPY. Expects each row as (dim, tuple(float,...)).
    Returns total inserted row count.
    """
    tbl_ident = ident_qualified(table)
    col_ident = psql.Identifier(column)
    total = 0

    stmt = psql.SQL("COPY {} ({}) FROM STDIN").format(tbl_ident, col_ident)

    with conn.cursor() as cur, cur.copy(stmt) as copy:
        for rec_dim, vec in rows:
            if rec_dim != dim:
                raise ValueError(f"Dimension mismatch: file has {rec_dim}, expected {dim}")
            lit = "[" + ",".join(f"{v:.6g}" for v in vec) + "]"  # pgvector text literal
            copy.write_row((lit,))
            total += 1
            if verbose and (total % batch_rows == 0):
                print(f"[copy] {total} rows streamed...")

    if verbose:
        print(f"[copy] finished: {total} rows")
    return total


def main():
    args = parse_args()
    cfg = DATASET_CONFIG[args.dataset]
    cache_dir: Path = args.cache_dir
    ensure_cache_dirs(cache_dir)

    archive_path = cache_dir / cfg["archive"]

    # Download .tar.gz if missing; reuse otherwise
    download_if_missing(cfg["url"], archive_path, args.verbose)

    # Extract if needed; reuse otherwise (.tar.gz only)
    extracted_root = extract_tar_gz_if_missing(
        archive_path, cache_dir, cfg["extracted"], cfg["splits"], args.verbose
    )

    # Resolve the path to the requested split (.fvecs)
    fvecs_path = get_split_path(extracted_root, args.subset, cfg["splits"])
    if args.verbose:
        print(f"[cache] Using split file: {fvecs_path}")

    # Connect via local UNIX socket (host unspecified), port fixed to 5432
    conn = psycopg.connect(
        dbname=args.dbname,
        user=args.user,
        port=5432,
        autocommit=True,
    )

    try:
        # Peek the first record to determine dim
        stream = iter_fvecs(fvecs_path)
        try:
            first_dim, first_vec = next(stream)
        except StopIteration:
            raise RuntimeError(f"Empty .fvecs file: {fvecs_path}")

        dim = first_dim

        with conn.cursor() as cur:
            ensure_extension_vector(cur)
            # Always CREATE TABLE; will raise if it already exists
            create_table_fail(cur, args.table, args.column, dim)

        # Chain the first record back into the stream
        def chained_rows() -> Generator[Tuple[int, Tuple[float, ...]], None, None]:
            yield first_dim, first_vec
            for rec in stream:
                yield rec

        # COPY streaming (no --limit; load all)
        inserted = copy_vectors(
            conn=conn,
            table=args.table,
            column=args.column,
            dim=dim,
            rows=chained_rows(),
            batch_rows=args.batch_rows,
            verbose=args.verbose,
        )

        if args.verbose:
            print(f"[done] inserted rows: {inserted} (subset={args.subset}, dim={dim})")

    finally:
        conn.close()


if __name__ == "__main__":
    main()

