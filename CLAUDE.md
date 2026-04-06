# pgvector HNSW candidate pruning — development guide

This repository develops patches that improve pgvector's HNSW implementation
by pruning candidates during graph traversal using per-edge neighbor metadata.
Three approaches are maintained in parallel:

| Patch file | Method |
|---|---|
| `pgvector_v0.8.0_hnsw_candidate_pruning_turboquant.patch.gz` | TurboQuant (random rotation + 2-bit quantization) |
| `pgvector_v0.8.0_hnsw_candidate_pruning_simhash.patch` | SimHash (locality-sensitive hashing) |
| `pgvector_v0.8.0_hnsw_candidate_pruning_pq.patch` | Product Quantization |

The TurboQuant patch contains a large precomputed rotation matrix
(2000×2000 floats) and is gzip-compressed to save space.  The `export`
subcommand compresses automatically when the patch exceeds 10 MiB
(configurable via `--gz-threshold`).  All three Dockerfiles accept both
plain `.patch` and `.patch.gz` files transparently.

---

## Repository layout

```
pgvector_v0.8.0_hnsw_candidate_pruning_<method>.patch[.gz]  # exported patch files
scripts/pgvector_dev.sh          # dev-dir lifecycle helper (setup / export / reset)
dockerfiles/
  build/Dockerfile               # compile patch → /artifacts (vector.so + SQL files)
  test/Dockerfile                # compile + run SQL regression tests + TAP tests
  run/Dockerfile                 # PostgreSQL + patched pgvector + SIFT1M dataset
```

---

## Development workflow

`scripts/pgvector_dev.sh` manages the per-patch working directory lifecycle
(setup / export / reset / status).  For the full subcommand reference and
option descriptions, see the header comment block at the top of the script
itself (`scripts/pgvector_dev.sh`).

### 1. First-time setup (clone pgvector and apply patch)

`scripts/pgvector_dev.sh setup` clones pgvector at the pinned tag into a
working directory named after the patch file, then applies the patch.

```bash
bash scripts/pgvector_dev.sh setup \
  --patch pgvector_v0.8.0_hnsw_candidate_pruning_turboquant.patch
```

This creates a temporary working directory
`.pgvector_dev_v0.8.0_hnsw_candidate_pruning_turboquant/` in the repository
root.  This directory is a gitignored local clone of pgvector with the patch
already applied; it is not committed to the repository.  Edit source files
there directly.

### 2. Edit source code

Work inside the generated working directory, e.g.:

```
.pgvector_dev_v0.8.0_hnsw_candidate_pruning_turboquant/src/hnswmeta.c
.pgvector_dev_v0.8.0_hnsw_candidate_pruning_turboquant/src/hnswtq.h
...
```

### 3. Export changes to the patch file

After editing, regenerate the patch file in the repository root:

```bash
bash scripts/pgvector_dev.sh export \
  --dir .pgvector_dev_v0.8.0_hnsw_candidate_pruning_turboquant \
  pgvector_v0.8.0_hnsw_candidate_pruning_turboquant.patch
```

If the resulting patch exceeds 10 MiB (default `--gz-threshold`), the
`export` subcommand automatically gzip-compresses it and writes
`<name>.patch.gz` instead.  The TurboQuant patch always exceeds this
threshold.  To change the threshold:

```bash
bash scripts/pgvector_dev.sh export \
  --dir .pgvector_dev_v0.8.0_hnsw_candidate_pruning_turboquant \
  --gz-threshold 0 \
  pgvector_v0.8.0_hnsw_candidate_pruning_turboquant.patch
```

Repeat steps 2–3 during iteration.

### 4. Reset the working directory (start over)

```bash
bash scripts/pgvector_dev.sh reset \
  --dir .pgvector_dev_v0.8.0_hnsw_candidate_pruning_turboquant
```

---

## Docker-based build / test / run

All compilation and testing must be done via Docker (do **not** run
`make USE_PGXS=1` directly on the host).  The build context for every
`docker build` command is the **repository root** (the directory containing
the `.patch` or `.patch.gz` file).  Build arguments, run examples, and
environment variable references are documented in detail in the header comment
of each Dockerfile.  All Dockerfiles accept both plain `.patch` and
gzip-compressed `.patch.gz` files via the `PATCH_FILE` build argument.

- **`dockerfiles/build/Dockerfile`** — Applies the patch and compiles `vector.so`;
  collects the resulting extension files under `/artifacts` inside the image.
  No server is started.

- **`dockerfiles/test/Dockerfile`** — Applies the patch, compiles, installs, and
  runs both `make installcheck` (SQL regression) and `make prove_installcheck`
  (TAP tests).  Exit code 0 means all tests passed.

- **`dockerfiles/run/Dockerfile`** — Applies the patch, compiles, installs, and
  starts a PostgreSQL server.  On first container start the SIFT1M (or
  siftsmall) dataset is downloaded, loaded, and an HNSW index is built —
  intended for benchmarking.

---

## Benchmarking workflow

The benchmark workflow uses the `run` image with data loading skipped
(`SIFT_SUBSETS=""`), so that ann-benchmarks manages the dataset itself.

### 1. Start PostgreSQL (no data loading)

```bash
docker run --rm \
  -e POSTGRES_PASSWORD=postgres \
  -e SIFT_SUBSETS="" \
  --shm-size=8g \
  -p 5432:5432 \
  pgvector-run:pg17-turboquant
```

### 2. Recall accuracy — `scripts/run_ann_benchmark.py`

Runs the ann-benchmarks suite against the running PostgreSQL instance and
prints a summary table of Recall, QPS, and latency percentiles.  Requires
`scripts/.ann-benchmarks` (clone of `https://github.com/erikbern/ann-benchmarks`)
to exist; see the script's docstring for details.

```bash
python scripts/run_ann_benchmark.py \
  --host 127.0.0.1 --port 5432 \
  --user postgres --password postgres --dbname postgres \
  --dataset sift-128-euclidean \
  --count 10 --runs 5
```

To re-display results from a previous run without re-running the benchmark:

```bash
python scripts/run_ann_benchmark.py \
  --dataset sift-128-euclidean --count 10 --results-only
```

### 3. Block I/O footprint — `scripts/run_sift1m_footprint_benchmark.py`

Connects to a PostgreSQL instance and sweeps `hnsw.ef_search` over a range of
values.  For each value it runs a fixed number of random ANN queries, collects
`EXPLAIN (ANALYZE, BUFFERS)` block statistics, and exports a CSV and an SVG
line chart (mean ± stdev of block reads per ef_search value).

The target table is `items`, which is created by ann-benchmarks (`run_ann_benchmark.py`)
when it loads the dataset into PostgreSQL.  Run step 2 before this step.

```bash
python scripts/run_sift1m_footprint_benchmark.py \
  --dataset sift1m \
  --host 127.0.0.1 --port 5432 \
  --dbname postgres --user postgres --password postgres \
  --table items --embedding-col embedding \
  --ef 10,20,40,80,120,200,400,800 \
  --runs 100 \
  --series total_refs,shared_read,temp_read \
  --output_prefix sift1m_hnsw_l2 \
  --verbose
```

Output files are written to the current directory:
`sift1m_hnsw_l2_<timestamp>.csv` and `sift1m_hnsw_l2_<timestamp>.svg`.
See the script's docstring for the full option reference.