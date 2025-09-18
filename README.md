![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
[![Build and test](https://github.com/maropu/pgvector_hnsw_candidate_pruning_patch/actions/workflows/BuildAndTests.yml/badge.svg)](https://github.com/maropu/pgvector_hnsw_candidate_pruning_patch/actions/workflows/BuildAndTests.yml)
[![Prebuilt binaries](https://github.com/maropu/pgvector_hnsw_candidate_pruning_patch/actions/workflows/ExtensionDistribution.yml/badge.svg)](https://github.com/maropu/pgvector_hnsw_candidate_pruning_patch/actions/workflows/ExtensionDistribution.yml)

## What this patch does and how to apply it?

This patch adds a candidate pruning logic to [pgvector](https://github.com/pgvector/pgvector)'s HNSW [1] implementation whose design is based on PASE [2].
PASE is an index structure for approximate nearest neighbor search, implemented as an external extension to a general-purpose RDBMS (PostgreSQL),
that follows the graph-based HNSW search algorithm. Since data structures in an RDBMS are typically managed as fixed-size disk blocks
(e.g., 8 KiB in PostgreSQL), a key feature of PASE is that it organizes the graph’s vertices and edges to align naturally with these disk blocks.

The search algorithm proposed in the HNSW paper [1] proceeds greedily: for a vertex representing a vector, it computes distances
between all of its adjacent candidate vectices and the query, then iteratively moves to the neighbor that most reduces the distance to the query.
In pgvector, which follows the PASE’s design, these adjacent vertices are often located on different disk blocks, leading to frequent random block accesses during search.
This becomes a major issue in an RDBMS, where sophisticated concurrency control for transaction processing makes both I/O overhead and lock contention critical concerns.
To mitigate this, the patch embeds metadata into each vertex that allows estimating distances between its neighbors and the query
without reading the disk blocks containing those neighbors.
Specifically, Each vertex tuple in a disk block stores 16 bytes of per-neighbor metadata:
(1) a 64-bit SimHash [3] of the $d$-dimensional edge vector $\Delta = (n - c)$, and (2) the edge length $\lVert \Delta \rVert$ (a double-precision floating-point value).
Here, $c \in \mathbb{R}^d$ is the current vertex vector, $n \in \mathbb{R}^d$ is a neighbor vector, and $q \in \mathbb{R}^d$ is the query vector; let $v = (q - c)$.

At search time, it computes the SimHash of the query vector $v$, estimates the angle $\hat{\theta}$ between $v$ and the edge vector $\Delta$ from their Hamming distance,
and sorts candidate vertices by estimated distance $\widehat{d}(q,n)$ computed from the cosine theorem. 
By using this estimated distance $\widehat{d}(q,n)$, the search can prioritize neighbors without fetching the disk blocks that contain their vectors.
Specifically, candidate neighbors are first sorted in ascending order of $\widehat{d}(q,n)$,
and only the top-k neighbors (k=3 by default) are accessed from disk to compute their exact distances,
while the remaining neighbors rely solely on the estimated distances. This strategy results in reducing random I/O and lock contention in PostgreSQL.

Apply the patch to pgvector and compile it as described below:

```shell
// Cehckout pgvector v0.8.0
$ git clone --depth 1 https://github.com/pgvector/pgvector.git
$ cd pgvector
$ git fetch --tags --depth 1 origin "v0.8.0"
$ git checkout "v0.8.0"

// Compile and install pgvector w/the patch
$ patch -p1 < pgvector_v0.8.0_hnsw_candidate_pruning.patch
$ make
$ make install
```

Note that **this patch is incompatible with the pgvector’s original index data format** because it adds 16 bytes per-neighbor metadata, and
it currently supports only the L2 distance (vector_l2_ops) on single-precision floating point vectors.

### Additional options

#### Index options

Specify HNSW additional one index parameters

- `neighbor_metadata` - whether to store neighbor metadata to estimate distances (on by default)

```sql
CREATE INDEX ON items USING hnsw (embedding vector_l2_ops) WITH (m = 16, ef_construction = 64, neighbor_metadata = on);
```

#### Query options

Specify HNSW additional two query parameters

- `hnsw.candidate_pruning` - enables candidate pruning for faster scans (on by default)
- `hnsw.distance_computation_topk ` - sets the number of neighbors to compute precise distances when using distance estimation (3 by default)

```sql
SET hnsw.distance_computation_topk = 3;
```

A higher value provides better recall at the cost of block accesses.

## Benchmark results

An evaluation compared the patch with the vanilla pgvector by measuring the number of block accesses required to achieve different levels of recall.
At Recall≈0.95, the patch achieved a reduction in block accesses of approximately 59% when k=7. The improvement became even more pronounced as recall approached 1.0,
where block accesses were reduced by about 72% for k=3, 68% for k=5, and 64% for k=7. These results indicate that the patch provides a consistent reduction
in block read while maintaining accuracy, with the benefits observed in the high-recall regime.

<img src="resources/sift1m_recall_blocks_tradeoff.png" width="600">

## Detailed design of this patch

TBW

## TODO

 - Improve the patch to further reduce the number of blocks read
 - Add benchmark results showing the recall-TPS (transactions per second) tradeoff and include them in the section **"Benchmark results"**
 - Document the implementation details and design considerations of this patch in the section **"Detailed design of this patch"**

## References

 - [1] Yu A. Malkov and D. A. Yashunin, "Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs", IEEE Trans. Pattern Anal. Mach. Intell. 42, 4, 2020.
 - [2] Wen Yang, Tao Li, Gai Fang, and Hong Wei, "PASE: PostgreSQL Ultra-High-Dimensional Approximate Nearest Neighbor Search Extension", In Proceedings of the 2020 ACM SIGMOD International Conference on Management of Data (SIGMOD '20), 2241–2253, 2020.
 - [3] Moses S. Charikar, "Similarity estimation techniques from rounding algorithms", In Proceedings of the thiry-fourth annual ACM symposium on Theory of computing (STOC), 2002.

