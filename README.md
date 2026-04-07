![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
[![Build and test](https://github.com/maropu/pgvector_hnsw_candidate_pruning_patch/actions/workflows/BuildAndTests.yml/badge.svg)](https://github.com/maropu/pgvector_hnsw_candidate_pruning_patch/actions/workflows/BuildAndTests.yml)
[![Prebuilt binaries](https://github.com/maropu/pgvector_hnsw_candidate_pruning_patch/actions/workflows/ExtensionDistribution.yml/badge.svg)](https://github.com/maropu/pgvector_hnsw_candidate_pruning_patch/actions/workflows/ExtensionDistribution.yml)

## What this patch does and how to apply it?

This work adds a candidate pruning logic to [pgvector](https://github.com/pgvector/pgvector)'s HNSW [1] implementation whose design is based on PASE [2].
PASE is an index structure for approximate nearest neighbor search, implemented as an external extension to a general-purpose RDBMS (PostgreSQL),
that follows the graph-based HNSW search algorithm. Since data structures in an RDBMS are typically managed as fixed-size blocks
(e.g., 8 KiB in PostgreSQL), a key feature of PASE is that it organizes the graph’s vertices and edges to align naturally with these blocks.

The search algorithm proposed in the HNSW paper [1] proceeds greedily: for a vertex representing a vector, it computes distances
between all of its adjacent candidate vectices and the query, then iteratively moves to the neighbor that most reduces the distance to the query.
In pgvector, which follows the PASE’s design, these adjacent vertices are often located on different blocks, leading to frequent random block accesses during search.
This becomes a major issue in an RDBMS, where sophisticated concurrency control for transaction processing makes both I/O overhead and lock contention critical concerns.
To mitigate this, the work provides three alternative patches that embed per-neighbor metadata into each vertex and use it to estimate distances to the query
without reading the blocks containing those neighbors. Neighbors of the current vertex are first ranked by an estimated distance $\hat d(q,n)$,
and only the top-k are fetched to compute exact distances; this reduces random I/O and contention while preserving accuracy.
This strategy is well known in earlier work as two-level search with hybrid distance [4] or re-ranking [6,7,8,9].

The first patch adopts a SimHash-based estimator: each vertex tuple stores 16 bytes of per-neighbor metadata consisting of a 96-bit SimHash [3] of the edge vector $\Delta = (n - c)$ and
the edge length $\|\Delta\|$. At query time, it computes the SimHash of the query offset $v = (q - c)$, estimates the angle $\hat{\theta}$ between $v$ and $\Delta$
from their Hamming distance, and derives an estimated $L_2$ distance $\hat d(q,n)$ via the cosine theorem.
Here, $c \in \mathbb{R}^d$ is the current vertex vector, $n \in \mathbb{R}^d$ is a neighbor vector, and $q \in \mathbb{R}^d$ is the query vector.
This design is training-free, compact, and computationally light.

Alternatively, the second patch employs Product Quantization (PQ) [5] as the estimator. The neighbor vector $n$ is split into $M$ equal-length parts $n_1,\ldots,n_M$.
For each part $j$, the index stores a one-byte code that identifies the nearest centroid in a learned codebook $C_j$ with $k$ centroids.
At query time, the PQ code is simply decoded: for each part $j$ we read the corresponding centroid and reconstruct an approximate neighbor $\tilde{n}$ by concatenating these centroids.
The estimated distance is then $\hat{d}(q,n) = \| q - \tilde{n} \|$.
Neighbors are ranked by $\hat{d}(q,n)$ and, as in the first patch, only the top-k are fetched to compute exact distances.
Compared to SimHash, PQ offers stronger estimation at the cost of build-time training and additional storage for codebooks.

The third patch uses TurboQuant (TQ) [10], an MSE-optimized online vector quantizer. Each edge stores 16 bytes of per-neighbor metadata:
4 bytes for the edge length $\|\Delta\|$ and 12 bytes of 2-bit quantized codes for up to $m = \min(d, 48)$ rotated coordinates of the edge vector $\Delta = (n - c)$.
During index construction, $\Delta$ is multiplied by a pre-computed random orthogonal matrix $\Pi$ and each rotated coordinate is scalar-quantized to 2 bits using Lloyd-Max centroids.
At query time, the query offset $v = (q - c)$ is likewise rotated, and the inner product $\langle v, \Delta \rangle$ is estimated from the quantized codes to derive an estimated $L_2$ distance $\hat d(q,n)$.
When $d > 48$, dimension subsampling is applied: only the first 48 rows of $\Pi$ are used, acting as a random projection that mixes all $d$ input dimensions into 48 rotated coordinates.
This design is training-free, compact (fixed 16 bytes per edge regardless of dimension), and its MSE bias conservatively overestimates distances, preserving recall.

Apply the patches to pgvector and compile them as described below:

```shell
// Cehckout pgvector v0.8.0
$ git clone --depth 1 https://github.com/pgvector/pgvector.git
$ cd pgvector
$ git fetch --tags --depth 1 origin "v0.8.0"
$ git checkout "v0.8.0"
```

The SimHash-based patch:
```shell
// Compile and install pgvector w/the the SimHash-based patch
$ patch -p1 < pgvector_v0.8.0_hnsw_candidate_pruning_simhash.patch
$ make
$ make install
```

The PQ-based patch:
```shell
// Compile and install pgvector w/the PQ-based patch
$ patch -p1 < pgvector_v0.8.0_hnsw_candidate_pruning_pq.patch
$ make
$ make install
```
The TurboQuant-based patch:
```shell
// Compile and install pgvector w/the TurboQuant-based patch
$ gunzip -c pgvector_v0.8.0_hnsw_candidate_pruning_turboquant.patch.gz | patch -p1
$ make
$ make install
```
Note that **these patches are incompatible with the pgvector’s original index data format** because they adds 16 bytes per-neighbor metadata, and
they currently support only the L2 distance (vector_l2_ops) on single-precision floating-point vectors.

### Additional options

#### Index options

Specify HNSW additional one index parameter:

- `neighbor_metadata` - whether to store neighbor metadata to estimate distances (on by default)

```sql
CREATE INDEX ON items USING hnsw (embedding vector_l2_ops) WITH (m = 16, ef_construction = 64, neighbor_metadata = on);
```

#### Query options

Specify HNSW additional two query parameters:

- `hnsw.candidate_pruning` - enables candidate pruning for faster scans (on by default)
- `hnsw.distance_computation_topk ` - sets the number of neighbors to compute precise distances when using distance estimation (3 by default)

```sql
SET hnsw.distance_computation_topk = 3;
```

A higher value provides better recall at the cost of block accesses.

## Benchmark results

This experiment compares vanilla pgvector with the three candidate-pruning variants (SimHash-based, PQ-based, and TurboQuant-based) on [SIFT1M](http://corpus-texmex.irisa.fr/) 10-NN,
using the number of block accesses required to keep target recall levels as the metric. The evaluation uses HNSW parameters m=24 and ef_construction=200.
The figure below plots the recall vs. block-access tradeoff for each variant at k=3, 5, and 7.

Around recall=0.95, PQ achieves the largest block-access reductions of approximately 81% (k=3), 75% (k=5), and 69% (k=7) relative to vanilla pgvector.
SimHash reduces block accesses by approximately 52% (k=3), 69% (k=5), and 66% (k=7),
while TurboQuant yields reductions of approximately 14% (k=3), 43% (k=5), and 59% (k=7).
At the highest recall each variant reaches (1.0 for SimHash and PQ, and approximately 0.997–0.999 for TurboQuant),
PQ again leads with about 84% (k=3), 77% (k=5), and 71% (k=7) fewer block accesses,
followed by TurboQuant at about 79% across all k values,
and SimHash at about 72% (k=3), 68% (k=5), and 64% (k=7).
Overall, PQ delivers the strongest block reduction across all recall levels and k values.
SimHash and TurboQuant are both training-free; SimHash achieves higher recall at lower k, while TurboQuant provides more uniform block savings across k values in the high-recall regime.

<img src="resources/sift1m_recall_blocks_tradeoff.png" width="600">

A limitation of this design is **the increase in index size due to per‑neighbor metadata**.
On SIFT1M, the vanilla pgvector index occupies 781 MiB, whereas enabling the 16‑byte neighbor metadata inflates the index to 1313 MiB for all three patches,
corresponding to an increase of approximately 68%. Addressing this storage overhead remains an important direction for future work.

## TODO

 - Address the challenge of index size expansion due to the addition of neighbor metadata
   - For follow-up work to tackle this issue, see https://github.com/maropu/pgvector_hnsw_two_stage_pq_patch
 - Improve the patches to further reduce the number of block accesses
 - Add benchmark results showing the recall-TPS (transactions per second) tradeoff and include them in the section "Benchmark results"
 - Add benchmark results on higher-dimensional datasets, as TurboQuant's algorithm is designed for high-dimensional spaces and may show stronger gains beyond SIFT1M's 128 dimensions

## References

 - [1] Yu A. Malkov and D. A. Yashunin. 2020. Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs. IEEE Trans. Pattern Anal. Mach. Intell. 42, 4 (April 2020), 824–836. https://doi.org/10.1109/TPAMI.2018.2889473.
 - [2] Wen Yang, Tao Li, Gai Fang, and Hong Wei. 2020. PASE: PostgreSQL Ultra-High-Dimensional Approximate Nearest Neighbor Search Extension. In Proceedings of the 2020 ACM SIGMOD International Conference on Management of Data (SIGMOD '20). Association for Computing Machinery, New York, NY, USA, 2241–2253. https://doi.org/10.1145/3318464.3386131.
 - [3] Moses S. Charikar. 2002. Similarity estimation techniques from rounding algorithms. In Proceedings of the thiry-fourth annual ACM symposium on Theory of computing (STOC '02). Association for Computing Machinery, New York, NY, USA, 380–388. https://doi.org/10.1145/509907.509965.
 - [4] Yichuan Wang, Shu Liu, Zhifei Li, Yongji Wu, Ziming Mao, Yilong Zhao, Xiao Yan, Zhiying Xu, Yang Zhou, Ion Stoica, Sewon Min, Matei Zaharia, and Joseph E. Gonzalez. 2025. LEANN: A Low-Storage Vector Index. arXiv preprint arXiv:2506.08276.
 - [5] Herve Jégou, Matthijs Douze, and Cordelia Schmid. 2011. Product Quantization for Nearest Neighbor Search. IEEE Transactions on Pattern Analysis and Machine Intelligence 33, 1 (2011), 117–128.
 - [6] Matthijs Douze, Alexandre Sablayrolles, and Hervé Jégou. 2018. Link and Code: Fast Indexing with Graphs and Compact Regression Codes. In Proceedings of the 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2018). IEEE, 3646–3654. https://doi.org/10.1109/CVPR.2018.00384.
 - [7] Hervé Jégou, Romain Tavenard, Matthijs Douze, and Laurent Amsaleg. 2011. Searching in one billion vectors: Re-rank with source coding. In Proceedings of the 2011 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP 2011). IEEE, 861–864. https://doi.org/10.1109/ICASSP.2011.5946540.
 - [8] Herve Jégou, Matthijs Douze, and Cordelia Schmid. 2011. Product Quantization for Nearest Neighbor Search. IEEE Transactions on Pattern Analysis and Machine Intelligence 33, 1 (2011), 117–128. https://doi.org/10.1109/TPAMI.2010.57.
 - [9] Bing Tian, Haikun Liu, Yuhang Tang, Shihai Xiao, Zhuohui Duan, Xiaofei Liao, Hai Jin, Xuecang Zhang, Junhua Zhu, and Yu Zhang. 2025. Towards high-throughput and low-latency billion-scale vector search via CPU/GPU collaborative filtering and re-ranking. In Proceedings of the 23rd USENIX Conference on File and Storage Technologies (FAST '25). USENIX Association, USA, Article 11, 171–185.
 - [10] Amir Zandieh, Majid Daliri, Vahab Mirrokni, and Majid Hadian. 2025. TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate. arXiv preprint arXiv:2504.19874.
