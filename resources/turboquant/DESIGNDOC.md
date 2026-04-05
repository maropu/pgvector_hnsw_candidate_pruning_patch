# Design Documentation: pgvector HNSW Candidate Pruning Optimization using TurboQuant (MSE-Optimized)

This document defines the necessary and sufficient specifications for implementing only the pure `QUANT_mse` and `DEQUANT_mse` in C, omitting the inner product bias correction (QJL) to minimize computational resource usage to the absolute limit, and integrates this distance estimation and Candidate Pruning utilizing the TurboQuant (TQ) algorithm into `pgvector`'s HNSW index.

**Source Paper:**
Amir Zandieh, Majid Daliri, Vahab Mirrokni, Majid Hadian. "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate." arXiv:2504.19874v1 [cs.LG], 28 Apr 2025.

---

## 1. TurboQuant Algorithm Specification (MSE-Optimized)

### Global Parameters (Pre-computed and Retained Data)
For implementation, the following two sets of data must be prepared as constants at compile-time or runtime.

* **Random Rotation Matrix (`Pi`)**
    * **Size:** $d \times d$ matrix ($d$ is the dimensionality of the vectors).
    * **Generation Method:** Generate a random matrix where all entries follow independent and identically distributed standard normal distributions, and perform QR decomposition on it to create an orthogonal matrix $\Pi \in \mathbb{R}^{d \times d}$.
* **Codebook (`c`)**
    * **Size:** 1D array of length $2^b$ ($b$ is the bit-width).
    * **Content:** A set of optimal representative values (centroids) distributed within the range of `[-1, 1]`.
    * **Specific Values:** Since each coordinate is approximated by a normal distribution $\mathcal{N}(0, 1/d)$ in high dimensions, the following representative values are provided for moderately high dimensions:
        * For $b=1$ (2 levels): $\{\pm\frac{\sqrt{2/\pi}}{\sqrt{d}}\}$
        * For $b=2$ (4 levels): $\{\pm\frac{0.453}{\sqrt{d}}, \pm\frac{1.51}{\sqrt{d}}\}$

### Encoding Process (`QUANT_mse`)
The process of compressing a vector into a low-bit index array.

* **Input**
    * Target vector `x` ($d$-dimensional).
    * **\*Important:** The algorithm in the paper assumes the input vector has a unit norm ($||x||_2 = 1$). For datasets that do not satisfy this, you must calculate and store the original L2 norm (in floating-point), normalize the vector to a length of **1**, and then feed it into the following process.
* **Processing Steps**
    1.  **Apply Rotation:** Multiply the input vector `x` by the rotation matrix `Pi` to compute vector `y`. Formula: $y \leftarrow \Pi \cdot x$.
    2.  **Scalar Quantization:** For each dimensional element `y_j` (`j = 0` to `d-1`) of `y`, find the index of the centroid in the codebook `c` that is closest in value. Formula: $idx_j \leftarrow \arg\min_{k \in [2^b]} |y_j - c_k|$.
* **Output**
    * The computed array of $d$ indices `idx`.
    * (During C implementation, these indices should be packed using bitwise operations to save memory.)

### Decoding Process (`DEQUANT_mse`)
The process of reconstructing the original vector (approximation) from the index array.

* **Input**
    * The index array `idx` output during encoding.
* **Processing Steps**
    1.  **Dequantization:** Look up the codebook value corresponding to each index `idx_j` to reconstruct the rotated vector `y_tilde`. Formula: $\tilde{y}_j \leftarrow c_{idx_j}$.
    2.  **Apply Inverse Rotation:** Multiply the reconstructed `y_tilde` by the transpose of the rotation matrix $\Pi^\top$ to return to the original coordinate system. Formula: $\tilde{x} \leftarrow \Pi^\top \cdot \tilde{y}$.
* **Output**
    * The reconstructed $d$-dimensional vector `x_tilde` (or $\tilde{x}$).
    * **\*Important:** If the L2 norm was separated and normalized during encoding, finally multiply all elements of `x_tilde` by the retained L2 norm here to restore the original scale.

---

## 2. pgvector HNSW Implementation Design

### Architecture Overview
This implementation introduces distance estimation and Candidate Pruning utilizing the TurboQuant (TQ) algorithm into `pgvector`'s HNSW index. During high-dimensional vector neighborhood searches, instead of performing exact distance calculations (high-cost operations using SIMD, etc.) for all candidates, it rapidly estimates distances using 2-bit quantized metadata stored in the edges (adjacency lists), narrowing down the exhaustive search to only the most promising top candidates.

### Generation of Compile-time Constants (Python Script Specification)
To reduce runtime computational costs, the random rotation matrix $\Pi$ is embedded as a compile-time constant in `src/hnswtq.c`. The script specifications for regeneration are as follows:

* **Objective:** Generate a $2000 \times 2000$ orthogonal matrix supporting up to 2000 dimensions.
* **Logic (Python Pseudo-code):**
    ```python
    import numpy as np

    np.random.seed(42) # Ensure reproducibility
    max_dim = 2000

    # 1. Generate a random matrix following N(0, 1)
    A = np.random.randn(max_dim, max_dim)

    # 2. Obtain orthogonal matrix Q via QR decomposition
    Q, R = np.linalg.qr(A)

    # 3. Uniquely determine Q using the signs of R's diagonal elements (Standardization)
    d = np.diagonal(R)
    ph = d / np.abs(d)
    Q = np.multiply(Q, ph)

    # 4. Output as a C language const float array
    # `const float hnsw_tq_rotation[2000][2000] = { ... };`
    ```
* **Implementation Note:** Theoretically, a matrix following $\mathcal{N}(0, 1/d)$ is required, but here we output a matrix based on $\mathcal{N}(0, 1)$. The scale correction of $1/\sqrt{d}$, which depends on the dimension $d$, is applied dynamically during execution in the C code.

### Data Structure and Storage Layout

**Metadata Page (`HnswMetaPageData`)**
During index construction, the variable-length metadata size is calculated from the target vector's dimension $d$ and persisted in the metadata page.
* **Added Field:** `uint16 neighborMetadataSize;`
* **Calculation Logic:** `sizeof(float4) + ceil((d * 2.0) / 8.0)`
    * For low dimensions (e.g., under 32 dimensions, below `HNSW_TQ_L2_MIN_ROT`), the pruning accuracy is unstable. Thus, TQ compression is skipped, and the logic falls back to `sizeof(float4)` (storing only exact distance upper bounds, etc.).

**Neighbor Tuple (`HnswNeighborTupleData`)**
Data is packed in the following layout immediately after the TID in each adjacency list entry.

| Offset | Type | Size | Content |
| :--- | :--- | :--- | :--- |
| `0` | `float4` | 4 bytes | **Sum of Squares**: The $L2$ norm of the difference vector. |
| `4` | `uint8[]` | Variable | **TQ Codes**: 2-bit quantized code sequence. Stores 4 dimensions per byte. |

### Detailed Function Specifications and Logic by Module

**TQ Core Operations Module (`src/hnswtq.c`, `src/hnswtq.h`)**
Handles the pure mathematical transformations and packing for the TQ algorithm.

* **Constant Definitions:**
    * `Lloyd-Max Centroids`: `const float hnsw_tq_centroids[4] = {-1.5104f, -0.4528f, 0.4528f, 1.5104f};`
* **`void TurboQuantProject(const float *vec, int dim, float scale, uint8 *out_codes)`**
    * **Role:** Rotates and scales the input vector, then quantizes and packs it into 2-bit codes.
    * **Logic:**
        1. Multiplies the input vector `vec` by the submatrix ($dim \times dim$) of `hnsw_tq_rotation`.
        2. Multiplies each resulting component by the argument `scale` (usually $1/\sqrt{dim}$ or a normalization coefficient).
        3. Compares each component with `hnsw_tq_centroids` and retrieves the closest index (0 to 3).
        4. Packs the 4 indices into 1 byte (`val = (idx3 << 6) | (idx2 << 4) | (idx1 << 2) | idx0;`) and stores it in `out_codes`.
* **`float HnswGetTQDistance(const uint8 *codes, const float *query_rot, int dim, float scale)`**
    * **Role:** Estimates the inner product/distance from the packed codes and the pre-rotated query.
    * **Logic:**
        1. Unpacks `codes` and restores the values of `hnsw_tq_centroids` from the indices.
        2. Calculates the dot product of the restored values and the corresponding dimensions of `query_rot`.
        3. Multiplies the result by `scale` to return the final estimated inner product (dot product).

**pgvector Metadata Integration Module (`src/hnswmeta.c`, `src/hnswmeta.h`)**
Bridges pgvector data types (`vector`, `halfvec`, etc.) with TQ operations.

* **`void HnswSetVectorL2NeighborMetadata(Datum vec, Datum neighbor, uint8 *metadata, int dim)`**
    * **Role:** Generates metadata for each edge during index insertion.
    * **Logic:**
        1. Calculates the difference vector $\delta$ between `vec` and `neighbor`.
        2. Calculates the $L2$ norm (Sum of Squares) of $\delta$ and writes it to the first 4 bytes of `metadata`.
        3. Calls `TurboQuantProject` using $1/\sqrt{dim}$ as the scale, writing the TQ codes to the remaining bytes.
* **`void HnswEstimateVectorL2Distances(Datum query, Datum *neighbors, int num_neighbors, uint8 **metadata, float *distances, int dim)`**
    * **Role:** Batch estimates squared L2 distances against multiple candidate points during a search.
    * **Logic:**
        1. **Query Preprocessing:** Calculates the squared L2 norm of `query` (`query_sum_of_squares`), applies the TQ rotation to `query` just once, and saves it in a temporary array `query_rot`.
        2. **Candidate Loop:** Loops through each candidate. Reads the stored squared norm of the candidate (`neighbor_sum_of_squares`) from the first 4 bytes of `metadata`.
        3. **Inner Product Estimation:** Calculates `scale = sqrt(neighbor_sum_of_squares) / sqrt(dim);`, and calls `HnswGetTQDistance` to get the estimated dot product (`estimated_dot_product`) between `query_rot` and the 2-bit codes.
        4. **L2 Distance Calculation:** Utilizes the vector property $L2^2(q, v) = \|q\|^2 + \|v\|^2 - 2\langle q, v \rangle$ to calculate the squared L2 distance using the following formula:
           `Estimated Distance = query_sum_of_squares + neighbor_sum_of_squares - (2.0 * estimated_dot_product)`
        5. Stores the calculated estimated distances in the `distances` array.

**Scan and Pruning Control Module (`src/hnswscan.c`)**
Integrates the pruning logic into the actual HNSW graph search (graph traversal).

* **Added GUC Parameters (defined in `src/hnsw.c`):**
    * `hnsw.candidate_pruning` (bool): Enable/disable pruning (default: `on`).
    * `hnsw.distance_computation_topk` (int): Number of top candidates to perform exact calculations on after sorting by estimated distance (e.g., 50).
* **Changes in `HnswSearchLayer`:**
    * When retrieving the candidate list within the search loop, it branches if `hnsw.candidate_pruning == true` and `neighborMetadataSize > 0`.
    * Calls `estimateDistances` via function pointer to rapidly calculate estimated distances for all elements in the candidate list.
    * Sorts the candidate list based on the estimated distances (usually quicksort, etc.).
    * Executes the traditional exact `HnswGetDistance` (heavy operations like SIMD) *only* on the sorted top `distance_computation_topk` candidates, adding the evaluation results to the search queue.