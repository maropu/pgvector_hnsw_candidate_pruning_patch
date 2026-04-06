# Design Documentation: pgvector HNSW Candidate Pruning Optimization using TurboQuant (MSE-Optimized)

This document defines the necessary and sufficient specifications for implementing only the pure `QUANT_mse` and `DEQUANT_mse` in C, omitting the inner product bias correction (QJL) to minimize computational resource usage to the absolute limit, and integrates this distance estimation and Candidate Pruning utilizing the TurboQuant (TQ) algorithm into `pgvector`'s HNSW index.

**Source Paper:**
Amir Zandieh, Majid Daliri, Vahab Mirrokni, Majid Hadian. "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate." arXiv:2504.19874v1 [cs.LG], 28 Apr 2025.

---

## 1. Design Rationale: Why Q_mse Only

This implementation adopts only TurboQuant's MSE-optimized quantizer ($Q_{\text{mse}}$, Algorithm 1 in the paper). The paper's inner-product-optimized quantizer ($Q_{\text{prod}}$, Algorithm 2) and the related PolarQuant approach were evaluated and deliberately excluded. This section documents the rationale.

### 1.1 Why Not Q_prod (QJL Residual Correction)

$Q_{\text{prod}}$ allocates one bit of the total budget to a QJL (Quantized Johnson-Lindenstrauss) transform on the MSE residual, producing an unbiased inner product estimator. With a target bit-width of $b$, the MSE stage uses only $b - 1$ bits, and the remaining 1 bit is spent on the QJL sign sketch.

**At b=2, variance increases outweigh bias removal.**
When $b = 2$ (our operating point), $Q_{\text{prod}}$ reduces the MSE stage to 1-bit quantization. The paper's Theorem 1 gives $D_{\text{mse}}(b=1) \approx 0.36$ versus $D_{\text{mse}}(b=2) \approx 0.117$—a 3× increase in reconstruction error. While the QJL stage removes the multiplicative bias, the inner product distortion bound from Theorem 2 is $D_{\text{prod}}(b=2) \approx 0.56/d$, versus $D_{\text{mse}}(b=2) \approx 0.117/d$ (plus a small bias). The net effect is higher variance in the distance estimate, which degrades pruning quality.

**Bias in Q_mse acts conservatively for HNSW pruning.**
The MSE quantizer's bias attenuates the estimated inner product ($\hat{\cos\theta} \approx \alpha \cos\theta$, $\alpha < 1$). In the L2 distance formula $\hat{d} = \|v\|^2 + \|\delta\|^2 - 2\|v\|\|\delta\|\hat{\cos\theta}$, this causes distance *overestimation*. For candidate pruning, overestimation is the safe direction: it may skip a candidate that was actually slightly better (reducing QPS), but it never promotes a poor candidate above a good one (preserving recall). An unbiased estimator with higher variance can cause both over- and under-estimation, the latter being harmful to recall.

**Implementation cost is prohibitive.**
$Q_{\text{prod}}$ requires:
- An additional random matrix $S \in \mathbb{R}^{d \times d}$ with i.i.d. $\mathcal{N}(0,1)$ entries (separate from the rotation matrix $\Pi$), adding ~$d^2 \times 4$ bytes of compile-time storage.
- Storing the residual norm $\|r\|_2$ per edge (+4 bytes/edge metadata).
- Computing $S \cdot v$ for each query direction at scan time ($O(d^2)$ per visited node, in addition to the existing rotation $\Pi \cdot v$).

These costs are disproportionate to the marginal accuracy gain at $b = 2$.

### 1.2 Why Not PolarQuant

PolarQuant (Han et al., arXiv:2502.02617, 2025) decomposes each vector into norm and direction, applies a random rotation to the direction, then quantizes the rotated coordinates. During dequantization, it re-normalizes the reconstructed direction to unit norm, which minimizes the MSE of the *direction* component.

**PolarQuant's re-normalization optimizes vector reconstruction, not scalar inner product estimation.**
Our use case estimates individual scalar inner products $\langle v, \delta \rangle$ from quantized codes, using the Lloyd-Max conditional expectation $E[Z \mid \text{bin } k]$ as the dequantized value for each coordinate. This is already the MMSE (minimum mean-squared error) estimator for each scalar, and is optimal for inner product estimation via linearity of expectation.

PolarQuant's re-normalization step (projecting back onto the unit sphere) is a nonlinear operation that improves the L2 reconstruction of the *full vector* but does not improve—and can degrade—the accuracy of coordinate-wise inner product accumulation. In our pipeline, we never reconstruct the full vector; we only compute $\sum_j w_j \cdot \hat{q}_j$ where $\hat{q}_j$ are the Lloyd-Max centroids. Applying re-normalization would break this linear accumulation structure.

**No additional benefit for the pruning pipeline.**
Since we estimate $\langle v, \delta \rangle$ as a sum of scalar products and never reconstruct $\delta$ explicitly, PolarQuant's direction-aware dequantization provides no advantage over coordinate-wise Lloyd-Max decoding.

---

## 2. TurboQuant Algorithm Specification (MSE-Optimized)

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

## 3. pgvector HNSW Implementation Design

### Architecture Overview
This implementation introduces distance estimation and Candidate Pruning utilizing the TurboQuant (TQ) algorithm into `pgvector`'s HNSW index. During high-dimensional vector neighborhood searches, instead of performing exact distance calculations (high-cost operations using SIMD, etc.) for all candidates, it rapidly estimates distances using 2-bit quantized metadata stored in the edges (adjacency lists), narrowing down the exhaustive search to only the most promising top candidates.

### Dimension Subsampling

The original TurboQuant paper quantizes all $d$ rotated coordinates. For the candidate pruning use case, the metadata budget per edge is fixed at `HNSW_NEIGHBOR_METADATA_MAX_BYTES` = 16 bytes (matching the SimHash and PQ patch methods). After subtracting `sizeof(float32)` = 4 bytes for the stored norm, 12 bytes remain for 2-bit codes, accommodating at most $m = 12 \times 8 / 2 = 48$ rotated coordinates.

When $d > 48$ (e.g., SIFT1M with $d = 128$), only the first $m = 48$ rows of the $N \times N$ rotation matrix are used. This is valid because the random rotation mixes all $d$ input dimensions into every rotated coordinate; the first $m$ coordinates act as an $m$-dimensional random projection of the $d$-dimensional input (analogous to the Johnson-Lindenstrauss property).

**Variance analysis.** The inner product estimation variance decomposes into subsampling and quantization terms:

$$\text{Var}(\widehat{IP}(m, b)) \approx \frac{(1 + \mathcal{C}(f_X, b))\|v\|^2\|\delta\|^2 + \langle v, \delta \rangle^2}{m}$$

where $\mathcal{C}(f_X, b)$ is the scalar MSE cost ($\approx 0.117$ for $b = 2$). The quantization contribution ($0.117$) is only ~12% of the subsampling contribution ($1.0$), so reducing $m$ from $d$ to 48 primarily increases the subsampling term. For SIFT1M ($d = 128$), the standard deviation increases by a factor of $\sqrt{128/48} \approx 1.63$.

**Practical benefits:**
- **Fixed metadata size**: 16 bytes per edge regardless of input dimension, matching the SimHash patch budget.
- **Reduced rotation cost**: $O(m \times d)$ instead of $O(d^2)$ at both build and scan time.
- **Acceptable accuracy**: For candidate ranking (not exact distance computation), the moderate variance increase is tolerable since the top-k selection only requires correct relative ordering among the best candidates.

**Macro definitions** (in `src/hnswtq.h`):
```c
#define HNSW_TQ_CODE_BYTES     (HNSW_NEIGHBOR_METADATA_MAX_BYTES - (int) sizeof(float))  /* 12 */
#define HNSW_TQ_MAX_ROT_DIMS   ((HNSW_TQ_CODE_BYTES * 8) / HNSW_TQ_BIT_WIDTH)           /* 48 */
```

All encode/decode paths use $m = \min(d,\, \texttt{HNSW\_TQ\_MAX\_ROT\_DIMS})$ as the effective number of rotated coordinates, and the decode divisor is $m / \sqrt{N}$ (not $d / \sqrt{N}$).

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
* **Implementation Note:** Theoretically, a matrix following $\mathcal{N}(0, 1/d)$ is required, but here we output a matrix based on $\mathcal{N}(0, 1)$. Because the rotation matrix is $N \times N$ (where $N$ = `HNSW_MAX_DIM` = 2000), each entry has magnitude $\sim 1/\sqrt{N}$. For a $d$-dimensional input ($d \leq N$), only the first $m = \min(d, \texttt{HNSW\_TQ\_MAX\_ROT\_DIMS})$ rows and $d$ columns are used (dimension subsampling), so each rotated coordinate has variance $\|\delta\|^2 / N$ (not $\|\delta\|^2 / d$). The encode normalizer uses $\sqrt{N}$ (not $\sqrt{d}$) to map to $\mathcal{N}(0, 1)$ before quantization. On the decode side, the $m$-row partial rotation captures only $m/N$ of the true inner product, so the decode divisor is $m / \sqrt{N}$.

### Data Structure and Storage Layout

**Metadata Page (`HnswMetaPageData`)**
During index construction, the variable-length metadata size is calculated from the target vector's dimension $d$ and persisted in the metadata page.
* **Added Field:** `uint16 neighborMetadataSize;`
* **Calculation Logic:** `sizeof(float4) + ceil((m * 2.0) / 8.0)` where $m = \min(d, \texttt{HNSW\_TQ\_MAX\_ROT\_DIMS})$.
    * The metadata size is capped at `HNSW_NEIGHBOR_METADATA_MAX_BYTES` (16 bytes) regardless of input dimension.
    * For low dimensions (e.g., under 8 dimensions, below `HNSW_TQ_L2_MIN_DIM`), the pruning accuracy is unstable. Thus, TQ compression is skipped, and the logic falls back to `sizeof(float4)` (storing only exact distance upper bounds, etc.).

**Neighbor Tuple (`HnswNeighborTupleData`)**
Data is packed in the following layout immediately after the TID in each adjacency list entry.

| Offset | Type | Size | Content |
| :--- | :--- | :--- | :--- |
| `0` | `float4` | 4 bytes | **Sum of Squares**: The $L2$ norm of the difference vector. |
| `4` | `uint8[]` | $\lceil 2m/8 \rceil$ bytes (max 12) | **TQ Codes**: 2-bit quantized code sequence for the first $m$ rotated coordinates. Stores 4 dimensions per byte. |

### Detailed Function Specifications and Logic by Module

**TQ Core Operations Module (`src/hnswtq.c`, `src/hnswtq.h`)**
Handles the pure mathematical transformations and packing for the TQ algorithm.

* **Constant Definitions:**
    * `Lloyd-Max Centroids`: `const float hnsw_tq_centroids[4] = {-1.5104f, -0.4528f, 0.4528f, 1.5104f};`
* **`void TurboQuantProject(const float *vec, int dim, float scale, uint8 *out_codes)`**
    * **Role:** Rotates and scales the input vector, then quantizes and packs it into 2-bit codes.
    * **Logic:**
        1. Computes $m = \min(d, \texttt{HNSW\_TQ\_MAX\_ROT\_DIMS})$.
        2. Multiplies the input vector `vec` by the first $m$ rows (and $d$ columns) of `hnsw_tq_rotation`.
        3. Multiplies each resulting component by the argument `scale` ($\sqrt{N}/\|\delta\|$ where $N$ = `HNSW_MAX_DIM`).
        4. Compares each component with `hnsw_tq_centroids` and retrieves the closest index (0 to 3).
        5. Packs the 4 indices into 1 byte (`val = (idx3 << 6) | (idx2 << 4) | (idx1 << 2) | idx0;`) and stores it in `out_codes`.
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
        3. Calls `TurboQuantProject` using $\sqrt{N}/\|\delta\|$ (where $N$ = `HNSW_MAX_DIM`) as the normalization scale, writing the TQ codes to the remaining bytes. Note: the rotation matrix is $N \times N$ orthogonal, so each entry has magnitude $\sim 1/\sqrt{N}$; using $\sqrt{N}$ (not $\sqrt{d}$) ensures the quantizer input follows $\mathcal{N}(0, 1)$.
* **`void HnswEstimateVectorL2Distances(Datum query, Datum *neighbors, int num_neighbors, uint8 **metadata, float *distances, int dim)`**
    * **Role:** Batch estimates squared L2 distances against multiple candidate points during a search.
    * **Logic:**
        1. **Query Preprocessing:** Calculates the squared L2 norm of `query` (`query_sum_of_squares`), applies the TQ rotation (first $m$ rows) to `query` just once, and saves it in a temporary array `query_rot[m]`.
        2. **Candidate Loop:** Loops through each candidate. Reads the stored squared norm of the candidate (`neighbor_sum_of_squares`) from the first 4 bytes of `metadata`.
        3. **Inner Product Estimation:** Calculates `inv_scale = m / sqrt(HNSW_MAX_DIM);`, and calls `HnswGetTQDistance` to get the estimated dot product (`estimated_dot_product`) between `query_rot` and the 2-bit codes. The divisor $m / \sqrt{N}$ compensates for two effects: (a) the encode scaling of $\sqrt{N}/\|\delta\|$ and (b) the $m$-row subblock of the $N \times N$ rotation capturing only $m/N$ of the true inner product.
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