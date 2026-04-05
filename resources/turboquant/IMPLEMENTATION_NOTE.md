# Implementation Specification: Lightweight Encode/Decode Processing for TurboQuant (MSE-Optimized)

**Purpose:** To define the necessary and sufficient specifications for implementing only the pure `QUANT_mse` and `DEQUANT_mse` in C, omitting the inner product bias correction (QJL) to minimize computational resource usage to the absolute limit.

**Source Paper:**
Amir Zandieh, Majid Daliri, Vahab Mirrokni, Majid Hadian. "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate." arXiv:2504.19874v1 [cs.LG], 28 Apr 2025.

---

### 1. Global Parameters (Pre-computed and Retained Data)

For implementation, the following two sets of data must be prepared as constants at compile-time or runtime.

* **Random Rotation Matrix (`Pi`)**
    * **Size:** `d x d` matrix (`d` is the dimensionality of the vectors).
    * **Generation Method:** Generate a random matrix where all entries follow independent and identically distributed standard normal distributions, and perform QR decomposition on it to create an orthogonal matrix $\Pi \in \mathbb{R}^{d \times d}$.
* **Codebook (`c`)**
    * **Size:** 1D array of length `2^b` (`b` is the bit-width).
    * **Content:** A set of optimal representative values (centroids) distributed within the range of `[-1, 1]`.
    * **Specific Values:** Since each coordinate is approximated by a normal distribution $\mathcal{N}(0, 1/d)$ in high dimensions, the following representative values are provided for moderately high dimensions:
        * For `b=1` (2 levels): $\{\pm\frac{\sqrt{2/\pi}}{\sqrt{d}}\}$
        * For `b=2` (4 levels): $\{\pm\frac{0.453}{\sqrt{d}}, \pm\frac{1.51}{\sqrt{d}}\}$

---

### 2. Encoding Process (`QUANT_mse`)

The process of compressing a vector into a low-bit index array.

* **Input**
    * Target vector `x` (`d`-dimensional).
    * **\*Important:** The algorithm in the paper assumes the input vector has a unit norm ($||x||_2 = 1$). For datasets that do not satisfy this, you must calculate and store the original L2 norm (in floating-point), normalize the vector to a length of **1**, and then feed it into the following process.
* **Processing Steps**
    1.  **Apply Rotation:** Multiply the input vector `x` by the rotation matrix `Pi` to compute vector `y`.
        * Formula: $y \leftarrow \Pi \cdot x$
    2.  **Scalar Quantization:** For each dimensional element `y_j` (`j = 0` to `d-1`) of `y`, find the index of the centroid in the codebook `c` that is closest in value.
        * Formula: $idx_j \leftarrow \arg\min_{k \in [2^b]} |y_j - c_k|$
* **Output**
    * The computed array of `d` indices `idx`.
    * (During C implementation, these indices should be packed using bitwise operations to save memory.)

---

### 3. Decoding Process (`DEQUANT_mse`)

The process of reconstructing the original vector (approximation) from the index array.

* **Input**
    * The index array `idx` output during encoding.
* **Processing Steps**
    1.  **Dequantization:** Look up the codebook value corresponding to each index `idx_j` to reconstruct the rotated vector `y_tilde`.
        * Formula: $\tilde{y}_j \leftarrow c_{idx_j}$
    2.  **Apply Inverse Rotation:** Multiply the reconstructed `y_tilde` by the transpose of the rotation matrix $\Pi^\top$ to return to the original coordinate system.
        * Formula: $\tilde{x} \leftarrow \Pi^\top \cdot \tilde{y}$
* **Output**
    * The reconstructed `d`-dimensional vector `x_tilde` (or $\tilde{x}$).
    * **\*Important:** If the L2 norm was separated and normalized during encoding, finally multiply all elements of `x_tilde` by the retained L2 norm here to restore the original scale.
