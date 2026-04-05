#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_hnswtq.py — generator for pgvector/src/hnswtq.c

This script materializes the compile-time constants required by the
TurboQuant HNSW candidate-pruning patch:

  1. A random rotation matrix Q (max_dim x max_dim), generated via
     QR decomposition of a Gaussian matrix (numpy seed).  The full square
     matrix is stored so that a single precomputed matrix can be reused
     for input vectors of any dimension d <= max_dim: a d-dimensional vector
     is conceptually zero-filled to max_dim before rotation, and only the
     first d rows of Q are applied (the zero-padded coordinates contribute
     nothing to the dot product, preserving the orthonormal row property).

  2. (Informational) The Lloyd-Max codebook for N(0,1) with 2^bit_width
     levels, solved via the iterative Lloyd-Max algorithm.  The codebook
     values are embedded directly as named macros in hnswtq.h; this script
     prints them for verification but does not write them (they are stable
     across seeds and dimensions).

What it generates:
  - src/hnswtq.c  containing:
        const float hnsw_tq_rotation[HNSW_MAX_DIM][HNSW_MAX_DIM]

Design notes:
  - max_dim  defaults to 2000 (HNSW_MAX_DIM in hnsw.h).
  - bit_width defaults to 2 (4 Lloyd-Max centroids).
  - The RNG seed is fixed so the generated file is reproducible.

Usage:
  python generate_hnswtq.py \\
      --max-dim   2000 \\
      --bit-width 2 \\
      --seed      42 \\
      --out       pgvector/src/hnswtq.c
"""

import argparse
import datetime
import platform
import sys

import numpy as np
from scipy.stats import norm

# Silence legacy-API deprecation warnings from numpy (we use np.random.seed
# intentionally to stay bit-for-bit compatible with the shipped hnswtq.c).
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="numpy")


# ---------------------------------------------------------------------------
# Lloyd-Max solver
# ---------------------------------------------------------------------------

def lloyd_max_gaussian(num_levels: int, sigma: float = 1.0, max_iter: int = 200):
    """
    Compute optimal Lloyd-Max quantizer centroids and decision boundaries
    for N(0, sigma^2) with *num_levels* reconstruction levels.

    Returns
    -------
    centroids   : ndarray of shape (num_levels,)
    boundaries  : ndarray of shape (num_levels + 1,)  (-inf ... +inf)
    """
    k = num_levels
    centroids = np.array([sigma * norm.ppf((2 * i + 1) / (2 * k)) for i in range(k)])

    for _ in range(max_iter):
        boundaries = np.empty(k + 1)
        boundaries[0] = -np.inf
        boundaries[k] = np.inf
        for i in range(1, k):
            boundaries[i] = (centroids[i - 1] + centroids[i]) / 2.0

        new_centroids = np.empty(k)
        for i in range(k):
            lo, hi = boundaries[i], boundaries[i + 1]
            lo_c = max(lo, -6 * sigma)
            hi_c = min(hi, 6 * sigma)
            num = norm.expect(lambda x: x, loc=0, scale=sigma, lb=lo_c, ub=hi_c)
            den = norm.cdf(hi, scale=sigma) - norm.cdf(lo, scale=sigma)
            new_centroids[i] = num / den if den > 1e-15 else (lo_c + hi_c) / 2.0

        if np.allclose(centroids, new_centroids, atol=1e-12):
            break
        centroids = new_centroids

    # Final boundaries
    boundaries = np.empty(k + 1)
    boundaries[0] = -np.inf
    boundaries[k] = np.inf
    for i in range(1, k):
        boundaries[i] = (centroids[i - 1] + centroids[i]) / 2.0

    return centroids, boundaries


# ---------------------------------------------------------------------------
# Rotation matrix
# ---------------------------------------------------------------------------

def gen_rotation_matrix(max_dim: int, seed: int) -> np.ndarray:
    """
    Generate a random orthogonal matrix via QR decomposition of a Gaussian
    matrix.  Returns the full (max_dim x max_dim) orthogonal matrix Q.

    The square matrix enables reuse across input dimensions: for a
    d-dimensional input vector (d <= max_dim), zero-fill to max_dim and
    apply the first d rows of Q.  Only the first d columns of each row
    contribute (zero-padded entries vanish), preserving orthonormality.

    Uses the legacy numpy.random API (np.random.seed + np.random.randn) so
    that the output is bit-for-bit identical to the shipped hnswtq.c.
    """
    np.random.seed(seed)
    G = np.random.randn(max_dim, max_dim).astype(np.float32)
    Q, R_mat = np.linalg.qr(G)
    # Make the decomposition unique by flipping signs so diag(R) > 0
    diag_sign = np.sign(np.diag(R_mat)).astype(np.float32)
    diag_sign[diag_sign == 0] = 1.0
    Q = (Q * diag_sign[np.newaxis, :]).astype(np.float32)

    # Sanity check: rows should be mutually orthonormal
    err = float(np.max(np.abs(Q @ Q.T - np.eye(max_dim))))
    if err > 1e-4:
        print(f"[WARNING] Orthogonality error {err:.2e} exceeds 1e-4 — "
              "check numpy/LAPACK version.", file=sys.stderr)

    return Q  # shape: (max_dim, max_dim)


# ---------------------------------------------------------------------------
# C file emitter
# ---------------------------------------------------------------------------

def c_float_literal(x: float) -> str:
    return f"{x:.8e}f"


def emit_c_file(out_path: str, Q: np.ndarray, max_dim: int,
                bit_width: int, seed: int):
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    py_ver = platform.python_version()
    np_ver = np.__version__

    num_levels = 2 ** bit_width
    centroids, boundaries = lloyd_max_gaussian(num_levels, sigma=1.0)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("/*\n")
        f.write(" * hnswtq.c — Auto-generated TurboQuant constants for pgvector (HNSW)\n")
        f.write(" *\n")
        f.write(f" * Generated : {ts}\n")
        f.write(f" * Arguments : max_dim={max_dim}, "
                f"bit_width={bit_width}, seed={seed}\n")
        f.write(f" * Python    : {py_ver}\n")
        f.write(f" * NumPy     : {np_ver}\n")
        f.write(" *\n")
        f.write(" * This file is generated by generate_hnswtq.py. Do not edit manually.\n")
        f.write(" */\n\n")

        f.write("#include \"hnswtq.h\"\n\n")

        # Print codebook as a comment for documentation
        f.write("/*\n")
        f.write(f" * Lloyd-Max codebook for N(0,1) with {num_levels} levels "
                f"(bit_width={bit_width}):\n")
        f.write(f" *   centroids  = [{', '.join(f'{c:.8f}' for c in centroids)}]\n")
        finite_bounds = boundaries[1:-1]
        f.write(f" *   boundaries = [{', '.join(f'{b:.8f}' for b in finite_bounds)}]\n")
        f.write(" *\n")
        f.write(" * These values are embedded as macros in hnswtq.h (HNSW_TQ_CENTROID_*,\n")
        f.write(" * HNSW_TQ_BOUNDARY_*) and are not repeated here.\n")
        f.write(" */\n\n")

        # Rotation matrix
        f.write("/*\n")
        f.write(f" * Random rotation matrix: {max_dim}x{max_dim} orthogonal matrix Q\n")
        f.write(f" * obtained by QR decomposition of a Gaussian matrix (numpy seed={seed}).\n")
        f.write(f" * Shape: [{max_dim}][{max_dim}].\n")
        f.write(f" *\n")
        f.write(f" * For a d-dimensional input vector (d <= {max_dim}): apply the first d rows\n")
        f.write(f" * of Q to the vector (conceptually zero-filled to {max_dim} dimensions).\n")
        f.write(f" * Only the first d columns of each row contribute, preserving orthonormality.\n")
        err = float(np.max(np.abs(Q @ Q.T - np.eye(max_dim))))
        f.write(f" *\n")
        f.write(f" * Orthogonality check (Q @ Q^T ≈ I_{max_dim}): max |QQ^T - I| = {err:.2e}\n")
        f.write(" */\n")
        f.write(f"const float hnsw_tq_rotation"
                f"[HNSW_MAX_DIM][HNSW_MAX_DIM] = {{\n")

        for i in range(max_dim):
            row = Q[i]
            f.write(f"\t/* row {i} */\n\t{{")
            vals = [c_float_literal(v) for v in row]
            # 10 values per line, indented
            for j, v in enumerate(vals):
                if j % 10 == 0 and j > 0:
                    f.write(",\n\t ")
                elif j > 0:
                    f.write(", ")
                f.write(v)
            f.write("}")
            if i < max_dim - 1:
                f.write(",")
            f.write("\n")

        f.write("};\n")

    print(f"[generate_hnswtq.py] wrote: {out_path} "
          f"(max_dim={max_dim}, seed={seed})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Generate pgvector/src/hnswtq.c with TurboQuant rotation matrix."
    )
    ap.add_argument(
        "--max-dim", type=int, default=2000,
        help="rotation matrix dimension D; generates a DxD orthogonal matrix "
             "reusable for any input dimension d <= D (default: 2000 = HNSW_MAX_DIM)"
    )
    ap.add_argument(
        "--bit-width", type=int, default=2,
        help="Lloyd-Max quantization bit-width (default: 2 => 4 levels)"
    )
    ap.add_argument(
        "--seed", type=int, default=42,
        help="numpy RNG seed for reproducibility (default: 42)"
    )
    ap.add_argument(
        "--out", type=str, required=True,
        help="output C file path (e.g., ../pgvector/src/hnswtq.c)"
    )
    args = ap.parse_args()

    if args.max_dim <= 0:
        print("--max-dim must be positive", file=sys.stderr)
        sys.exit(1)
    if args.bit_width not in (1, 2, 3, 4):
        print("--bit-width must be one of 1, 2, 3, 4", file=sys.stderr)
        sys.exit(1)

    Q = gen_rotation_matrix(args.max_dim, args.seed)
    emit_c_file(args.out, Q, args.max_dim, args.bit_width, args.seed)


if __name__ == "__main__":
    main()
