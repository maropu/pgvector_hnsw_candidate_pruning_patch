#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize the SIFT1M dataset using PCA and export a rotating 3D scatter plot as a GIF.

Features:
- Loads SIFT1M data stored in FAISS .fvecs format: [int32 dim][float32 * dim] repeated
- Optionally applies:
    * PQ  (Product Quantization) [1]
    * OPQ (Optimized Product Quantization, non-parametric alternating optimization) [2]
  to compress & reconstruct the vectors before visualization
- Performs PCA to reduce dimensionality to 3D
- Plots the PCA-transformed data as a 3D scatter plot
- Exports it as an animated GIF file

Note:
  This script is inspired by the observation "the SIFT1M set has two distinct clusters
  (this can be visualized by the first two principal components), ..."
  in the OPQ paper.

[1] Herve Jégou et al. 2011. Product Quantization for Nearest Neighbor Search.
    IEEE Transactions on Pattern Analysis and Machine Intelligence 33, 1 (2011), 117–128.
[2] Tiezheng Ge, et al., "Optimized Product Quantization for Approximate Nearest Neighbor Search."
    In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2013, pp. 2946–2953.
    https://doi.org/10.1109/CVPR.2013.379

Example usage:
  # Visualize original SIFT vectors (no quantization)
  python visualize_sift1m.py --input sift_base.fvecs --n-samples 30000

  # Visualize PQ reconstructed vectors
  python visualize_sift1m.py --input sift_base.fvecs --n-samples 30000 --use pq --pq-m 8
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass
from typing import Optional


# Number of centroids per subspace (fixed)
NUM_CENTROIDS_PER_SUBSPACE = 256


# ----------------------------------------------------------------------
# FAISS .fvecs loader
# ----------------------------------------------------------------------
def read_fvecs(fname: str) -> np.ndarray:
    """
    Read a FAISS .fvecs file and return a float32 numpy array of shape (N, D).

    The .fvecs format is:
      [int32 dim][float32 * dim] repeated N times.
    """
    a = np.fromfile(fname, dtype=np.int32)
    if a.size == 0:
        raise ValueError("Empty file or invalid path.")

    d = a[0]
    if d != 128:
        raise ValueError(f"Unexpected dimension {d}. Expected 128 for SIFT1M data.")

    n = a.size // (d + 1)
    if n * (d + 1) != a.size:
        raise ValueError(f"File size mismatch: n={n}, d={d}, ints={a.size}")

    a = a.reshape(n, d + 1)
    fv = a[:, 1:].view(np.float32)

    return fv.copy()


# ----------------------------------------------------------------------
# Naive Product Quantization implementation
# ----------------------------------------------------------------------
@dataclass
class PQConfig:
    """Configuration for Product Quantization."""
    dim: int   # original dimension D
    M: int     # number of subquantizers


class PQ:
    """
    Naive Product Quantizer implementation.

    - The space R^D is split into M disjoint subspaces of size dsub = D / M.
    - Each subspace has its own codebook of NUM_CENTROIDS_PER_SUBSPACE centroids.
    - A vector x is encoded as M integers (indices of centroids).
    """

    def __init__(self, config: PQConfig, seed=42):
        self.config = config
        self.seed: int = seed
        self.dsub: Optional[int] = None
        # Codebooks: shape (M, NUM_CENTROIDS_PER_SUBSPACE, dsub), filled after fit()
        self.codebooks: Optional[np.ndarray] = None
        self.fitted: bool = False

    def _check_and_init(self, D: int) -> None:
        """
        Check dimensionality consistency and initialize internal buffers.
        """
        if D != self.config.dim:
            raise ValueError(f"Input dim {D} != config.dim {self.config.dim}")
        if D % self.config.M != 0:
            raise ValueError(f"dim={D} must be divisible by M={self.config.M}")
        self.dsub = D // self.config.M
        if self.codebooks is None:
            self.codebooks = np.empty(
                (self.config.M, NUM_CENTROIDS_PER_SUBSPACE, self.dsub),
                dtype=np.float32,
            )

    def fit(self, X: np.ndarray, n_iter: int = 20, verbose: bool = False) -> None:
        """
        Learn subspace codebooks via k-means.

        Parameters
        ----------
        X : np.ndarray, shape (N, D)
            Training vectors in the space to which PQ is applied
            (e.g., rotated vectors if using OPQ).
        n_iter : int
            Max iterations for k-means (per subspace).
        verbose : bool
            If True, print progress messages.
        """
        if X.ndim != 2:
            raise ValueError("X must be a 2D array [N, D]")
        N, D = X.shape
        X = np.asarray(X, dtype=np.float32)

        self._check_and_init(D)
        assert self.dsub is not None
        dsub = self.dsub

        if verbose:
            print(f"[PQ.fit] N={N}, D={D}, M={self.config.M}, "
                  f"dsub={dsub}, Ks={NUM_CENTROIDS_PER_SUBSPACE}")

        # Fit KMeans for each subspace independently
        for m in range(self.config.M):
            if verbose:
                print(f"[PQ.fit] Training subquantizer {m + 1}/{self.config.M}")
            start = m * dsub
            end = (m + 1) * dsub
            X_sub = X[:, start:end]  # (N, dsub)

            kmeans = KMeans(
                n_clusters=NUM_CENTROIDS_PER_SUBSPACE,
                n_init=1,              # one run is enough for a correctness check
                max_iter=n_iter,
                random_state=self.seed,
                verbose=0,
            )
            kmeans.fit(X_sub)
            self.codebooks[m] = kmeans.cluster_centers_.astype(np.float32)

        self.fitted = True

    def encode(self, X: np.ndarray) -> np.ndarray:
        """
        Encode vectors into PQ codes.

        Parameters
        ----------
        X : np.ndarray, shape (N, D)

        Returns
        -------
        codes : np.ndarray, shape (N, M), dtype=int32
        """
        if not self.fitted or self.codebooks is None:
            raise RuntimeError("PQ is not fitted yet.")
        if X.ndim != 2:
            raise ValueError("X must be a 2D array [N, D]")

        N, D = X.shape
        X = np.asarray(X, dtype=np.float32)
        self._check_and_init(D)
        assert self.dsub is not None
        dsub = self.dsub

        codes = np.empty((N, self.config.M), dtype=np.int32)

        # For each subspace, assign the nearest centroid
        for m in range(self.config.M):
            start = m * dsub
            end = (m + 1) * dsub
            X_sub = X[:, start:end]  # (N, dsub)
            centroids = self.codebooks[m]  # (NUM_CENTROIDS_PER_SUBSPACE, dsub)

            # Naive squared distance with broadcasting:
            # dist^2(x, c) = sum((x - c)^2)
            diff = X_sub[:, None, :] - centroids[None, :, :]  # (N, NUM_CENTROIDS_PER_SUBSPACE, dsub)
            dist2 = np.sum(diff * diff, axis=2)               # (N, NUM_CENTROIDS_PER_SUBSPACE)
            codes[:, m] = np.argmin(dist2, axis=1)

        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """
        Decode PQ codes back to reconstructed vectors in the PQ space.

        Parameters
        ----------
        codes : np.ndarray, shape (N, M)

        Returns
        -------
        X_rec : np.ndarray, shape (N, D), dtype=float32
            Reconstructed vectors in the same space where PQ was trained
            (e.g., rotated coordinates if used inside OPQ).
        """
        if not self.fitted or self.codebooks is None:
            raise RuntimeError("PQ is not fitted yet.")
        if codes.ndim != 2:
            raise ValueError("codes must be a 2D array [N, M]")

        N, M = codes.shape
        if M != self.config.M:
            raise ValueError(f"codes.shape[1]={M} != M={self.config.M}")

        assert self.dsub is not None
        dsub = self.dsub
        D = self.config.dim

        X_rec = np.empty((N, D), dtype=np.float32)

        # For each subspace, lookup centroids according to codes[:, m]
        for m in range(self.config.M):
            start = m * dsub
            end = (m + 1) * dsub
            centroids = self.codebooks[m]  # (NUM_CENTROIDS_PER_SUBSPACE, dsub)
            idx = codes[:, m]              # (N,)
            X_rec[:, start:end] = centroids[idx]

        return X_rec


# ----------------------------------------------------------------------
# Non-parametric OPQ
# ----------------------------------------------------------------------
@dataclass
class OPQConfig:
    """Configuration for non-parametric OPQ."""
    dim: int               # original dimension D
    M: int                 # number of subquantizers
    pq_kmeans_iters: int   # iterations for k-means inside PQ
    opq_outer_iters: int   # outer alternating iterations


class OPQ:
    """
    Non-parametric Optimized Product Quantization.

    This class jointly learns:
      - an orthogonal rotation matrix R (D, D)
      - PQ codebooks in the rotated space

    The optimization alternates between:
      (i) PQ training on rotated data RX
      (ii) Procrustes update of R:
            R = argmin_{R^T R = I} ||R X - Y||_F^2
          where Y is the PQ reconstruction of RX.

    Reconstruction to the original space is:
      X_rec ≈ (PQ decode in rotated space) @ R
    """

    def __init__(self, config: OPQConfig):
        self.config = config
        self.pq = PQ(PQConfig(dim=config.dim, M=config.M))
        # Rotation matrix R: initialized as identity (D x D)
        self.R: np.ndarray = np.eye(config.dim, dtype=np.float32)
        self.fitted: bool = False

    def _procrustes_update(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Solve the orthogonal Procrustes problem:
          argmin_R ||R X - Y||_F^2  s.t. R^T R = I

        Parameters
        ----------
        X : np.ndarray, shape (N, D)
            Original data.
        Y : np.ndarray, shape (N, D)
            Current PQ reconstruction in the rotated space.

        Returns
        -------
        R_new : np.ndarray, shape (D, D)
            Updated rotation matrix.
        """
        # We treat X and Y row-wise (N, D). Procrustes formulation uses column-wise,
        # but the equivalent covariance is: covariance = Y^T X.
        cov = Y.T @ X  # (D, D)
        U, _, Vt = np.linalg.svd(cov, full_matrices=True)
        R_new = U @ Vt

        # Optionally enforce det(R) = 1 (proper rotation, avoid reflections).
        if np.linalg.det(R_new) < 0:
            U[:, -1] *= -1.0
            R_new = U @ Vt

        return R_new.astype(np.float32)

    def fit(self, X: np.ndarray, verbose: bool = True) -> None:
        """
        Learn rotation R and PQ codebooks from training data X.

        Parameters
        ----------
        X : np.ndarray, shape (N, D)
            Training data in the original space.
        verbose : bool
            If True, print progress messages and reconstruction errors per iteration.
        """
        if X.ndim != 2:
            raise ValueError("X must be a 2D array [N, D]")
        N, D = X.shape
        if D != self.config.dim:
            raise ValueError(f"Input dim {D} != config.dim {self.config.dim}")
        X = np.asarray(X, dtype=np.float32)

        # Initialize R as identity (could also use PCA-based or OPQP initialization).
        self.R = np.eye(D, dtype=np.float32)

        for it in range(self.config.opq_outer_iters):
            if verbose:
                print(f"[OPQ.fit] Outer iteration {it + 1}/{self.config.opq_outer_iters}")

            # Step (i): PQ training on rotated data
            # ------------------------------------
            # Rotate original data: X_rot = R X
            # In row-major form: X_rot = X @ R^T
            X_rot = X @ self.R.T

            # (i-a) Train PQ codebooks on rotated data
            self.pq.fit(
                X_rot,
                n_iter=self.config.pq_kmeans_iters,
                verbose=verbose,
            )

            # (i-b) Encode and decode to obtain PQ reconstruction Y in rotated space
            codes = self.pq.encode(X_rot)
            Y = self.pq.decode(codes)  # (N, D), rotated-space reconstruction

            # Compute reconstruction error in rotated space
            diff_rot = X_rot - Y
            mse = float(np.mean(np.sum(diff_rot * diff_rot, axis=1)))

            # Step (ii): Procrustes update for R
            # ----------------------------------
            # We want R_new that best maps original X to Y:
            #   R_new = argmin ||R_new X - Y||_F^2
            R_new = self._procrustes_update(X, Y)

            # After updating R, measure reconstruction in original space
            # by mapping Y back: X_rec ≈ Y @ R_new
            X_rec = Y @ R_new
            diff_new = X - X_rec
            mse_new = float(np.mean(np.sum(diff_new * diff_new, axis=1)))

            if verbose:
                print(f"[OPQ.fit] MSE: {mse_new:.6f} (diff={mse_new - mse:.6f})")

            self.R = R_new

        self.fitted = True

    def encode(self, X: np.ndarray) -> np.ndarray:
        """
        Encode data in the original space into OPQ codes.

        Steps:
        1. Rotate X by R: X_rot = X @ R^T
        2. Apply PQ encoding on X_rot.

        Parameters
        ----------
        X : np.ndarray, shape (N, D)

        Returns
        -------
        codes : np.ndarray, shape (N, M)
        """
        if not self.fitted:
            raise RuntimeError("OPQ model is not fitted yet.")
        if X.ndim != 2:
            raise ValueError("X must be a 2D array [N, D]")

        X = np.asarray(X, dtype=np.float32)
        if X.shape[1] != self.config.dim:
            raise ValueError(f"Input dim {X.shape[1]} != config.dim {self.config.dim}")

        X_rot = X @ self.R.T
        codes = self.pq.encode(X_rot)
        return codes

    def decode_rotated(self, codes: np.ndarray) -> np.ndarray:
        """
        Decode OPQ codes into reconstructed vectors in the rotated space.

        Parameters
        ----------
        codes : np.ndarray, shape (N, M)

        Returns
        -------
        Y : np.ndarray, shape (N, D)
            Reconstructed vectors in the rotated coordinate system.
        """
        if not self.fitted:
            raise RuntimeError("OPQ model is not fitted yet.")
        return self.pq.decode(codes)

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """
        Decode OPQ codes into reconstructed vectors in the original space.

        Steps:
        1. Decode codes in rotated space: Y (approx. R X)
        2. Map back to original space by applying R: X_rec ≈ Y @ R

        Parameters
        ----------
        codes : np.ndarray, shape (N, M)

        Returns
        -------
        X_rec : np.ndarray, shape (N, D)
            Approximate reconstruction in the original space.
        """
        Y = self.decode_rotated(codes)
        X_rec = Y @ self.R
        return X_rec

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply the learned rotation R to original data X.

        Parameters
        ----------
        X : np.ndarray, shape (N, D)

        Returns
        -------
        X_rot : np.ndarray, shape (N, D)
            Rotated data, X_rot = X @ R^T.
        """
        if not self.fitted:
            raise RuntimeError("OPQ model is not fitted yet.")
        if X.ndim != 2:
            raise ValueError("X must be a 2D array [N, D]")

        X = np.asarray(X, dtype=np.float32)
        if X.shape[1] != self.config.dim:
            raise ValueError(f"Input dim {X.shape[1]} != config.dim {self.config.dim}")

        return X @ self.R.T


# ----------------------------------------------------------------------
# Main: PCA + Visualization (+ Optional PQ / OPQ)
# ----------------------------------------------------------------------
def main():
    """
    Run PCA and generate the rotating 3D scatter GIF.

    Steps:
    1. Load SIFT1M data from .fvecs.
    2. Randomly subsample for faster processing (and quantizer training).
    3. Depending on --use flag:
        - None: visualize original vectors
        - "pq":  visualize PQ-reconstructed vectors
        - "opq": visualize OPQ-reconstructed vectors
    4. Run PCA to 3D.
    5. Plot and export a rotating 3D scatter plot as a GIF.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize SIFT1M via PCA "
                    "(optionally after PQ or non-parametric OPQ) and save as GIF"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to sift_learn.fvecs or sift_base.fvecs",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=30000,
        help="Number of samples to visualize (randomly drawn from the dataset).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="sift1m_pca.gif",
        help="Output GIF filename.",
    )
    parser.add_argument(
        "--use",
        choices=["pq", "opq"],
        default=None,
        help=(
            "Vector representation to visualize:\n"
            "  pq  : PQ-reconstructed vectors\n"
            "  opq : OPQ-reconstructed vectors\n"
            "If omitted, original vectors are visualized."
        ),
    )
    parser.add_argument(
        "--pq-m",
        type=int,
        default=8,
        help="Number of subquantizers M for PQ/OPQ (dimension must be divisible by M).",
    )
    parser.add_argument(
        "--pq-kmeans-iters",
        type=int,
        default=20,
        help="Max iterations for k-means in PQ training.",
    )
    parser.add_argument(
        "--opq-outer-iters",
        type=int,
        default=10,
        help="Number of outer alternating iterations for OPQ.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print(f"Loading data from: {args.input}")
    X = read_fvecs(args.input)  # shape (N, 128)
    print(f"Loaded shape: {X.shape}")

    # Randomly sample data for faster PCA/plotting (and quantizer training)
    if X.shape[0] > args.n_samples:
        idx = np.random.choice(X.shape[0], args.n_samples, replace=False)
        X = X[idx]
        print(f"Sampled: {X.shape[0]} points")

    # ------------------------------------------------------------------
    # Vector representation selection: original / PQ / OPQ
    # ------------------------------------------------------------------
    if args.use is None:
        X_for_pca = X
        print("Using original vectors for PCA visualization (no quantization).")

    elif args.use == "pq":
        print(f"Applying Product Quantization: M={args.pq_m}, Ks={NUM_CENTROIDS_PER_SUBSPACE}")
        pq_cfg = PQConfig(dim=X.shape[1], M=args.pq_m)
        pq = PQ(pq_cfg)

        # For this visualization script, we simply use the sampled data as training data.
        pq.fit(X, n_iter=args.pq_kmeans_iters, verbose=True)

        print("Encoding with PQ...")
        codes = pq.encode(X)
        print(f"Codes shape: {codes.shape}, dtype={codes.dtype}")

        print("Decoding (reconstructing) from PQ codes...")
        X_rec = pq.decode(codes)

        # Report reconstruction error
        diff = X - X_rec
        mse = float(np.mean(np.sum(diff * diff, axis=1)))
        print(f"PQ reconstruction MSE (mean squared L2 error): {mse:.6f}")

        X_for_pca = X_rec
        print("Using PQ-reconstructed vectors for PCA visualization.")

    elif args.use == "opq":
        print(f"Applying non-parametric OPQ: M={args.pq_m}, Ks={NUM_CENTROIDS_PER_SUBSPACE}, "
              f"outer_iters={args.opq_outer_iters}")
        opq_cfg = OPQConfig(
            dim=X.shape[1],
            M=args.pq_m,
            pq_kmeans_iters=args.pq_kmeans_iters,
            opq_outer_iters=args.opq_outer_iters,
        )
        opq = OPQ(opq_cfg)

        # Fit OPQ on the sampled data
        opq.fit(X, verbose=True)

        print("Encoding with OPQ...")
        codes = opq.encode(X)
        print(f"Codes shape: {codes.shape}, dtype={codes.dtype}")

        print("Decoding (reconstructing) from OPQ codes...")
        X_rec = opq.decode(codes)

        # Report reconstruction error in original space
        diff = X - X_rec
        mse = float(np.mean(np.sum(diff * diff, axis=1)))
        print(f"OPQ reconstruction MSE (mean squared L2 error, original space): {mse:.6f}")

        X_for_pca = X_rec
        print("Using OPQ-reconstructed vectors for PCA visualization.")

    else:
        raise ValueError(f"Unknown --use option: {args.use}")

    # ------------------------------------------------------------------
    # PCA to 3D (on original or reconstructed vectors)
    # ------------------------------------------------------------------
    print("Running PCA (n_components=3)...")
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_for_pca)
    print("Explained variance ratio:", pca.explained_variance_ratio_)

    # ------------------------------------------------------------------
    # 3D scatter plot + rotating GIF
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    if args.use is None:
        title_suffix = " (original)"
    elif args.use == "pq":
        title_suffix = f" (PQ/M={args.pq_m})"
    else:
        title_suffix = f" (OPQ/M={args.pq_m})"

    ax.set_title(f"SIFT1M PCA Projection {title_suffix}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")

    scatter = ax.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        X_pca[:, 2],
        s=1,
        alpha=0.35,
        c="steelblue",
    )

    def update(frame: int):
        """
        Update function for the animation; rotates the azimuth angle.
        """
        ax.view_init(elev=20, azim=frame)
        return scatter,

    frames = 120  # number of frames for a full rotation
    ani = FuncAnimation(fig, update, frames=frames, interval=100, blit=True)

    print(f"Saving GIF to {args.output} ...")
    ani.save(args.output, writer="pillow", fps=15)
    print("Done!")


if __name__ == "__main__":
    main()
