#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize SIFT1M dataset using PCA and export a rotating 3D scatter plot as a GIF.

- Loads SIFT1M data stored in FAISS .fvecs format: [int32 dim][float32 * dim] repeated
- Performs PCA to reduce the dimensionality to 3D
- Plots the PCA-transformed data as a 3D scatter plot
- Exports it as an animated GIF file

Note:
  This script is inspired by the observation "... the SIFT1M set has two distinct clusters (this can be visualized by the first two principal components), ..." in:
    Tiezheng Ge, Kaiming He, Qifa Ke, and Jian Sun.
    "Optimized Product Quantization for Approximate Nearest Neighbor Search."
    In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2013, pp. 2946–2953.
    https://doi.org/10.1109/CVPR.2013.379

Usage:
  python visualize_sift1m.py --input sift_base.fvecs --n-samples 30000
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.animation import FuncAnimation


def read_fvecs(fname):
    """Read FAISS .fvecs file and return a float32 numpy array."""
    a = np.fromfile(fname, dtype=np.int32)
    if a.size == 0:
        raise ValueError("Empty file or invalid path.")

    if a[0] != 128:
        raise ValueError(f"Unexpected dimension {d}. Expected 128 for SIFT1M data.")

    n = a.size // (a[0] + 1)
    if n * (a[0] + 1) != a.size:
        raise ValueError(f"File size mismatch: n={n}, d={d}, ints={a.size}")

    a = a.reshape(n, a[0] + 1)
    fv = a[:, 1:].view(np.float32)

    return fv.copy()


def main():
    """Run PCA and generate the rotating 3D scatter GIF."""
    import argparse
    parser = argparse.ArgumentParser(description="Visualize SIFT1M via PCA and save as GIF")
    parser.add_argument("--input", required=True, help="Path to sift_learn.fvecs or sift_base.fvecs")
    parser.add_argument("--n-samples", type=int, default=30000, help="Number of samples to visualize")
    parser.add_argument("--output", type=str, default="sift1m_pca.gif", help="Output GIF filename")
    args = parser.parse_args()

    print(f"Loading data from: {args.input}")
    X = read_fvecs(args.input)
    print(f"Loaded shape: {X.shape}")

    # Randomly sample data for faster PCA/plotting
    if X.shape[0] > args.n_samples:
        idx = np.random.choice(X.shape[0], args.n_samples, replace=False)
        X = X[idx]
        print(f"Sampled: {X.shape[0]} points")

    # PCA to 3D
    print("Running PCA (n_components=3)...")
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)
    print("Explained variance ratio:", pca.explained_variance_ratio_)

    # 3D scatter plot setup
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("SIFT1M PCA Projection (3D)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], s=1, alpha=0.35, c="steelblue")

    # Animation: rotate azimuth angle to create a smooth 360° view
    def update(frame):
        ax.view_init(elev=20, azim=frame)
        return scatter,

    frames = 120  # number of frames for a full rotation
    ani = FuncAnimation(fig, update, frames=frames, interval=100, blit=True)

    print(f"Saving GIF to {args.output} ...")
    ani.save(args.output, writer="pillow", fps=15)
    print("Done!")


if __name__ == "__main__":
    main()
