#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

def plot_results():
    # vanilla
    recall_vanilla = [0.833, 0.920, 0.971, 0.992, 0.997, 0.999, 1.000, 1.000]
    blocks_vanilla = [1528.08, 2269.80, 3456.58, 5850.28, 8457.66, 14005.32, 26703.02, 49756.98]

    # k=3
    recall_tq_k3 = [0.1347, 0.221, 0.3314, 0.45059, 0.5244, 0.6153, 0.7245, 0.8175]
    blocks_tq_k3 = [238.020, 393.300, 654.520, 1209.140, 1719.840, 2616.260, 4949.980, 9380.260]

    # k=5
    recall_tq_k5 = []
    blocks_tq_k5 = []

    # k=7
    recall_tq_k7 = []
    blocks_tq_k7 = []

    # Plot results
    plt.plot(recall_vanilla, blocks_vanilla, marker='s', color="black", linewidth=2, label="vanilla pgvector")
    plt.plot(recall_tq_k3, blocks_tq_k3, marker='v', color="#00441B", label="w/TurboQuant (k=3)", linewidth=1.8, markersize=7)
    plt.plot(recall_tq_k5, blocks_tq_k5, marker='^', color="#238B45", label="w/TurboQuant (k=5)", linewidth=1.8, markersize=7)
    plt.plot(recall_tq_k7, blocks_tq_k7, marker='o', color="#74C476", label="w/TurboQuant (k=7)", linewidth=1.8, markersize=7)

    plt.xlabel("Recall")
    plt.ylabel("#Blocks")
    plt.yscale("log")
    plt.title("Recall-#Blocks tradeoff (SIFT1M,10-NN,m=24,ef_construction=200)")
    plt.legend()
    plt.grid(True, which="both")

    # Save to file
    plt.savefig("sift1m_recall_blocks_tradeoff.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    plot_results()
