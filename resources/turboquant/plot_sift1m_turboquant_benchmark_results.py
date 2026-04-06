#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

def plot_results():
    # vanilla
    recall_vanilla = [0.833, 0.920, 0.971, 0.992, 0.997, 0.999, 1.000, 1.000]
    blocks_vanilla = [1528.08, 2269.80, 3456.58, 5850.28, 8457.66, 14005.32, 26703.02, 49756.98]

    # k=1
    recall_tq_k1 = [0.4277, 0.5715, 0.7072, 0.8182, 0.8700, 0.9207, 0.9652, 0.9876]
    blocks_tq_k1 = [171.260, 263.660, 457.900, 836.380, 1192.800, 1920.320, 3717.140, 7339.960]

    # k=3
    recall_tq_k3 = [0.6033, 0.7354, 0.8422, 0.9170, 0.9469, 0.9727, 0.9912, 0.9977]
    blocks_tq_k3 = [264.940, 428.880, 701.900, 1182.260, 1684.720, 2654.940, 5097.560, 10011.740]

    # k=5
    recall_tq_k5 = [0.6644, 0.7876, 0.8810, 0.9437, 0.9670, 0.9842, 0.9954, 0.9990]
    blocks_tq_k5 = [357.700, 524.400, 855.400, 1448.400, 2039.620, 3259.900, 6308.620, 12385.500]

    # k=7
    recall_tq_k7 = [0.6968, 0.8182, 0.9036, 0.9581, 0.9774, 0.9900, 0.9973, 0.9995]
    blocks_tq_k7 = [445.440, 654.900, 1029.080, 1743.840, 2486.940, 3959.380, 7677.680, 15157.040]

    # Plot results
    plt.plot(recall_vanilla, blocks_vanilla, marker='s', color="black", linewidth=2, label="vanilla pgvector")
    plt.plot(recall_tq_k1, blocks_tq_k1, marker='D', color="#FCBBA1", label="w/TurboQuant (k=1)", linewidth=1.8, markersize=7)
    plt.plot(recall_tq_k3, blocks_tq_k3, marker='v', color="#FB6A4A", label="w/TurboQuant (k=3)", linewidth=1.8, markersize=7)
    plt.plot(recall_tq_k5, blocks_tq_k5, marker='^', color="#CB181D", label="w/TurboQuant (k=5)", linewidth=1.8, markersize=7)
    plt.plot(recall_tq_k7, blocks_tq_k7, marker='o', color="#67000D", label="w/TurboQuant (k=7)", linewidth=1.8, markersize=7)

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
